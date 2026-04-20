from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import curve_fit
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------

class CoresetSelection(ABC):

    def __init__(self, coreset_fraction):
        self.coreset_fraction = coreset_fraction

    @abstractmethod
    def select_coreset(self, dataset) -> list[int]:
        pass


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class FullDatasetSelection(CoresetSelection):
    def __init__(self, coreset_fraction=1.0):
        super().__init__(coreset_fraction)

    def select_coreset(self, dataset):
        return list(range(len(dataset)))


class RandomCoresetSelection(CoresetSelection):
    def __init__(self, coreset_fraction):
        super().__init__(coreset_fraction)

    def select_coreset(self, dataset):
        np.random.seed(0)
        return np.random.choice(
            len(dataset), int(self.coreset_fraction * len(dataset)), replace=False
        ).tolist()


# ---------------------------------------------------------------------------
# MRMC helpers (adapted for tabular / MLP, CPU-friendly)
# ---------------------------------------------------------------------------

def _neg_exp_model(r, q, w):
    return q * np.power(w, -r)


def _fit_mrmc_scores(loss_sequences):
    """Fit neg-exp curve per sample; return MRMC criterion scores (N,)."""
    N, R = loss_sequences.shape
    epochs = np.arange(1, R + 1, dtype=float)
    scores = np.zeros(N, dtype=float)
    for i in tqdm(range(N), desc="Computing MRMC scores", unit="sample"):
        L = loss_sequences[i]
        try:
            p0 = [max(L[0], 1e-6), max(1.01, L[0] / max(L[-1], 1e-9))]
            popt, _ = curve_fit(
                _neg_exp_model, epochs, L,
                p0=p0,
                bounds=([0, 1.0], [np.inf, np.inf]),
                maxfev=2000,
            )
            q, w = popt
            scores[i] = q * (1.0 - w ** (-R))
        except (RuntimeError, ValueError):
            scores[i] = max(L[0] - L[-1], 0.0)
    return scores


def _extract_features(model, loader, device, penultimate_layer_name=None):
    """
    Extract features from a named layer (or the model output if None).
    Works with any nn.Module — no ResNet-specific assumptions.
    """
    model.eval()
    all_feats, all_labels = [], []
    feature_map = {}

    def hook_fn(_module, _input, output):
        feature_map['feat'] = output.detach()

    handle = None
    if penultimate_layer_name is not None:
        layer = dict(model.named_modules())[penultimate_layer_name]
        handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            out = model(inputs)
            feats = feature_map['feat'] if penultimate_layer_name is not None else out
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    if handle is not None:
        handle.remove()
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def _train_proxy(features, labels, num_classes, device, lr=0.01, epochs=10):
    """Train a single linear layer on the initial MRMC subset C'."""
    feat_dim = features.shape[1]
    proxy = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = optim.SGD(proxy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, labels), batch_size=64, shuffle=True)
    proxy.train()
    for _ in range(epochs):
        for feat, lab in loader:
            feat, lab = feat.to(device), lab.to(device)
            optimizer.zero_grad()
            criterion(proxy(feat), lab).backward()
            optimizer.step()
    return proxy


def _stratified_top_k(scores, labels, k):
    """Select k indices preserving the class distribution of the full dataset."""
    indices = []
    classes, counts = np.unique(labels, return_counts=True)
    class_quota = np.round((counts / len(labels)) * k).astype(int)
    diff = k - class_quota.sum()
    class_quota[np.argmax(counts)] += diff
    for cls, quota in zip(classes, class_quota):
        cls_idx = np.where(labels == cls)[0]
        top = cls_idx[np.argsort(scores[cls_idx])[::-1][:quota]]
        indices.extend(top.tolist())
    return indices


def _run_mrmc_selection(
    model_fn,
    dataset,
    coreset_size,
    device,
    R,
    rho,
    gamma,
    num_classes,
    penultimate_layer_name,
    batch_size,
    lr,
    stratified=False,
):
    """Core MRMC selection logic (Algorithm 1), adapted for tabular data."""
    N = len(dataset)
    model = model_fn()
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ordered_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_sequences = np.zeros((N, R), dtype=np.float32)

    # --- Phase 1: warm-up training + loss collection ---
    for r in range(R):
        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(inputs), labels).mean().backward()
            optimizer.step()

        model.eval()
        idx = 0
        with torch.no_grad():
            for inputs, labels in ordered_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                losses = criterion(model(inputs), labels).cpu().numpy()
                bs = len(losses)
                loss_sequences[idx:idx + bs, r] = losses
                idx += bs
        # print(f"  Warm-up epoch {r+1}/{R}  mean_loss={loss_sequences[:, r].mean():.4f}")

    # --- Phase 2: MRMC scores ---
    mrmc_scores = _fit_mrmc_scores(loss_sequences)

    all_labels_np = dataset.tensors[1].numpy()

    def _select(scores, indices, k):
        if stratified:
            return _stratified_top_k(scores, all_labels_np[indices], k)
        return np.argsort(scores)[::-1][:k].tolist()

    # --- Phase 3: pure MRMC (rho=1) ---
    if rho >= 1.0:
        all_indices = np.arange(N)
        selected = _select(mrmc_scores, all_indices, coreset_size)
        return [int(all_indices[i]) for i in selected] if stratified else [int(i) for i in np.argsort(mrmc_scores)[::-1][:coreset_size]]

    # --- Phase 4: MRMC-R (regularized) ---
    initial_size = max(1, int(round(rho * coreset_size)))
    remaining_size = coreset_size - initial_size

    all_indices = np.arange(N)
    c_prime_local = _select(mrmc_scores, all_indices, initial_size)
    c_prime_indices = [int(all_indices[i]) for i in c_prime_local] if stratified else [int(i) for i in np.argsort(mrmc_scores)[::-1][:initial_size]]
    c_prime_set = set(c_prime_indices)
    remaining_pool = [i for i in range(N) if i not in c_prime_set]

    print("Extracting features for proxy model...")
    all_features, all_labels = _extract_features(model, ordered_loader, device, penultimate_layer_name)

    print("Training proxy model...")
    proxy = _train_proxy(
        all_features[c_prime_indices],
        all_labels[c_prime_indices],
        num_classes, device,
    )

    pool_features = all_features[remaining_pool]
    pool_labels = all_labels[remaining_pool]

    proxy.eval()
    reg_losses = []
    ce = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for start in range(0, len(pool_features), 256):
            feat = pool_features[start:start + 256].to(device)
            lab = pool_labels[start:start + 256].to(device)
            reg_losses.extend(ce(proxy(feat), lab).cpu().numpy())
    reg_scores = np.exp(-np.array(reg_losses))

    pool_indices = np.array(remaining_pool)
    combined = mrmc_scores[pool_indices] - gamma * reg_scores
    top_local = _select(combined, pool_indices, remaining_size)
    top_remaining = [int(pool_indices[i]) for i in top_local] if stratified else [int(pool_indices[i]) for i in np.argsort(combined)[::-1][:remaining_size]]

    return c_prime_indices + top_remaining


# ---------------------------------------------------------------------------
# MRMC Selectors
# ---------------------------------------------------------------------------

class MRMCOriginalCoresetSelection(CoresetSelection):
    """
    MRMC coreset selection (Algorithm 1) adapted for tabular datasets.

    Args:
        model_fn:               callable with no args that returns a fresh nn.Module
        penultimate_layer_name: name of the layer to extract features from for MRMC-R
                                (e.g. 'fc2' for WineMLP). Only used when rho < 1.
    """
    def __init__(self, coreset_fraction, R, rho, gamma, model_fn, device,
                 penultimate_layer_name=None, batch_size=64, lr=0.001, num_classes=None):
        super().__init__(coreset_fraction)
        self.R = R
        self.rho = rho
        self.gamma = gamma
        self.model_fn = model_fn
        self.device = device
        self.penultimate_layer_name = penultimate_layer_name
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes

    def select_coreset(self, dataset):
        coreset_size = int(self.coreset_fraction * len(dataset))
        num_classes = self.num_classes
        if num_classes is None:
            # infer from TensorDataset labels
            num_classes = int(dataset.tensors[1].max().item()) + 1
        return _run_mrmc_selection(
            model_fn=self.model_fn,
            dataset=dataset,
            coreset_size=coreset_size,
            device=self.device,
            R=self.R,
            rho=self.rho,
            gamma=self.gamma,
            num_classes=num_classes,
            penultimate_layer_name=self.penultimate_layer_name,
            batch_size=self.batch_size,
            lr=self.lr,
        )


class MRMCOriginalStratifiedCoresetSelection(CoresetSelection):
    """
    MRMC coreset selection with stratified top-k selection.
    Preserves the original class distribution in the coreset, which prevents
    class-imbalanced datasets from producing a biased selection.
    """
    def __init__(self, coreset_fraction, R, rho, gamma, model_fn, device,
                 penultimate_layer_name=None, batch_size=64, lr=0.001, num_classes=None):
        super().__init__(coreset_fraction)
        self.R = R
        self.rho = rho
        self.gamma = gamma
        self.model_fn = model_fn
        self.device = device
        self.penultimate_layer_name = penultimate_layer_name
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes

    def select_coreset(self, dataset):
        coreset_size = int(self.coreset_fraction * len(dataset))
        num_classes = self.num_classes
        if num_classes is None:
            num_classes = int(dataset.tensors[1].max().item()) + 1
        return _run_mrmc_selection(
            model_fn=self.model_fn,
            dataset=dataset,
            coreset_size=coreset_size,
            device=self.device,
            R=self.R,
            rho=self.rho,
            gamma=self.gamma,
            num_classes=num_classes,
            penultimate_layer_name=self.penultimate_layer_name,
            batch_size=self.batch_size,
            lr=self.lr,
            stratified=True,
        )


def _run_mrmc_kmeans_selection(
    model_fn, dataset, coreset_size, device,
    R, penultimate_layer_name, batch_size, lr,
):
    N = len(dataset)
    model = model_fn()
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ordered_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_sequences = np.zeros((N, R), dtype=np.float32)

    for r in range(R):
        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(inputs), labels).mean().backward()
            optimizer.step()

        model.eval()
        idx = 0
        with torch.no_grad():
            for inputs, labels in ordered_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                losses = criterion(model(inputs), labels).cpu().numpy()
                loss_sequences[idx:idx + len(losses), r] = losses
                idx += len(losses)
        print(f"  Warm-up epoch {r+1}/{R}  mean_loss={loss_sequences[:, r].mean():.4f}")

    mrmc_scores = _fit_mrmc_scores(loss_sequences)

    print("Extracting features for K-Means clustering...")
    all_features, _ = _extract_features(model, ordered_loader, device, penultimate_layer_name)
    all_features_np = all_features.numpy()
    all_labels_np = dataset.tensors[1].numpy()

    classes, counts = np.unique(all_labels_np, return_counts=True)
    class_quota = np.round((counts / N) * coreset_size).astype(int)
    class_quota[np.argmax(counts)] += coreset_size - class_quota.sum()

    selected = []
    for cls, quota in tqdm(zip(classes, class_quota), total=len(classes), desc="K-Means per class"):
        cls_indices = np.where(all_labels_np == cls)[0]
        cls_features = all_features_np[cls_indices]
        cls_scores = mrmc_scores[cls_indices]

        km = MiniBatchKMeans(n_clusters=quota, random_state=0,
                             batch_size=min(10_000, len(cls_indices)), n_init=3)
        cluster_ids = km.fit_predict(cls_features)

        for c in range(quota):
            mask = cluster_ids == c
            if not mask.any():
                continue
            best_local = int(np.argmax(cls_scores[mask]))
            selected.append(int(cls_indices[np.where(mask)[0][best_local]]))

    return selected


class MRMCKMeansCoresetSelection(CoresetSelection):
    """
    MRMC + K-Means diversity coreset selection.

    Clusters each class's feature space into K groups (MiniBatchKMeans on
    penultimate-layer features), then picks the highest-MRMC sample per cluster.
    This combines spatial coverage with informativeness, fixing the redundancy
    problem of vanilla MRMC where hard samples cluster near the same boundaries.
    """
    def __init__(self, coreset_fraction, R, model_fn, device,
                 penultimate_layer_name=None, batch_size=64, lr=0.001):
        super().__init__(coreset_fraction)
        self.R = R
        self.model_fn = model_fn
        self.device = device
        self.penultimate_layer_name = penultimate_layer_name
        self.batch_size = batch_size
        self.lr = lr

    def select_coreset(self, dataset):
        coreset_size = int(self.coreset_fraction * len(dataset))
        return _run_mrmc_kmeans_selection(
            model_fn=self.model_fn,
            dataset=dataset,
            coreset_size=coreset_size,
            device=self.device,
            R=self.R,
            penultimate_layer_name=self.penultimate_layer_name,
            batch_size=self.batch_size,
            lr=self.lr,
        )


# ---------------------------------------------------------------------------
# MRMC + Typicality
# ---------------------------------------------------------------------------

def _compute_typicality_scores(features_np, labels_np):
    """
    Per-sample typicality = exp(-normalized_dist_to_class_centroid).

    Features are z-score normalized per dimension first so that no single
    feature dominates the distance.  Higher score = more representative of
    the class cluster (i.e. the sample is near the class mean).
    """
    # z-score normalize feature columns
    mu = features_np.mean(axis=0)
    sigma = features_np.std(axis=0) + 1e-8
    normed = (features_np - mu) / sigma

    classes = np.unique(labels_np)
    typicality = np.zeros(len(labels_np), dtype=np.float32)
    for cls in classes:
        mask = labels_np == cls
        cls_feats = normed[mask]
        centroid = cls_feats.mean(axis=0)
        dists = np.linalg.norm(cls_feats - centroid, axis=1)
        # scale by within-class std so the exponent is unit-free
        scale = dists.std() + 1e-8
        typicality[mask] = np.exp(-dists / scale)
    return typicality


def _run_mrmc_typicality_selection(
    model_fn, dataset, coreset_size, device,
    R, penultimate_layer_name, batch_size, lr, alpha=0.5,
):
    """MRMC warm-up then typicality-weighted stratified selection."""
    N = len(dataset)
    model = model_fn()
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ordered_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_sequences = np.zeros((N, R), dtype=np.float32)

    for r in range(R):
        model.train()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(inputs), labels).mean().backward()
            optimizer.step()

        model.eval()
        idx = 0
        with torch.no_grad():
            for inputs, labels in ordered_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                losses = criterion(model(inputs), labels).cpu().numpy()
                loss_sequences[idx:idx + len(losses), r] = losses
                idx += len(losses)
        print(f"  Warm-up epoch {r+1}/{R}  mean_loss={loss_sequences[:, r].mean():.4f}")

    mrmc_scores = _fit_mrmc_scores(loss_sequences)

    print("Extracting features for typicality scoring...")
    all_features, _ = _extract_features(model, ordered_loader, device, penultimate_layer_name)
    all_features_np = all_features.numpy()
    all_labels_np = dataset.tensors[1].numpy()

    typicality_scores = _compute_typicality_scores(all_features_np, all_labels_np)

    def _norm01(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-8)

    # Multiply informativeness by typicality^alpha.
    # alpha=0 → pure MRMC; alpha=1 → equal geometric weight on both.
    combined = _norm01(mrmc_scores) * (_norm01(typicality_scores) ** alpha)

    return _stratified_top_k(combined, all_labels_np, coreset_size)


class MRMCTypicalityCoresetSelection(CoresetSelection):
    """
    MRMC + Typicality coreset selection.

    Uses the same learning-dynamics warm-up as vanilla MRMC, but re-weights
    each sample's score by how *typical* it is of its class in penultimate-layer
    feature space:

        score_i = norm(mrmc_i) * norm(typicality_i)^alpha

    where typicality_i = exp(-dist_to_class_centroid / within_class_std).

    Motivation for reducing overfitting
    ------------------------------------
    Vanilla MRMC preferentially selects the hardest samples.  Many of those
    samples are hard because they are outliers or carry noisy labels — not
    because they lie near a meaningful decision boundary.  Training on such
    samples causes the model to memorize noise rather than learn structure.

    By down-weighting atypical hard samples we steer selection toward samples
    that are *informative and representative*, reducing the chance of fitting
    to edge-case noise while retaining the benefit of difficulty-based scoring.

    Args:
        alpha: typicality exponent in [0, 1].
               0 → pure MRMC (no typicality penalty).
               1 → equal multiplicative weight on MRMC and typicality.
               Default 0.5 balances informativeness and representativeness.
    """
    def __init__(self, coreset_fraction, R, model_fn, device,
                 penultimate_layer_name=None, batch_size=64, lr=0.001, alpha=0.5):
        super().__init__(coreset_fraction)
        self.R = R
        self.model_fn = model_fn
        self.device = device
        self.penultimate_layer_name = penultimate_layer_name
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha

    def select_coreset(self, dataset):
        coreset_size = int(self.coreset_fraction * len(dataset))
        return _run_mrmc_typicality_selection(
            model_fn=self.model_fn,
            dataset=dataset,
            coreset_size=coreset_size,
            device=self.device,
            R=self.R,
            penultimate_layer_name=self.penultimate_layer_name,
            batch_size=self.batch_size,
            lr=self.lr,
            alpha=self.alpha,
        )


class MRMCAdaptiveCoresetSelection(CoresetSelection):
    def __init__(self, coreset_fraction, R, rho, gamma, model_fn, device,
                 penultimate_layer_name=None, batch_size=64, lr=0.001, num_classes=None):
        super().__init__(coreset_fraction)
        self.R = R
        self.rho = rho
        self.gamma = gamma
        self.model_fn = model_fn
        self.device = device
        self.penultimate_layer_name = penultimate_layer_name
        self.batch_size = batch_size
        self.lr = lr
        self.num_classes = num_classes

    def select_coreset(self, dataset):
        raise NotImplementedError("MRMC Adaptive Coreset Selection not implemented yet")
