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
    def __init__(self, coreset_fraction, seed=0):
        super().__init__(coreset_fraction)
        self.seed = seed

    def select_coreset(self, dataset):
        rng = np.random.default_rng(self.seed)
        return rng.choice(
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
    loader = DataLoader(TensorDataset(features, labels), batch_size=256, shuffle=True)
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


def _score_stratified_select(
    scores,
    labels,
    k,
    num_bins=5,
    prune_bottom=0.1,
    prune_top=0.0,
    per_class=True,
    rng=None,
):
    """
    CCS-style score-stratified coreset selection.

    Bins MRMC scores into equal-population strata and samples uniformly from
    each stratum. This fixes the pathology where pure top-k concentrates on
    the hardest samples (often noisy/ambiguous) and ignores the easy
    distribution — a known failure mode at aggressive pruning ratios.

    Args:
        scores:        (N,) MRMC (or any) importance score, higher = harder.
        labels:        (N,) class labels, used when per_class=True.
        k:             total coreset size.
        num_bins:      number of equal-population score bins.
        prune_bottom:  drop this fraction of lowest-score samples before
                       binning (trivial/easy samples the model already got).
        prune_top:     drop this fraction of highest-score samples before
                       binning (typically noisy / mislabeled outliers).
        per_class:     if True, bin within each class independently, so class
                       balance is preserved alongside score coverage.
        rng:           np.random.Generator (seeded for reproducibility).
    Returns:
        list[int] of selected indices.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    def _stratify_pool(pool_idx, quota):
        """Bin pool_idx by score quantile, sample `quota` uniformly across bins."""
        if quota <= 0 or len(pool_idx) == 0:
            return []
        pool_scores = scores[pool_idx]
        # prune low/high score tails
        order = np.argsort(pool_scores)
        lo = int(prune_bottom * len(order))
        hi = len(order) - int(prune_top * len(order))
        kept_local = order[lo:hi]
        kept_idx = pool_idx[kept_local]
        kept_scores = pool_scores[kept_local]
        if len(kept_idx) <= quota:
            return kept_idx.tolist()

        # equal-population bins via quantiles
        B = min(num_bins, len(kept_idx))
        quantiles = np.quantile(kept_scores, np.linspace(0, 1, B + 1))
        quantiles[-1] += 1e-9  # include top edge
        bin_assign = np.digitize(kept_scores, quantiles[1:-1])

        # allocate quota evenly across bins, remainder → random bins
        base = quota // B
        remainder = quota - base * B
        per_bin = np.full(B, base, dtype=int)
        if remainder:
            per_bin[rng.choice(B, size=remainder, replace=False)] += 1

        chosen = []
        leftover = 0
        for b in range(B):
            members = np.where(bin_assign == b)[0]
            want = per_bin[b] + leftover
            if len(members) <= want:
                chosen.extend(kept_idx[members].tolist())
                leftover = want - len(members)
            else:
                pick = rng.choice(members, size=want, replace=False)
                chosen.extend(kept_idx[pick].tolist())
                leftover = 0
        # if we still owe samples (bins ran thin), fill from remaining kept pool
        if len(chosen) < quota:
            remaining = np.setdiff1d(kept_idx, np.array(chosen, dtype=kept_idx.dtype))
            extra = rng.choice(remaining, size=quota - len(chosen), replace=False)
            chosen.extend(extra.tolist())
        return chosen[:quota]

    if not per_class:
        return _stratify_pool(np.arange(len(scores)), k)

    classes, counts = np.unique(labels, return_counts=True)
    class_quota = np.round((counts / len(labels)) * k).astype(int)
    class_quota[np.argmax(counts)] += k - class_quota.sum()

    selected = []
    for cls, quota in zip(classes, class_quota):
        cls_pool = np.where(labels == cls)[0]
        selected.extend(_stratify_pool(cls_pool, int(quota)))
    return selected


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
        print(f"  Warm-up epoch {r+1}/{R}  mean_loss={loss_sequences[:, r].mean():.4f}")

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
        indices = [int(all_indices[i]) for i in selected] if stratified else [int(i) for i in np.argsort(mrmc_scores)[::-1][:coreset_size]]
        return indices, mrmc_scores

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

    return c_prime_indices + top_remaining, mrmc_scores


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
        indices, mrmc_scores = _run_mrmc_selection(
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
        self.last_scores = mrmc_scores
        return indices


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
        indices, mrmc_scores = _run_mrmc_selection(
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
        self.last_scores = mrmc_scores
        return indices


def _run_mrmc_kmeans_selection(
    model_fn, dataset, coreset_size, device,
    R, penultimate_layer_name, batch_size, lr,
    normalize_features=False,
    alpha=1.0,
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
    if normalize_features:
        all_features_np = all_features_np / (np.linalg.norm(all_features_np, axis=1, keepdims=True) + 1e-8)
    all_labels_np = dataset.tensors[1].numpy()

    classes, counts = np.unique(all_labels_np, return_counts=True)
    class_quota = np.round((counts / N) * coreset_size).astype(int)
    class_quota[np.argmax(counts)] += coreset_size - class_quota.sum()

    selected = []
    for cls, quota in tqdm(zip(classes, class_quota), total=len(classes), desc="K-Means per class"):
        if quota == 0:
            continue
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
            member_indices = np.where(mask)[0]
            if alpha >= 1.0:
                best_local = int(np.argmax(cls_scores[mask]))
            else:
                dists = np.linalg.norm(cls_features[mask] - km.cluster_centers_[c], axis=1)
                proximity = 1.0 / (dists + 1e-8)

                mrmc_vals = cls_scores[mask]
                mrmc_norm = (mrmc_vals - mrmc_vals.min()) / (mrmc_vals.ptp() + 1e-8)
                prox_norm = (proximity - proximity.min()) / (proximity.ptp() + 1e-8)

                blended = alpha * mrmc_norm + (1.0 - alpha) * prox_norm
                best_local = int(np.argmax(blended))
            selected.append(int(cls_indices[member_indices[best_local]]))

    return selected, mrmc_scores


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
        indices, mrmc_scores = _run_mrmc_kmeans_selection(
            model_fn=self.model_fn,
            dataset=dataset,
            coreset_size=coreset_size,
            device=self.device,
            R=self.R,
            penultimate_layer_name=self.penultimate_layer_name,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        self.last_scores = mrmc_scores
        return indices


class MRMCKMeansNormalizedCoresetSelection(CoresetSelection):
    """
    MRMC + K-Means with L2-normalized features before clustering.

    Identical to MRMCKMeansCoresetSelection but normalizes penultimate-layer
    features to unit length before running MiniBatchKMeans. This prevents
    high-variance dimensions from dominating cluster assignments and improves
    diversity when feature magnitudes vary across the dataset.
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
        indices, mrmc_scores = _run_mrmc_kmeans_selection(
            model_fn=self.model_fn,
            dataset=dataset,
            coreset_size=coreset_size,
            device=self.device,
            R=self.R,
            penultimate_layer_name=self.penultimate_layer_name,
            batch_size=self.batch_size,
            lr=self.lr,
            normalize_features=True,
        )
        self.last_scores = mrmc_scores
        return indices


class MRMCKMeansBlendedCoresetSelection(CoresetSelection):
    """
    MRMC + K-Means with a blended per-cluster selection score.

    Within each cluster, selects the sample that maximizes:
        alpha * norm_mrmc_score + (1 - alpha) * norm_proximity_score

    where proximity = 1 / (distance_to_centroid + eps).

    alpha=1.0  → pure MRMC (same as MRMCKMeansCoresetSelection)
    alpha=0.0  → pure centroid-closest (maximum representativeness)
    alpha=0.5  → balanced informativeness + coverage

    Setting alpha < 1 reduces the bias toward hard/borderline samples that
    causes vanilla MRMC-KMeans to underperform random selection.
    """
    def __init__(self, coreset_fraction, R, model_fn, device,
                 penultimate_layer_name=None, batch_size=64, lr=0.001,
                 alpha=0.5, normalize_features=True):
        super().__init__(coreset_fraction)
        self.R = R
        self.model_fn = model_fn
        self.device = device
        self.penultimate_layer_name = penultimate_layer_name
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.normalize_features = normalize_features

    def select_coreset(self, dataset):
        coreset_size = int(self.coreset_fraction * len(dataset))
        indices, mrmc_scores = _run_mrmc_kmeans_selection(
            model_fn=self.model_fn,
            dataset=dataset,
            coreset_size=coreset_size,
            device=self.device,
            R=self.R,
            penultimate_layer_name=self.penultimate_layer_name,
            batch_size=self.batch_size,
            lr=self.lr,
            normalize_features=self.normalize_features,
            alpha=self.alpha,
        )
        self.last_scores = mrmc_scores
        return indices


class MRMCScoreStratifiedCoresetSelection(CoresetSelection):
    """
    MRMC + CCS-style score-stratified selection.

    Instead of picking the top-k highest MRMC scores (which concentrate on the
    hardest / most ambiguous samples at aggressive pruning ratios), this bins
    MRMC scores into equal-population strata and samples uniformly across bins.
    Combines informativeness (MRMC score signal) with coverage across the
    difficulty spectrum.

    Addresses the paper's stated limitation that MRMC's optimization term
    -Σ Δl_j implicitly assumes a constant initial loss across samples — an
    assumption that breaks on datasets where some samples are inherently
    easier than others (e.g. MNIST digits 0/1 vs 4/9).
    """
    def __init__(self, coreset_fraction, R, model_fn, device,
                 penultimate_layer_name=None, batch_size=64, lr=0.001,
                 num_bins=5, prune_bottom=0.1, prune_top=0.0,
                 per_class=True, seed=0, num_classes=None):
        super().__init__(coreset_fraction)
        self.R = R
        self.model_fn = model_fn
        self.device = device
        self.penultimate_layer_name = penultimate_layer_name
        self.batch_size = batch_size
        self.lr = lr
        self.num_bins = num_bins
        self.prune_bottom = prune_bottom
        self.prune_top = prune_top
        self.per_class = per_class
        self.seed = seed
        self.num_classes = num_classes

    def select_coreset(self, dataset):
        N = len(dataset)
        coreset_size = int(self.coreset_fraction * N)

        model = self.model_fn().to(self.device)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        ordered_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        loss_sequences = np.zeros((N, self.R), dtype=np.float32)

        for r in range(self.R):
            model.train()
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                criterion(model(inputs), labels).mean().backward()
                optimizer.step()

            model.eval()
            idx = 0
            with torch.no_grad():
                for inputs, labels in ordered_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    losses = criterion(model(inputs), labels).cpu().numpy()
                    loss_sequences[idx:idx + len(losses), r] = losses
                    idx += len(losses)
            print(f"  Warm-up epoch {r+1}/{self.R}  mean_loss={loss_sequences[:, r].mean():.4f}")

        mrmc_scores = _fit_mrmc_scores(loss_sequences)
        self.last_scores = mrmc_scores
        all_labels_np = dataset.tensors[1].numpy()

        rng = np.random.default_rng(self.seed)
        return _score_stratified_select(
            scores=mrmc_scores,
            labels=all_labels_np,
            k=coreset_size,
            num_bins=self.num_bins,
            prune_bottom=self.prune_bottom,
            prune_top=self.prune_top,
            per_class=self.per_class,
            rng=rng,
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
