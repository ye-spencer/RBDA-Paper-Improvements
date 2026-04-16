from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import curve_fit


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
    for i in range(N):
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
    print("Computing MRMC scores...")
    mrmc_scores = _fit_mrmc_scores(loss_sequences)

    # --- Phase 3: pure MRMC (rho=1) ---
    sorted_by_mrmc = np.argsort(mrmc_scores)[::-1]
    if rho >= 1.0:
        return sorted_by_mrmc[:coreset_size].tolist()

    # --- Phase 4: MRMC-R (regularized) ---
    initial_size = max(1, int(round(rho * coreset_size)))
    remaining_size = coreset_size - initial_size

    c_prime_indices = sorted_by_mrmc[:initial_size].tolist()
    remaining_pool = sorted_by_mrmc[initial_size:].tolist()

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

    combined = mrmc_scores[remaining_pool] - gamma * reg_scores
    top_remaining_local = np.argsort(combined)[::-1][:remaining_size]
    top_remaining = [remaining_pool[k] for k in top_remaining_local]

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
