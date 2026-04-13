"""
MRMC: Maximum Reduction as Maximum Contribution
Core-set Selection for Deep Learning Through Squared Loss Minimization

Reference Version Created Through AI - Not Validated

Implementation of Algorithm 1 from:
"Efficient Core-set Selection for Deep Learning Through Squared Loss Minimization"
Jianting Chen, ICML 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from scipy.optimize import curve_fit
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import argparse


# ---------------------------------------------------------------------------
# Exponential curve fitting for MRMC criterion
# ---------------------------------------------------------------------------

def neg_exp_model(r, q, w):
    """Negative exponential loss model: l(r) = q * w^(-r), w > 1."""
    return q * np.power(w, -r)


def fit_mrmc_criterion(loss_sequences):
    """
    Fit a negative exponential model to each sample's loss sequence and
    compute the MRMC score: phi_MRMC(z_j) = q_j * (1 - w_j^(-R))

    Args:
        loss_sequences: np.ndarray of shape (N, R) — per-sample loss at each epoch

    Returns:
        scores: np.ndarray of shape (N,) — MRMC criterion values
    """
    N, R = loss_sequences.shape
    epochs = np.arange(1, R + 1, dtype=float)
    scores = np.zeros(N, dtype=float)

    for i in range(N):
        L = loss_sequences[i]
        try:
            # Initial guess: q = L[0], w slightly > 1
            p0 = [max(L[0], 1e-6), max(1.01, L[0] / max(L[-1], 1e-9))]
            popt, _ = curve_fit(
                neg_exp_model, epochs, L,
                p0=p0,
                bounds=([0, 1.0], [np.inf, np.inf]),
                maxfev=2000,
            )
            q, w = popt
            scores[i] = q * (1.0 - w ** (-R))
        except (RuntimeError, ValueError):
            # Fallback: use raw loss reduction if fitting fails
            scores[i] = max(L[0] - L[-1], 0.0)

    return scores


# ---------------------------------------------------------------------------
# Proxy model for regularization
# ---------------------------------------------------------------------------

class LinearProxyModel(nn.Module):
    """Single linear (output) layer for the proxy model."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_proxy_model(features, labels, num_classes, device,
                      lr=0.01, batch_size=512, epochs=10):
    """
    Train a lightweight linear proxy model on the initial subset C'.

    Args:
        features: torch.Tensor of shape (|C'|, D)
        labels:   torch.Tensor of shape (|C'|,)
        num_classes: int
        device: torch.device

    Returns:
        proxy: trained LinearProxyModel
    """
    feature_dim = features.shape[1]
    proxy = LinearProxyModel(feature_dim, num_classes).to(device)
    optimizer = optim.SGD(proxy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    proxy.train()
    for _ in range(epochs):
        for feat, lab in loader:
            feat, lab = feat.to(device), lab.to(device)
            optimizer.zero_grad()
            loss = criterion(proxy(feat), lab)
            loss.backward()
            optimizer.step()

    return proxy


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def extract_features(model, loader, device):
    """
    Extract penultimate-layer features and labels from a trained model.

    Returns:
        features: torch.Tensor (N, D)
        labels:   torch.Tensor (N,)
    """
    model.eval()
    all_feats, all_labels = [], []

    # Temporarily hook the penultimate layer
    feature_map = {}

    def hook_fn(module, input, output):
        feature_map['feat'] = output.detach()

    # For ResNet, the average pool output is the feature
    handle = model.avgpool.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            model(images)
            feats = feature_map['feat'].squeeze(-1).squeeze(-1)  # (B, D)
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    handle.remove()
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Core MRMC selection algorithm (Algorithm 1)
# ---------------------------------------------------------------------------

def select_coreset(
    model,
    dataset,
    coreset_size,
    device,
    R=20,
    rho=1.0,
    gamma=2.0,
    batch_size=128,
    num_workers=2,
    num_classes=10,
):
    """
    MRMC core-set selection (Algorithm 1).

    Args:
        model:        PyTorch model (will be trained for R epochs internally)
        dataset:      full torch Dataset
        coreset_size: int, |C| = floor(omega * n)
        device:       torch.device
        R:            int, number of warm-up training epochs
        rho:          float in (0,1]. rho=1 → no regularization (pure MRMC)
        gamma:        float, trade-off between MRMC and regularization
        batch_size:   int
        num_workers:  int
        num_classes:  int

    Returns:
        coreset_indices: list of int — indices into dataset forming the core-set
    """
    N = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)

    # ------------------------------------------------------------------
    # Step 1: Train model on full dataset for R epochs; collect loss sequences
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=R,
                                                      eta_min=1e-4)

    loss_sequences = np.zeros((N, R), dtype=np.float32)

    # Ordered loader created once; reused every epoch for loss collection
    ordered_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    model.train()
    for r in range(R):
        epoch_losses = np.zeros(N, dtype=np.float32)
        # Training pass (shuffled)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Loss collection pass (ordered)
        model.eval()
        idx = 0
        with torch.no_grad():
            for images, labels in ordered_loader:
                images, labels = images.to(device), labels.to(device)
                losses = criterion(model(images), labels).cpu().numpy()
                bs = len(losses)
                epoch_losses[idx:idx + bs] = losses
                idx += bs
        model.train()

        loss_sequences[:, r] = epoch_losses
        print(f"  Warm-up epoch {r+1}/{R}  mean_loss={epoch_losses.mean():.4f}")

    # ------------------------------------------------------------------
    # Step 2: Compute MRMC criterion for each sample
    # ------------------------------------------------------------------
    print("Computing MRMC scores ...")
    mrmc_scores = fit_mrmc_criterion(loss_sequences)

    # ------------------------------------------------------------------
    # Step 3: Core-set selection
    # ------------------------------------------------------------------
    if rho >= 1.0:
        # No regularization: top |C| by MRMC score
        coreset_indices = np.argsort(mrmc_scores)[::-1][:coreset_size].tolist()
        return coreset_indices

    # ------------------------------------------------------------------
    # Step 4 (MRMC-R): Regularized selection
    # ------------------------------------------------------------------
    initial_size = max(1, int(round(rho * coreset_size)))
    remaining_size = coreset_size - initial_size

    # 4a. Initial subset C' — top rho*|C| by MRMC
    sorted_by_mrmc = np.argsort(mrmc_scores)[::-1]
    c_prime_indices = sorted_by_mrmc[:initial_size].tolist()
    remaining_pool = sorted_by_mrmc[initial_size:].tolist()

    # 4b. Extract features for all samples and train proxy on C'
    print("Extracting features for proxy model ...")
    model.eval()
    ordered_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)
    all_features, all_labels = extract_features(model, ordered_loader, device)

    c_prime_features = all_features[c_prime_indices]
    c_prime_labels = all_labels[c_prime_indices]

    print("Training proxy model ...")
    proxy = train_proxy_model(c_prime_features, c_prime_labels,
                              num_classes=num_classes, device=device)

    # 4c. Compute regularization scores for D \ C'
    # Keep pool tensors on CPU; move to device in batches to avoid OOM
    pool_features = all_features[remaining_pool]
    pool_labels = all_labels[remaining_pool]

    ce_criterion = nn.CrossEntropyLoss(reduction='none')
    proxy.eval()
    reg_losses = []
    bs = 512
    with torch.no_grad():
        for start in range(0, len(pool_features), bs):
            feat = pool_features[start:start + bs].to(device)
            lab = pool_labels[start:start + bs].to(device)
            l = ce_criterion(proxy(feat), lab).cpu().numpy()
            reg_losses.extend(l)
    reg_losses = np.array(reg_losses)
    reg_scores = np.exp(-reg_losses)  # phi_reg = exp(-L(z_i, theta'))

    # 4d. Combined score: phi(z_j) = phi_MRMC(z_j) - gamma * phi_reg(z_j)
    pool_mrmc = mrmc_scores[remaining_pool]
    combined_scores = pool_mrmc - gamma * reg_scores

    # Select top (1-rho)*|C| from remaining pool
    top_remaining_local = np.argsort(combined_scores)[::-1][:remaining_size]
    top_remaining_indices = [remaining_pool[k] for k in top_remaining_local]

    coreset_indices = c_prime_indices + top_remaining_indices
    return coreset_indices


# ---------------------------------------------------------------------------
# Full training on the selected core-set
# ---------------------------------------------------------------------------

def train_on_coreset(model, coreset_dataset, test_loader, device,
                     epochs=200, batch_size=128, num_workers=2,
                     lr=0.1, momentum=0.9, weight_decay=5e-4):
    """Train model on the core-set and evaluate on the test set."""
    loader = DataLoader(coreset_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                      eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == epochs:
            acc = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch}/{epochs}  test_acc={acc:.2f}%")
            if epoch == epochs:
                return acc


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_cifar10(data_root='./data'):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(data_root, train=True,
                                              download=True,
                                              transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(data_root, train=False,
                                             download=True,
                                             transform=test_transform)
    return train_set, test_set, 10


def get_cifar100(data_root='./data'):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    train_set = torchvision.datasets.CIFAR100(data_root, train=True,
                                               download=True,
                                               transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(data_root, train=False,
                                              download=True,
                                              transform=test_transform)
    return train_set, test_set, 100


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MRMC Core-set Selection')
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--data-root', default='./data',
                        help='Path to dataset root')
    parser.add_argument('--selection-ratio', type=float, default=0.5,
                        help='Fraction of training data to keep (omega)')
    parser.add_argument('--R', type=int, default=20,
                        help='Warm-up epochs for MRMC score computation')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Fraction of core-set from pure MRMC (1=no reg)')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Trade-off between MRMC and regularization')
    parser.add_argument('--train-epochs', type=int, default=200,
                        help='Epochs for final training on core-set')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset
    if args.dataset == 'cifar10':
        train_set, test_set, num_classes = get_cifar10(args.data_root)
    else:
        train_set, test_set, num_classes = get_cifar100(args.data_root)

    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=2)

    coreset_size = int(args.selection_ratio * len(train_set))
    print(f"\nDataset: {args.dataset}  N={len(train_set)}  "
          f"omega={args.selection_ratio}  |C|={coreset_size}")
    print(f"Hyperparams: R={args.R}  rho={args.rho}  gamma={args.gamma}\n")

    # Build model (ResNet-18 for CIFAR)
    model = models.resnet18(weights=None, num_classes=num_classes).to(device)

    # ------------------------------------------------------------------
    # Core-set selection
    # ------------------------------------------------------------------
    print("=== Phase 1: Core-set Selection ===")
    coreset_indices = select_coreset(
        model=model,
        dataset=train_set,
        coreset_size=coreset_size,
        device=device,
        R=args.R,
        rho=args.rho,
        gamma=args.gamma,
        batch_size=args.batch_size,
        num_workers=2,
        num_classes=num_classes,
    )
    print(f"Selected {len(coreset_indices)} samples.\n")

    # ------------------------------------------------------------------
    # Train on core-set from scratch
    # ------------------------------------------------------------------
    print("=== Phase 2: Training on Core-set ===")
    coreset_dataset = Subset(train_set, coreset_indices)

    # Re-initialise model weights for a fair comparison
    model = models.resnet18(weights=None, num_classes=num_classes).to(device)

    final_acc = train_on_coreset(
        model=model,
        coreset_dataset=coreset_dataset,
        test_loader=test_loader,
        device=device,
        epochs=args.train_epochs,
        batch_size=args.batch_size,
    )
    print(f"\nFinal test accuracy: {final_acc:.2f}%")


if __name__ == '__main__':
    main()
