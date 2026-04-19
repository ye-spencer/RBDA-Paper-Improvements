"""
Testing Pipeline

For different experiments, we select different datasets (and select the best hyperparameters for each dataset). Then, we compare the accuracy of different coreset selection methods, including random (bottom baseline), original MRMC (control), adaptive MRMC (improvement), and full dataset (top baseline).

Results are pickled to results/run_{seed}.pkl for later analysis by the plot_*.py scripts.
"""
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits, fetch_openml, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from coreset_selection import (
    RandomCoresetSelection,
    FullDatasetSelection,
    MRMCOriginalCoresetSelection,
    MRMCOriginalStratifiedCoresetSelection,
    MRMCKMeansCoresetSelection,
    MRMCKMeansBlendedCoresetSelection,
    MRMCScoreStratifiedCoresetSelection,
)
from models import WineMLP


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SEEDS = [0]  # extend to [0, 1, 2] for multi-seed averaging in the plots


def load_dataset(name):
    if name == 'digits':
        X, y = load_digits(return_X_y=True)
    elif name == 'mnist':
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data, mnist.target.astype(int)
    elif name == 'covtype':
        data = fetch_covtype()
        X, y = data.data, (data.target - 1).astype(int)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, criterion, optimizer, train_loader, epochs, device):
    train_losses, train_accs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return train_losses, train_accs


def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for lbl, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                per_class_total[int(lbl)] = per_class_total.get(int(lbl), 0) + 1
                if lbl == pred:
                    per_class_correct[int(lbl)] = per_class_correct.get(int(lbl), 0) + 1

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    per_class_acc = {c: per_class_correct.get(c, 0) / per_class_total[c] for c in per_class_total}
    return epoch_loss, epoch_acc, per_class_acc


def build_selectors(coreset_fraction, R, rho, gamma, generate_model_func,
                    device, num_classes, seed):
    return [
        RandomCoresetSelection(coreset_fraction, seed=seed),
        FullDatasetSelection(),
        MRMCOriginalCoresetSelection(
            coreset_fraction, R, rho, gamma,
            model_fn=generate_model_func, device=device,
            penultimate_layer_name='fc2', num_classes=num_classes,
        ),
        MRMCOriginalStratifiedCoresetSelection(
            coreset_fraction, R, rho, gamma,
            model_fn=generate_model_func, device=device,
            penultimate_layer_name='fc2', num_classes=num_classes,
        ),
        MRMCKMeansCoresetSelection(
            coreset_fraction, R,
            model_fn=generate_model_func, device=device,
            penultimate_layer_name='fc2',
        ),
        MRMCKMeansBlendedCoresetSelection(
            coreset_fraction, R,
            model_fn=generate_model_func, device=device,
            penultimate_layer_name='fc2', alpha=0.5,
        ),
        MRMCScoreStratifiedCoresetSelection(
            coreset_fraction, R,
            model_fn=generate_model_func, device=device,
            penultimate_layer_name='fc2',
            num_bins=5, prune_bottom=0.1, prune_top=0.0,
            per_class=True, seed=seed,
        ),
    ]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Training hyperparameters ---
    batch_size = 16
    epochs = 10
    lr = 0.001

    # --- Dataset ---
    DATASET = 'mnist'
    X_raw, y_raw = load_dataset(DATASET)
    num_features = X_raw.shape[1]
    num_classes = len(set(y_raw))
    print("\n Dataset loaded: ---")
    print(f"Features shape: {X_raw.shape}")
    print(f"Labels shape:   {y_raw.shape}")
    print(f"Classes:        {set(y_raw)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=0, stratify=y_raw
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    train_labels_np = y_train_t.numpy()

    # --- Experiment variations ---
    R = 6
    rho = 0.5
    gamma = 2.0
    coreset_fraction = 0.1
    generate_model_func = lambda features=num_features, num_classes=num_classes: WineMLP(features, num_classes)

    for seed in SEEDS:
        print(f"\n===== Seed {seed} =====")
        set_global_seed(seed)

        results = {
            '_meta': {
                'seed': seed,
                'dataset': DATASET,
                'coreset_fraction': coreset_fraction,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'R': R, 'rho': rho, 'gamma': gamma,
                'train_labels': train_labels_np,
                'n_train': len(train_dataset),
                'n_test': len(test_dataset),
                'num_classes': num_classes,
            }
        }

        selectors = build_selectors(
            coreset_fraction, R, rho, gamma,
            generate_model_func, device,
            num_classes, seed,
        )

        for selector in selectors:
            name = selector.__class__.__name__
            print(f"\n {name} Coreset: ")

            set_global_seed(seed)  # reset before selection for reproducibility
            t0 = time.time()
            coreset_indices = selector.select_coreset(train_dataset)
            selection_time = time.time() - t0
            print(f"Coreset selection time: {selection_time:.4f} seconds")

            train_subset = torch.utils.data.Subset(train_dataset, coreset_indices)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            set_global_seed(seed)  # reset before training to make runs comparable
            model = generate_model_func().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            train_losses, train_accs = train_model(
                model, criterion, optimizer, train_loader, epochs, device
            )
            test_loss, test_acc, per_class_acc = evaluate_model(
                model, criterion, test_loader, device
            )
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

            results[name] = {
                'test_acc': test_acc,
                'test_loss': test_loss,
                'per_class_test_acc': per_class_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'selection_time': selection_time,
                'coreset_indices': np.asarray(coreset_indices, dtype=np.int64),
                'mrmc_scores': getattr(selector, 'last_scores', None),
            }

        out_path = os.path.join(RESULTS_DIR, f'run_{seed}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
