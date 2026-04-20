"""
Testing Pipeline

For different experiments, we select different datasets (and select the best hyperparameters for each dataset). Then, we compare the accuracy of different coreset selection methods, including random (bottom baseline), original MRMC (control), adaptive MRMC (improvement), and full dataset (top baseline).

"""
from sklearn.datasets import load_digits, load_wine, load_breast_cancer, fetch_openml, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import WineMLP
import torch.nn as nn
import torch.optim as optim
from coreset_selection import RandomCoresetSelection, FullDatasetSelection, MRMCOriginalCoresetSelection, MRMCOriginalStratifiedCoresetSelection, MRMCKMeansCoresetSelection, MRMCAdaptiveCoresetSelection
import time


def _subsample_per_class(X, y, counts, seed=0):
    """Return a subset of (X, y) with `counts[c]` samples drawn from each class c.

    If a class has fewer samples than requested, all of its samples are taken.
    Indices are shuffled so class order is not preserved in the output.
    """
    rng = np.random.RandomState(seed)
    parts = []
    for cls, n in counts.items():
        cls_idx = np.where(y == cls)[0]
        n = min(n, len(cls_idx))
        parts.append(rng.choice(cls_idx, size=n, replace=False))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_dataset(name):
    """
    Returns (X_raw, y_raw) as numpy arrays.

    Datasets:
      'wine'                    - 178 samples,  13 features,  3 classes  (small, balanced)
      'breast_cancer'           - 569 samples,  30 features,  2 classes  (small, binary)
      'digits'                  - 1797 samples, 64 features,  10 classes (small, multiclass)
      'adult'                   - 48842 samples,14 features,  2 classes  (medium, imbalanced)
      'letter'                  - 20000 samples,16 features,  26 classes (medium, balanced)
      'mnist'                   - 70000 samples,784 features, 10 classes (large, image-derived)
      'mnist_sample_balanced'   - 27500 samples, 784 features, 10 classes (2750 per class)
      'mnist_sample_unbalanced' - 27500 samples, 784 features, 10 classes (500..5500 per class, 10:1 ratio)
      'covtype'                 - 581012 samples,54 features, 7 classes  (large, imbalanced)
    """
    if name == 'wine':
        X, y = load_wine(return_X_y=True)
    elif name == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
    elif name == 'digits':
        X, y = load_digits(return_X_y=True)
    elif name == 'adult':
        adult = fetch_openml('adult', version=2, as_frame=False, parser='liac-arff')
        X = adult.data.astype(float)
        y = (adult.target == '>50K').astype(int)
    elif name == 'letter':
        letter = fetch_openml('letter', version=1, as_frame=False, parser='liac-arff')
        X = letter.data.astype(float)
        y = (letter.target.view('U1').ravel().view(np.uint32) - ord('A')).astype(int)
    elif name in ('mnist', 'mnist_sample_balanced', 'mnist_sample_unbalanced'):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data, mnist.target.astype(int)
        if name == 'mnist_sample_balanced':
            X, y = _subsample_per_class(X, y, {c: 2200 for c in range(10)}, seed=0)
        elif name == 'mnist_sample_unbalanced':
            X, y = _subsample_per_class(X, y, {c: (c + 1) * 500 for c in range(10)}, seed=0)
    elif name == 'covtype':
        data = fetch_covtype()
        X, y = data.data, (data.target - 1).astype(int)  # labels are 1-7, shift to 0-6
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def train_model(model, criterion, optimizer, train_loader, epochs, device):
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

        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc
        
    
def prepare_datasets(name):
    """Load, split, scale, and tensorize a dataset once for reuse across trials."""
    X_raw, y_raw = load_dataset(name)
    num_features = X_raw.shape[1]
    num_classes = len(set(y_raw))

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=0, stratify=y_raw
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset, num_features, num_classes


def main(rho_val, gamma_val, train_dataset, test_dataset, num_features, num_classes, device):

    ### --- Find Best Normal ML Hyperparameters for Dataset --- ###

    batch_size = 128
    epochs = 25
    lr = 0.001

    ### --- End Find Best Normal ML Hyperparameters for Dataset --- ###

    ### --- EXPERIMENT VARIATIONS --- ###

    # MRMC Hyperparameters
    R = 10
    rho = rho_val
    gamma = gamma_val

    # Coreset Fraction
    coreset_fraction = 0.3

    # Model Function
    generate_model_func = lambda features=num_features, num_classes=num_classes: WineMLP(features, num_classes)

    ### --- END EXPERIMENT VARIATIONS --- ###

    CORESET_SELECTION = [
        RandomCoresetSelection(coreset_fraction),
        FullDatasetSelection(),
        # MRMCOriginalCoresetSelection(
        #     coreset_fraction, R, rho, gamma,
        #     model_fn=generate_model_func,
        #     device=device,
        #     penultimate_layer_name='fc2',
        #     num_classes=num_classes,
        # ),
        # MRMCOriginalStratifiedCoresetSelection(
        #     coreset_fraction, R, rho, gamma,
        #     model_fn=generate_model_func,
        #     device=device,
        #     penultimate_layer_name='fc2',
        #     num_classes=num_classes,
        # ),
        # MRMCKMeansCoresetSelection(
        #     coreset_fraction, R,
        #     model_fn=generate_model_func,
        #     device=device,
        #     penultimate_layer_name='fc2',
        # ),
        # MRMCAdaptiveCoresetSelection(coreset_fraction, R, rho, gamma, generate_model_func, device),
    ]

    for coreset_selector in CORESET_SELECTION:

        print(f"\n {coreset_selector.__class__.__name__} Coreset: ")

        start_time = time.time()
        coreset_indices = coreset_selector.select_coreset(train_dataset)
        end_time = time.time()
        # print(f"Coreset selection time: {end_time - start_time:.4f} seconds")

        train_subset = torch.utils.data.Subset(train_dataset, coreset_indices)

        pin = device.type == 'cuda'
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=pin)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=pin)

        model = generate_model_func().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)


        train_model(model, criterion, optimizer, train_loader, epochs, device)

        test_loss, test_acc = evaluate_model(model, criterion, test_loader, device)

        #print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        return test_acc


if __name__ == "__main__":

    DATASET = 'mnist_sample_balanced'   # 'digits' | 'mnist' | 'covtype'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, num_features, num_classes = prepare_datasets(DATASET)


    cumulative_acc = 0
    for i in range(10):
        attempt = main(0.5, 3, train_dataset, test_dataset, num_features, num_classes, device)
        cumulative_acc += attempt
        print(f"Attempt {i+1}: {attempt}")
    print(f"Average Accuracy: {cumulative_acc / 10}")
    exit()

    rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    gammas = [1, 2, 3, 4, 5, 6]
    repeats = 3
    trials_per_rho = len(gammas) * repeats

    results_per_rho = []
    for rho in rhos:
        cumulative_acc = 0
        print(f"\n Rho: {rho:.1f} ---")
        min_acc = float('inf')
        max_acc = float('-inf')
        for gamma in gammas:
            print(f"Gamma: {gamma}")
            for i in range(repeats):
                test_acc = main(rho, gamma, train_dataset, test_dataset, num_features, num_classes, device)
                cumulative_acc += test_acc
                min_acc = min(min_acc, test_acc)
                max_acc = max(max_acc, test_acc)
        results_per_rho.append((cumulative_acc / trials_per_rho, min_acc, max_acc))
    for i in range(len(results_per_rho)):
        print(f"Rho: {i * 0.1 + 0.1:.1f}, Avg Test Accuracy: {results_per_rho[i][0]:.4f}, Min Test Accuracy: {results_per_rho[i][1]:.4f}, Max Test Accuracy: {results_per_rho[i][2]:.4f}")
        