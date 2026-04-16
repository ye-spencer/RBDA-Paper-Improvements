"""
Testing Pipeline

For different experiments, we select different datasets (and select the best hyperparameters for each dataset). Then, we compare the accuracy of different coreset selection methods, including random (bottom baseline), original MRMC (control), adaptive MRMC (improvement), and full dataset (top baseline).

"""
from sklearn.datasets import load_digits, fetch_openml, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import WineMLP
import torch.nn as nn
import torch.optim as optim
from coreset_selection import RandomCoresetSelection, FullDatasetSelection, MRMCOriginalCoresetSelection, MRMCAdaptiveCoresetSelection
import time


def load_dataset(name):
    """
    Returns (X_raw, y_raw) as numpy arrays.
    Supported: 'digits', 'mnist'
    """
    if name == 'digits':
        X, y = load_digits(return_X_y=True)
    elif name == 'mnist':
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data, mnist.target.astype(int)
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


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
        
    
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### --- Find Best Normal ML Hyperparameters for Dataset --- ### 
    
    batch_size = 16
    epochs = 10
    lr = 0.001
    
    ### --- End Find Best Normal ML Hyperparameters for Dataset --- ###

    ### Pick Dataset ###
    DATASET = 'covtype'   # 'digits' | 'mnist' | 'covtype'
    X_raw, y_raw = load_dataset(DATASET)
    num_features = X_raw.shape[1]
    num_classes = len(set(y_raw))
    print("\n Dataset loaded: ---")
    print(f"Features shape: {X_raw.shape}")
    print(f"Labels shape:   {y_raw.shape}")
    print(f"Classes:        {set(y_raw)}\n")


    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=0, stratify=y_raw)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)


    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)


    ### --- EXPERIMENT VARIATIONS --- ###

    # MRMC Hyperparameters
    R = 6
    rho = 0.5
    gamma = 2.0

    # Coreset Fraction
    coreset_fraction = 0.1

    # Model Function
    generate_model_func = lambda features=num_features, num_classes=num_classes: WineMLP(features, num_classes)

    ### --- END EXPERIMENT VARIATIONS --- ###

    CORESET_SELECTION = [
        RandomCoresetSelection(coreset_fraction),
        FullDatasetSelection(),
        MRMCOriginalCoresetSelection(
            coreset_fraction, R, rho, gamma,
            model_fn=generate_model_func,
            device=device,
            penultimate_layer_name='fc2',
            num_classes=num_classes,
        ),
        # MRMCAdaptiveCoresetSelection(coreset_fraction, R, rho, gamma, generate_model_func, device),
    ]

    for coreset_selector in CORESET_SELECTION:

        print(f"\n {coreset_selector.__class__.__name__} Coreset: ")

        start_time = time.time()
        coreset_indices = coreset_selector.select_coreset(train_dataset)
        end_time = time.time()
        print(f"Coreset selection time: {end_time - start_time:.4f} seconds")

        train_subset = torch.utils.data.Subset(train_dataset, coreset_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        model = generate_model_func()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)


        train_model(model, criterion, optimizer, train_loader, epochs, device)

        test_loss, test_acc = evaluate_model(model, criterion, test_loader, device)

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()