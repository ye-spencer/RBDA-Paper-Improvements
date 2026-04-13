"""
Version of MRMC created from the original paper by hand

"""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import WineMLP


def train_model(model, X_train, y_train, epochs, lr, batch_size, device):
    pass
    
def main():

    batch_size = 16
    epochs = 10
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Pick Dataset ###
    X, y = load_wine(return_X_y=True)
    print("\n Wine dataset loaded: ---")
    print(f"Features shape: {X.shape}")  # (178, 13)
    print(f"Labels shape:   {y.shape}")  # (178,)
    print(f"Classes:        {set(y)}")   # {0, 1, 2}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    model = WineMLP()
    print(model)

    # Count parameters to get a feel for the model's size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params}")


if __name__ == "__main__":
    main()