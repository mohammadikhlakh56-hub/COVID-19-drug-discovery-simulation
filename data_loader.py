import torch
from torchvision import datasets, transforms
import numpy as np

def get_mnist_data(batch_size=1, train_size=100, test_size=50):
    """
    Loads MNIST and filters for digits 0 and 1.
    Resizes images to 4x4 or similar small size to match qubit count if needed, 
    but here we'll keep them 28x28 and let the classical layer handle reduction.
    """
    
    # Transform: Normalize and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter for 0s and 1s only
    def filter_binary(dataset, num_samples):
        indices = np.where((dataset.targets == 0) | (dataset.targets == 1))[0]
        # Limit samples for speed in quantum simulation
        indices = indices[:num_samples]
        dataset.data = dataset.data[indices]
        dataset.targets = dataset.targets[indices]
        return dataset

    train_dataset = filter_binary(train_dataset, train_size)
    test_dataset = filter_binary(test_dataset, test_size)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples (0s and 1s).")
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_mnist_data()
    for data, target in train_loader:
        print(f"Data shape: {data.shape}, Target: {target}")
        break
