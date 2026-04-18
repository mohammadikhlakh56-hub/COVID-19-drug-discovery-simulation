import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_mnist_data
from hybrid_model import HybridModel

def train():
    # Parameters
    epochs = 5
    lr = 0.01
    
    # Load data
    train_loader, test_loader = get_mnist_data(batch_size=1, train_size=50, test_size=10)
    
    # Initialize model
    model = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss() # EstimatorQNN usually returns a single value for regression-like behavior; we'll adapt for classification

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Map target to -1, 1 if needed by measurement, or just use 0, 1
            # Standard EstimatorQNN with Z measurement usually range [-1, 1]
            # Let's map target 0 -> -1, target 1 -> 1
            target_mapped = (target.float() * 2) - 1
            
            loss = loss_func(output, target_mapped.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    print("\nEvaluating...")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # Threshold at 0 for classification (-1 vs 1)
            pred = 1 if output.item() >= 0 else 0
            if pred == target.item():
                correct += 1
                
    print(f"Test Accuracy: {correct/len(test_loader) * 100:.2f}%")

if __name__ == "__main__":
    train()
