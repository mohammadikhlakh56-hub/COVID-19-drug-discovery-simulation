"""
train.py
--------
Training script for the Hybrid Classical-Quantum MNIST classifier (0 vs 1).

Usage:
    python train.py

The script trains with a tiny subset (50 train / 10 test samples) because
quantum circuit simulation is exponentially slow on classical hardware.
Increase train_size / test_size only if you have GPU-accelerated simulation.

Compatible with: torch >= 2.0, qiskit >= 1.0, qiskit-machine-learning >= 0.7
"""

import torch
import torch.optim as optim
import torch.nn as nn

from data_loader import get_mnist_data
from hybrid_model import HybridModel


def train():
    # ── Hyper-parameters ───────────────────────────────────────────────────────
    epochs      = 5
    lr          = 0.01
    batch_size  = 1     # keep at 1 — quantum batching is slow on CPU
    train_size  = 50    # small subset for practical runtime
    test_size   = 10

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_mnist_data(
        batch_size=batch_size,
        train_size=train_size,
        test_size=test_size,
    )

    # ── Model, optimiser, loss ─────────────────────────────────────────────────
    model     = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # MSELoss with labels remapped to {-1, +1} matches the QNN's [-1, 1] output range
    loss_func = nn.MSELoss()

    print(f"\nStarting training  (epochs={epochs}, lr={lr}) ...")
    print("─" * 55)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output = model(data)              # shape: (B, 1)

            # Remap labels: 0 → -1,  1 → +1
            # This matches the quantum layer's [-1, 1] expectation-value output.
            target_mapped = (target.float() * 2 - 1).unsqueeze(1)  # (B, 1)

            loss = loss_func(output, target_mapped)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Live progress (print every 10 samples)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Sample {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}  ──  Avg Loss: {avg_loss:.4f}")

    # ── Evaluation ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    print("Evaluating on test set...")
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)          # shape: (1, 1)
            # Threshold at 0: output >= 0 → class 1, else class 0
            pred = 1 if output.item() >= 0 else 0
            if pred == target.item():
                correct += 1

    accuracy = correct / len(test_loader) * 100
    print(f"Test Accuracy: {correct}/{len(test_loader)} = {accuracy:.2f}%")
    print("─" * 55)


if __name__ == "__main__":
    train()
