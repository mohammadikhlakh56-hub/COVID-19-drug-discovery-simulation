"""
hybrid_model.py
---------------
Hybrid Classical-Quantum Neural Network for binary MNIST classification (0 vs 1).

Architecture:
  Classical encoder  →  4-qubit Quantum layer (VQC)  →  Binary output

Classical encoder reduces 28×28 images to 4 feature values, which are used
as rotation angles for 4 qubits.  The quantum layer returns an expectation
value in [-1, +1] that serves as the binary class score.

Compatible with: torch >= 2.0, qiskit >= 1.0, qiskit-machine-learning >= 0.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _build_qnn() -> EstimatorQNN:
    """
    Builds a 4-qubit Quantum Neural Network circuit.

    The circuit has two kinds of parameters:
      - input_params  (4)  : feature-map angles, supplied at forward-pass time
      - weight_params (8)  : trainable variational angles, learnt by the optimiser

    Layout (per qubit): Ry(input) → Ry(weight_0) → Rz(weight_1) → CX entanglement

    Returns
    -------
    EstimatorQNN
        Ready-to-use QNN with TorchConnector-compatible interface.
    """
    n_qubits = 4
    n_weights = n_qubits * 2  # 2 trainable rotations per qubit

    inputs  = ParameterVector("x", n_qubits)
    weights = ParameterVector("θ", n_weights)

    qc = QuantumCircuit(n_qubits)

    # Feature map: encode classical features as Ry rotations
    for i in range(n_qubits):
        qc.ry(inputs[i], i)

    # Entanglement layer
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)  # wrap-around entanglement

    # Variational layer: two trainable rotations per qubit
    for i in range(n_qubits):
        qc.ry(weights[2 * i],     i)
        qc.rz(weights[2 * i + 1], i)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=inputs.params,
        weight_params=weights.params,
        input_gradients=True,  # required for backprop through TorchConnector
    )
    return qnn


class HybridModel(nn.Module):
    """
    Hybrid Classical-Quantum model for binary image classification.

    Classical CNN encoder:
        Conv(1→6, k=5) → MaxPool → Conv(6→16, k=5) → MaxPool
        → Flatten → Linear(256→64) → Linear(64→4)

    Quantum layer:
        4-qubit EstimatorQNN via TorchConnector
        Output: scalar expectation value in [-1, +1]

    The scalar output is directly used with MSELoss against {-1, +1} labels.
    """

    def __init__(self):
        super().__init__()

        # ── Classical encoder ──────────────────────────────────────────────────
        # Input: (B, 1, 28, 28)
        # After conv1 + pool: (B, 6, 12, 12)
        # After conv2 + pool: (B, 16,  4,  4)  → flatten → 256
        self.conv1   = nn.Conv2d(1,  6, kernel_size=5)   # → (B,  6, 24, 24)
        self.conv2   = nn.Conv2d(6, 16, kernel_size=5)   # → (B, 16, 10, 10) after relu+pool×2
        self.dropout = nn.Dropout2d(p=0.25)
        self.fc1     = nn.Linear(16 * 4 * 4, 64)         # 256 → 64
        self.fc2     = nn.Linear(64, 4)                  # 64  → 4 qubit inputs

        # ── Quantum layer ──────────────────────────────────────────────────────
        self.qnn     = _build_qnn()
        self.q_layer = TorchConnector(self.qnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, 28, 28) — normalised MNIST image batch.

        Returns
        -------
        torch.Tensor
            Shape (B, 1) — expectation value per sample in [-1, +1].
        """
        # Classical path
        x = F.relu(F.max_pool2d(self.conv1(x), 2))         # (B,  6, 12, 12)
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))  # (B, 16,  4,  4)
        x = x.view(x.size(0), -1)                          # (B, 256)
        x = F.relu(self.fc1(x))                            # (B,  64)
        x = self.fc2(x)                                    # (B,   4)

        # Scale to [-π, π] so features span the full rotation range
        x = torch.tanh(x) * torch.pi

        # Quantum path — TorchConnector expects (B, n_inputs)
        x = self.q_layer(x)                                # (B,   1)

        return x


if __name__ == "__main__":
    model = HybridModel()
    print(model)
    # Sanity-check a single forward pass
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"Output shape : {output.shape}")   # expected: torch.Size([1, 1])
    print(f"Output value : {output.item():.4f}")  # expected: in [-1, 1]
