import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        
        # Classical part: Reduces 28x28 (784) input to 4 features for our 4-qubit circuit
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 4)  # 4 features for 4 qubits

        # Quantum part
        self.qnn = self.create_qnn()
        self.q_layer = TorchConnector(self.qnn)

    def create_qnn(self):
        # Define a 4-qubit QNN circuit
        # QNNCircuit is a convenient way to create a variational circuit
        # with feature map and ansatz.
        qnn_circuit = QNNCircuit(num_qubits=4)
        
        # We use the EstimatorQNN which returns expectation values
        qnn = EstimatorQNN(circuit=qnn_circuit)
        return qnn

    def forward(self, x):
        # Classical layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output size (batch, 4)
        
        # Quantum layer
        # TorchConnector expects input in the shape (batch, num_inputs)
        x = self.q_layer(x) # Output size (batch, 1) or similar depending on measurement
        
        # Final classification
        # Since it's binary, we can use a single output with sigmoid or 2 outputs for CrossEntropy
        # TorchConnector for EstimatorQNN with 1 weight usually returns 1 value.
        # Let's add a small classical layer at the end to map to 2 classes.
        return x

if __name__ == "__main__":
    model = HybridModel()
    print(model)
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
