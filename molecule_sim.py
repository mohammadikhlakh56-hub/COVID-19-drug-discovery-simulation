from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def simulate_h2_2qubit():
    print("Running 2-qubit H2 simulation with StatevectorEstimator...")
    
    # 2-qubit H2 Hamiltonian (Parity mapping, distance 0.735A)
    # H = -1.052 * I + 0.398 * Z0 - 0.398 * Z1 - 0.011 * Z0Z1 + 0.181 * X0X1
    paulis = ["II", "ZI", "IZ", "ZZ", "XX"]
    coeffs = [-1.052, 0.398, -0.398, -0.011, 0.181]
    
    qubit_op = SparsePauliOp(paulis, coeffs=coeffs)

    ansatz = TwoLocal(num_qubits=2, rotation_blocks="ry", entanglement_blocks="cx", reps=1)
    
    optimizer = SLSQP(maxiter=30)
    estimator = Estimator()
    vqe = VQE(estimator, ansatz, optimizer)
    
    print("Optimization in progress...")
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    print("\n--- Molecular Simulation Results (2-Qubit H2) ---")
    print(f"Computed Ground State Energy: {result.eigenvalue.real:.6f} Hartree")
    print("------------------------------------\n")
    return result

if __name__ == "__main__":
    try:
        simulate_h2_2qubit()
    except Exception as e:
        import traceback
        traceback.print_exc()



