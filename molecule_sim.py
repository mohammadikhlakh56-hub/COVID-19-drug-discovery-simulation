"""
molecule_sim.py
---------------
Simulates the ground-state energy of an H2 molecule using a 2-qubit
Hamiltonian and the VQE algorithm with the SLSQP optimizer.

Compatible with: qiskit >= 1.0, qiskit-algorithms >= 0.3
"""

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def simulate_h2_2qubit():
    """
    2-qubit H2 molecule ground-state simulation.
    
    Hamiltonian (Parity mapping, bond length 0.735 Å):
        H = -1.052·II + 0.398·ZI - 0.398·IZ - 0.011·ZZ + 0.181·XX
    
    Expected ground-state energy ≈ -1.137 Hartree.
    """
    print("Running 2-qubit H2 simulation with StatevectorEstimator...")

    # ── Hamiltonian ────────────────────────────────────────────────────────────
    paulis = ["II", "ZI", "IZ", "ZZ", "XX"]
    coeffs = [-1.052, 0.398, -0.398, -0.011, 0.181]
    qubit_op = SparsePauliOp(paulis, coeffs=coeffs)

    # ── Ansatz ─────────────────────────────────────────────────────────────────
    ansatz = TwoLocal(num_qubits=2, rotation_blocks="ry",
                      entanglement_blocks="cx", reps=1)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = SLSQP(maxiter=30)

    # ── Estimator primitive (V2) ───────────────────────────────────────────────
    # StatevectorEstimator is the V2 primitive; VQE accepts it directly in
    # qiskit-algorithms >= 0.3.
    estimator = StatevectorEstimator()

    # ── VQE ────────────────────────────────────────────────────────────────────
    vqe = VQE(estimator, ansatz, optimizer)

    print("Optimization in progress...")
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    energy = result.eigenvalue.real
    print("\n--- Molecular Simulation Results (2-Qubit H2) ---")
    print(f"Computed Ground State Energy : {energy:.6f} Hartree")
    print(f"Reference (FCI)              : -1.137270  Hartree")
    print(f"Error                        : {abs(energy - (-1.13727)):.6f} Hartree")
    print("--------------------------------------------------\n")
    return result


if __name__ == "__main__":
    try:
        simulate_h2_2qubit()
    except Exception as e:
        import traceback
        traceback.print_exc()
