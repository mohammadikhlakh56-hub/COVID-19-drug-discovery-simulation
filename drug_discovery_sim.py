"""
drug_discovery_sim.py
---------------------
Simulates protein-drug binding energy using a 3-qubit Hamiltonian
and the VQE algorithm with the SLSQP optimizer.

Compatible with: qiskit >= 1.0, qiskit-algorithms >= 0.3
"""

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def simulate_binding_energy():
    """
    Simulates the interaction energy between two simplified molecular fragments.

    In drug discovery, this predicts how well a drug candidate (ligand) binds
    to a disease-related protein. A lower (more negative) energy indicates a
    stronger, more stable binding — the primary goal of drug optimisation.

    Returns
    -------
    VQEResult
        Full VQE result object containing eigenvalue and optimal parameters.
    """
    print("Simulating Protein-Drug Binding Interaction...")

    # ── Hamiltonian ────────────────────────────────────────────────────────────
    # 3-qubit system representing a simplified interaction Hamiltonian.
    # H = H_protein + H_drug + H_interaction
    # Coefficients are in Hartree units.
    paulis = ["III", "ZII", "IZI", "IIZ", "ZZI", "IZZ", "ZIZ", "XXX"]
    coeffs = [-1.5, 0.4, 0.4, 0.2, -0.05, -0.05, -0.02, 0.1]
    qubit_op = SparsePauliOp(paulis, coeffs=coeffs)

    # ── Ansatz ─────────────────────────────────────────────────────────────────
    ansatz = TwoLocal(num_qubits=3, rotation_blocks="ry",
                      entanglement_blocks="cx", reps=2)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = SLSQP(maxiter=50)

    # ── Estimator primitive (V2) ───────────────────────────────────────────────
    estimator = StatevectorEstimator()

    # ── VQE ────────────────────────────────────────────────────────────────────
    vqe = VQE(estimator, ansatz, optimizer)

    print("Computing lowest energy binding configuration (VQE)...")
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    energy = result.eigenvalue.real
    print("\n--- Medical Drug Discovery Simulation Results ---")
    print(f"Computed Binding Energy Level : {energy:.6f} Hartree")
    print("Note: Lower (more negative) energy = stronger, more stable binding.")
    print("Strong binding → drug 'jams' the disease protein's machinery.")
    print("-------------------------------------------------\n")
    return result


if __name__ == "__main__":
    try:
        simulate_binding_energy()
    except Exception as e:
        import traceback
        traceback.print_exc()
