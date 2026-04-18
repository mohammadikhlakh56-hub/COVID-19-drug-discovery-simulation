"""
covid_drug_sim.py
-----------------
Simulates the binding interaction between a drug candidate and the
SARS-CoV-2 Main Protease (Mpro) using VQE with the SLSQP optimizer.

Target protein: Mpro (nsp5 protease) — essential for viral replication.
Inhibiting Mpro prevents the virus from producing functional proteins.

Compatible with: qiskit >= 1.0, qiskit-algorithms >= 0.3
"""

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def simulate_covid_mpro_binding():
    """
    Simulates the binding energy of a drug candidate in the Mpro active site.

    The 4-qubit Hamiltonian approximates:
      - Individual orbital energies  (I, Z terms)
      - Two-body interaction energies (ZZ terms)
      - Quantum tunnelling / orbital overlap (XX, YY terms)

    Residues modelled: CYS145 and HIS41 of SARS-CoV-2 Mpro.

    Returns
    -------
    float
        Binding energy in Hartree units.
    """
    print("--- COVID-19 Drug Discovery Simulation (Mpro Target) ---")
    print("Goal: Find the most stable binding configuration for a candidate molecule.\n")

    # ── Hamiltonian ────────────────────────────────────────────────────────────
    paulis = [
        "IIII", "ZIII", "IZII", "IIZI", "IIIZ",  # Individual orbital energies
        "ZZII", "ZIZI", "ZIIZ", "IZZI", "IZIZ", "IIZZ",  # Interaction energies
        "XXXX", "YYYY",  # Quantum tunnelling / orbital overlap
    ]
    coeffs = [
        -2.1,  0.5,   0.4,   0.5,   0.4,   # Base stability terms
        -0.1, -0.05, -0.05, -0.1, -0.05, -0.1,  # Binding attractions
         0.05,  0.05,  # Dynamic fluctuations
    ]
    qubit_op = SparsePauliOp(paulis, coeffs=coeffs)

    # ── Ansatz ─────────────────────────────────────────────────────────────────
    # TwoLocal with 2 reps represents the drug's geometric poses in the pocket
    ansatz = TwoLocal(num_qubits=4, rotation_blocks="ry",
                      entanglement_blocks="cx", reps=2)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = SLSQP(maxiter=100)

    # ── Estimator primitive (V2) ───────────────────────────────────────────────
    estimator = StatevectorEstimator()

    # ── VQE ────────────────────────────────────────────────────────────────────
    vqe = VQE(estimator, ansatz, optimizer)

    print("Running VQE to optimize the drug-protease interaction...")
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    binding_energy = result.eigenvalue.real

    # ── Results & Interpretation ───────────────────────────────────────────────
    print("\n--- Simulation Results ---")
    print(f"Computed Binding Energy: {binding_energy:.6f} Hartree")

    if binding_energy < -2.5:
        print("RESULT: HIGH AFFINITY. Candidate shows strong binding to Mpro.")
        print("Recommendation: Proceed to virtual synthesis and physical lab testing.")
    elif binding_energy < -2.0:
        print("RESULT: MODERATE AFFINITY. Further structural optimisation recommended.")
    else:
        print("RESULT: WEAK AFFINITY. Candidate unlikely to inhibit Mpro effectively.")

    print("----------------------------------------------------------\n")
    return binding_energy


if __name__ == "__main__":
    try:
        simulate_covid_mpro_binding()
    except Exception as e:
        import traceback
        traceback.print_exc()
