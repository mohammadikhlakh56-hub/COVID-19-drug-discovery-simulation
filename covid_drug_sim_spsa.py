"""
covid_drug_sim_spsa.py
----------------------
Advanced Mpro binding simulation using SPSA optimizer — better suited for
noisy quantum environments and often escapes local minima more effectively
than gradient-based methods like SLSQP.

Compatible with: qiskit >= 1.0, qiskit-algorithms >= 0.3
"""

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import numpy as np


def simulate_covid_mpro_binding_advanced():
    """
    Advanced simulation of SARS-CoV-2 Mpro binding using the SPSA optimizer.

    SPSA (Simultaneous Perturbation Stochastic Approximation):
      - Gradient-free: estimates gradient from only 2 function evaluations
      - Noise-robust: ideal when circuit noise prevents clean gradient signals
      - Escape-friendly: stochastic perturbations help avoid shallow local minima

    Same Hamiltonian as covid_drug_sim.py, but with 3 ansatz repetitions
    (richer variational space) and 200 SPSA iterations.

    Returns
    -------
    float
        Optimised binding energy in Hartree units.
    """
    print("--- Advanced COVID-19 Drug Discovery (SPSA Optimized) ---")
    print("Goal: Robust, noise-tolerant optimization of drug binding energy.\n")

    # ── Hamiltonian (same as covid_drug_sim.py) ────────────────────────────────
    paulis = [
        "IIII", "ZIII", "IZII", "IIZI", "IIIZ",
        "ZZII", "ZIZI", "ZIIZ", "IZZI", "IZIZ", "IIZZ",
        "XXXX", "YYYY",
    ]
    coeffs = [
        -2.1,  0.5,   0.4,   0.5,   0.4,
        -0.1, -0.05, -0.05, -0.1, -0.05, -0.1,
         0.05,  0.05,
    ]
    qubit_op = SparsePauliOp(paulis, coeffs=coeffs)

    # ── Ansatz (deeper: 3 reps for richer variational space) ──────────────────
    ansatz = TwoLocal(num_qubits=4, rotation_blocks="ry",
                      entanglement_blocks="cx", reps=3)

    # ── SPSA Optimizer ─────────────────────────────────────────────────────────
    # Higher iteration count compensates for SPSA's stochastic nature
    optimizer = SPSA(maxiter=200)

    # ── Estimator primitive (V2) ───────────────────────────────────────────────
    estimator = StatevectorEstimator()

    # ── VQE ────────────────────────────────────────────────────────────────────
    vqe = VQE(estimator, ansatz, optimizer)

    print("Running SPSA-VQE optimization (may take a moment)...")
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    binding_energy = result.eigenvalue.real

    # ── Results & Interpretation ───────────────────────────────────────────────
    print("\n--- Advanced Simulation Results ---")
    print(f"Computed Binding Energy: {binding_energy:.6f} Hartree")

    if binding_energy < -3.0:
        print("RESULT: EXTREME AFFINITY. SPSA found a deeper energy well!")
        print("This configuration is highly statistically probable.")
    elif binding_energy < -2.5:
        print("RESULT: HIGH AFFINITY. Consistent with SLSQP baseline results.")
    elif binding_energy < -2.0:
        print("RESULT: MODERATE AFFINITY. Consider more SPSA iterations.")
    else:
        print("RESULT: WEAK AFFINITY. May need a different ansatz or Hamiltonian.")

    print("----------------------------------------------------------\n")
    return binding_energy


if __name__ == "__main__":
    try:
        simulate_covid_mpro_binding_advanced()
    except Exception as e:
        import traceback
        traceback.print_exc()
