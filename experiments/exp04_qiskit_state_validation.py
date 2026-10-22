from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

INPUT_CSV = RESULTS_DIR / "exp02_quantum_probabilities.csv"

OUTPUT_CSV = RESULTS_DIR / "exp04_qiskit_measurements.csv"
OUTPUT_JSON = RESULTS_DIR / "exp04_summary.json"

FIG_FULL = FIGURES_DIR / "exp04_theory_vs_qiskit_full.png"
FIG_TOP = FIGURES_DIR / "exp04_theory_vs_qiskit_top.png"


# =====================================================================
# Experiment configuration
# =====================================================================

SHOTS = 100_000
SEED = 20261022

TOP_K = 16


# =====================================================================
# Utility functions
# =====================================================================

def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Total Variation Distance (TVD)

        TVD(p, q) = 1/2 * sum_k |p_k - q_k|

    TVD = 0 means identical probability distributions.
    """
    return float(
        0.5 * np.sum(np.abs(p - q))
    )


def probability_rmse(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Root Mean Square Error between two probability distributions.
    """
    return float(
        np.sqrt(
            np.mean(
                (p - q) ** 2
            )
        )
    )


def classical_fidelity(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Classical probability-distribution fidelity:

        F = (sum_k sqrt(p_k q_k))^2

    F = 1 means identical distributions.
    """
    coefficient = np.sum(
        np.sqrt(p * q)
    )

    return float(
        coefficient**2
    )


def safe_hex(candidate: int) -> str:
    """
    Format 8-bit candidate as two-digit hexadecimal.
    """
    return f"{candidate:02X}"


# =====================================================================
# Main
# =====================================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Load exp02 correlation-state information
    # -----------------------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}\n\n"
            "Run exp02_correlation_state.py first."
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = {
        "candidate",
        "candidate_hex",
        "quantum_amplitude",
        "quantum_probability",
        "is_true_key",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    # Computational-basis order:
    #
    # 0, 1, ..., 255
    #
    # corresponds to
    #
    # |00000000>, |00000001>, ..., |11111111>
    #
    df = (
        df.sort_values("candidate")
        .reset_index(drop=True)
    )

    n_candidates = len(df)

    # -----------------------------------------------------------------
    # Check candidate-space dimension
    # -----------------------------------------------------------------

    n_qubits = int(
        round(
            math.log2(
                n_candidates
            )
        )
    )

    if 2**n_qubits != n_candidates:
        raise ValueError(
            "Number of candidates must be "
            "a power of two."
        )

    print()
    print(
        f"Candidate space : {n_candidates}"
    )

    print(
        f"Number of qubits: {n_qubits}"
    )

    # -----------------------------------------------------------------
    # Load target amplitudes
    # -----------------------------------------------------------------

    target_amplitudes = (
        df["quantum_amplitude"]
        .to_numpy(
            dtype=np.complex128
        )
    )

    theoretical_probabilities = (
        df["quantum_probability"]
        .to_numpy(
            dtype=np.float64
        )
    )

    # -----------------------------------------------------------------
    # Check normalization
    # -----------------------------------------------------------------

    state_norm = float(
        np.sum(
            np.abs(
                target_amplitudes
            ) ** 2
        )
    )

    probability_sum = float(
        theoretical_probabilities.sum()
    )

    if not np.isclose(
        state_norm,
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Input amplitudes are not normalized. "
            f"Norm = {state_norm}"
        )

    if not np.isclose(
        probability_sum,
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Input probabilities do not sum to 1. "
            f"Sum = {probability_sum}"
        )

    # =================================================================
    # Step 1
    # Construct ideal quantum correlation state
    # =================================================================

    preparation_circuit = QuantumCircuit(
        n_qubits
    )

    state_preparation = StatePreparation(
        target_amplitudes
    )

    preparation_circuit.append(
        state_preparation,
        range(n_qubits),
    )

    # =================================================================
    # Step 2
    # Exact statevector validation
    # =================================================================
    #
    # Before sampling, verify that Qiskit's StatePreparation generated
    # the desired signed amplitudes.
    #
    # This checks amplitude signs/phases as well as probabilities.
    #
    # =================================================================

    exact_state = Statevector.from_instruction(
        preparation_circuit
    )

    exact_amplitudes = (
        exact_state.data
    )

    overlap = np.vdot(
        target_amplitudes,
        exact_amplitudes,
    )

    state_fidelity = float(
        np.abs(overlap) ** 2
    )

    max_amplitude_error = float(
        np.max(
            np.abs(
                target_amplitudes
                - exact_amplitudes
            )
        )
    )

    print()
    print(
        "[Exact statevector validation]"
    )

    print(
        f"State fidelity       : "
        f"{state_fidelity:.12f}"
    )

    print(
        f"Max amplitude error  : "
        f"{max_amplitude_error:.3e}"
    )

    # =================================================================
    # Step 3
    # Add measurements
    # =================================================================

    measurement_circuit = (
        preparation_circuit.copy()
    )

    measurement_circuit.measure_all()

    # =================================================================
    # Step 4
    # Aer simulator
    # =================================================================

    simulator = AerSimulator()

    transpiled_circuit = transpile(
        measurement_circuit,
        backend=simulator,
        optimization_level=1,
        seed_transpiler=SEED,
    )

    # -----------------------------------------------------------------
    # Circuit statistics
    # -----------------------------------------------------------------

    transpiled_depth = (
        transpiled_circuit.depth()
    )

    transpiled_size = (
        transpiled_circuit.size()
    )

    operation_counts_raw = (
        transpiled_circuit.count_ops()
    )

    operation_counts = {
        str(key): int(value)
        for key, value
        in operation_counts_raw.items()
    }

    print()
    print(
        "[Transpiled circuit]"
    )

    print(
        f"Depth                : "
        f"{transpiled_depth}"
    )

    print(
        f"Number of operations : "
        f"{transpiled_size}"
    )

    print(
        f"Operation counts     : "
        f"{operation_counts}"
    )

    # =================================================================
    # Step 5
    # Finite-shot simulation
    # =================================================================

    job = simulator.run(
        transpiled_circuit,
        shots=SHOTS,
        seed_simulator=SEED,
    )

    result = job.result()

    counts = result.get_counts(
        transpiled_circuit
    )

    # =================================================================
    # Step 6
    # Convert Qiskit bitstrings to candidate indices
    # =================================================================

    measured_counts = np.zeros(
        n_candidates,
        dtype=np.int64,
    )

    for bitstring, count in counts.items():

        # Remove spaces in case Qiskit uses multiple
        # classical registers.
        clean_bitstring = (
            bitstring.replace(
                " ",
                "",
            )
        )

        candidate = int(
            clean_bitstring,
            2,
        )

        measured_counts[
            candidate
        ] += count

    measured_probabilities = (
        measured_counts
        / SHOTS
    )

    # =================================================================
    # Step 7
    # Distribution comparison
    # =================================================================

    tvd = total_variation_distance(
        theoretical_probabilities,
        measured_probabilities,
    )

    rmse = probability_rmse(
        theoretical_probabilities,
        measured_probabilities,
    )

    fidelity = classical_fidelity(
        theoretical_probabilities,
        measured_probabilities,
    )

    max_probability_error = float(
        np.max(
            np.abs(
                theoretical_probabilities
                - measured_probabilities
            )
        )
    )

    # =================================================================
    # True key
    # =================================================================

    true_mask = (
        df["is_true_key"]
        .astype(bool)
        .to_numpy()
    )

    true_indices = np.flatnonzero(
        true_mask
    )

    if len(true_indices) != 1:
        raise RuntimeError(
            "Expected exactly one true-key candidate."
        )

    true_index = int(
        true_indices[0]
    )

    true_candidate = int(
        df.iloc[
            true_index
        ]["candidate"]
    )

    true_hex = safe_hex(
        true_candidate
    )

    true_theoretical_probability = float(
        theoretical_probabilities[
            true_index
        ]
    )

    true_measured_probability = float(
        measured_probabilities[
            true_index
        ]
    )

    true_count = int(
        measured_counts[
            true_index
        ]
    )

    # =================================================================
    # Probability ranking
    # =================================================================

    theoretical_rank = (
        pd.Series(
            theoretical_probabilities
        )
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
        .to_numpy()
    )

    measured_rank = (
        pd.Series(
            measured_probabilities
        )
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
        .to_numpy()
    )

    true_theoretical_rank = int(
        theoretical_rank[
            true_index
        ]
    )

    true_measured_rank = int(
        measured_rank[
            true_index
        ]
    )

    # =================================================================
    # Save candidate-level results
    # =================================================================

    output_df = pd.DataFrame(
        {
            "candidate":
                df["candidate"].astype(int),

            "candidate_hex":
                [
                    safe_hex(int(x))
                    for x
                    in df["candidate"]
                ],

            "target_amplitude":
                np.real(
                    target_amplitudes
                ),

            "theoretical_probability":
                theoretical_probabilities,

            "measured_count":
                measured_counts,

            "measured_probability":
                measured_probabilities,

            "absolute_probability_error":
                np.abs(
                    theoretical_probabilities
                    - measured_probabilities
                ),

            "theoretical_rank":
                theoretical_rank,

            "measured_rank":
                measured_rank,

            "is_true_key":
                df[
                    "is_true_key"
                ].astype(bool),
        }
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =================================================================
    # Summary JSON
    # =================================================================

    summary = {

        "experiment":
            "exp04_qiskit_state_validation",

        "shots":
            SHOTS,

        "seed":
            SEED,

        "n_candidates":
            n_candidates,

        "n_qubits":
            n_qubits,

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            true_hex,

        "true_theoretical_probability":
            true_theoretical_probability,

        "true_measured_probability":
            true_measured_probability,

        "true_measured_count":
            true_count,

        "true_theoretical_rank":
            true_theoretical_rank,

        "true_measured_rank":
            true_measured_rank,

        "exact_state_fidelity":
            state_fidelity,

        "max_amplitude_error":
            max_amplitude_error,

        "total_variation_distance":
            tvd,

        "probability_rmse":
            rmse,

        "probability_fidelity":
            fidelity,

        "max_probability_error":
            max_probability_error,

        "transpiled_depth":
            int(
                transpiled_depth
            ),

        "transpiled_size":
            int(
                transpiled_size
            ),

        "transpiled_operation_counts":
            operation_counts,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # =================================================================
    # Figure 1
    # Full probability distribution
    # =================================================================

    candidates = np.arange(
        n_candidates
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        candidates,
        theoretical_probabilities,
        linewidth=1.5,
        label="Theory",
    )

    ax.scatter(
        candidates,
        measured_probabilities,
        s=10,
        alpha=0.7,
        label="Qiskit / Aer",
    )

    ax.set_xlabel(
        "8-bit subkey candidate"
    )

    ax.set_ylabel(
        "Measurement probability"
    )

    ax.set_title(
        "Theoretical vs Qiskit Correlation-State Distribution"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_FULL,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Top-K candidates
    # =================================================================

    top_indices = np.argsort(
        theoretical_probabilities
    )[::-1][:TOP_K]

    top_theory = (
        theoretical_probabilities[
            top_indices
        ]
    )

    top_measured = (
        measured_probabilities[
            top_indices
        ]
    )

    labels = [
        f"0x{safe_hex(int(i))}"
        for i
        in top_indices
    ]

    x = np.arange(
        TOP_K
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        x - width / 2,
        top_theory,
        width=width,
        label="Theory",
    )

    ax.bar(
        x + width / 2,
        top_measured,
        width=width,
        label="Qiskit / Aer",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Subkey candidate"
    )

    ax.set_ylabel(
        "Measurement probability"
    )

    ax.set_title(
        f"Top {TOP_K} Correlation-State Candidates"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_TOP,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Console report
    # =================================================================

    print()
    print("=" * 78)
    print(
        "EXP04: Qiskit Correlation-State Validation"
    )
    print("=" * 78)

    print()
    print(
        "[Quantum system]"
    )

    print(
        f"Candidates             : "
        f"{n_candidates}"
    )

    print(
        f"Qubits                 : "
        f"{n_qubits}"
    )

    print(
        f"Shots                  : "
        f"{SHOTS}"
    )

    print()
    print(
        "[True key]"
    )

    print(
        f"Candidate              : "
        f"0x{true_hex}"
    )

    print(
        f"Theoretical probability: "
        f"{true_theoretical_probability:.8f} "
        f"({100 * true_theoretical_probability:.4f}%)"
    )

    print(
        f"Measured probability   : "
        f"{true_measured_probability:.8f} "
        f"({100 * true_measured_probability:.4f}%)"
    )

    print(
        f"Measured count         : "
        f"{true_count}"
    )

    print(
        f"Theoretical rank       : "
        f"{true_theoretical_rank}"
    )

    print(
        f"Measured rank          : "
        f"{true_measured_rank}"
    )

    print()
    print(
        "[Exact state validation]"
    )

    print(
        f"State fidelity         : "
        f"{state_fidelity:.12f}"
    )

    print(
        f"Max amplitude error    : "
        f"{max_amplitude_error:.3e}"
    )

    print()
    print(
        "[Finite-shot distribution comparison]"
    )

    print(
        f"TVD                    : "
        f"{tvd:.8f}"
    )

    print(
        f"Probability RMSE       : "
        f"{rmse:.8e}"
    )

    print(
        f"Distribution fidelity  : "
        f"{fidelity:.8f}"
    )

    print(
        f"Max probability error  : "
        f"{max_probability_error:.8f}"
    )

    print()
    print(
        "[Circuit complexity]"
    )

    print(
        f"Transpiled depth       : "
        f"{transpiled_depth}"
    )

    print(
        f"Transpiled size        : "
        f"{transpiled_size}"
    )

    print(
        f"Operation counts       : "
        f"{operation_counts}"
    )

    print()
    print(
        "[Output files]"
    )

    print(
        f"Candidate results      : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary JSON           : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Full distribution fig. : "
        f"{FIG_FULL}"
    )

    print(
        f"Top candidates fig.    : "
        f"{FIG_TOP}"
    )

    print()
    print(
        "[PASS] exp04 completed successfully."
    )


if __name__ == "__main__":
    main()