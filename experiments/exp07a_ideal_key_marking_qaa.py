from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

EXP05_CSV = RESULTS_DIR / "exp05_coherent_probabilities.csv"

OUTPUT_CSV = RESULTS_DIR / "exp07a_ideal_key_qaa.csv"
OUTPUT_JSON = RESULTS_DIR / "exp07a_summary.json"

FIG_PROB = FIGURES_DIR / "exp07a_true_key_probability.png"
FIG_COST = FIGURES_DIR / "exp07a_key_extraction_cost.png"


# =====================================================================
# Configuration
# =====================================================================

MAX_ITERATIONS = 20

TARGET_LEVELS = [
    0.50,
    0.90,
    0.95,
    0.99,
]


# =====================================================================
# Utility functions
# =====================================================================

def qaa_probability(
    initial_probability: float,
    iterations: int,
) -> float:
    """
    Ideal Quantum Amplitude Amplification (QAA).

    If

        sin^2(theta) = p

    then after t QAA iterations,

        P(t) = sin^2((2t + 1) theta).
    """

    theta = math.asin(
        math.sqrt(initial_probability)
    )

    return math.sin(
        (2 * iterations + 1) * theta
    ) ** 2


def first_iteration_for_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> int | None:

    indices = np.flatnonzero(
        probabilities >= threshold
    )

    if len(indices) == 0:
        return None

    return int(indices[0])


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

    # =================================================================
    # Step 1
    # Load correlation-state key distribution from EXP05
    # =================================================================

    if not EXP05_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP05_CSV}\n\n"
            "Run EXP05 first."
        )

    df = pd.read_csv(
        EXP05_CSV
    )

    required_columns = {
        "candidate",
        "candidate_hex",
        "coherent_probability",
        "is_true_key",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    df = (
        df
        .sort_values("candidate")
        .reset_index(drop=True)
    )

    probabilities = (
        df["coherent_probability"]
        .to_numpy(
            dtype=np.float64
        )
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    # =================================================================
    # Step 2
    # Identify the true key
    # =================================================================

    true_indices = np.flatnonzero(
        df["is_true_key"]
        .astype(bool)
        .to_numpy()
    )

    if len(true_indices) != 1:

        raise RuntimeError(
            "Expected exactly one true key."
        )

    true_index = int(
        true_indices[0]
    )

    true_candidate = int(
        df.iloc[
            true_index
        ]["candidate"]
    )

    p_true = float(
        probabilities[
            true_index
        ]
    )

    # -------------------------------------------------------------
    # Strongest wrong-key competitor
    # -------------------------------------------------------------

    wrong_probs = probabilities.copy()

    wrong_probs[
        true_index
    ] = -1.0

    second_index = int(
        np.argmax(
            wrong_probs
        )
    )

    second_candidate = int(
        df.iloc[
            second_index
        ]["candidate"]
    )

    p_second = float(
        probabilities[
            second_index
        ]
    )

    # =================================================================
    # Step 3
    # Initial amplitude geometry
    # =================================================================

    theta = math.asin(
        math.sqrt(
            p_true
        )
    )

    continuous_optimum = (
        math.pi
        / (4.0 * theta)
        - 0.5
    )

    approximate_optimum = int(
        round(
            continuous_optimum
        )
    )

    # =================================================================
    # Step 4
    # Sweep ideal key-level QAA
    # =================================================================

    iterations = np.arange(
        0,
        MAX_ITERATIONS + 1,
        dtype=int,
    )

    qaa_probabilities = np.asarray(
        [
            qaa_probability(
                initial_probability=p_true,
                iterations=int(t),
            )
            for t in iterations
        ],
        dtype=np.float64,
    )

    # =================================================================
    # Step 5
    # Logical extraction-cost proxies
    # =================================================================
    #
    # If A_key prepares the correlation state,
    # textbook AA uses approximately
    #
    #     1 + 2t
    #
    # calls to A_key / A_key^dagger family
    #
    # for one amplified attempt.
    #
    # Since an amplified attempt still succeeds only with P_QAA(t),
    #
    # expected cost per true-key hit:
    #
    #     (1 + 2t) / P_QAA(t)
    #
    #
    # IMPORTANT:
    # This is NOT a physical gate count.
    # =================================================================

    key_state_family_calls = (
        1
        + 2 * iterations
    )

    expected_calls_per_true_hit = (
        key_state_family_calls
        / qaa_probabilities
    )

    expected_attempts_per_true_hit = (
        1.0
        / qaa_probabilities
    )

    # =================================================================
    # Step 6
    # Best probability point
    # =================================================================

    best_probability_idx = int(
        np.argmax(
            qaa_probabilities
        )
    )

    best_probability_iteration = int(
        iterations[
            best_probability_idx
        ]
    )

    best_probability = float(
        qaa_probabilities[
            best_probability_idx
        ]
    )

    # =================================================================
    # Step 7
    # Cost-optimal key-level QAA
    # =================================================================

    cost_optimal_idx = int(
        np.argmin(
            expected_calls_per_true_hit
        )
    )

    cost_optimal_iteration = int(
        iterations[
            cost_optimal_idx
        ]
    )

    cost_optimal_probability = float(
        qaa_probabilities[
            cost_optimal_idx
        ]
    )

    cost_optimal_expected_calls = float(
        expected_calls_per_true_hit[
            cost_optimal_idx
        ]
    )

    # =================================================================
    # Step 8
    # Direct-sampling baseline
    # =================================================================

    # Expected number of correlation-state measurements
    # required to observe the true key once.

    direct_expected_samples_per_hit = (
        1.0 / p_true
    )

    # In the logical proxy:
    #
    # direct sampling:
    #
    #     one prepared correlation state per sample
    #
    # so expected state-preparation count is also 1/p_true.

    direct_expected_state_calls = (
        direct_expected_samples_per_hit
    )

    ideal_qaa_reduction = (
        direct_expected_state_calls
        / cost_optimal_expected_calls
    )

    # =================================================================
    # Step 9
    # Threshold analysis
    # =================================================================

    threshold_results = {}

    for level in TARGET_LEVELS:

        iteration = (
            first_iteration_for_threshold(
                probabilities=qaa_probabilities,
                threshold=level,
            )
        )

        name = (
            f"{int(level * 100)}pct"
        )

        if iteration is None:

            threshold_results[
                name
            ] = {
                "first_iteration":
                    None,
            }

            continue

        threshold_results[
            name
        ] = {

            "first_iteration":
                iteration,

            "success_probability":
                float(
                    qaa_probabilities[
                        iteration
                    ]
                ),

            "key_state_family_calls_per_attempt":
                int(
                    key_state_family_calls[
                        iteration
                    ]
                ),

            "expected_key_state_family_calls_per_true_hit":
                float(
                    expected_calls_per_true_hit[
                        iteration
                    ]
                ),
        }

    # =================================================================
    # Step 10
    # Save iteration-level CSV
    # =================================================================

    output_df = pd.DataFrame(
        {
            "qaa_iterations":
                iterations,

            "true_key_probability":
                qaa_probabilities,

            "key_state_family_calls_per_attempt":
                key_state_family_calls,

            "expected_attempts_per_true_key_hit":
                expected_attempts_per_true_hit,

            "expected_key_state_family_calls_per_true_hit":
                expected_calls_per_true_hit,
        }
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =================================================================
    # Step 11
    # Save summary
    # =================================================================

    summary = {

        "experiment":
            "exp07a_ideal_key_marking_qaa",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "initial_true_key_probability":
            p_true,

        "second_candidate":
            second_candidate,

        "second_candidate_hex":
            f"{second_candidate:02X}",

        "second_candidate_probability":
            p_second,

        "initial_true_to_second_probability_ratio":
            p_true / p_second,

        "theta_rad":
            theta,

        "continuous_optimal_iteration_estimate":
            continuous_optimum,

        "rounded_optimal_iteration_estimate":
            approximate_optimum,

        "best_probability_iteration":
            best_probability_iteration,

        "best_probability":
            best_probability,

        "cost_optimal_iteration":
            cost_optimal_iteration,

        "cost_optimal_true_key_probability":
            cost_optimal_probability,

        "cost_optimal_expected_key_state_family_calls":
            cost_optimal_expected_calls,

        "direct_sampling_expected_samples_per_true_hit":
            direct_expected_samples_per_hit,

        "ideal_qaa_logical_reduction_factor":
            ideal_qaa_reduction,

        "thresholds":
            threshold_results,

        "critical_assumption": (
            "The QAA oracle directly marks the true key. "
            "This requires prior knowledge of the correct key and "
            "therefore does not constitute a realizable cryptanalytic "
            "attack. EXP07A is only an idealized extraction upper bound."
        ),

        "important_cost_warning": (
            "The key-state-family call metric is a logical proxy. "
            "It assumes access to a unitary correlation-state "
            "preparation operator and its inverse. It must not be "
            "directly multiplied by the expected postselection cost "
            "from EXP06C, because a coherent outer QAA requires a "
            "fully unitary preparation routine rather than a "
            "measurement/postselection procedure."
        ),
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
    # True-key probability under ideal QAA
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        iterations,
        qaa_probabilities,
        marker="o",
        linewidth=1.5,
        label="Ideal key-marking QAA",
    )

    ax.axhline(
        p_true,
        linestyle=":",
        linewidth=1.0,
        label=(
            f"Initial true-key probability "
            f"= {100 * p_true:.2f}%"
        ),
    )

    ax.axvline(
        best_probability_iteration,
        linestyle="--",
        linewidth=1.0,
        label=(
            f"Max probability t="
            f"{best_probability_iteration}"
        ),
    )

    ax.axvline(
        cost_optimal_iteration,
        linestyle="-.",
        linewidth=1.0,
        label=(
            f"Cost-optimal t="
            f"{cost_optimal_iteration}"
        ),
    )

    ax.set_xlabel(
        "Ideal key-level QAA iterations"
    )

    ax.set_ylabel(
        "True-key measurement probability"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "EXP07A: Ideal True-Key Amplitude Amplification"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_PROB,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Expected logical cost per true-key hit
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        iterations,
        expected_calls_per_true_hit,
        marker="o",
        linewidth=1.5,
        label="Ideal QAA logical cost",
    )

    ax.axhline(
        direct_expected_state_calls,
        linestyle="--",
        linewidth=1.0,
        label=(
            "Direct-sampling baseline"
        ),
    )

    ax.axvline(
        cost_optimal_iteration,
        linestyle=":",
        linewidth=1.0,
        label=(
            f"Cost-optimal t="
            f"{cost_optimal_iteration}"
        ),
    )

    ax.set_xlabel(
        "Ideal key-level QAA iterations"
    )

    ax.set_ylabel(
        "Expected correlation-state-family calls per true-key hit"
    )

    ax.set_title(
        "Ideal Key Extraction: Direct Sampling vs QAA"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_COST,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Console report
    # =================================================================

    print("=" * 78)
    print(
        "EXP07A: Ideal True-Key Marking QAA"
    )
    print("=" * 78)

    print()
    print("[Initial correlation-state distribution]")

    print(
        f"True key                         : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"Initial P(true key)              : "
        f"{p_true:.8f} "
        f"({100 * p_true:.4f}%)"
    )

    print(
        f"Strongest wrong key              : "
        f"0x{second_candidate:02X}"
    )

    print(
        f"P(second key)                    : "
        f"{p_second:.8f} "
        f"({100 * p_second:.4f}%)"
    )

    print()
    print("[Direct sampling]")

    print(
        f"Expected samples / true-key hit  : "
        f"{direct_expected_samples_per_hit:.4f}"
    )

    print()
    print("[Ideal key-level QAA geometry]")

    print(
        f"theta                            : "
        f"{theta:.8f} rad"
    )

    print(
        f"Continuous optimum estimate      : "
        f"{continuous_optimum:.4f}"
    )

    print(
        f"Rounded optimum estimate         : "
        f"{approximate_optimum}"
    )

    print()
    print("[Maximum-probability point]")

    print(
        f"QAA iterations                   : "
        f"{best_probability_iteration}"
    )

    print(
        f"P(true key)                      : "
        f"{best_probability:.8f} "
        f"({100 * best_probability:.4f}%)"
    )

    print()
    print("[Cost-optimal point]")

    print(
        f"QAA iterations                   : "
        f"{cost_optimal_iteration}"
    )

    print(
        f"P(true key)                      : "
        f"{cost_optimal_probability:.8f} "
        f"({100 * cost_optimal_probability:.4f}%)"
    )

    print(
        f"Expected logical calls / hit     : "
        f"{cost_optimal_expected_calls:.4f}"
    )

    print(
        f"Ideal reduction vs direct sample : "
        f"{ideal_qaa_reduction:.4f} x"
    )

    print()
    print("[Thresholds]")

    for level in TARGET_LEVELS:

        name = (
            f"{int(level * 100)}pct"
        )

        result = (
            threshold_results[
                name
            ]
        )

        print()

        print(
            f"Target true-key probability "
            f"{100 * level:.0f}%"
        )

        if (
            result[
                "first_iteration"
            ]
            is None
        ):

            print(
                "  Not reached within sweep."
            )

        else:

            print(
                f"  First iteration             : "
                f"{result['first_iteration']}"
            )

            print(
                f"  Achieved probability        : "
                f"{result['success_probability']:.8f}"
            )

            print(
                f"  Expected logical calls/hit  : "
                f"{result['expected_key_state_family_calls_per_true_hit']:.4f}"
            )

    print()
    print("[CRITICAL ASSUMPTION]")

    print(
        "This experiment assumes an oracle that already "
        "knows and marks the correct key."
    )

    print(
        "Therefore EXP07A is NOT a realizable "
        "cryptanalytic attack."
    )

    print(
        "It provides an idealized upper bound on how "
        "efficient key extraction could become."
    )

    print()
    print("[Output files]")

    print(
        f"Iteration results               : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary JSON                    : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Probability figure              : "
        f"{FIG_PROB}"
    )

    print(
        f"Cost figure                     : "
        f"{FIG_COST}"
    )

    print()
    print(
        "[PASS] EXP07A completed successfully."
    )


if __name__ == "__main__":
    main()