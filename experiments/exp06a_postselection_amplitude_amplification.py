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

INPUT_JSON = RESULTS_DIR / "exp05_summary.json"

OUTPUT_CSV = RESULTS_DIR / "exp06a_amplitude_amplification.csv"
OUTPUT_JSON = RESULTS_DIR / "exp06a_summary.json"

FIG_SUCCESS = FIGURES_DIR / "exp06a_aa_success_probability.png"
FIG_COMPARISON = FIGURES_DIR / "exp06a_naive_vs_aa.png"


# =====================================================================
# Configuration
# =====================================================================

TARGET_SUCCESS_LEVELS = [
    0.50,
    0.90,
    0.95,
    0.99,
]

# Number of iterations beyond the first theoretical peak
EXTRA_ITERATIONS = 40


# =====================================================================
# Utility functions
# =====================================================================

def amplitude_amplification_probability(
    initial_probability: float,
    iterations: int,
) -> float:
    """
    Ideal amplitude amplification success probability.

    Let

        sin^2(theta) = p

    where p is the initial good-state probability.

    After t Grover/amplitude-amplification iterations,

        P_AA(t) = sin^2((2t + 1) theta).
    """

    theta = math.asin(
        math.sqrt(
            initial_probability
        )
    )

    return math.sin(
        (2 * iterations + 1)
        * theta
    ) ** 2


def naive_runs_for_success(
    p: float,
    target_success: float,
) -> int:
    """
    Number of independent repetitions required so that

        P(at least one success) >= target_success.

    For independent repetitions:

        P_success = 1 - (1-p)^R
    """

    if p <= 0:
        raise ValueError(
            "Probability must be positive."
        )

    if p >= 1:
        return 1

    return math.ceil(
        math.log(
            1.0 - target_success
        )
        /
        math.log(
            1.0 - p
        )
    )


def first_iteration_for_success(
    probabilities: np.ndarray,
    threshold: float,
) -> int | None:
    """
    Return the first AA iteration count whose success probability
    reaches the requested threshold.
    """

    indices = np.flatnonzero(
        probabilities >= threshold
    )

    if len(indices) == 0:
        return None

    return int(
        indices[0]
    )


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
    # Load EXP05 result
    # -----------------------------------------------------------------

    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_JSON}\n\n"
            "Run EXP05 first."
        )

    with open(
        INPUT_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        exp05 = json.load(f)

    required_fields = {
        "postselection_probability",
        "true_postselected_probability",
        "unconditional_true_key_probability_per_full_run",
    }

    missing = (
        required_fields
        - set(exp05.keys())
    )

    if missing:
        raise ValueError(
            "EXP05 summary is missing fields:\n"
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------------------
    # Extract probabilities
    # -----------------------------------------------------------------

    p_good = float(
        exp05[
            "postselection_probability"
        ]
    )

    p_true_given_good = float(
        exp05[
            "true_postselected_probability"
        ]
    )

    p_true_full = float(
        exp05[
            "unconditional_true_key_probability_per_full_run"
        ]
    )

    if not (
        0.0 < p_good < 1.0
    ):
        raise ValueError(
            "Postselection probability must lie in (0, 1)."
        )

    # =================================================================
    # Step 1
    # Compute amplitude angle
    # =================================================================

    theta = math.asin(
        math.sqrt(
            p_good
        )
    )

    # -----------------------------------------------------------------
    # Approximate optimal iteration count
    #
    # We want
    #
    #   (2t + 1) theta ≈ pi / 2
    #
    # giving
    #
    #   t ≈ pi/(4 theta) - 1/2
    # -----------------------------------------------------------------

    t_opt_continuous = (
        math.pi
        / (4.0 * theta)
        - 0.5
    )

    t_opt_rounded = int(
        round(
            t_opt_continuous
        )
    )

    # Include enough iterations to show Grover overshoot
    t_max = (
        max(
            10,
            2 * t_opt_rounded
            + EXTRA_ITERATIONS
        )
    )

    iterations = np.arange(
        0,
        t_max + 1,
        dtype=int,
    )

    # =================================================================
    # Step 2
    # Ideal AA success probability
    # =================================================================

    aa_probabilities = np.asarray(
        [
            amplitude_amplification_probability(
                initial_probability=p_good,
                iterations=int(t),
            )
            for t in iterations
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------------------
    # Probability of obtaining true key in one amplified run
    #
    # Important:
    #
    # AA only amplifies the GOOD SECTOR.
    #
    # It does not modify the relative distribution of key candidates
    # inside that sector.
    #
    # Therefore:
    #
    # P(true key after AA)
    #
    #   = P_AA(good)
    #     * P(true key | good)
    # -----------------------------------------------------------------

    true_key_probabilities = (
        aa_probabilities
        * p_true_given_good
    )

    # =================================================================
    # Step 3
    # Locate the numerical optimum
    # =================================================================

    best_index = int(
        np.argmax(
            aa_probabilities
        )
    )

    best_iteration = int(
        iterations[
            best_index
        ]
    )

    best_good_probability = float(
        aa_probabilities[
            best_index
        ]
    )

    best_true_probability = float(
        true_key_probabilities[
            best_index
        ]
    )

    # =================================================================
    # Step 4
    # Threshold analysis
    # =================================================================

    threshold_results = {}

    for level in TARGET_SUCCESS_LEVELS:

        aa_iteration = (
            first_iteration_for_success(
                probabilities=aa_probabilities,
                threshold=level,
            )
        )

        naive_runs = (
            naive_runs_for_success(
                p=p_good,
                target_success=level,
            )
        )

        name = (
            f"{int(level * 100)}pct"
        )

        threshold_results[
            name
        ] = {
            "target_success_probability":
                level,

            "naive_repetitions":
                naive_runs,

            "first_aa_iteration":
                aa_iteration,

            "aa_success_probability":
                (
                    None
                    if aa_iteration is None
                    else float(
                        aa_probabilities[
                            aa_iteration
                        ]
                    )
                ),
        }

    # =================================================================
    # Step 5
    # Textbook AA operation-count proxy
    # =================================================================
    #
    # A typical amplitude-amplification iteration contains
    #
    #   A
    #   S_good
    #   A^dagger
    #   S_0
    #
    # around the initial preparation A.
    #
    # Therefore, if only calls to A / A^dagger are counted:
    #
    #   preparation-family calls = 1 + 2t
    #
    # This is NOT a physical gate count.
    #
    # It is only a logical cost proxy.
    # =================================================================

    preparation_family_calls = (
        1
        + 2 * iterations
    )

    # =================================================================
    # Step 6
    # Save iteration-level results
    # =================================================================

    output_df = pd.DataFrame(
        {
            "aa_iterations":
                iterations,

            "aa_good_sector_probability":
                aa_probabilities,

            "true_key_conditional_probability":
                np.full(
                    len(iterations),
                    p_true_given_good,
                ),

            "unconditional_true_key_probability":
                true_key_probabilities,

            "preparation_family_calls_proxy":
                preparation_family_calls,
        }
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =================================================================
    # Step 7
    # Summary
    # =================================================================

    summary = {

        "experiment":
            "exp06a_postselection_amplitude_amplification",

        "initial_good_sector_probability":
            p_good,

        "initial_true_key_given_good_probability":
            p_true_given_good,

        "initial_unconditional_true_key_probability":
            p_true_full,

        "theta_rad":
            theta,

        "continuous_optimal_iteration_estimate":
            t_opt_continuous,

        "rounded_optimal_iteration_estimate":
            t_opt_rounded,

        "numerical_best_iteration":
            best_iteration,

        "numerical_best_good_sector_probability":
            best_good_probability,

        "true_key_probability_at_best_iteration":
            best_true_probability,

        "conditional_true_key_probability_after_aa":
            p_true_given_good,

        "key_distribution_changed_inside_good_sector":
            False,

        "threshold_comparison":
            threshold_results,

        "important_interpretation": (
            "Amplitude amplification increases the probability "
            "of reaching the correlation-state sector, but does "
            "not increase the relative probability of the correct "
            "key within that sector."
        ),

        "cost_warning": (
            "The AA iteration count is not a physical gate count. "
            "Each iteration requires state preparation/inversion "
            "and reflections. Coherent data access and attack-oracle "
            "costs must be included in an end-to-end comparison."
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
    # AA success probability versus iteration
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        iterations,
        aa_probabilities,
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Correlation-sector probability",
    )

    ax.axvline(
        best_iteration,
        linestyle="--",
        linewidth=1.0,
        label=(
            f"Best iteration = "
            f"{best_iteration}"
        ),
    )

    ax.axhline(
        p_good,
        linestyle=":",
        linewidth=1.0,
        label=(
            f"Initial probability = "
            f"{p_good:.3e}"
        ),
    )

    ax.set_xlabel(
        "Amplitude-amplification iterations"
    )

    ax.set_ylabel(
        "Probability of correlation sector"
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_title(
        "EXP06A: Correlation-Sector Amplitude Amplification"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_SUCCESS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Good-sector probability and true-key probability
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        iterations,
        aa_probabilities,
        linewidth=1.5,
        label="P(correlation sector)",
    )

    ax.plot(
        iterations,
        true_key_probabilities,
        linewidth=1.5,
        label="P(correlation sector AND true key)",
    )

    ax.axhline(
        p_true_given_good,
        linestyle="--",
        linewidth=1.0,
        label=(
            "Maximum possible true-key probability "
            "without key-selective amplification"
        ),
    )

    ax.set_xlabel(
        "Amplitude-amplification iterations"
    )

    ax.set_ylabel(
        "Probability"
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_title(
        "Preparation Amplification Does Not Solve Key Extraction"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_COMPARISON,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Console report
    # =================================================================

    print("=" * 78)
    print(
        "EXP06A: Correlation-Sector Amplitude Amplification"
    )
    print("=" * 78)

    print()
    print("[EXP05 input]")

    print(
        f"Initial P(good sector)              : "
        f"{p_good:.10e}"
    )

    print(
        f"P(true key | good sector)           : "
        f"{p_true_given_good:.8f} "
        f"({100 * p_true_given_good:.4f}%)"
    )

    print(
        f"Initial P(good AND true key)        : "
        f"{p_true_full:.10e}"
    )

    print()
    print("[Amplitude geometry]")

    print(
        f"theta                               : "
        f"{theta:.10f} rad"
    )

    print(
        f"Continuous optimum estimate         : "
        f"{t_opt_continuous:.4f}"
    )

    print(
        f"Rounded optimum estimate            : "
        f"{t_opt_rounded}"
    )

    print()
    print("[Numerical optimum]")

    print(
        f"Best AA iteration                   : "
        f"{best_iteration}"
    )

    print(
        f"P(good) at best iteration           : "
        f"{best_good_probability:.10f} "
        f"({100 * best_good_probability:.6f}%)"
    )

    print(
        f"P(true key | good) after AA         : "
        f"{p_true_given_good:.10f} "
        f"({100 * p_true_given_good:.6f}%)"
    )

    print(
        f"P(good AND true key) after AA       : "
        f"{best_true_probability:.10f} "
        f"({100 * best_true_probability:.6f}%)"
    )

    print()
    print("[Naive repetition vs ideal AA]")

    for name, result in threshold_results.items():

        level = (
            100
            * result[
                "target_success_probability"
            ]
        )

        print()
        print(
            f"Target success = {level:.0f}%"
        )

        print(
            f"  Naive repetitions                 : "
            f"{result['naive_repetitions']}"
        )

        print(
            f"  First AA iteration                : "
            f"{result['first_aa_iteration']}"
        )

        if (
            result[
                "aa_success_probability"
            ]
            is not None
        ):

            print(
                f"  AA success probability            : "
                f"{result['aa_success_probability']:.8f}"
            )

    print()
    print("[Key result]")

    print(
        "Amplitude amplification strongly increases "
        "P(correlation sector),"
    )

    print(
        "but P(true key | correlation sector) remains "
        "unchanged."
    )

    print()

    print(
        f"Before AA:"
    )

    print(
        f"    P(good)                         = "
        f"{p_good:.6e}"
    )

    print(
        f"    P(true | good)                  = "
        f"{p_true_given_good:.6f}"
    )

    print()

    print(
        f"Near optimal AA:"
    )

    print(
        f"    P(good)                         = "
        f"{best_good_probability:.6f}"
    )

    print(
        f"    P(true | good)                  = "
        f"{p_true_given_good:.6f}"
    )

    print()
    print("[Important cost warning]")

    print(
        "The AA iteration number is NOT equivalent "
        "to a physical circuit depth or oracle cost."
    )

    print(
        "Each AA iteration requires state-preparation "
        "inversion and reflections."
    )

    print(
        "Data-access and reversible cryptanalytic-oracle "
        "costs remain part of the end-to-end attack."
    )

    print()
    print("[Output files]")

    print(
        f"Iteration results                   : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary JSON                        : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"AA success figure                   : "
        f"{FIG_SUCCESS}"
    )

    print(
        f"Preparation/key comparison figure   : "
        f"{FIG_COMPARISON}"
    )

    print()
    print(
        "[PASS] EXP06A completed successfully."
    )


if __name__ == "__main__":
    main()