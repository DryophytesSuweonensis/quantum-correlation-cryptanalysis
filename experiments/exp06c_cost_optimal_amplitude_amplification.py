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

EXP05_JSON = RESULTS_DIR / "exp05_summary.json"
EXP06A_JSON = RESULTS_DIR / "exp06a_summary.json"

OUTPUT_CSV = RESULTS_DIR / "exp06c_cost_optimal_aa.csv"
OUTPUT_JSON = RESULTS_DIR / "exp06c_summary.json"

FIG_COST = FIGURES_DIR / "exp06c_cost_per_usable_sample.png"
FIG_TRADEOFF = FIGURES_DIR / "exp06c_success_cost_tradeoff.png"


# =====================================================================
# Configuration
# =====================================================================

MAX_ITERATIONS = 80


# =====================================================================
# Utility functions
# =====================================================================

def aa_success_probability(
    p_initial: float,
    iterations: int,
) -> float:
    """
    Ideal amplitude-amplification success probability.

        sin^2(theta) = p_initial

    After t AA iterations:

        P_good(t) = sin^2((2t + 1) theta)
    """

    theta = math.asin(
        math.sqrt(
            p_initial
        )
    )

    return math.sin(
        (2 * iterations + 1)
        * theta
    ) ** 2


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
    # Load previous results
    # -----------------------------------------------------------------

    if not EXP05_JSON.exists():
        raise FileNotFoundError(
            f"Missing:\n{EXP05_JSON}\n\n"
            "Run EXP05 first."
        )

    if not EXP06A_JSON.exists():
        raise FileNotFoundError(
            f"Missing:\n{EXP06A_JSON}\n\n"
            "Run EXP06A first."
        )

    with open(
        EXP05_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        exp05 = json.load(f)

    with open(
        EXP06A_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        exp06a = json.load(f)

    # -----------------------------------------------------------------
    # Input probability
    # -----------------------------------------------------------------

    p_initial = float(
        exp05[
            "postselection_probability"
        ]
    )

    t_max_success = int(
        exp06a[
            "numerical_best_iteration"
        ]
    )

    p_max_success = float(
        exp06a[
            "numerical_best_good_sector_probability"
        ]
    )

    if not (
        0.0 < p_initial < 1.0
    ):
        raise ValueError(
            "Initial postselection probability must lie in (0,1)."
        )

    theta = math.asin(
        math.sqrt(
            p_initial
        )
    )

    # =================================================================
    # Evaluate all iteration counts
    # =================================================================

    rows = []

    for t in range(
        MAX_ITERATIONS + 1
    ):

        p_good = (
            aa_success_probability(
                p_initial=p_initial,
                iterations=t,
            )
        )

        # -------------------------------------------------------------
        # Cost proxies
        # -------------------------------------------------------------
        #
        # Preparation-family proxy:
        #
        #   1 + 2t
        #
        # corresponding conceptually to:
        #
        #   initial A
        #   + repeated A / A^dagger usage
        #
        # This is NOT a physical gate count.
        # -------------------------------------------------------------

        prep_family_calls = (
            1 + 2 * t
        )

        # Expected number of attempts required
        # to obtain one successful usable sample.

        expected_attempts_per_success = (
            1.0 / p_good
        )

        # Expected preparation-family calls
        # per successful usable correlation-state sample.

        prep_calls_per_usable_sample = (
            prep_family_calls
            / p_good
        )

        # AA iterations expended per successful sample.

        aa_iterations_per_usable_sample = (
            t / p_good
        )

        rows.append(
            {
                "aa_iterations":
                    t,

                "good_sector_probability":
                    p_good,

                "prep_family_calls_per_attempt":
                    prep_family_calls,

                "expected_attempts_per_usable_sample":
                    expected_attempts_per_success,

                "expected_prep_family_calls_per_usable_sample":
                    prep_calls_per_usable_sample,

                "expected_aa_iterations_per_usable_sample":
                    aa_iterations_per_usable_sample,
            }
        )

    df = pd.DataFrame(
        rows
    )

    # =================================================================
    # Find cost-optimal iteration
    # =================================================================

    cost_column = (
        "expected_prep_family_calls_per_usable_sample"
    )

    optimal_idx = int(
        df[
            cost_column
        ].idxmin()
    )

    optimal_row = df.loc[
        optimal_idx
    ]

    t_cost_optimal = int(
        optimal_row[
            "aa_iterations"
        ]
    )

    p_cost_optimal = float(
        optimal_row[
            "good_sector_probability"
        ]
    )

    cost_optimal = float(
        optimal_row[
            cost_column
        ]
    )

    aa_iter_cost_optimal = float(
        optimal_row[
            "expected_aa_iterations_per_usable_sample"
        ]
    )

    # =================================================================
    # Compare against no-AA and max-success AA
    # =================================================================

    no_aa_row = df.iloc[0]

    no_aa_cost = float(
        no_aa_row[
            cost_column
        ]
    )

    max_success_row = df[
        df[
            "aa_iterations"
        ]
        == t_max_success
    ].iloc[0]

    max_success_cost = float(
        max_success_row[
            cost_column
        ]
    )

    max_success_probability_check = float(
        max_success_row[
            "good_sector_probability"
        ]
    )

    # Improvement factors

    cost_reduction_vs_no_aa = (
        no_aa_cost
        / cost_optimal
    )

    cost_reduction_vs_max_success = (
        max_success_cost
        / cost_optimal
    )

    # =================================================================
    # Save CSV
    # =================================================================

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =================================================================
    # Save summary
    # =================================================================

    summary = {

        "experiment":
            "exp06c_cost_optimal_amplitude_amplification",

        "initial_good_sector_probability":
            p_initial,

        "theta_rad":
            theta,

        "max_iterations_tested":
            MAX_ITERATIONS,

        "cost_metric":
            "expected preparation-family calls per usable sample",

        "cost_formula":
            "(1 + 2t) / P_good(t)",

        "cost_optimal_iteration":
            t_cost_optimal,

        "cost_optimal_good_sector_probability":
            p_cost_optimal,

        "cost_optimal_expected_prep_calls_per_usable_sample":
            cost_optimal,

        "cost_optimal_expected_aa_iterations_per_usable_sample":
            aa_iter_cost_optimal,

        "no_aa_expected_prep_calls_per_usable_sample":
            no_aa_cost,

        "max_success_iteration":
            t_max_success,

        "max_success_probability":
            p_max_success,

        "max_success_probability_recomputed":
            max_success_probability_check,

        "max_success_expected_prep_calls_per_usable_sample":
            max_success_cost,

        "cost_reduction_vs_no_aa":
            cost_reduction_vs_no_aa,

        "cost_reduction_vs_max_success_aa":
            cost_reduction_vs_max_success,

        "important_interpretation": (
            "The amplitude-amplification iteration count that "
            "maximizes success probability is not necessarily the "
            "iteration count that minimizes expected cost per usable "
            "correlation-state sample."
        ),

        "cost_warning": (
            "The metric uses a logical preparation-family-call proxy "
            "(1 + 2t). It is not a physical gate count. Reversible "
            "data access, reflections, oracle depth, ancilla overhead, "
            "and hardware noise are not included."
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
    # Expected preparation cost per usable sample
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        df[
            "aa_iterations"
        ],
        df[
            cost_column
        ],
        linewidth=1.5,
        label="Expected prep-family calls / usable sample",
    )

    ax.axvline(
        t_cost_optimal,
        linestyle="--",
        linewidth=1.0,
        label=(
            f"Cost-optimal t = "
            f"{t_cost_optimal}"
        ),
    )

    ax.axvline(
        t_max_success,
        linestyle=":",
        linewidth=1.0,
        label=(
            f"Max-success t = "
            f"{t_max_success}"
        ),
    )

    ax.set_xlabel(
        "Amplitude-amplification iterations"
    )

    ax.set_ylabel(
        "Expected preparation-family calls per usable sample"
    )

    ax.set_title(
        "EXP06C: Cost-Optimal Amplitude Amplification"
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
    # Figure 2
    # Success probability vs cost
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.scatter(
        df[
            "expected_prep_family_calls_per_usable_sample"
        ],
        df[
            "good_sector_probability"
        ],
        s=24,
    )

    ax.scatter(
        [cost_optimal],
        [p_cost_optimal],
        s=90,
        marker="o",
        label=(
            f"Cost-optimal t={t_cost_optimal}"
        ),
    )

    ax.scatter(
        [max_success_cost],
        [max_success_probability_check],
        s=90,
        marker="s",
        label=(
            f"Max-success t={t_max_success}"
        ),
    )

    ax.set_xlabel(
        "Expected preparation-family calls per usable sample"
    )

    ax.set_ylabel(
        "Good-sector probability per attempt"
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_title(
        "Success Probability–Cost Tradeoff"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_TRADEOFF,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Console report
    # =================================================================

    print("=" * 78)
    print(
        "EXP06C: Cost-Optimal Amplitude Amplification"
    )
    print("=" * 78)

    print()
    print("[Initial condition]")

    print(
        f"P(good) without AA                  : "
        f"{p_initial:.10e}"
    )

    print(
        f"Expected prep calls / usable sample : "
        f"{no_aa_cost:.4f}"
    )

    print()
    print("[Maximum-success AA]")

    print(
        f"AA iterations                       : "
        f"{t_max_success}"
    )

    print(
        f"P(good)                             : "
        f"{max_success_probability_check:.8f} "
        f"({100 * max_success_probability_check:.4f}%)"
    )

    print(
        f"Expected prep calls / usable sample : "
        f"{max_success_cost:.4f}"
    )

    print()
    print("[Cost-optimal AA]")

    print(
        f"AA iterations                       : "
        f"{t_cost_optimal}"
    )

    print(
        f"P(good)                             : "
        f"{p_cost_optimal:.8f} "
        f"({100 * p_cost_optimal:.4f}%)"
    )

    print(
        f"Expected prep calls / usable sample : "
        f"{cost_optimal:.4f}"
    )

    print(
        f"Expected AA iter. / usable sample   : "
        f"{aa_iter_cost_optimal:.4f}"
    )

    print()
    print("[Cost reduction]")

    print(
        f"vs no AA                            : "
        f"{cost_reduction_vs_no_aa:.2f} x"
    )

    print(
        f"vs max-success AA                   : "
        f"{cost_reduction_vs_max_success:.4f} x"
    )

    print()
    print("[Interpretation]")

    print(
        "The iteration count that maximizes P(good) "
        "is not the cost-optimal operating point."
    )

    print(
        "Stopping earlier can reduce expected cost per "
        "usable correlation-state sample."
    )

    print()
    print("[Output files]")

    print(
        f"Iteration sweep CSV                  : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary JSON                         : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Cost figure                          : "
        f"{FIG_COST}"
    )

    print(
        f"Tradeoff figure                      : "
        f"{FIG_TRADEOFF}"
    )

    print()
    print(
        "[PASS] EXP06C completed successfully."
    )


if __name__ == "__main__":
    main()