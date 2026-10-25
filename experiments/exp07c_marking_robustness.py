from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

sys.path.insert(0, str(SRC_DIR))

from heys_spn import encrypt_block
from linear_attack import run_linear_attack


EXP05_CSV = RESULTS_DIR / "exp05_coherent_probabilities.csv"

OUTPUT_CSV = RESULTS_DIR / "exp07c_marking_robustness.csv"
OUTPUT_JSON = RESULTS_DIR / "exp07c_summary.json"

FIG_RETENTION = FIGURES_DIR / "exp07c_true_key_retention.png"
FIG_FALSE_POSITIVE = FIGURES_DIR / "exp07c_false_positives.png"
FIG_CONDITIONAL = FIGURES_DIR / "exp07c_true_given_marked.png"


# =====================================================================
# Configuration
# =====================================================================

SEED = 20261023

N_REFERENCE_PAIRS = 10_000
N_EXP05_DATA = 8_192

N_HELDOUT = (
    N_REFERENCE_PAIRS
    - N_EXP05_DATA
)

ROUND_KEYS = (
    0x3A94,
    0xD63F,
    0xB127,
    0x8E5C,
    0xA2B4,
)

# Validation sample count M
VALIDATION_SIZES = [
    32,
    64,
    128,
    256,
    512,
    1024,
]

# Bias threshold tau
BIAS_THRESHOLDS = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
]

# Independent random validation subsets
N_TRIALS = 500


# =====================================================================
# Utilities
# =====================================================================

def generate_known_pairs(
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(
        seed
    )

    plaintexts = rng.choice(
        2**16,
        size=n_pairs,
        replace=False,
    ).astype(np.uint16)

    ciphertexts = np.asarray(
        [
            encrypt_block(
                int(p),
                ROUND_KEYS,
            )
            for p in plaintexts
        ],
        dtype=np.uint16,
    )

    return plaintexts, ciphertexts


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
    # Load EXP05 correlation-state distribution
    # =================================================================

    if not EXP05_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP05_CSV}\n\n"
            "Run EXP05 first."
        )

    state_df = pd.read_csv(
        EXP05_CSV
    )

    state_df = (
        state_df
        .sort_values("candidate")
        .reset_index(drop=True)
    )

    base_probabilities = (
        state_df[
            "coherent_probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    base_probabilities = (
        base_probabilities
        / base_probabilities.sum()
    )

    true_indices = np.flatnonzero(
        state_df[
            "is_true_key"
        ]
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
        state_df.iloc[
            true_index
        ]["candidate"]
    )

    true_k2 = (
        true_candidate >> 4
    ) & 0xF

    true_k4 = (
        true_candidate
        & 0xF
    )

    p_true_initial = float(
        base_probabilities[
            true_index
        ]
    )

    # =================================================================
    # Step 2
    # Rebuild same 10,000-pair pool
    # =================================================================

    plaintexts, ciphertexts = (
        generate_known_pairs(
            n_pairs=N_REFERENCE_PAIRS,
            seed=20261022,
        )
    )

    heldout_plaintexts = (
        plaintexts[
            N_EXP05_DATA:
        ]
    )

    heldout_ciphertexts = (
        ciphertexts[
            N_EXP05_DATA:
        ]
    )

    if len(heldout_plaintexts) != N_HELDOUT:

        raise RuntimeError(
            "Unexpected held-out size."
        )

    rng = np.random.default_rng(
        SEED
    )

    # =================================================================
    # Step 3
    # Robustness experiment
    # =================================================================

    rows = []

    for m in VALIDATION_SIZES:

        if m > N_HELDOUT:
            continue

        print()
        print(
            f"[M={m}]"
        )

        # -------------------------------------------------------------
        # Accumulators for each tau
        # -------------------------------------------------------------

        stats = {
            tau: {
                "true_mark_count": 0,
                "false_positive_sum": 0,
                "marked_count_sum": 0,
                "true_given_marked_sum": 0.0,
                "true_given_marked_success_trials": 0,
                "true_bias_sum": 0.0,
                "true_abs_bias_sum": 0.0,
            }
            for tau in BIAS_THRESHOLDS
        }

        # =============================================================
        # Random subset trials
        # =============================================================

        for trial in range(
            N_TRIALS
        ):

            subset_indices = (
                rng.choice(
                    N_HELDOUT,
                    size=m,
                    replace=False,
                )
            )

            val_p = (
                heldout_plaintexts[
                    subset_indices
                ]
            )

            val_c = (
                heldout_ciphertexts[
                    subset_indices
                ]
            )

            validation_df = (
                run_linear_attack(
                    plaintexts=val_p,
                    ciphertexts=val_c,
                    true_k2=true_k2,
                    true_k4=true_k4,
                )
            )

            validation_df = (
                validation_df
                .sort_values(
                    "candidate"
                )
                .reset_index(
                    drop=True
                )
            )

            biases = (
                validation_df[
                    "bias"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            abs_biases = (
                np.abs(
                    biases
                )
            )

            true_bias = float(
                biases[
                    true_index
                ]
            )

            true_abs_bias = float(
                abs_biases[
                    true_index
                ]
            )

            # =========================================================
            # Threshold sweep
            # =========================================================

            for tau in BIAS_THRESHOLDS:

                marked = (
                    abs_biases
                    >= tau
                )

                n_marked = int(
                    np.sum(
                        marked
                    )
                )

                true_marked = bool(
                    marked[
                        true_index
                    ]
                )

                false_positive_count = (
                    n_marked
                    - int(
                        true_marked
                    )
                )

                marked_mass = float(
                    np.sum(
                        base_probabilities[
                            marked
                        ]
                    )
                )

                if (
                    true_marked
                    and marked_mass > 0
                ):

                    true_given_marked = (
                        p_true_initial
                        / marked_mass
                    )

                else:

                    true_given_marked = 0.0

                stat = stats[
                    tau
                ]

                stat[
                    "true_mark_count"
                ] += int(
                    true_marked
                )

                stat[
                    "false_positive_sum"
                ] += (
                    false_positive_count
                )

                stat[
                    "marked_count_sum"
                ] += (
                    n_marked
                )

                stat[
                    "true_bias_sum"
                ] += (
                    true_bias
                )

                stat[
                    "true_abs_bias_sum"
                ] += (
                    true_abs_bias
                )

                if true_marked:

                    stat[
                        "true_given_marked_sum"
                    ] += (
                        true_given_marked
                    )

                    stat[
                        "true_given_marked_success_trials"
                    ] += 1

        # =============================================================
        # Aggregate statistics
        # =============================================================

        for tau in BIAS_THRESHOLDS:

            stat = stats[
                tau
            ]

            true_retention_probability = (
                stat[
                    "true_mark_count"
                ]
                / N_TRIALS
            )

            mean_false_positives = (
                stat[
                    "false_positive_sum"
                ]
                / N_TRIALS
            )

            mean_marked_candidates = (
                stat[
                    "marked_count_sum"
                ]
                / N_TRIALS
            )

            mean_true_bias = (
                stat[
                    "true_bias_sum"
                ]
                / N_TRIALS
            )

            mean_true_abs_bias = (
                stat[
                    "true_abs_bias_sum"
                ]
                / N_TRIALS
            )

            successful_trials = (
                stat[
                    "true_given_marked_success_trials"
                ]
            )

            if successful_trials > 0:

                mean_true_given_marked = (
                    stat[
                        "true_given_marked_sum"
                    ]
                    / successful_trials
                )

            else:

                mean_true_given_marked = 0.0

            # ---------------------------------------------------------
            # A combined quantity:
            #
            # Expected true-key probability after ideal amplification
            # when verifier failure is included.
            #
            # If true key is not marked -> zero contribution.
            #
            # ---------------------------------------------------------

            effective_true_probability = (
                true_retention_probability
                * mean_true_given_marked
            )

            rows.append(
                {
                    "validation_size":
                        m,

                    "bias_threshold":
                        tau,

                    "n_trials":
                        N_TRIALS,

                    "true_key_retention_probability":
                        true_retention_probability,

                    "mean_false_positive_count":
                        mean_false_positives,

                    "mean_marked_candidates":
                        mean_marked_candidates,

                    "mean_true_validation_bias":
                        mean_true_bias,

                    "mean_true_abs_validation_bias":
                        mean_true_abs_bias,

                    "mean_true_probability_given_marked":
                        mean_true_given_marked,

                    "effective_true_probability":
                        effective_true_probability,
                }
            )

    # =================================================================
    # Step 4
    # Save CSV
    # =================================================================

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =================================================================
    # Step 5
    # Identify useful operating points
    # =================================================================

    # Require true key retention >= 90%
    robust_df = result_df[
        result_df[
            "true_key_retention_probability"
        ] >= 0.90
    ].copy()

    if len(robust_df) > 0:

        best_robust_row = (
            robust_df
            .sort_values(
                [
                    "effective_true_probability",
                    "mean_false_positive_count",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

    else:

        best_robust_row = None

    # =================================================================
    # Step 6
    # Summary JSON
    # =================================================================

    summary = {

        "experiment":
            "exp07c_marking_robustness",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "initial_true_key_probability":
            p_true_initial,

        "heldout_pool_size":
            N_HELDOUT,

        "validation_sizes":
            VALIDATION_SIZES,

        "bias_thresholds":
            BIAS_THRESHOLDS,

        "random_trials_per_configuration":
            N_TRIALS,

        "marking_rule":
            "mark k if |validation_bias(k)| >= tau",

        "main_metrics": [
            "true_key_retention_probability",
            "mean_false_positive_count",
            "mean_true_probability_given_marked",
            "effective_true_probability",
        ],

        "important_interpretation": (
            "EXP07C measures whether the validation-based "
            "marking rule remains selective across random "
            "held-out subsets, rather than relying on one "
            "particular validation split."
        ),
    }

    if best_robust_row is not None:

        summary[
            "best_configuration_with_retention_ge_90pct"
        ] = {

            "validation_size":
                int(
                    best_robust_row[
                        "validation_size"
                    ]
                ),

            "bias_threshold":
                float(
                    best_robust_row[
                        "bias_threshold"
                    ]
                ),

            "true_key_retention_probability":
                float(
                    best_robust_row[
                        "true_key_retention_probability"
                    ]
                ),

            "mean_false_positive_count":
                float(
                    best_robust_row[
                        "mean_false_positive_count"
                    ]
                ),

            "mean_true_probability_given_marked":
                float(
                    best_robust_row[
                        "mean_true_probability_given_marked"
                    ]
                ),

            "effective_true_probability":
                float(
                    best_robust_row[
                        "effective_true_probability"
                    ]
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
    # True-key retention probability
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for m in VALIDATION_SIZES:

        part = result_df[
            result_df[
                "validation_size"
            ] == m
        ]

        ax.plot(
            part[
                "bias_threshold"
            ],
            part[
                "true_key_retention_probability"
            ],
            marker="o",
            label=f"M={m}",
        )

    ax.axhline(
        0.90,
        linestyle="--",
        linewidth=1.0,
        label="90% retention",
    )

    ax.set_xlabel(
        "Absolute validation-bias threshold"
    )

    ax.set_ylabel(
        "True-key retention probability"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "EXP07C: True-Key Retention Robustness"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_RETENTION,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Mean false positives
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for m in VALIDATION_SIZES:

        part = result_df[
            result_df[
                "validation_size"
            ] == m
        ]

        ax.plot(
            part[
                "bias_threshold"
            ],
            part[
                "mean_false_positive_count"
            ],
            marker="o",
            label=f"M={m}",
        )

    ax.set_xlabel(
        "Absolute validation-bias threshold"
    )

    ax.set_ylabel(
        "Mean number of false-positive keys"
    )

    ax.set_title(
        "EXP07C: False-Positive Marking"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_FALSE_POSITIVE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 3
    # Mean P(true | marked)
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for m in VALIDATION_SIZES:

        part = result_df[
            result_df[
                "validation_size"
            ] == m
        ]

        ax.plot(
            part[
                "bias_threshold"
            ],
            part[
                "mean_true_probability_given_marked"
            ],
            marker="o",
            label=f"M={m}",
        )

    ax.axhline(
        p_true_initial,
        linestyle="--",
        linewidth=1.0,
        label=(
            f"Initial P(true) "
            f"= {100 * p_true_initial:.2f}%"
        ),
    )

    ax.set_xlabel(
        "Absolute validation-bias threshold"
    )

    ax.set_ylabel(
        "Mean P(true key | marked)"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "EXP07C: Conditional True-Key Concentration"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_CONDITIONAL,
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
        "EXP07C: Robustness of Validation-Based Marking"
    )
    print("=" * 78)

    print()
    print("[Configuration]")

    print(
        f"True key                       : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"Held-out pool                  : "
        f"{N_HELDOUT}"
    )

    print(
        f"Random trials / configuration  : "
        f"{N_TRIALS}"
    )

    print()

    if best_robust_row is not None:

        print(
            "[Best configuration with >=90% true-key retention]"
        )

        print(
            f"Validation size M              : "
            f"{int(best_robust_row['validation_size'])}"
        )

        print(
            f"Bias threshold tau             : "
            f"{best_robust_row['bias_threshold']:.4f}"
        )

        print(
            f"True-key retention             : "
            f"{best_robust_row['true_key_retention_probability']:.4f}"
        )

        print(
            f"Mean false positives           : "
            f"{best_robust_row['mean_false_positive_count']:.2f}"
        )

        print(
            f"Mean P(true | marked)          : "
            f"{best_robust_row['mean_true_probability_given_marked']:.4f}"
        )

        print(
            f"Effective true probability     : "
            f"{best_robust_row['effective_true_probability']:.4f}"
        )

    else:

        print(
            "No tested configuration achieved "
            ">=90% true-key retention."
        )

    print()
    print("[Output files]")

    print(
        f"Robustness CSV                 : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary JSON                   : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Retention figure               : "
        f"{FIG_RETENTION}"
    )

    print(
        f"False-positive figure          : "
        f"{FIG_FALSE_POSITIVE}"
    )

    print(
        f"Conditional-probability figure : "
        f"{FIG_CONDITIONAL}"
    )

    print()
    print(
        "[PASS] EXP07C completed successfully."
    )


if __name__ == "__main__":
    main()