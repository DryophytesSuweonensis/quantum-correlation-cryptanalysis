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

OUTPUT_SWEEP_CSV = RESULTS_DIR / "exp07b_marking_sweep.csv"
OUTPUT_CANDIDATE_CSV = RESULTS_DIR / "exp07b_validation_candidates.csv"
OUTPUT_JSON = RESULTS_DIR / "exp07b_summary.json"

FIG_MARKED = FIGURES_DIR / "exp07b_marked_candidates.png"
FIG_TRUE_PROB = FIGURES_DIR / "exp07b_true_key_probability.png"


# =====================================================================
# Configuration
# =====================================================================

SEED = 20261022

# Same pool used previously
N_REFERENCE_PAIRS = 10_000

# EXP05 used the first 8192
N_EXP05_DATA = 8_192

# Remaining samples:
#
# 10000 - 8192 = 1808
#
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

# Number of held-out validation pairs used
VALIDATION_SIZES = [
    32,
    64,
    128,
    256,
    512,
    1024,
    N_HELDOUT,
]

# Absolute validation-bias thresholds
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

MAX_QAA_ITERATIONS = 100


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

    return (
        plaintexts,
        ciphertexts,
    )


def qaa_probability(
    initial_probability: float,
    iterations: int,
) -> float:
    """
    Ideal QAA probability for the entire marked subspace.

    If

        sin^2(theta) = p_marked

    then

        P_marked(t)
        =
        sin^2((2t+1)theta)
    """

    if initial_probability <= 0:
        return 0.0

    if initial_probability >= 1:
        return 1.0

    theta = math.asin(
        math.sqrt(
            initial_probability
        )
    )

    return math.sin(
        (2 * iterations + 1)
        * theta
    ) ** 2


def find_best_marked_qaa(
    marked_probability: float,
) -> tuple[int, float]:

    if marked_probability <= 0:
        return 0, 0.0

    probabilities = np.asarray(
        [
            qaa_probability(
                marked_probability,
                t,
            )
            for t in range(
                MAX_QAA_ITERATIONS + 1
            )
        ]
    )

    index = int(
        np.argmax(
            probabilities
        )
    )

    return (
        index,
        float(
            probabilities[index]
        ),
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
    # Reconstruct same 10,000-pair pool
    # =================================================================

    plaintexts, ciphertexts = (
        generate_known_pairs(
            n_pairs=N_REFERENCE_PAIRS,
            seed=SEED,
        )
    )

    # -------------------------------------------------------------
    # EXP05:
    #
    # indices 0 ... 8191
    #
    # EXP07B:
    #
    # indices 8192 ... 9999
    # -------------------------------------------------------------

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

    if (
        len(heldout_plaintexts)
        != N_HELDOUT
    ):
        raise RuntimeError(
            "Unexpected held-out size."
        )

    # =================================================================
    # Step 3
    # Run validation attack for each M
    # =================================================================

    candidate_rows = []
    sweep_rows = []

    for m in VALIDATION_SIZES:

        print()
        print(
            f"[Validation size M={m}]"
        )

        val_p = (
            heldout_plaintexts[
                :m
            ]
        )

        val_c = (
            heldout_ciphertexts[
                :m
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

        validation_bias = (
            validation_df[
                "bias"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        abs_validation_bias = (
            np.abs(
                validation_bias
            )
        )

        # -------------------------------------------------------------
        # Under the null:
        #
        # Var(bias) ≈ 1/(4M)
        #
        # std ≈ 1/(2 sqrt(M))
        #
        # Useful diagnostic only.
        # -------------------------------------------------------------

        null_std = (
            1.0
            / (
                2.0
                * math.sqrt(m)
            )
        )

        z_scores = (
            abs_validation_bias
            / null_std
        )

        # Save candidate-level validation result
        for k in range(
            len(validation_df)
        ):

            candidate_rows.append(
                {
                    "validation_size":
                        m,

                    "candidate":
                        int(
                            validation_df.iloc[
                                k
                            ][
                                "candidate"
                            ]
                        ),

                    "candidate_hex":
                        f"{int(validation_df.iloc[k]['candidate']):02X}",

                    "validation_bias":
                        float(
                            validation_bias[k]
                        ),

                    "abs_validation_bias":
                        float(
                            abs_validation_bias[k]
                        ),

                    "null_std":
                        null_std,

                    "abs_z_score":
                        float(
                            z_scores[k]
                        ),

                    "correlation_state_probability":
                        float(
                            base_probabilities[
                                k
                            ]
                        ),

                    "is_true_key":
                        bool(
                            k == true_index
                        ),
                }
            )

        # =============================================================
        # Step 4
        # Sweep marking threshold tau
        # =============================================================

        for tau in BIAS_THRESHOLDS:

            marked = (
                abs_validation_bias
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

            # ---------------------------------------------------------
            # Probability mass of all marked keys in the
            # EXP05 correlation state
            # ---------------------------------------------------------

            marked_mass = float(
                np.sum(
                    base_probabilities[
                        marked
                    ]
                )
            )

            # ---------------------------------------------------------
            # If QAA perfectly amplifies the MARKED SUBSPACE,
            #
            # the relative amplitudes among marked keys do NOT change.
            #
            # Therefore:
            #
            # P(true | marked)
            #
            # =
            # p_true / p_marked
            #
            # if true key is marked.
            # ---------------------------------------------------------

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

            # ---------------------------------------------------------
            # Ideal QAA for the entire marked set
            # ---------------------------------------------------------

            (
                best_qaa_iteration,
                best_marked_probability,
            ) = find_best_marked_qaa(
                marked_mass
            )

            # After QAA:
            #
            # P(true)
            #
            # =
            # P(marked after QAA)
            # * P(true | marked)
            #
            ideal_true_probability_after_qaa = (
                best_marked_probability
                * true_given_marked
            )

            # Absolute upper bound:
            #
            # if marked sector could be amplified to exactly 1
            #
            upper_bound_true_probability = (
                true_given_marked
            )

            sweep_rows.append(
                {
                    "validation_size":
                        m,

                    "bias_threshold":
                        tau,

                    "true_validation_bias":
                        float(
                            validation_bias[
                                true_index
                            ]
                        ),

                    "true_abs_validation_bias":
                        float(
                            abs_validation_bias[
                                true_index
                            ]
                        ),

                    "true_abs_z_score":
                        float(
                            z_scores[
                                true_index
                            ]
                        ),

                    "true_key_marked":
                        true_marked,

                    "n_marked_candidates":
                        n_marked,

                    "false_positive_count":
                        false_positive_count,

                    "marked_probability_mass":
                        marked_mass,

                    "true_probability_before_marking":
                        p_true_initial,

                    "true_probability_given_marked":
                        true_given_marked,

                    "best_ideal_qaa_iteration":
                        best_qaa_iteration,

                    "best_marked_sector_probability":
                        best_marked_probability,

                    "ideal_true_probability_after_qaa":
                        ideal_true_probability_after_qaa,

                    "upper_bound_true_probability":
                        upper_bound_true_probability,
                }
            )

    # =================================================================
    # Step 5
    # Save CSVs
    # =================================================================

    candidate_df = pd.DataFrame(
        candidate_rows
    )

    sweep_df = pd.DataFrame(
        sweep_rows
    )

    candidate_df.to_csv(
        OUTPUT_CANDIDATE_CSV,
        index=False,
    )

    sweep_df.to_csv(
        OUTPUT_SWEEP_CSV,
        index=False,
    )

    # =================================================================
    # Step 6
    # Find best selectivity cases
    # =================================================================
    #
    # This is evaluation only.
    #
    # We are allowed to inspect the true-key label here because
    # the purpose is to evaluate verifier quality, not to construct
    # the attack oracle using secret-key knowledge.
    # =================================================================

    valid_df = sweep_df[
        sweep_df[
            "true_key_marked"
        ]
    ].copy()

    if len(valid_df) > 0:

        best_selectivity_row = (
            valid_df
            .sort_values(
                [
                    "n_marked_candidates",
                    "validation_size",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .iloc[0]
        )

        best_probability_row = (
            valid_df
            .sort_values(
                "upper_bound_true_probability",
                ascending=False,
            )
            .iloc[0]
        )

    else:

        best_selectivity_row = None
        best_probability_row = None

    # =================================================================
    # Step 7
    # Summary JSON
    # =================================================================

    summary = {

        "experiment":
            "exp07b_heldout_validation_marking",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "initial_true_key_probability":
            p_true_initial,

        "heldout_pair_count":
            N_HELDOUT,

        "validation_sizes":
            VALIDATION_SIZES,

        "bias_thresholds":
            BIAS_THRESHOLDS,

        "marking_rule":
            "mark k if |validation_bias(k)| >= tau",

        "validation_data_usage":
            (
                "Held-out plaintext/ciphertext pairs not used "
                "for EXP05 correlation-state construction."
            ),

        "critical_limitation":
            (
                "EXP07B evaluates the marking rule classically. "
                "The reversible quantum implementation and its "
                "oracle cost are not yet included."
            ),
    }

    if best_selectivity_row is not None:

        summary[
            "best_selectivity_configuration"
        ] = {

            "validation_size":
                int(
                    best_selectivity_row[
                        "validation_size"
                    ]
                ),

            "bias_threshold":
                float(
                    best_selectivity_row[
                        "bias_threshold"
                    ]
                ),

            "n_marked_candidates":
                int(
                    best_selectivity_row[
                        "n_marked_candidates"
                    ]
                ),

            "false_positive_count":
                int(
                    best_selectivity_row[
                        "false_positive_count"
                    ]
                ),

            "true_probability_given_marked":
                float(
                    best_selectivity_row[
                        "true_probability_given_marked"
                    ]
                ),
        }

    if best_probability_row is not None:

        summary[
            "best_true_probability_upper_bound_configuration"
        ] = {

            "validation_size":
                int(
                    best_probability_row[
                        "validation_size"
                    ]
                ),

            "bias_threshold":
                float(
                    best_probability_row[
                        "bias_threshold"
                    ]
                ),

            "n_marked_candidates":
                int(
                    best_probability_row[
                        "n_marked_candidates"
                    ]
                ),

            "true_probability_upper_bound":
                float(
                    best_probability_row[
                        "upper_bound_true_probability"
                    ]
                ),

            "ideal_true_probability_after_qaa":
                float(
                    best_probability_row[
                        "ideal_true_probability_after_qaa"
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
    # Number of marked candidates vs threshold
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for m in VALIDATION_SIZES:

        part = sweep_df[
            sweep_df[
                "validation_size"
            ] == m
        ]

        ax.plot(
            part[
                "bias_threshold"
            ],
            part[
                "n_marked_candidates"
            ],
            marker="o",
            label=f"M={m}",
        )

    ax.set_xlabel(
        "Absolute validation-bias threshold"
    )

    ax.set_ylabel(
        "Number of marked key candidates"
    )

    ax.set_title(
        "EXP07B: Marking Selectivity"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_MARKED,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Ideal true-key probability after marked-set QAA
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for m in VALIDATION_SIZES:

        part = sweep_df[
            sweep_df[
                "validation_size"
            ] == m
        ]

        ax.plot(
            part[
                "bias_threshold"
            ],
            part[
                "ideal_true_probability_after_qaa"
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
        "Ideal true-key probability after marked-set QAA"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "Validation-Based Marking and Ideal QAA"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_TRUE_PROB,
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
        "EXP07B: Held-Out Validation-Based Key Marking"
    )
    print("=" * 78)

    print()
    print("[Data split]")

    print(
        f"EXP05 training / correlation pairs : "
        f"{N_EXP05_DATA}"
    )

    print(
        f"Held-out validation pairs          : "
        f"{N_HELDOUT}"
    )

    print()
    print("[Initial key distribution]")

    print(
        f"True key                           : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"Initial P(true key)                : "
        f"{p_true_initial:.8f} "
        f"({100 * p_true_initial:.4f}%)"
    )

    print()
    print("[Marking rule]")

    print(
        "Mark candidate k if:"
    )

    print(
        "    |validation_bias(k)| >= tau"
    )

    print()

    if best_selectivity_row is not None:

        print(
            "[Best selectivity while retaining true key]"
        )

        print(
            f"Validation pairs M                 : "
            f"{int(best_selectivity_row['validation_size'])}"
        )

        print(
            f"Bias threshold tau                 : "
            f"{best_selectivity_row['bias_threshold']:.4f}"
        )

        print(
            f"Marked candidates                  : "
            f"{int(best_selectivity_row['n_marked_candidates'])}"
        )

        print(
            f"False positives                    : "
            f"{int(best_selectivity_row['false_positive_count'])}"
        )

        print(
            f"P(true | marked)                   : "
            f"{best_selectivity_row['true_probability_given_marked']:.8f}"
        )

    else:

        print(
            "WARNING: No tested marking configuration "
            "retained the true key."
        )

    print()

    if best_probability_row is not None:

        print(
            "[Best ideal true-key probability]"
        )

        print(
            f"Validation pairs M                 : "
            f"{int(best_probability_row['validation_size'])}"
        )

        print(
            f"Bias threshold tau                 : "
            f"{best_probability_row['bias_threshold']:.4f}"
        )

        print(
            f"Marked candidates                  : "
            f"{int(best_probability_row['n_marked_candidates'])}"
        )

        print(
            f"True-key upper bound after marking : "
            f"{best_probability_row['upper_bound_true_probability']:.8f}"
        )

        print(
            f"Ideal QAA true-key probability     : "
            f"{best_probability_row['ideal_true_probability_after_qaa']:.8f}"
        )

    print()
    print("[CRITICAL LIMITATION]")

    print(
        "The validation scores are computed classically "
        "in EXP07B."
    )

    print(
        "This experiment tests whether the statistical "
        "marking rule is selective enough."
    )

    print(
        "It does NOT yet implement the reversible "
        "validation oracle."
    )

    print()
    print("[Output files]")

    print(
        f"Marking sweep CSV                  : "
        f"{OUTPUT_SWEEP_CSV}"
    )

    print(
        f"Candidate validation CSV           : "
        f"{OUTPUT_CANDIDATE_CSV}"
    )

    print(
        f"Summary JSON                       : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Marking figure                     : "
        f"{FIG_MARKED}"
    )

    print(
        f"True-key probability figure        : "
        f"{FIG_TRUE_PROB}"
    )

    print()
    print(
        "[PASS] EXP07B completed successfully."
    )


if __name__ == "__main__":
    main()