from __future__ import annotations

from pathlib import Path
import json

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
EXP05_JSON = RESULTS_DIR / "exp05_summary.json"
EXP06A_JSON = RESULTS_DIR / "exp06a_summary.json"

OUTPUT_CSV = RESULTS_DIR / "exp06b_end_to_end_summary.csv"
OUTPUT_JSON = RESULTS_DIR / "exp06b_summary.json"

FIG_KRP = FIGURES_DIR / "exp06b_krp_vs_samples.png"
FIG_COST = FIGURES_DIR / "exp06b_cost_comparison.png"


# =====================================================================
# Configuration
# =====================================================================

SEED = 20261023

# Number of independent Monte Carlo key-recovery experiments
N_TRIALS = 5000

# Number of SUCCESSFULLY PREPARED correlation-state samples
#
# Important:
# These are not raw circuit executions.
# Each sample means that the data register was successfully found
# in the correlation sector.
SAMPLE_COUNTS = [
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
]

KRP_THRESHOLDS = [
    0.50,
    0.90,
    0.95,
    0.99,
]


# =====================================================================
# Utility functions
# =====================================================================

def run_key_recovery_trials(
    probabilities: np.ndarray,
    true_index: int,
    n_samples: int,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """
    Repeated finite-sample key recovery from the prepared
    correlation state.

    For each trial:

      1. Measure the key register n_samples times.
      2. Build a candidate histogram.
      3. Determine whether the true key is:
           - unique mode      -> Strict KRP
           - among top modes  -> Top-set KRP
           - randomly selected among tied modes -> Tie-aware KRP

    The input probability distribution is conditional on having
    successfully reached the correlation sector.
    """

    strict_successes = 0
    top_set_successes = 0
    tie_aware_score = 0.0

    tie_trials = 0

    total_true_counts = 0

    for _ in range(n_trials):

        counts = rng.multinomial(
            n=n_samples,
            pvals=probabilities,
        )

        true_count = int(
            counts[true_index]
        )

        total_true_counts += true_count

        max_count = int(
            np.max(counts)
        )

        mode_indices = np.flatnonzero(
            counts == max_count
        )

        n_modes = len(
            mode_indices
        )

        true_is_mode = (
            true_index in mode_indices
        )

        # -------------------------------------------------------------
        # Strict KRP
        # -------------------------------------------------------------

        if (
            n_modes == 1
            and mode_indices[0] == true_index
        ):
            strict_successes += 1

        # -------------------------------------------------------------
        # Top-set KRP
        # -------------------------------------------------------------

        if true_is_mode:
            top_set_successes += 1

        # -------------------------------------------------------------
        # Tie-aware KRP
        # -------------------------------------------------------------

        if true_is_mode:
            tie_aware_score += (
                1.0 / n_modes
            )

        if n_modes > 1:
            tie_trials += 1

    return {

        "usable_samples":
            n_samples,

        "trials":
            n_trials,

        "strict_krp":
            strict_successes
            / n_trials,

        "tie_aware_krp":
            tie_aware_score
            / n_trials,

        "top_set_krp":
            top_set_successes
            / n_trials,

        "tie_rate":
            tie_trials
            / n_trials,

        "mean_true_key_count":
            total_true_counts
            / n_trials,
    }


def first_samples_for_threshold(
    df: pd.DataFrame,
    column: str,
    threshold: float,
) -> int | None:

    selected = df[
        df[column] >= threshold
    ]

    if len(selected) == 0:
        return None

    return int(
        selected.iloc[0][
            "usable_samples"
        ]
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
    # Load EXP05 conditional key distribution
    # =================================================================

    if not EXP05_CSV.exists():
        raise FileNotFoundError(
            f"Missing:\n{EXP05_CSV}\n\n"
            "Run EXP05 first."
        )

    if not EXP05_JSON.exists():
        raise FileNotFoundError(
            f"Missing:\n{EXP05_JSON}"
        )

    if not EXP06A_JSON.exists():
        raise FileNotFoundError(
            f"Missing:\n{EXP06A_JSON}\n\n"
            "Run EXP06A first."
        )

    key_df = pd.read_csv(
        EXP05_CSV
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

    # =================================================================
    # Step 2
    # Extract conditional key probabilities
    # =================================================================

    required_columns = {
        "candidate",
        "coherent_probability",
        "is_true_key",
    }

    missing = (
        required_columns
        - set(key_df.columns)
    )

    if missing:
        raise ValueError(
            f"EXP05 CSV missing columns: "
            f"{sorted(missing)}"
        )

    key_df = (
        key_df
        .sort_values("candidate")
        .reset_index(drop=True)
    )

    probabilities = (
        key_df[
            "coherent_probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    true_indices = np.flatnonzero(
        key_df[
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
        key_df.iloc[
            true_index
        ]["candidate"]
    )

    p_true_given_good = float(
        probabilities[
            true_index
        ]
    )

    # =================================================================
    # Step 3
    # Preparation probabilities from EXP05 / EXP06A
    # =================================================================

    p_good_naive = float(
        exp05[
            "postselection_probability"
        ]
    )

    aa_iterations = int(
        exp06a[
            "numerical_best_iteration"
        ]
    )

    p_good_aa = float(
        exp06a[
            "numerical_best_good_sector_probability"
        ]
    )

    # Textbook logical proxy from EXP06A:
    #
    # one initial A
    # + t uses of A
    # + t uses of A^dagger
    #
    # -> 1 + 2t preparation-family calls
    #
    prep_family_calls_per_aa_attempt = (
        1
        + 2 * aa_iterations
    )

    # =================================================================
    # Step 4
    # Monte Carlo key-recovery simulation
    # =================================================================

    rng = np.random.default_rng(
        SEED
    )

    rows = []

    for n_samples in SAMPLE_COUNTS:

        result = run_key_recovery_trials(
            probabilities=probabilities,
            true_index=true_index,
            n_samples=n_samples,
            n_trials=N_TRIALS,
            rng=rng,
        )

        # -------------------------------------------------------------
        # Expected number of full preparation attempts required
        #
        # To obtain R successful correlation-state samples:
        #
        # E[N_attempts] = R / p_good
        # -------------------------------------------------------------

        naive_expected_attempts = (
            n_samples
            / p_good_naive
        )

        aa_expected_attempts = (
            n_samples
            / p_good_aa
        )

        # -------------------------------------------------------------
        # Logical AA cost
        # -------------------------------------------------------------

        expected_total_aa_iterations = (
            aa_expected_attempts
            * aa_iterations
        )

        expected_total_prep_family_calls = (
            aa_expected_attempts
            * prep_family_calls_per_aa_attempt
        )

        # -------------------------------------------------------------
        # Naive preparation proxy
        #
        # Each naive full attempt requires one preparation.
        # -------------------------------------------------------------

        naive_preparation_calls_proxy = (
            naive_expected_attempts
        )

        # -------------------------------------------------------------
        # Cost-reduction ratios
        # -------------------------------------------------------------

        raw_attempt_reduction = (
            naive_expected_attempts
            / aa_expected_attempts
        )

        prep_proxy_reduction = (
            naive_preparation_calls_proxy
            / expected_total_prep_family_calls
        )

        result.update(
            {
                "expected_naive_full_attempts":
                    naive_expected_attempts,

                "expected_aa_full_attempts":
                    aa_expected_attempts,

                "aa_iterations_per_attempt":
                    aa_iterations,

                "prep_family_calls_per_aa_attempt":
                    prep_family_calls_per_aa_attempt,

                "expected_total_aa_iterations":
                    expected_total_aa_iterations,

                "expected_total_aa_prep_family_calls":
                    expected_total_prep_family_calls,

                "naive_preparation_calls_proxy":
                    naive_preparation_calls_proxy,

                "raw_attempt_reduction_factor":
                    raw_attempt_reduction,

                "prep_proxy_reduction_factor":
                    prep_proxy_reduction,
            }
        )

        rows.append(
            result
        )

        print(
            f"samples={n_samples:5d} | "
            f"tie-aware KRP="
            f"{result['tie_aware_krp']:.4f} | "
            f"naive attempts="
            f"{naive_expected_attempts:10.1f} | "
            f"AA prep calls="
            f"{expected_total_prep_family_calls:10.1f}"
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =================================================================
    # Step 5
    # KRP threshold analysis
    # =================================================================

    threshold_results = {}

    for threshold in KRP_THRESHOLDS:

        n_samples = (
            first_samples_for_threshold(
                df=summary_df,
                column="tie_aware_krp",
                threshold=threshold,
            )
        )

        name = (
            f"{int(threshold * 100)}pct"
        )

        if n_samples is None:

            threshold_results[
                name
            ] = {
                "required_usable_samples":
                    None,
            }

            continue

        row = summary_df[
            summary_df[
                "usable_samples"
            ] == n_samples
        ].iloc[0]

        threshold_results[
            name
        ] = {

            "required_usable_samples":
                n_samples,

            "tie_aware_krp":
                float(
                    row[
                        "tie_aware_krp"
                    ]
                ),

            "expected_naive_full_attempts":
                float(
                    row[
                        "expected_naive_full_attempts"
                    ]
                ),

            "expected_aa_full_attempts":
                float(
                    row[
                        "expected_aa_full_attempts"
                    ]
                ),

            "expected_total_aa_iterations":
                float(
                    row[
                        "expected_total_aa_iterations"
                    ]
                ),

            "expected_total_aa_prep_family_calls":
                float(
                    row[
                        "expected_total_aa_prep_family_calls"
                    ]
                ),
        }

    # =================================================================
    # Step 6
    # Save metadata
    # =================================================================

    summary = {

        "experiment":
            "exp06b_end_to_end_key_recovery",

        "seed":
            SEED,

        "n_trials":
            N_TRIALS,

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "true_key_probability_given_correlation_sector":
            p_true_given_good,

        "naive_correlation_sector_probability":
            p_good_naive,

        "aa_correlation_sector_probability":
            p_good_aa,

        "aa_iterations_per_attempt":
            aa_iterations,

        "prep_family_calls_per_aa_attempt":
            prep_family_calls_per_aa_attempt,

        "krp_thresholds":
            threshold_results,

        "important_interpretation": (
            "Amplitude amplification changes the cost of "
            "preparing usable correlation-state samples, "
            "but does not change the conditional key distribution. "
            "Therefore, KRP as a function of usable samples is the "
            "same key-extraction problem as before amplification."
        ),

        "cost_warning": (
            "The reported preparation-family calls and AA iterations "
            "are logical cost proxies, not physical gate counts. "
            "Reversible data access, attack-oracle implementation, "
            "reflections, ancilla management, and verification costs "
            "are not yet included."
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
    # KRP versus usable correlation-state samples
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    x = summary_df[
        "usable_samples"
    ].to_numpy()

    ax.plot(
        x,
        summary_df[
            "tie_aware_krp"
        ],
        marker="o",
        label="Tie-aware KRP",
    )

    ax.plot(
        x,
        summary_df[
            "strict_krp"
        ],
        marker="s",
        label="Strict KRP",
    )

    ax.plot(
        x,
        summary_df[
            "top_set_krp"
        ],
        marker="^",
        label="Top-set KRP",
    )

    ax.axhline(
        0.90,
        linestyle="--",
        linewidth=1.0,
        label="90% recovery",
    )

    ax.axhline(
        0.95,
        linestyle=":",
        linewidth=1.0,
        label="95% recovery",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_xlabel(
        "Usable correlation-state samples"
    )

    ax.set_ylabel(
        "Key-Recovery Probability"
    )

    ax.set_title(
        "EXP06B: Key Recovery After Correlation-State Amplification"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_KRP,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Naive vs AA-assisted preparation cost
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        x,
        summary_df[
            "expected_naive_full_attempts"
        ],
        marker="o",
        label="Naive postselection: full attempts",
    )

    ax.plot(
        x,
        summary_df[
            "expected_total_aa_prep_family_calls"
        ],
        marker="s",
        label="AA-assisted: preparation-family calls",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Usable correlation-state samples"
    )

    ax.set_ylabel(
        "Expected logical preparation cost"
    )

    ax.set_title(
        "Naive Postselection vs AA-Assisted Preparation"
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

    print()
    print("=" * 78)
    print(
        "EXP06B: End-to-End Key Recovery After Amplitude Amplification"
    )
    print("=" * 78)

    print()
    print("[Conditional key distribution]")

    print(
        f"True key                           : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"P(true key | correlation sector)  : "
        f"{p_true_given_good:.8f} "
        f"({100 * p_true_given_good:.4f}%)"
    )

    print()
    print("[Preparation probabilities]")

    print(
        f"Naive P(correlation sector)        : "
        f"{p_good_naive:.10e}"
    )

    print(
        f"AA P(correlation sector)           : "
        f"{p_good_aa:.10f}"
    )

    print(
        f"AA iterations / attempt            : "
        f"{aa_iterations}"
    )

    print(
        f"Preparation-family calls / attempt : "
        f"{prep_family_calls_per_aa_attempt}"
    )

    print()
    print("[Tie-aware KRP thresholds]")

    for threshold in KRP_THRESHOLDS:

        name = (
            f"{int(threshold * 100)}pct"
        )

        data = threshold_results[
            name
        ]

        samples = data.get(
            "required_usable_samples"
        )

        print()

        print(
            f"Target KRP = "
            f"{100 * threshold:.0f}%"
        )

        if samples is None:

            print(
                "  Threshold not reached "
                "within tested sample range."
            )

            continue

        print(
            f"  Required usable samples          : "
            f"{samples}"
        )

        print(
            f"  Observed tie-aware KRP           : "
            f"{data['tie_aware_krp']:.6f}"
        )

        print(
            f"  Naive expected full attempts     : "
            f"{data['expected_naive_full_attempts']:.1f}"
        )

        print(
            f"  AA expected full attempts        : "
            f"{data['expected_aa_full_attempts']:.1f}"
        )

        print(
            f"  Total AA iterations              : "
            f"{data['expected_total_aa_iterations']:.1f}"
        )

        print(
            f"  AA prep-family call proxy        : "
            f"{data['expected_total_aa_prep_family_calls']:.1f}"
        )

    print()
    print("[Interpretation]")

    print(
        "Amplitude amplification changes how expensive it is "
        "to obtain a usable correlation-state sample."
    )

    print(
        "It does NOT modify the key distribution inside that state."
    )

    print(
        "Therefore, the number of usable samples required for "
        "key recovery is determined by the same finite-sampling "
        "problem studied previously."
    )

    print()
    print("[Output files]")

    print(
        f"Summary CSV                       : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary JSON                      : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"KRP figure                        : "
        f"{FIG_KRP}"
    )

    print(
        f"Cost-comparison figure            : "
        f"{FIG_COST}"
    )

    print()
    print(
        "[PASS] EXP06B completed successfully."
    )


if __name__ == "__main__":
    main()