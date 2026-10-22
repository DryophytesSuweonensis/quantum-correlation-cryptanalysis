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

INPUT_CSV = RESULTS_DIR / "exp02_quantum_probabilities.csv"

OUTPUT_SUMMARY_CSV = RESULTS_DIR / "exp03_sampling_summary.csv"
OUTPUT_JSON = RESULTS_DIR / "exp03_summary.json"

FIG_KRP = FIGURES_DIR / "exp03_krp_vs_shots.png"
FIG_HIT = FIGURES_DIR / "exp03_hit_probability_vs_shots.png"


# =====================================================================
# Experiment configuration
# =====================================================================

SEED = 20261022

# Number of quantum measurements per experiment
SHOT_COUNTS = [
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

# Number of independent Monte Carlo experiments
N_TRIALS = 5000


# =====================================================================
# Utility functions
# =====================================================================

def theoretical_hit_probability(
    p_true: float,
    shots: int,
) -> float:
    """
    Probability that the true key appears at least once
    in 'shots' independent measurements.

        P(hit >= 1) = 1 - (1 - p_true)^shots
    """
    return 1.0 - (1.0 - p_true) ** shots


def run_sampling_experiment(
    probabilities: np.ndarray,
    true_index: int,
    shots: int,
    n_trials: int,
    rng: np.random.Generator,
) -> dict:
    """
    Perform repeated multinomial sampling.

    Each trial represents one complete quantum measurement experiment
    consisting of 'shots' measurements.

    We record:

    1. hit_probability:
       Did the true key appear at least once?

    2. strict_krp:
       Was the true key the UNIQUE most frequently observed candidate?

    3. tie_aware_krp:
       If multiple candidates tie for maximum count and the true key is
       among them, assign success probability 1 / number_of_tied_modes.

       This corresponds to random tie-breaking.

    4. top_set_krp:
       Was the true key among the candidates tied for maximum count?
    """

    hit_successes = 0
    strict_successes = 0
    top_set_successes = 0

    tie_aware_score = 0.0

    total_true_counts = 0

    # Keep track of how often ties occur
    tie_trials = 0

    for _ in range(n_trials):

        # Multinomial sampling:
        #
        # counts[k] = number of times candidate k was measured
        #
        counts = rng.multinomial(
            n=shots,
            pvals=probabilities,
        )

        true_count = int(counts[true_index])

        total_true_counts += true_count

        # -------------------------------------------------------------
        # Metric 1: true key observed at least once
        # -------------------------------------------------------------

        if true_count > 0:
            hit_successes += 1

        # -------------------------------------------------------------
        # Find measurement mode
        # -------------------------------------------------------------

        max_count = int(np.max(counts))

        mode_indices = np.flatnonzero(
            counts == max_count
        )

        n_modes = len(mode_indices)

        true_is_mode = true_index in mode_indices

        # -------------------------------------------------------------
        # Metric 2: strict KRP
        #
        # True key must be the unique mode.
        # -------------------------------------------------------------

        if n_modes == 1 and mode_indices[0] == true_index:
            strict_successes += 1

        # -------------------------------------------------------------
        # Metric 3: top-set KRP
        #
        # True key only needs to belong to the highest-count set.
        # -------------------------------------------------------------

        if true_is_mode:
            top_set_successes += 1

        # -------------------------------------------------------------
        # Metric 4: tie-aware KRP
        #
        # Randomly selecting one candidate among tied modes gives
        # expected success = 1 / n_modes if true key is tied.
        # -------------------------------------------------------------

        if true_is_mode:
            tie_aware_score += 1.0 / n_modes

        if n_modes > 1:
            tie_trials += 1

    # -----------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------

    empirical_hit_probability = (
        hit_successes / n_trials
    )

    strict_krp = (
        strict_successes / n_trials
    )

    top_set_krp = (
        top_set_successes / n_trials
    )

    tie_aware_krp = (
        tie_aware_score / n_trials
    )

    tie_rate = (
        tie_trials / n_trials
    )

    mean_true_count = (
        total_true_counts / n_trials
    )

    return {
        "shots": shots,
        "trials": n_trials,

        "empirical_hit_probability":
            empirical_hit_probability,

        "strict_krp":
            strict_krp,

        "tie_aware_krp":
            tie_aware_krp,

        "top_set_krp":
            top_set_krp,

        "tie_rate":
            tie_rate,

        "mean_true_count":
            mean_true_count,
    }


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
    # Load exp02 probability distribution
    # -----------------------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}\n\n"
            "Run exp02_correlation_state.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "candidate",
        "candidate_hex",
        "quantum_probability",
        "is_true_key",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    # Sort by computational basis index
    df = (
        df.sort_values("candidate")
        .reset_index(drop=True)
    )

    probabilities = (
        df["quantum_probability"]
        .to_numpy(dtype=np.float64)
    )

    # Numerical normalization
    probabilities = (
        probabilities
        / probabilities.sum()
    )

    # -----------------------------------------------------------------
    # Identify true key
    # -----------------------------------------------------------------

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
            "Expected exactly one true-key candidate, "
            f"but found {len(true_indices)}."
        )

    true_index = int(
        true_indices[0]
    )

    true_row = df.iloc[
        true_index
    ]

    true_candidate = int(
        true_row["candidate"]
    )

    true_candidate_hex = str(
        true_row["candidate_hex"]
    )

    p_true = float(
        probabilities[true_index]
    )

    # -----------------------------------------------------------------
    # Second-highest candidate
    # -----------------------------------------------------------------

    wrong_probabilities = (
        probabilities.copy()
    )

    wrong_probabilities[
        true_index
    ] = -1.0

    second_index = int(
        np.argmax(
            wrong_probabilities
        )
    )

    second_row = df.iloc[
        second_index
    ]

    second_candidate = int(
        second_row["candidate"]
    )

    second_candidate_hex = str(
        second_row["candidate_hex"]
    )

    p_second = float(
        probabilities[
            second_index
        ]
    )

    probability_gap = (
        p_true - p_second
    )

    # -----------------------------------------------------------------
    # Random number generator
    # -----------------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    # -----------------------------------------------------------------
    # Run Monte Carlo experiments
    # -----------------------------------------------------------------

    rows = []

    for shots in SHOT_COUNTS:

        result = (
            run_sampling_experiment(
                probabilities=probabilities,
                true_index=true_index,
                shots=shots,
                n_trials=N_TRIALS,
                rng=rng,
            )
        )

        result[
            "theoretical_hit_probability"
        ] = theoretical_hit_probability(
            p_true=p_true,
            shots=shots,
        )

        result[
            "expected_true_count"
        ] = shots * p_true

        rows.append(
            result
        )

        print(
            f"shots={shots:5d} | "
            f"strict KRP="
            f"{result['strict_krp']:.4f} | "
            f"tie-aware KRP="
            f"{result['tie_aware_krp']:.4f} | "
            f"hit="
            f"{result['empirical_hit_probability']:.4f}"
        )

    summary_df = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
    )

    # -----------------------------------------------------------------
    # Find approximate shot thresholds
    # -----------------------------------------------------------------

    def first_shots_for_threshold(
        column: str,
        threshold: float,
    ):
        selected = summary_df[
            summary_df[column]
            >= threshold
        ]

        if len(selected) == 0:
            return None

        return int(
            selected.iloc[0][
                "shots"
            ]
        )

    thresholds = {}

    for target in [
        0.50,
        0.90,
        0.95,
        0.99,
    ]:

        name = (
            f"shots_for_"
            f"{int(target * 100)}pct_"
            f"tie_aware_krp"
        )

        thresholds[name] = (
            first_shots_for_threshold(
                column="tie_aware_krp",
                threshold=target,
            )
        )

    # -----------------------------------------------------------------
    # Save experiment metadata
    # -----------------------------------------------------------------

    metadata = {

        "experiment":
            "exp03_finite_shot_sampling",

        "seed":
            SEED,

        "n_trials_per_shot_count":
            N_TRIALS,

        "shot_counts":
            SHOT_COUNTS,

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            true_candidate_hex,

        "true_probability":
            p_true,

        "second_candidate":
            second_candidate,

        "second_candidate_hex":
            second_candidate_hex,

        "second_probability":
            p_second,

        "probability_gap":
            probability_gap,

        "krp_definition": (
            "Probability that the true key is selected "
            "as the measurement-histogram mode."
        ),

        "tie_handling": (
            "tie_aware_krp assumes uniform random "
            "selection among candidates tied for "
            "maximum count."
        ),

        **thresholds,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # =================================================================
    # Figure 1:
    # Key-Recovery Probability versus shots
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    shots_array = (
        summary_df[
            "shots"
        ].to_numpy()
    )

    ax.plot(
        shots_array,
        summary_df[
            "tie_aware_krp"
        ],
        marker="o",
        label="Tie-aware KRP",
    )

    ax.plot(
        shots_array,
        summary_df[
            "strict_krp"
        ],
        marker="s",
        label="Strict KRP",
    )

    ax.plot(
        shots_array,
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
        0.0,
        1.02,
    )

    ax.set_xlabel(
        "Number of measurement shots"
    )

    ax.set_ylabel(
        "Key-Recovery Probability"
    )

    ax.set_title(
        "Finite-Shot Key Recovery"
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
    # Figure 2:
    # At-least-one-hit probability
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        shots_array,
        summary_df[
            "empirical_hit_probability"
        ],
        marker="o",
        label="Monte Carlo",
    )

    ax.plot(
        shots_array,
        summary_df[
            "theoretical_hit_probability"
        ],
        linestyle="--",
        label="Theory",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylim(
        0.0,
        1.02,
    )

    ax.set_xlabel(
        "Number of measurement shots"
    )

    ax.set_ylabel(
        "Probability true key appears at least once"
    )

    ax.set_title(
        "True-Key Observation Probability"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_HIT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Console summary
    # =================================================================

    print()
    print("=" * 78)
    print("EXP03: Finite-Shot Quantum Key-Recovery Simulation")
    print("=" * 78)

    print()
    print("[Input distribution]")

    print(
        f"True key                     : "
        f"0x{true_candidate_hex}"
    )

    print(
        f"True-key probability         : "
        f"{p_true:.8f} "
        f"({100 * p_true:.4f}%)"
    )

    print(
        f"Second candidate             : "
        f"0x{second_candidate_hex}"
    )

    print(
        f"Second probability           : "
        f"{p_second:.8f} "
        f"({100 * p_second:.4f}%)"
    )

    print(
        f"Probability gap              : "
        f"{probability_gap:.8f}"
    )

    print()
    print("[Monte Carlo configuration]")

    print(
        f"Trials per shot count        : "
        f"{N_TRIALS}"
    )

    print(
        f"Random seed                  : "
        f"{SEED}"
    )

    print()
    print("[Tie-aware KRP thresholds]")

    for target in [
        50,
        90,
        95,
        99,
    ]:

        key = (
            f"shots_for_"
            f"{target}pct_"
            f"tie_aware_krp"
        )

        value = thresholds[
            key
        ]

        print(
            f"{target:2d}% recovery                 : "
            f"{value}"
        )

    print()
    print("[Output files]")

    print(
        f"Summary CSV                  : "
        f"{OUTPUT_SUMMARY_CSV}"
    )

    print(
        f"Metadata JSON                : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"KRP figure                   : "
        f"{FIG_KRP}"
    )

    print(
        f"Hit-probability figure       : "
        f"{FIG_HIT}"
    )

    print()
    print(
        "[PASS] exp03 completed successfully."
    )


if __name__ == "__main__":
    main()