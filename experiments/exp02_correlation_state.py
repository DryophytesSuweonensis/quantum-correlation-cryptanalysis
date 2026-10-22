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

INPUT_CSV = RESULTS_DIR / "candidate_correlations.csv"

OUTPUT_CSV = RESULTS_DIR / "exp02_quantum_probabilities.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "exp02_summary.json"

FIG_FULL = FIGURES_DIR / "exp02_probability_distribution.png"
FIG_TOP = FIGURES_DIR / "exp02_top_candidates.png"


# =====================================================================
# Configuration
# =====================================================================

TOP_K = 16

CONFIDENCE_LEVELS = [
    0.90,
    0.95,
    0.99,
]


# =====================================================================
# Utility functions
# =====================================================================

def required_hits_shots(
    probability: float,
    confidence: float,
) -> int | None:
    """
    Number of independent measurements required so that the candidate
    is observed at least once with probability >= confidence.

    For one-shot candidate probability p,

        P(not observed after R shots) = (1 - p)^R

    Therefore,

        1 - (1-p)^R >= confidence.
    """
    if probability <= 0.0:
        return None

    if probability >= 1.0:
        return 1

    failure_probability = 1.0 - confidence

    shots = math.log(failure_probability) / math.log(1.0 - probability)

    return math.ceil(shots)


def shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Shannon entropy in bits.

        H = -sum p log2 p
    """
    p = probabilities[probabilities > 0]

    return float(-np.sum(p * np.log2(p)))


def inverse_participation_ratio(probabilities: np.ndarray) -> float:
    """
    Effective number of populated basis states:

        N_eff = 1 / sum p_k^2

    Uniform distribution over N states gives N_eff = N.
    """
    return float(1.0 / np.sum(probabilities**2))


# =====================================================================
# Main
# =====================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load exp01 results
    # -----------------------------------------------------------------

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}\n\n"
            "Run exp01_classical_baseline.py first."
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "candidate",
        "candidate_hex",
        "bias",
        "abs_bias",
        "is_true_key",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "The input CSV is missing required columns:\n"
            f"{sorted(missing)}"
        )

    # Candidate basis order:
    #
    # |00000000>, |00000001>, ..., |11111111>
    #
    # Therefore we sort using integer candidate ID,
    # not the classical bias rank from exp01.
    df = df.sort_values("candidate").reset_index(drop=True)

    n_candidates = len(df)

    # -----------------------------------------------------------------
    # Classical correlation/bias vector
    # -----------------------------------------------------------------

    bias = df["bias"].to_numpy(dtype=np.float64)

    energy = float(np.sum(bias**2))

    if energy <= 0:
        raise RuntimeError(
            "All candidate biases are zero. "
            "A correlation state cannot be constructed."
        )

    norm = math.sqrt(energy)

    # -----------------------------------------------------------------
    # Ideal quantum correlation state
    # -----------------------------------------------------------------
    #
    # |psi_C> = sum_k a_k |k>
    #
    # where
    #
    # a_k = bias(k) / sqrt(sum_j bias(j)^2)
    #
    # The SIGN of the bias is retained in the amplitude.
    #
    # Measurement probabilities:
    #
    # p_k = |a_k|^2
    #
    # -----------------------------------------------------------------

    amplitudes = bias / norm

    probabilities = amplitudes**2

    # Numerical sanity checks
    amp_norm = float(np.sum(amplitudes**2))
    prob_sum = float(np.sum(probabilities))

    if not np.isclose(amp_norm, 1.0, atol=1e-12):
        raise RuntimeError(
            f"State normalization failed: {amp_norm}"
        )

    if not np.isclose(prob_sum, 1.0, atol=1e-12):
        raise RuntimeError(
            f"Probability normalization failed: {prob_sum}"
        )

    df["quantum_amplitude"] = amplitudes
    df["quantum_probability"] = probabilities

    # -----------------------------------------------------------------
    # Probability ranking
    # -----------------------------------------------------------------

    prob_rank = (
        pd.Series(probabilities)
        .rank(method="min", ascending=False)
        .astype(int)
        .to_numpy()
    )

    df["probability_rank"] = prob_rank

    # -----------------------------------------------------------------
    # True-key statistics
    # -----------------------------------------------------------------

    true_rows = df[df["is_true_key"].astype(bool)]

    if len(true_rows) != 1:
        raise RuntimeError(
            "Expected exactly one true-key candidate, "
            f"but found {len(true_rows)}."
        )

    true_row = true_rows.iloc[0]

    true_candidate = int(true_row["candidate"])
    true_candidate_hex = str(true_row["candidate_hex"])

    p_star = float(true_row["quantum_probability"])
    a_star = float(true_row["quantum_amplitude"])

    # -----------------------------------------------------------------
    # Find strongest competing candidate
    # -----------------------------------------------------------------

    wrong_df = df[~df["is_true_key"].astype(bool)].copy()

    second_idx = wrong_df["quantum_probability"].idxmax()
    second_row = df.loc[second_idx]

    second_candidate = int(second_row["candidate"])
    second_candidate_hex = str(second_row["candidate_hex"])

    p_second = float(second_row["quantum_probability"])

    probability_gap = p_star - p_second

    probability_ratio = (
        p_star / p_second
        if p_second > 0
        else float("inf")
    )

    # -----------------------------------------------------------------
    # Uniform baseline
    # -----------------------------------------------------------------

    p_uniform = 1.0 / n_candidates

    uniform_gain = p_star / p_uniform

    # -----------------------------------------------------------------
    # Distribution descriptors
    # -----------------------------------------------------------------

    entropy = shannon_entropy(probabilities)

    max_entropy = math.log2(n_candidates)

    normalized_entropy = (
        entropy / max_entropy
        if max_entropy > 0
        else 0.0
    )

    effective_candidates = inverse_participation_ratio(
        probabilities
    )

    # -----------------------------------------------------------------
    # Shot requirement: "observe true key at least once"
    # -----------------------------------------------------------------

    hit_shots = {}

    for confidence in CONFIDENCE_LEVELS:
        shots = required_hits_shots(
            probability=p_star,
            confidence=confidence,
        )

        key = f"shots_for_{int(confidence * 100)}pct_hit"

        hit_shots[key] = shots

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    output_columns = [
        "candidate",
        "candidate_hex",
        "bias",
        "abs_bias",
        "quantum_amplitude",
        "quantum_probability",
        "probability_rank",
        "is_true_key",
    ]

    # Include original rank if present
    if "rank" in df.columns:
        output_columns.insert(
            output_columns.index("is_true_key"),
            "rank",
        )

    df[output_columns].to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = {
        "experiment": "exp02_ideal_correlation_state",
        "n_candidates": int(n_candidates),

        "true_candidate": true_candidate,
        "true_candidate_hex": true_candidate_hex,

        "true_bias": float(true_row["bias"]),
        "true_abs_bias": float(true_row["abs_bias"]),

        "true_quantum_amplitude": a_star,
        "true_quantum_probability": p_star,

        "second_candidate": second_candidate,
        "second_candidate_hex": second_candidate_hex,
        "second_quantum_probability": p_second,

        "probability_gap": probability_gap,
        "true_to_second_probability_ratio": probability_ratio,

        "uniform_probability": p_uniform,
        "true_probability_gain_over_uniform": uniform_gain,

        "shannon_entropy_bits": entropy,
        "maximum_entropy_bits": max_entropy,
        "normalized_entropy": normalized_entropy,

        "effective_number_of_candidates": effective_candidates,

        "state_norm": amp_norm,
        "probability_sum": prob_sum,

        **hit_shots,
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    # -----------------------------------------------------------------
    # Figure 1:
    # Probability distribution over all candidates
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    candidates = df["candidate"].to_numpy()

    ax.plot(
        candidates,
        probabilities,
        linewidth=1.0,
    )

    ax.scatter(
        [true_candidate],
        [p_star],
        s=80,
        marker="o",
        label=f"True key 0x{true_candidate_hex}",
        zorder=5,
    )

    ax.set_xlabel(
        "8-bit subkey candidate"
    )

    ax.set_ylabel(
        "Ideal quantum measurement probability"
    )

    ax.set_title(
        "Ideal Correlation-State Measurement Distribution"
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

    # -----------------------------------------------------------------
    # Figure 2:
    # Top-K candidates
    # -----------------------------------------------------------------

    top_df = (
        df.sort_values(
            "quantum_probability",
            ascending=False,
        )
        .head(TOP_K)
        .copy()
    )

    labels = [
        f"0x{x}"
        for x in top_df["candidate_hex"]
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x = np.arange(len(top_df))

    ax.bar(
        x,
        top_df["quantum_probability"].to_numpy(),
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Subkey candidate"
    )

    ax.set_ylabel(
        "Quantum measurement probability"
    )

    ax.set_title(
        f"Top {TOP_K} Correlation-State Candidates"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_TOP,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------

    print("=" * 78)
    print("EXP02: Ideal Quantum Correlation-State Analysis")
    print("=" * 78)

    print()
    print("[State construction]")
    print(
        "|psi_C> = sum_k bias(k) |k> "
        "/ sqrt(sum_j bias(j)^2)"
    )

    print()
    print(f"Number of candidates           : {n_candidates}")
    print(f"State norm                     : {amp_norm:.12f}")
    print(f"Probability sum                : {prob_sum:.12f}")

    print()
    print("[True-key result]")
    print(
        f"True candidate                 : "
        f"0x{true_candidate_hex}"
    )

    print(
        f"Classical bias                 : "
        f"{float(true_row['bias']):.8f}"
    )

    print(
        f"Normalized amplitude           : "
        f"{a_star:.8f}"
    )

    print(
        f"Measurement probability        : "
        f"{p_star:.8f} "
        f"({100 * p_star:.4f}%)"
    )

    print()
    print("[Strongest competitor]")

    print(
        f"Second candidate               : "
        f"0x{second_candidate_hex}"
    )

    print(
        f"Second probability             : "
        f"{p_second:.8f} "
        f"({100 * p_second:.4f}%)"
    )

    print(
        f"Probability gap                : "
        f"{probability_gap:.8f}"
    )

    print(
        f"True / second probability      : "
        f"{probability_ratio:.4f} x"
    )

    print()
    print("[Uniform baseline]")

    print(
        f"Uniform probability            : "
        f"{p_uniform:.8f} "
        f"({100 * p_uniform:.4f}%)"
    )

    print(
        f"True-key gain over uniform     : "
        f"{uniform_gain:.4f} x"
    )

    print()
    print("[Distribution concentration]")

    print(
        f"Shannon entropy                : "
        f"{entropy:.6f} bits"
    )

    print(
        f"Maximum entropy                : "
        f"{max_entropy:.6f} bits"
    )

    print(
        f"Normalized entropy             : "
        f"{normalized_entropy:.6f}"
    )

    print(
        f"Effective candidate count      : "
        f"{effective_candidates:.2f}"
    )

    print()
    print("[Shots for at least one true-key observation]")

    for confidence in CONFIDENCE_LEVELS:
        key = f"shots_for_{int(confidence * 100)}pct_hit"

        print(
            f"{100 * confidence:5.1f}% confidence              : "
            f"{hit_shots[key]} shots"
        )

    print()
    print("[Output files]")
    print(f"Probability CSV                : {OUTPUT_CSV}")
    print(f"Summary JSON                   : {OUTPUT_SUMMARY}")
    print(f"Full-distribution figure       : {FIG_FULL}")
    print(f"Top-candidate figure           : {FIG_TOP}")

    print()
    print("[PASS] exp02 completed successfully.")


if __name__ == "__main__":
    main()