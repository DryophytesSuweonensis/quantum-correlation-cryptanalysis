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

from heys_spn import encrypt_block, target_last_round_subkey
from linear_attack import (
    INV_SBOX_ARRAY,
    plaintext_linear_parity,
    run_linear_attack,
    u4_target_parity,
)


OUTPUT_CSV = RESULTS_DIR / "exp05_coherent_probabilities.csv"
OUTPUT_JSON = RESULTS_DIR / "exp05_summary.json"

FIG_ALL = FIGURES_DIR / "exp05_classical_vs_coherent.png"
FIG_TOP = FIGURES_DIR / "exp05_coherent_top_candidates.png"


# =====================================================================
# Configuration
# =====================================================================

SEED = 20261022

# Same 10,000-pair pool used in exp01.
N_REFERENCE_PAIRS = 10_000

# Use the first 8192 samples.
#
# 8192 = 2^13, so the data-index register can be represented
# exactly by 13 qubits and transformed with H^⊗13.
N_DATA = 8_192

ROUND_KEYS = (
    0x3A94,
    0xD63F,
    0xB127,
    0x8E5C,
    0xA2B4,
)

N_KEY_QUBITS = 8
N_CANDIDATES = 2**N_KEY_QUBITS

TOP_K = 16


# =====================================================================
# Generate the same known P/C pairs as exp01
# =====================================================================

def generate_known_pairs(
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:

    if n_pairs > 2**16:
        raise ValueError(
            "The 16-bit Heys SPN has only 65536 distinct plaintexts."
        )

    rng = np.random.default_rng(seed)

    plaintexts = rng.choice(
        2**16,
        size=n_pairs,
        replace=False,
    ).astype(np.uint16)

    ciphertexts = np.asarray(
        [
            encrypt_block(int(p), ROUND_KEYS)
            for p in plaintexts
        ],
        dtype=np.uint16,
    )

    return plaintexts, ciphertexts


# =====================================================================
# Normalized Fast Walsh-Hadamard Transform
# =====================================================================

def normalized_fwht_rows(
    x: np.ndarray,
) -> np.ndarray:
    """
    Apply the normalized Walsh-Hadamard Transform independently
    to every row.

    This simulates H^(tensor d) on the data-index register.

    x.shape = (number_of_keys, number_of_data_points)
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    ).copy()

    if x.ndim != 2:
        raise ValueError(
            "x must be a 2-D array."
        )

    n_rows, n_cols = x.shape

    if (
        n_cols == 0
        or (n_cols & (n_cols - 1)) != 0
    ):
        raise ValueError(
            "Number of columns must be a power of two."
        )

    h = 1

    while h < n_cols:

        view = x.reshape(
            n_rows,
            -1,
            2,
            h,
        )

        left = view[:, :, 0, :].copy()
        right = view[:, :, 1, :].copy()

        view[:, :, 0, :] = (
            left + right
        )

        view[:, :, 1, :] = (
            left - right
        )

        x = view.reshape(
            n_rows,
            n_cols,
        )

        h *= 2

    # Normalized Hadamard
    x /= math.sqrt(n_cols)

    return x


# =====================================================================
# Construct the coherent relation-phase oracle table
# =====================================================================

def build_relation_signs(
    plaintexts: np.ndarray,
    ciphertexts: np.ndarray,
) -> np.ndarray:
    """
    Construct

        s(k, i) = (-1)^r(k,i)

    for every subkey candidate k and known P/C pair i.

    The Heys linear relation is

        r(k,i)
        =
        P5 xor P7 xor P8
        xor U4,6 xor U4,8
        xor U4,14 xor U4,16.

    A coherent quantum phase oracle would perform

        |k>|i>
            ->
        (-1)^r(k,i) |k>|i>.

    Here we simulate the exact action of that oracle at the
    state-vector level.

    IMPORTANT:
    coherent data lookup and reversible oracle construction
    are NOT treated as free resources.
    """

    plaintexts = np.asarray(
        plaintexts,
        dtype=np.uint16,
    )

    ciphertexts = np.asarray(
        ciphertexts,
        dtype=np.uint16,
    )

    if plaintexts.shape != ciphertexts.shape:
        raise ValueError(
            "plaintexts and ciphertexts must have same shape."
        )

    n_data = len(plaintexts)

    # -------------------------------------------------------------
    # Plaintext part:
    #
    # P5 xor P7 xor P8
    # -------------------------------------------------------------

    p_parity = np.fromiter(
        (
            plaintext_linear_parity(int(p))
            for p in plaintexts
        ),
        dtype=np.uint8,
        count=n_data,
    )

    # -------------------------------------------------------------
    # Ciphertext nibble 2 and nibble 4
    # -------------------------------------------------------------

    c2 = (
        (ciphertexts >> 8)
        & 0xF
    ).astype(np.uint8)

    c4 = (
        ciphertexts
        & 0xF
    ).astype(np.uint8)

    # rows   = key candidates
    # cols   = known P/C pairs

    signs = np.empty(
        (
            N_CANDIDATES,
            n_data,
        ),
        dtype=np.int8,
    )

    # -------------------------------------------------------------
    # Evaluate relation for every candidate
    # -------------------------------------------------------------

    for k2 in range(16):

        u2 = INV_SBOX_ARRAY[
            c2 ^ k2
        ]

        u2_parity = (
            u4_target_parity(u2)
        )

        for k4 in range(16):

            u4 = INV_SBOX_ARRAY[
                c4 ^ k4
            ]

            u4_parity = (
                u4_target_parity(u4)
            )

            relation = (
                p_parity
                ^ u2_parity
                ^ u4_parity
            )

            candidate = (
                (k2 << 4)
                | k4
            )

            # relation = 0 -> +1 phase
            # relation = 1 -> -1 phase

            signs[candidate, :] = (
                1
                - 2 * relation.astype(
                    np.int8
                )
            )

    return signs


# =====================================================================
# Metrics
# =====================================================================

def state_fidelity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    overlap = np.vdot(
        a,
        b,
    )

    return float(
        np.abs(overlap) ** 2
    )


def total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    return float(
        0.5
        * np.sum(
            np.abs(p - q)
        )
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

    # -------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------

    if N_DATA > N_REFERENCE_PAIRS:
        raise ValueError(
            "N_DATA cannot exceed N_REFERENCE_PAIRS."
        )

    if N_DATA & (N_DATA - 1):
        raise ValueError(
            "N_DATA must be a power of two."
        )

    n_data_qubits = int(
        math.log2(N_DATA)
    )

    total_logical_qubits = (
        N_KEY_QUBITS
        + n_data_qubits
    )

    true_k2, true_k4 = (
        target_last_round_subkey(
            ROUND_KEYS
        )
    )

    true_candidate = (
        (true_k2 << 4)
        | true_k4
    )

    print("=" * 78)
    print(
        "EXP05: Coherent Correlation Extraction"
    )
    print("=" * 78)

    print()
    print("[Configuration]")

    print(
        f"Reference pair pool          : "
        f"{N_REFERENCE_PAIRS}"
    )

    print(
        f"Pairs used                   : "
        f"{N_DATA}"
    )

    print(
        f"Key candidates               : "
        f"{N_CANDIDATES}"
    )

    print(
        f"Key-register qubits          : "
        f"{N_KEY_QUBITS}"
    )

    print(
        f"Data-index qubits            : "
        f"{n_data_qubits}"
    )

    print(
        f"Conceptual logical qubits    : "
        f"{total_logical_qubits}"
    )

    print(
        f"True subkey                  : "
        f"0x{true_candidate:02X}"
    )

    # =================================================================
    # Step 1
    # Reproduce the exp01 known-P/C pool
    # =================================================================

    plaintext_pool, ciphertext_pool = (
        generate_known_pairs(
            n_pairs=N_REFERENCE_PAIRS,
            seed=SEED,
        )
    )

    # Use first 8192 samples from the exact same pool.
    plaintexts = (
        plaintext_pool[:N_DATA]
    )

    ciphertexts = (
        ciphertext_pool[:N_DATA]
    )

    # =================================================================
    # Step 2
    # Classical reference using the SAME 8192 pairs
    # =================================================================

    classical_df = run_linear_attack(
        plaintexts=plaintexts,
        ciphertexts=ciphertexts,
        true_k2=true_k2,
        true_k4=true_k4,
    )

    classical_df = (
        classical_df
        .sort_values("candidate")
        .reset_index(drop=True)
    )

    biases = (
        classical_df["bias"]
        .to_numpy(
            dtype=np.float64
        )
    )

    # -------------------------------------------------------------
    # Relation between bias and correlation
    #
    # C(k)
    # =
    # E[(-1)^r]
    #
    # =
    # P(r=0) - P(r=1)
    #
    # =
    # 2 * bias(k)
    # -------------------------------------------------------------

    correlations = (
        2.0 * biases
    )

    correlation_norm = float(
        np.linalg.norm(
            correlations
        )
    )

    reference_amplitudes = (
        correlations
        / correlation_norm
    )

    reference_probabilities = (
        reference_amplitudes**2
    )

    # =================================================================
    # Step 3
    # Apply the coherent phase oracle
    # =================================================================
    #
    # Initial state:
    #
    #   1 / sqrt(KD)
    #   sum_k sum_i |k>|i>
    #
    # After relation phase oracle:
    #
    #   1 / sqrt(KD)
    #   sum_k sum_i
    #   (-1)^r(k,i) |k>|i>
    #
    # =================================================================

    signs = build_relation_signs(
        plaintexts=plaintexts,
        ciphertexts=ciphertexts,
    )

    joint_state_after_oracle = (
        signs.astype(np.float64)
        / math.sqrt(
            N_CANDIDATES
            * N_DATA
        )
    )

    oracle_state_norm = float(
        np.sum(
            joint_state_after_oracle**2
        )
    )

    # =================================================================
    # Step 4
    # Apply Hadamard transform to data-index register
    # =================================================================

    joint_state_after_hadamard = (
        normalized_fwht_rows(
            joint_state_after_oracle
        )
    )

    transformed_state_norm = float(
        np.sum(
            joint_state_after_hadamard**2
        )
    )

    # -------------------------------------------------------------
    # Data register = |000...0>
    #
    # The corresponding amplitude is
    #
    # A(k, 0)
    # =
    # 1 / (D sqrt(K))
    # sum_i (-1)^r(k,i)
    #
    # =
    # C(k) / sqrt(K)
    # -------------------------------------------------------------

    zero_data_amplitudes = (
        joint_state_after_hadamard[
            :,
            0,
        ]
    )

    expected_zero_data_amplitudes = (
        correlations
        / math.sqrt(
            N_CANDIDATES
        )
    )

    max_zero_amplitude_error = float(
        np.max(
            np.abs(
                zero_data_amplitudes
                - expected_zero_data_amplitudes
            )
        )
    )

    # =================================================================
    # Step 5
    # Postselect data register = |000...0>
    # =================================================================

    postselection_probability = float(
        np.sum(
            zero_data_amplitudes**2
        )
    )

    if postselection_probability <= 0:
        raise RuntimeError(
            "Postselection probability is zero."
        )

    coherent_amplitudes = (
        zero_data_amplitudes
        / math.sqrt(
            postselection_probability
        )
    )

    coherent_probabilities = (
        coherent_amplitudes**2
    )

    # =================================================================
    # Step 6
    # Compare coherent result against classical correlation vector
    # =================================================================

    fidelity = state_fidelity(
        reference_amplitudes,
        coherent_amplitudes,
    )

    tvd = total_variation_distance(
        reference_probabilities,
        coherent_probabilities,
    )

    max_amplitude_error = float(
        np.max(
            np.abs(
                reference_amplitudes
                - coherent_amplitudes
            )
        )
    )

    # =================================================================
    # Step 7
    # Candidate ranking
    # =================================================================

    reference_ranks = (
        pd.Series(
            reference_probabilities
        )
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
        .to_numpy()
    )

    coherent_ranks = (
        pd.Series(
            coherent_probabilities
        )
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
        .to_numpy()
    )

    true_bias = float(
        biases[
            true_candidate
        ]
    )

    true_correlation = float(
        correlations[
            true_candidate
        ]
    )

    true_probability = float(
        coherent_probabilities[
            true_candidate
        ]
    )

    true_rank = int(
        coherent_ranks[
            true_candidate
        ]
    )

    top_candidate = int(
        np.argmax(
            coherent_probabilities
        )
    )

    # =================================================================
    # Step 8
    # Cost of naive postselection
    # =================================================================

    expected_runs_per_postselection = (
        1.0
        / postselection_probability
    )

    runs_for_95pct_postselection = (
        math.ceil(
            math.log(
                1.0 - 0.95
            )
            / math.log(
                1.0
                - postselection_probability
            )
        )
    )

    # Probability of seeing:
    #
    #   data register = 0
    #   AND
    #   key = true key
    #
    # in a single full run.
    unconditional_true_probability = float(
        zero_data_amplitudes[
            true_candidate
        ] ** 2
    )

    # =================================================================
    # Step 9
    # Save candidate-level CSV
    # =================================================================

    output_df = pd.DataFrame(
        {
            "candidate":
                np.arange(
                    N_CANDIDATES,
                    dtype=int,
                ),

            "candidate_hex":
                [
                    f"{k:02X}"
                    for k
                    in range(
                        N_CANDIDATES
                    )
                ],

            "classical_bias":
                biases,

            "classical_correlation":
                correlations,

            "reference_amplitude":
                reference_amplitudes,

            "coherent_amplitude":
                coherent_amplitudes,

            "reference_probability":
                reference_probabilities,

            "coherent_probability":
                coherent_probabilities,

            "reference_rank":
                reference_ranks,

            "coherent_rank":
                coherent_ranks,

            "is_true_key":
                (
                    np.arange(
                        N_CANDIDATES
                    )
                    == true_candidate
                ),
        }
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # =================================================================
    # Step 10
    # Save JSON summary
    # =================================================================

    summary = {

        "experiment":
            "exp05_coherent_correlation_extraction",

        "seed":
            SEED,

        "reference_pair_pool":
            N_REFERENCE_PAIRS,

        "n_data":
            N_DATA,

        "n_candidates":
            N_CANDIDATES,

        "key_register_qubits":
            N_KEY_QUBITS,

        "data_index_qubits":
            n_data_qubits,

        "conceptual_logical_qubits":
            total_logical_qubits,

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "true_bias":
            true_bias,

        "true_correlation":
            true_correlation,

        "true_postselected_probability":
            true_probability,

        "true_coherent_rank":
            true_rank,

        "top_candidate":
            top_candidate,

        "top_candidate_hex":
            f"{top_candidate:02X}",

        "oracle_state_norm":
            oracle_state_norm,

        "transformed_state_norm":
            transformed_state_norm,

        "postselection_probability":
            postselection_probability,

        "expected_runs_per_successful_postselection":
            expected_runs_per_postselection,

        "runs_for_95pct_at_least_one_postselection":
            runs_for_95pct_postselection,

        "unconditional_true_key_probability_per_full_run":
            unconditional_true_probability,

        "coherent_vs_classical_state_fidelity":
            fidelity,

        "coherent_vs_classical_tvd":
            tvd,

        "max_postselected_amplitude_error":
            max_amplitude_error,

        "max_zero_register_amplitude_identity_error":
            max_zero_amplitude_error,

        "important_limitation": (
            "The phase oracle and coherent access to "
            "known plaintext/ciphertext data are simulated "
            "at the state level. Their reversible circuit "
            "or qRAM cost is not free and is not included "
            "in this experiment."
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
    # All candidates
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    candidates = np.arange(
        N_CANDIDATES
    )

    ax.plot(
        candidates,
        reference_probabilities,
        linewidth=1.5,
        label="Classical correlation reference",
    )

    ax.scatter(
        candidates,
        coherent_probabilities,
        s=10,
        label="Coherent extraction",
    )

    ax.set_xlabel(
        "8-bit subkey candidate"
    )

    ax.set_ylabel(
        "Postselected key probability"
    )

    ax.set_title(
        "Classical Correlations vs Coherently Extracted State"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_ALL,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Top candidates
    # =================================================================

    top_indices = np.argsort(
        reference_probabilities
    )[::-1][:TOP_K]

    x = np.arange(
        TOP_K
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        x - width / 2,
        reference_probabilities[
            top_indices
        ],
        width=width,
        label="Classical reference",
    )

    ax.bar(
        x + width / 2,
        coherent_probabilities[
            top_indices
        ],
        width=width,
        label="Coherent extraction",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            f"0x{k:02X}"
            for k
            in top_indices
        ],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Subkey candidate"
    )

    ax.set_ylabel(
        "Postselected key probability"
    )

    ax.set_title(
        f"Top {TOP_K} Coherent Correlation-State Candidates"
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
    print("[State identity checks]")

    print(
        f"Norm after phase oracle       : "
        f"{oracle_state_norm:.12f}"
    )

    print(
        f"Norm after Hadamard           : "
        f"{transformed_state_norm:.12f}"
    )

    print(
        f"Max A(k,0)-C(k)/sqrt(K) error : "
        f"{max_zero_amplitude_error:.3e}"
    )

    print()
    print("[Postselected correlation state]")

    print(
        f"Postselection probability     : "
        f"{postselection_probability:.10f}"
    )

    print(
        f"Expected full runs / success  : "
        f"{expected_runs_per_postselection:.2f}"
    )

    print(
        f"Runs for >=95% postselection  : "
        f"{runs_for_95pct_postselection}"
    )

    print(
        f"State fidelity vs classical   : "
        f"{fidelity:.12f}"
    )

    print(
        f"TVD vs classical              : "
        f"{tvd:.3e}"
    )

    print(
        f"Max postselected amp. error   : "
        f"{max_amplitude_error:.3e}"
    )

    print()
    print("[True key]")

    print(
        f"Candidate                     : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"Bias                          : "
        f"{true_bias:.8f}"
    )

    print(
        f"Correlation                   : "
        f"{true_correlation:.8f}"
    )

    print(
        f"Conditional probability       : "
        f"{true_probability:.8f} "
        f"({100 * true_probability:.4f}%)"
    )

    print(
        f"Conditional rank              : "
        f"{true_rank}"
    )

    print(
        f"Unconditional P(data=0,key*)  : "
        f"{unconditional_true_probability:.10e}"
    )

    print()
    print("[Important limitation]")

    print(
        "This experiment simulates the coherent relation-phase "
        "oracle at the state level."
    )

    print(
        "Coherent P/C lookup and reversible oracle construction "
        "are NOT assumed to be free in an end-to-end cost model."
    )

    print()
    print("[Output files]")

    print(
        f"Candidate results             : "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary JSON                  : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"All-candidate figure          : "
        f"{FIG_ALL}"
    )

    print(
        f"Top-candidate figure          : "
        f"{FIG_TOP}"
    )

    print()

    if (
        true_rank == 1
        and fidelity > 1.0 - 1e-10
    ):
        print(
            "[PASS] Coherent extraction reproduces "
            "the correlation state and ranks "
            "the true key #1."
        )

    elif fidelity > 1.0 - 1e-10:

        print(
            "[PASS] Coherent extraction reproduces "
            "the correlation state, but the true key "
            "is not rank #1 for this data size."
        )

    else:

        print(
            "[WARNING] Coherent extraction does not "
            "match the classical reference."
        )


if __name__ == "__main__":
    main()