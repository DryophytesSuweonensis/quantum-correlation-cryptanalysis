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
EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

sys.path.insert(
    0,
    str(EXPERIMENTS_DIR),
)

import exp08_marking_strategy_benchmark as exp08


EXP05_CSV = (
    RESULTS_DIR
    / "exp05_coherent_probabilities.csv"
)

EXP08_BANK_CSV = (
    RESULTS_DIR
    / "exp08_approximation_bank.csv"
)

EXP10_BASELINE_CSV = (
    RESULTS_DIR
    / "exp10_baseline_comparison.csv"
)

EXP11_BREAK_EVEN_CSV = (
    RESULTS_DIR
    / "exp11_break_even.csv"
)


# =====================================================================
# Outputs
# =====================================================================

OUTPUT_RAW_COST_CSV = (
    RESULTS_DIR
    / "exp12_raw_transform_cost.csv"
)

OUTPUT_INNER_AA_CSV = (
    RESULTS_DIR
    / "exp12_sector_amplification_cost.csv"
)

OUTPUT_TARGETS_CSV = (
    RESULTS_DIR
    / "exp12_state_prep_targets.csv"
)

OUTPUT_PLACEMENT_CSV = (
    RESULTS_DIR
    / "exp12_exp11_placement.csv"
)

OUTPUT_SUMMARY_JSON = (
    RESULTS_DIR
    / "exp12_summary.json"
)


FIG_SECTOR_SUCCESS = (
    FIGURES_DIR
    / "exp12_sector_success.png"
)

FIG_STATE_PREP_COST = (
    FIGURES_DIR
    / "exp12_state_prep_cost.png"
)

FIG_COST_PER_GOOD = (
    FIGURES_DIR
    / "exp12_cost_per_good_sector.png"
)

FIG_BREAK_EVEN = (
    FIGURES_DIR
    / "exp12_exp11_break_even.png"
)


# =====================================================================
# Cipher / data configuration
# =====================================================================

SEED_DATA = 20261022

N_REFERENCE_PAIRS = 10_000
N_EXP05_DATA = 8_192

KEY_QUBITS = 8
INDEX_QUBITS = 13

N_CANDIDATES = 2**KEY_QUBITS
N_DATA = 2**INDEX_QUBITS

assert N_DATA == N_EXP05_DATA


# =====================================================================
# Coherent data word
#
# The canonical relation requires:
#
#     P5, P7, P8
#     ciphertext nibble 2
#     ciphertext nibble 4
#
# Therefore coherent QROM does not need the full 16+16 bits.
#
# It only needs:
#
#     3 + 4 + 4 = 11 bits.
# =====================================================================

PLAINTEXT_RELATION_BITS = (
    5,
    7,
    8,
)

QROM_OUTPUT_BITS = (
    len(PLAINTEXT_RELATION_BITS)
    + 4
    + 4
)

assert QROM_OUTPUT_BITS == 11


# =====================================================================
# Cost model
# =====================================================================

# Same S-box scale used conceptually in EXP10.
SBOX_LOOKUP_TE = 16

# Raw QROM Toffoli-equivalent assumptions.
#
# A coherent read-only lookup over N classical words has cost
# linear in N in a straightforward unary-iteration construction.
#
# These are intentionally scenarios, NOT claims of exact synthesized
# physical gate counts.
#
# Cost below is PER QROM CALL.
QROM_SCENARIOS = {
    "optimistic": {
        "te_per_qrom_call":
            N_DATA - 1,
    },

    "baseline": {
        "te_per_qrom_call":
            2 * (N_DATA - 1),
    },

    "pessimistic": {
        "te_per_qrom_call":
            4 * (N_DATA - 1),
    },
}


# =====================================================================
# Inner amplitude-amplification sweep
# =====================================================================

MAX_INNER_AA_ITERATIONS = 80

SECTOR_TARGETS = [
    0.90,
    0.95,
    0.99,
    0.999,
]

# We will use this as the representative A in the EXP11 placement.
DEFAULT_STATE_PREP_TARGET = 0.99


# =====================================================================
# Helpers
# =====================================================================

def bit_msb16(
    x: np.ndarray,
    position: int,
) -> np.ndarray:
    """
    Heys convention:

        position 1  = MSB
        position 16 = LSB
    """

    shift = (
        16 - position
    )

    return (
        (
            x
            >> shift
        )
        & 1
    ).astype(
        np.uint16
    )


def load_true_candidate():
    """
    Read the true candidate from EXP05.
    """

    if not EXP05_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP05_CSV}\n\n"
            "Run EXP05 first."
        )

    df = pd.read_csv(
        EXP05_CSV
    )

    true_rows = df[
        df[
            "is_true_key"
        ].astype(bool)
    ]

    if len(
        true_rows
    ) != 1:

        raise RuntimeError(
            "Expected exactly one true key."
        )

    row = true_rows.iloc[0]

    return int(
        row[
            "candidate"
        ]
    )


def load_canonical_approximation():
    """
    Load A0 canonical approximation from EXP08.
    """

    if not EXP08_BANK_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP08_BANK_CSV}\n\n"
            "Run EXP08 first."
        )

    df = pd.read_csv(
        EXP08_BANK_CSV
    )

    canonical = df[
        df[
            "approximation_id"
        ]
        == "A0_canonical"
    ]

    if len(
        canonical
    ) != 1:

        raise RuntimeError(
            "Could not identify A0_canonical."
        )

    row = canonical.iloc[0]

    return {
        "approximation_id":
            str(
                row[
                    "approximation_id"
                ]
            ),

        "plaintext_mask":
            int(
                row[
                    "plaintext_mask"
                ]
            ),

        "u4_mask":
            int(
                row[
                    "u4_mask"
                ]
            ),

        "mask_nibble2":
            int(
                row[
                    "mask_nibble2"
                ]
            ),

        "mask_nibble4":
            int(
                row[
                    "mask_nibble4"
                ]
            ),

        "predicted_abs_correlation":
            float(
                row[
                    "predicted_abs_correlation"
                ]
            ),

        "source":
            str(
                row[
                    "source"
                ]
            ),
    }


# =====================================================================
# Reconstruct EXP05 empirical sector probability
# =====================================================================

def reconstruct_exp05_sector_statistics(
    canonical_approximation,
    true_candidate: int,
):
    """
    Reconstruct EXP05 using the same first 8192 known P/C pairs.

    For each candidate k:

        C_hat(k)
        =
        (1/D) sum_i (-1)^r(k,i)

    In the state after H on the data register:

        amplitude(k, y=0)
        =
        C_hat(k) / sqrt(K)

    therefore

        P_good
        =
        (1/K) sum_k C_hat(k)^2

    and

        P(k | good)
        =
        C_hat(k)^2
        / sum_j C_hat(j)^2.
    """

    plaintexts, ciphertexts = (
        exp08.generate_known_pairs(
            N_REFERENCE_PAIRS,
            SEED_DATA,
        )
    )

    exp05_p = (
        plaintexts[
            :N_EXP05_DATA
        ]
    )

    exp05_c = (
        ciphertexts[
            :N_EXP05_DATA
        ]
    )

    signs = (
        exp08.precompute_relation_signs(
            [
                canonical_approximation
            ],
            exp05_p,
            exp05_c,
        )
    )

    # shape:
    #
    # [1, 256, 8192]
    correlations = np.mean(
        signs[
            0
        ],
        axis=1,
        dtype=np.float64,
    )

    squared = (
        correlations**2
    )

    p_good = float(
        np.sum(
            squared
        )
        / N_CANDIDATES
    )

    conditional = (
        squared
        / np.sum(
            squared
        )
    )

    p_true_given_good = float(
        conditional[
            true_candidate
        ]
    )

    true_correlation = float(
        correlations[
            true_candidate
        ]
    )

    return {
        "plaintexts":
            exp05_p,

        "ciphertexts":
            exp05_c,

        "correlations":
            correlations,

        "p_good":
            p_good,

        "p_true_given_good":
            p_true_given_good,

        "true_correlation":
            true_correlation,
    }


# =====================================================================
# Build actual 11-bit QROM table
# =====================================================================

def build_qrom_words(
    plaintexts: np.ndarray,
    ciphertexts: np.ndarray,
):
    """
    Pack the minimum required information for the canonical relation:

        [P5 P7 P8 | C_nibble2 | C_nibble4]

    into one 11-bit integer per sample.
    """

    p5 = bit_msb16(
        plaintexts,
        5,
    )

    p7 = bit_msb16(
        plaintexts,
        7,
    )

    p8 = bit_msb16(
        plaintexts,
        8,
    )

    plaintext_word = (
        (
            p5 << 2
        )
        |
        (
            p7 << 1
        )
        |
        p8
    )

    c2 = (
        (
            ciphertexts
            >> 8
        )
        & 0xF
    ).astype(
        np.uint16
    )

    c4 = (
        ciphertexts
        & 0xF
    ).astype(
        np.uint16
    )

    words = (
        (
            plaintext_word
            << 8
        )
        |
        (
            c2 << 4
        )
        |
        c4
    ).astype(
        np.uint16
    )

    return words


def count_qrom_ones(
    words: np.ndarray,
):
    """
    Naive XOR-loading Clifford proxy:

        each 1 in a classical QROM word
        corresponds to one controlled bit flip
        once the address condition is selected.

    This is only a CNOT-like data-loading proxy.
    """

    total_ones = int(
        sum(
            int(x).bit_count()
            for x in words
        )
    )

    average_ones = (
        total_ones
        / len(
            words
        )
    )

    return (
        total_ones,
        average_ones,
    )


# =====================================================================
# Raw coherent correlation transform U_corr
# =====================================================================

def raw_transform_resources(
    qrom_ones: int,
    scenario_name: str,
    scenario: dict,
):
    """
    Raw coherent transform:

        |0>_K |0>_D
          -> H_K H_D
          -> coherent QROM(i)
          -> partial inverse S-boxes
          -> relation phase (-1)^r
          -> uncompute S-boxes
          -> uncompute QROM
          -> H_D

    Important:
        this produces the FULL Walsh state.
        It does NOT yet prepare the normalized y=0 correlation sector.
    """

    # -------------------------------------------------------------
    # QROM
    #
    # One lookup + one unlookup.
    # -------------------------------------------------------------

    qrom_calls = 2

    qrom_te = (
        qrom_calls
        * int(
            scenario[
                "te_per_qrom_call"
            ]
        )
    )

    # Data-bit loading/unloading CNOT proxy.
    qrom_cnot = (
        2
        * qrom_ones
    )

    # -------------------------------------------------------------
    # Partial inverse S-box
    #
    # Two attacked nibbles:
    #
    #     C2 xor k2
    #     C4 xor k4
    #
    # Two computations + two uncomputations.
    # -------------------------------------------------------------

    sbox_calls = 4

    sbox_te = (
        sbox_calls
        * SBOX_LOOKUP_TE
    )

    # -------------------------------------------------------------
    # Relation parity
    #
    # P5 xor P7 xor P8
    # xor selected U4 bits.
    #
    # canonical U masks are 0x5 and 0x5:
    #
    # 2 U bits in nibble 2
    # 2 U bits in nibble 4
    #
    # Total relation terms = 7.
    #
    # Compute parity into phase ancilla then uncompute.
    # -------------------------------------------------------------

    relation_terms = 7

    relation_cnot = (
        2
        * relation_terms
    )

    # -------------------------------------------------------------
    # Hadamards
    #
    # initial H on 8 key qubits
    # initial H on 13 data-index qubits
    # final H on 13 data-index qubits
    # -------------------------------------------------------------

    hadamards = (
        KEY_QUBITS
        + INDEX_QUBITS
        + INDEX_QUBITS
    )

    total_te = (
        qrom_te
        + sbox_te
    )

    total_cnot = (
        qrom_cnot
        + relation_cnot
    )

    # -------------------------------------------------------------
    # Logical-qubit / workspace proxy
    #
    # persistent:
    #     key      8
    #     index   13
    #
    # workspace:
    #     QROM output          11
    #     two inverse S-boxes   8
    #     parity phase ancilla  1
    #     unary-iterator work  12
    #     reflection work       1
    #
    # Work registers can be reused after uncomputation.
    # -------------------------------------------------------------

    qrom_work_qubits = (
        INDEX_QUBITS
        - 1
    )

    sbox_output_qubits = 8
    relation_ancilla = 1
    reflection_ancilla = 1

    logical_qubits = (
        KEY_QUBITS
        + INDEX_QUBITS
        + QROM_OUTPUT_BITS
        + sbox_output_qubits
        + relation_ancilla
        + qrom_work_qubits
        + reflection_ancilla
    )

    return {
        "scenario":
            scenario_name,

        "qrom_entries":
            N_DATA,

        "qrom_output_bits":
            QROM_OUTPUT_BITS,

        "qrom_calls":
            qrom_calls,

        "qrom_te":
            qrom_te,

        "qrom_cnot_proxy":
            qrom_cnot,

        "sbox_calls":
            sbox_calls,

        "sbox_te":
            sbox_te,

        "relation_cnot_proxy":
            relation_cnot,

        "hadamard_count":
            hadamards,

        "raw_transform_te":
            total_te,

        "raw_transform_cnot_proxy":
            total_cnot,

        "logical_qubits_proxy":
            logical_qubits,
    }


# =====================================================================
# Multi-controlled reflection proxies
# =====================================================================

def multi_controlled_phase_te(
    n_controls: int,
) -> int:
    """
    Clean-ancilla linear Toffoli-equivalent proxy.

    This is intentionally a simple logical-resource estimate,
    not a hardware-native synthesis count.
    """

    if n_controls <= 1:
        return 0

    return max(
        1,
        2 * n_controls - 3,
    )


def zero_sector_reflection_te():
    """
    Reflection conditioned on:

        data register == |0^13>
    """

    return multi_controlled_phase_te(
        INDEX_QUBITS
    )


def all_zero_reflection_te():
    """
    Reflection about the initial all-zero key+index state:

        |0^(8+13)>
    """

    return multi_controlled_phase_te(
        KEY_QUBITS
        + INDEX_QUBITS
    )


# =====================================================================
# Inner amplitude amplification
# =====================================================================

def amplified_probability(
    initial_probability: float,
    iterations: int,
) -> float:

    p = float(
        np.clip(
            initial_probability,
            0.0,
            1.0,
        )
    )

    if p <= 0:
        return 0.0

    if p >= 1:
        return 1.0

    theta = math.asin(
        math.sqrt(
            p
        )
    )

    return math.sin(
        (
            2 * iterations
            + 1
        )
        * theta
    ) ** 2


def inner_aa_resources(
    raw_te: float,
    raw_cnot: float,
    raw_h: int,
    iterations: int,
):
    """
    Inner AA used to amplify the zero-Walsh sector.

    Starting from |0>:

        U
        [ S_good U^-1 S_0 U ]^t

    Therefore:

        raw-U-family calls = 1 + 2t

    where S_good marks data == 0^13.
    """

    t = int(
        iterations
    )

    raw_calls = (
        1 + 2 * t
    )

    good_reflections = t
    zero_reflections = t

    good_reflection_te = (
        zero_sector_reflection_te()
    )

    zero_reflection_te = (
        all_zero_reflection_te()
    )

    reflection_te = (
        t
        * (
            good_reflection_te
            + zero_reflection_te
        )
    )

    total_te = (
        raw_calls
        * raw_te
        + reflection_te
    )

    total_cnot = (
        raw_calls
        * raw_cnot
    )

    total_h = (
        raw_calls
        * raw_h
    )

    return {
        "inner_aa_iterations":
            t,

        "raw_transform_family_calls":
            raw_calls,

        "good_sector_reflections":
            good_reflections,

        "zero_state_reflections":
            zero_reflections,

        "reflection_te":
            reflection_te,

        "state_prep_te":
            total_te,

        "state_prep_cnot_proxy":
            total_cnot,

        "state_prep_hadamards":
            total_h,
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

    # =================================================================
    # Step 1
    # Reconstruct EXP05 statistics
    # =================================================================

    true_candidate = (
        load_true_candidate()
    )

    canonical = (
        load_canonical_approximation()
    )

    exp05 = (
        reconstruct_exp05_sector_statistics(
            canonical,
            true_candidate,
        )
    )

    p_good_initial = float(
        exp05[
            "p_good"
        ]
    )

    p_true_given_good = float(
        exp05[
            "p_true_given_good"
        ]
    )

    true_correlation = float(
        exp05[
            "true_correlation"
        ]
    )

    print()
    print("=" * 80)
    print(
        "EXP12: Correlation-State Preparation Cost"
    )
    print("=" * 80)

    print()
    print(
        "[Reconstructed EXP05]"
    )

    print(
        f"True candidate                : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"True correlation              : "
        f"{true_correlation:.8f}"
    )

    print(
        f"Initial P(data=0^13)          : "
        f"{p_good_initial:.10f} "
        f"({100 * p_good_initial:.6f}%)"
    )

    print(
        f"P(true | data=0^13)           : "
        f"{p_true_given_good:.8f} "
        f"({100 * p_true_given_good:.4f}%)"
    )

    # =================================================================
    # Step 2
    # Build minimal coherent QROM table
    # =================================================================

    qrom_words = (
        build_qrom_words(
            exp05[
                "plaintexts"
            ],
            exp05[
                "ciphertexts"
            ],
        )
    )

    if np.max(
        qrom_words
    ) >= (
        1
        << QROM_OUTPUT_BITS
    ):

        raise RuntimeError(
            "QROM word exceeds configured bit width."
        )

    (
        qrom_ones,
        average_qrom_ones,
    ) = count_qrom_ones(
        qrom_words
    )

    print()
    print(
        "[Minimal coherent data lookup]"
    )

    print(
        f"Entries                       : "
        f"{len(qrom_words)}"
    )

    print(
        f"Output bits / entry           : "
        f"{QROM_OUTPUT_BITS}"
    )

    print(
        f"Total 1-bits in table         : "
        f"{qrom_ones}"
    )

    print(
        f"Average 1-bits / entry        : "
        f"{average_qrom_ones:.4f}"
    )

    # =================================================================
    # Step 3
    # Raw transform resource scenarios
    # =================================================================

    raw_rows = []

    raw_lookup = {}

    for (
        scenario_name,
        scenario,
    ) in QROM_SCENARIOS.items():

        row = raw_transform_resources(
            qrom_ones=qrom_ones,
            scenario_name=
                scenario_name,
            scenario=scenario,
        )

        raw_rows.append(
            row
        )

        raw_lookup[
            scenario_name
        ] = row

    raw_df = pd.DataFrame(
        raw_rows
    )

    raw_df.to_csv(
        OUTPUT_RAW_COST_CSV,
        index=False,
    )

    print()
    print(
        "[Raw coherent transform U_corr]"
    )

    print(
        raw_df[
            [
                "scenario",
                "raw_transform_te",
                "raw_transform_cnot_proxy",
                "hadamard_count",
                "logical_qubits_proxy",
            ]
        ].to_string(
            index=False
        )
    )

    # =================================================================
    # Step 4
    # Inner sector AA sweep
    # =================================================================

    aa_rows = []

    for scenario_name in (
        QROM_SCENARIOS.keys()
    ):

        raw = (
            raw_lookup[
                scenario_name
            ]
        )

        for t in range(
            MAX_INNER_AA_ITERATIONS
            + 1
        ):

            p_sector = (
                amplified_probability(
                    p_good_initial,
                    t,
                )
            )

            resources = (
                inner_aa_resources(
                    raw_te=float(
                        raw[
                            "raw_transform_te"
                        ]
                    ),

                    raw_cnot=float(
                        raw[
                            "raw_transform_cnot_proxy"
                        ]
                    ),

                    raw_h=int(
                        raw[
                            "hadamard_count"
                        ]
                    ),

                    iterations=t,
                )
            )

            expected_te_per_good = (
                resources[
                    "state_prep_te"
                ]
                / max(
                    p_sector,
                    1e-15,
                )
            )

            aa_rows.append(
                {
                    "scenario":
                        scenario_name,

                    "inner_aa_iterations":
                        t,

                    "zero_sector_probability":
                        p_sector,

                    "conditional_true_key_probability":
                        p_true_given_good,

                    "unconditional_true_key_probability":
                        (
                            p_sector
                            * p_true_given_good
                        ),

                    **resources,

                    "expected_te_per_good_sector_measurement":
                        expected_te_per_good,
                }
            )

    aa_df = pd.DataFrame(
        aa_rows
    )

    aa_df.to_csv(
        OUTPUT_INNER_AA_CSV,
        index=False,
    )

    # =================================================================
    # Step 5
    # Find special operating points
    # =================================================================

    target_rows = []

    for scenario_name in (
        QROM_SCENARIOS.keys()
    ):

        part = aa_df[
            aa_df[
                "scenario"
            ]
            == scenario_name
        ].copy()

        # -------------------------------------------------------------
        # Cost-optimal sector extraction point
        # -------------------------------------------------------------

        cost_optimal = (
            part
            .sort_values(
                "expected_te_per_good_sector_measurement"
            )
            .iloc[0]
        )

        target_rows.append(
            {
                "scenario":
                    scenario_name,

                "target_label":
                    "cost_optimal",

                "target_sector_probability":
                    np.nan,

                **cost_optimal.to_dict(),
            }
        )

        # -------------------------------------------------------------
        # Maximum sector probability in sweep
        # -------------------------------------------------------------

        maximum = (
            part
            .sort_values(
                "zero_sector_probability",
                ascending=False,
            )
            .iloc[0]
        )

        target_rows.append(
            {
                "scenario":
                    scenario_name,

                "target_label":
                    "max_success",

                "target_sector_probability":
                    np.nan,

                **maximum.to_dict(),
            }
        )

        # -------------------------------------------------------------
        # Earliest t reaching requested probabilities
        # -------------------------------------------------------------

        for target in SECTOR_TARGETS:

            feasible = part[
                part[
                    "zero_sector_probability"
                ]
                >= target
            ]

            if len(
                feasible
            ) == 0:

                continue

            earliest = (
                feasible
                .sort_values(
                    "inner_aa_iterations"
                )
                .iloc[0]
            )

            target_rows.append(
                {
                    "scenario":
                        scenario_name,

                    "target_label":
                        (
                            f"Pgood_{target:.3f}"
                        ),

                    "target_sector_probability":
                        target,

                    **earliest.to_dict(),
                }
            )

    targets_df = pd.DataFrame(
        target_rows
    )

    targets_df.to_csv(
        OUTPUT_TARGETS_CSV,
        index=False,
    )

    # =================================================================
    # Step 6
    # EXP11 placement
    # =================================================================

    placement_rows = []

    exp10_available = (
        EXP10_BASELINE_CSV.exists()
    )

    exp11_available = (
        EXP11_BREAK_EVEN_CSV.exists()
    )

    if exp10_available:

        exp10_df = pd.read_csv(
            EXP10_BASELINE_CSV
        )

        m3_rows = exp10_df[
            exp10_df[
                "method"
            ]
            == "M3_llr"
        ]

        if len(
            m3_rows
        ) == 1:

            m3_oracle_te = float(
                m3_rows.iloc[0][
                    "phase_oracle_te_proxy"
                ]
            )

        else:

            m3_oracle_te = np.nan

    else:

        m3_oracle_te = np.nan

    if exp11_available:

        break_even_df = pd.read_csv(
            EXP11_BREAK_EVEN_CSV
        )

    else:

        break_even_df = None

    for scenario_name in (
        QROM_SCENARIOS.keys()
    ):

        label = (
            f"Pgood_{DEFAULT_STATE_PREP_TARGET:.3f}"
        )

        rows = targets_df[
            (
                targets_df[
                    "scenario"
                ]
                == scenario_name
            )
            &
            (
                targets_df[
                    "target_label"
                ]
                == label
            )
        ]

        if len(
            rows
        ) != 1:

            continue

        row = rows.iloc[0]

        actual_prep_te = float(
            row[
                "state_prep_te"
            ]
        )

        if np.isfinite(
            m3_oracle_te
        ):

            prep_over_m3_oracle = (
                actual_prep_te
                / m3_oracle_te
            )

        else:

            prep_over_m3_oracle = (
                np.nan
            )

        if break_even_df is not None:

            for _, break_row in (
                break_even_df.iterrows()
            ):

                break_even_te = float(
                    break_row[
                        "break_even_state_prep_te"
                    ]
                )

                if np.isfinite(
                    break_even_te
                ):

                    above_break_even = (
                        actual_prep_te
                        >= break_even_te
                    )

                    ratio_to_break_even = (
                        actual_prep_te
                        / break_even_te
                    )

                else:

                    above_break_even = False
                    ratio_to_break_even = np.nan

                placement_rows.append(
                    {
                        "scenario":
                            scenario_name,

                        "state_prep_target":
                            DEFAULT_STATE_PREP_TARGET,

                        "inner_aa_iterations":
                            int(
                                row[
                                    "inner_aa_iterations"
                                ]
                            ),

                        "achieved_zero_sector_probability":
                            float(
                                row[
                                    "zero_sector_probability"
                                ]
                            ),

                        "state_prep_te":
                            actual_prep_te,

                        "M3_oracle_te":
                            m3_oracle_te,

                        "state_prep_over_M3_oracle":
                            prep_over_m3_oracle,

                        "outer_method":
                            str(
                                break_row[
                                    "method"
                                ]
                            ),

                        "EXP11_break_even_te":
                            break_even_te,

                        "state_prep_over_break_even":
                            ratio_to_break_even,

                        "above_EXP11_break_even":
                            bool(
                                above_break_even
                            ),
                    }
                )

        else:

            placement_rows.append(
                {
                    "scenario":
                        scenario_name,

                    "state_prep_target":
                        DEFAULT_STATE_PREP_TARGET,

                    "inner_aa_iterations":
                        int(
                            row[
                                "inner_aa_iterations"
                            ]
                        ),

                    "achieved_zero_sector_probability":
                        float(
                            row[
                                "zero_sector_probability"
                            ]
                        ),

                    "state_prep_te":
                        actual_prep_te,

                    "M3_oracle_te":
                        m3_oracle_te,

                    "state_prep_over_M3_oracle":
                        prep_over_m3_oracle,

                    "outer_method":
                        "",

                    "EXP11_break_even_te":
                        np.nan,

                    "state_prep_over_break_even":
                        np.nan,

                    "above_EXP11_break_even":
                        False,
                }
            )

    placement_df = pd.DataFrame(
        placement_rows
    )

    placement_df.to_csv(
        OUTPUT_PLACEMENT_CSV,
        index=False,
    )

    # =================================================================
    # Step 7
    # Figure: zero-Walsh sector probability
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    baseline_part = aa_df[
        aa_df[
            "scenario"
        ]
        == "baseline"
    ]

    ax.plot(
        baseline_part[
            "inner_aa_iterations"
        ],
        baseline_part[
            "zero_sector_probability"
        ],
        marker="o",
        markersize=4,
    )

    for target in SECTOR_TARGETS:

        ax.axhline(
            target,
            linestyle="--",
            linewidth=0.9,
            label=(
                f"{100 * target:.1f}% target"
            ),
        )

    ax.set_xlabel(
        "Inner sector-AA iterations"
    )

    ax.set_ylabel(
        "P(data register = 0^13)"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "EXP12: Zero-Walsh Sector Amplification"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_SECTOR_SUCCESS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 8
    # Figure: state preparation TE vs t
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for scenario_name in (
        QROM_SCENARIOS.keys()
    ):

        part = aa_df[
            aa_df[
                "scenario"
            ]
            == scenario_name
        ]

        ax.plot(
            part[
                "inner_aa_iterations"
            ],
            part[
                "state_prep_te"
            ],
            label=scenario_name,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Inner sector-AA iterations"
    )

    ax.set_ylabel(
        "State-preparation TE proxy"
    )

    ax.set_title(
        "EXP12: Correlation-State Preparation Cost"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_STATE_PREP_COST,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 9
    # Figure: expected cost per successful sector
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for scenario_name in (
        QROM_SCENARIOS.keys()
    ):

        part = aa_df[
            aa_df[
                "scenario"
            ]
            == scenario_name
        ]

        ax.plot(
            part[
                "inner_aa_iterations"
            ],
            part[
                "expected_te_per_good_sector_measurement"
            ],
            label=scenario_name,
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Inner sector-AA iterations"
    )

    ax.set_ylabel(
        "Expected TE proxy per successful zero-sector output"
    )

    ax.set_title(
        "EXP12: Cost per Usable Correlation-Sector State"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_COST_PER_GOOD,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 10
    # Figure: EXP11 break-even placement
    # =================================================================

    if (
        len(
            placement_df
        ) > 0
        and
        "EXP11_break_even_te"
        in placement_df.columns
        and
        np.any(
            np.isfinite(
                placement_df[
                    "EXP11_break_even_te"
                ]
            )
        )
    ):

        methods = list(
            placement_df[
                "outer_method"
            ].drop_duplicates()
        )

        scenarios = list(
            QROM_SCENARIOS.keys()
        )

        x = np.arange(
            len(
                scenarios
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        prep_values = []

        for scenario_name in scenarios:

            part = placement_df[
                placement_df[
                    "scenario"
                ]
                == scenario_name
            ]

            prep_values.append(
                float(
                    part.iloc[0][
                        "state_prep_te"
                    ]
                )
            )

        ax.plot(
            x,
            prep_values,
            marker="o",
            linewidth=2,
            label=(
                f"EXP12 A "
                f"(Pgood >= "
                f"{100 * DEFAULT_STATE_PREP_TARGET:.1f}%)"
            ),
        )

        for method in methods:

            part = placement_df[
                placement_df[
                    "outer_method"
                ]
                == method
            ]

            if len(
                part
            ) == 0:

                continue

            break_even = float(
                part.iloc[0][
                    "EXP11_break_even_te"
                ]
            )

            ax.axhline(
                break_even,
                linestyle="--",
                label=(
                    f"EXP11 break-even: "
                    f"{method}"
                ),
            )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            scenarios
        )

        ax.set_yscale(
            "log"
        )

        ax.set_xlabel(
            "QROM cost assumption"
        )

        ax.set_ylabel(
            "TE proxy"
        )

        ax.set_title(
            "EXP12: Actual State-Preparation Cost "
            "versus EXP11 Break-Even"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIG_BREAK_EVEN,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # =================================================================
    # Step 11
    # Console operating-point report
    # =================================================================

    print()
    print(
        "[Important inner-AA operating points]"
    )

    display_cols = [
        "scenario",
        "target_label",
        "inner_aa_iterations",
        "zero_sector_probability",
        "state_prep_te",
        "expected_te_per_good_sector_measurement",
    ]

    print(
        targets_df[
            display_cols
        ].to_string(
            index=False
        )
    )

    if len(
        placement_df
    ) > 0:

        print()
        print(
            "[Placement relative to EXP11]"
        )

        placement_cols = [
            "scenario",
            "outer_method",
            "inner_aa_iterations",
            "achieved_zero_sector_probability",
            "state_prep_te",
            "state_prep_over_M3_oracle",
            "EXP11_break_even_te",
            "state_prep_over_break_even",
            "above_EXP11_break_even",
        ]

        print(
            placement_df[
                placement_cols
            ].to_string(
                index=False
            )
        )

    # =================================================================
    # Step 12
    # Summary JSON
    # =================================================================

    baseline_raw = raw_lookup[
        "baseline"
    ]

    baseline_targets = targets_df[
        targets_df[
            "scenario"
        ]
        == "baseline"
    ]

    baseline_special = {}

    for _, row in (
        baseline_targets.iterrows()
    ):

        baseline_special[
            str(
                row[
                    "target_label"
                ]
            )
        ] = {
            "inner_aa_iterations":
                int(
                    row[
                        "inner_aa_iterations"
                    ]
                ),

            "zero_sector_probability":
                float(
                    row[
                        "zero_sector_probability"
                    ]
                ),

            "state_prep_te":
                float(
                    row[
                        "state_prep_te"
                    ]
                ),

            "expected_te_per_good_sector_measurement":
                float(
                    row[
                        "expected_te_per_good_sector_measurement"
                    ]
                ),
        }

    summary = {
        "experiment":
            "exp12_correlation_state_preparation_cost",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "reconstructed_exp05": {
            "initial_zero_walsh_sector_probability":
                p_good_initial,

            "true_probability_given_zero_sector":
                p_true_given_good,

            "true_candidate_correlation":
                true_correlation,
        },

        "coherent_data_access": {
            "entries":
                N_DATA,

            "index_qubits":
                INDEX_QUBITS,

            "output_bits":
                QROM_OUTPUT_BITS,

            "stored_information":
                (
                    "P5, P7, P8, ciphertext nibble 2, "
                    "ciphertext nibble 4"
                ),

            "total_one_bits":
                qrom_ones,

            "average_one_bits_per_entry":
                average_qrom_ones,
        },

        "baseline_raw_transform": {
            "TE_proxy":
                int(
                    baseline_raw[
                        "raw_transform_te"
                    ]
                ),

            "CNOT_proxy":
                int(
                    baseline_raw[
                        "raw_transform_cnot_proxy"
                    ]
                ),

            "Hadamards":
                int(
                    baseline_raw[
                        "hadamard_count"
                    ]
                ),

            "logical_qubits_proxy":
                int(
                    baseline_raw[
                        "logical_qubits_proxy"
                    ]
                ),
        },

        "baseline_sector_preparation_points":
            baseline_special,

        "default_correlation_state_target":
            DEFAULT_STATE_PREP_TARGET,

        "exp11_placement":
            (
                placement_df.to_dict(
                    orient="records"
                )
                if len(
                    placement_df
                ) > 0
                else []
            ),

        "important_interpretation": [
            (
                "The raw coherent transform is not the same as "
                "the normalized EXP05 correlation state. The desired "
                "correlation amplitudes reside in the data-register "
                "zero-Walsh sector."
            ),
            (
                "A coherent outer-QAA preparation unitary therefore "
                "requires either sector amplitude amplification or "
                "a joint outer oracle that explicitly includes the "
                "data-register zero condition."
            ),
            (
                "EXP12 models the nested approach: first amplify the "
                "zero-Walsh sector, then use the resulting unitary as "
                "the state-preparation family A."
            ),
            (
                "The QROM cost is scenario-dependent because no "
                "gate-level hardware-specific QROM synthesis has been "
                "fixed."
            ),
            (
                "QROM stores only the eleven classical bits actually "
                "needed by the canonical linear relation rather than "
                "full plaintext and ciphertext blocks."
            ),
        ],

        "limitations": [
            (
                "TE values are logical Toffoli-equivalent resource "
                "proxies, not physical fault-tolerant runtimes."
            ),
            (
                "QROM, inverse S-box, and multi-controlled-reflection "
                "costs are analytical resource models rather than "
                "transpiled hardware circuits."
            ),
            (
                "The near-deterministic sector preparation still "
                "contains residual non-zero-Walsh amplitude whenever "
                "Pgood < 1."
            ),
            (
                "EXP13 should model the outer good predicate as "
                "(data register = 0) AND verifier(key), avoiding any "
                "assumption that residual non-zero-Walsh amplitude "
                "has the desired key distribution."
            ),
        ],
    }

    with open(
        OUTPUT_SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    # =================================================================
    # Final console report
    # =================================================================

    print()
    print(
        "[Baseline raw transform]"
    )

    print(
        f"Raw U_corr TE proxy           : "
        f"{baseline_raw['raw_transform_te']:.0f}"
    )

    print(
        f"Raw U_corr CNOT proxy         : "
        f"{baseline_raw['raw_transform_cnot_proxy']:.0f}"
    )

    print(
        f"Logical-qubit proxy           : "
        f"{baseline_raw['logical_qubits_proxy']}"
    )

    print()
    print(
        "[Interpretation]"
    )

    print(
        "Raw U_corr alone does NOT prepare the normalized "
        "correlation-key state."
    )

    print(
        "It prepares the full key x Walsh-index state, with the "
        "desired correlation distribution living in data=0^13."
    )

    print(
        "Inner sector amplitude amplification is therefore included "
        "when estimating the coherent state-preparation family A."
    )

    print()
    print(
        "[Output files]"
    )

    print(
        f"Raw transform costs       : "
        f"{OUTPUT_RAW_COST_CSV}"
    )

    print(
        f"Inner-AA sweep            : "
        f"{OUTPUT_INNER_AA_CSV}"
    )

    print(
        f"Important targets         : "
        f"{OUTPUT_TARGETS_CSV}"
    )

    print(
        f"EXP11 placement           : "
        f"{OUTPUT_PLACEMENT_CSV}"
    )

    print(
        f"Summary                   : "
        f"{OUTPUT_SUMMARY_JSON}"
    )

    print(
        f"Sector-success figure     : "
        f"{FIG_SECTOR_SUCCESS}"
    )

    print(
        f"State-prep-cost figure    : "
        f"{FIG_STATE_PREP_COST}"
    )

    print(
        f"Cost-per-good figure      : "
        f"{FIG_COST_PER_GOOD}"
    )

    if FIG_BREAK_EVEN.exists():

        print(
            f"Break-even figure         : "
            f"{FIG_BREAK_EVEN}"
        )

    print()
    print(
        "[PASS] EXP12 completed successfully."
    )


if __name__ == "__main__":
    main()