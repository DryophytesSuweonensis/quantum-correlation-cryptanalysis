from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from itertools import product
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm, chi2, binom


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

sys.path.insert(0, str(SRC_DIR))

from heys_spn import encrypt_block


EXP05_CSV = RESULTS_DIR / "exp05_coherent_probabilities.csv"

OUTPUT_BANK_CSV = RESULTS_DIR / "exp08_approximation_bank.csv"
OUTPUT_BENCHMARK_CSV = RESULTS_DIR / "exp08_marking_benchmark.csv"
OUTPUT_BEST_CSV = RESULTS_DIR / "exp08_best_robust_points.csv"
OUTPUT_JSON = RESULTS_DIR / "exp08_summary.json"

FIG_EFFECTIVE = FIGURES_DIR / "exp08_effective_probability.png"
FIG_FP = FIGURES_DIR / "exp08_false_positives.png"
FIG_TRADEOFF = FIGURES_DIR / "exp08_retention_selectivity.png"


# =====================================================================
# Experiment configuration
# =====================================================================

SEED_DATA = 20261022
SEED_MONTE_CARLO = 20261024
SEED_NULL = 20261025

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

# Number of linear approximations used jointly
N_APPROXIMATIONS = 8

# Structural linear-trail search
BEAM_WIDTH = 64
TOP_LOCAL_TRANSITIONS = 4

# Robustness benchmark
VALIDATION_SIZES = [
    32,
    64,
    128,
    256,
    512,
    1024,
]

# 200 is intentionally lower than EXP07C because each trial now
# evaluates several linear approximations simultaneously.
N_TRIALS = 200

# All methods are compared at approximately the same nominal
# wrong-key false-positive rate.
NOMINAL_FPRS = [
    0.50,
    0.25,
    0.10,
    0.05,
    0.02,
    0.01,
]

# Consensus method:
# one component is considered individually significant if |z| >= 2.
CONSENSUS_COMPONENT_Z = 2.0

# Monte-Carlo calibration size for the LLR null distribution
N_NULL_CALIBRATION = 100_000

# A robust operating point must retain the true key in at least
# this fraction of random held-out subsets.
ROBUST_RETENTION = 0.90


# =====================================================================
# Heys S-box / P-box
# =====================================================================

SBOX = np.asarray(
    [
        0xE, 0x4, 0xD, 0x1,
        0x2, 0xF, 0xB, 0x8,
        0x3, 0xA, 0x6, 0xC,
        0x5, 0x9, 0x0, 0x7,
    ],
    dtype=np.uint8,
)

INV_SBOX = np.empty(
    16,
    dtype=np.uint8,
)

for x, y in enumerate(SBOX):
    INV_SBOX[y] = x


# Heys bit permutation:
#
# position 1 = MSB
# position 16 = LSB
#
PBOX = (
    1, 5, 9, 13,
    2, 6, 10, 14,
    3, 7, 11, 15,
    4, 8, 12, 16,
)


# =====================================================================
# Parity lookup tables
# =====================================================================

PARITY4 = np.asarray(
    [
        int(x).bit_count() & 1
        for x in range(16)
    ],
    dtype=np.uint8,
)

PARITY16 = np.asarray(
    [
        int(x).bit_count() & 1
        for x in range(2**16)
    ],
    dtype=np.uint8,
)


# =====================================================================
# Basic utilities
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


def mask_nibbles(
    mask: int,
) -> tuple[int, int, int, int]:

    return (
        (mask >> 12) & 0xF,
        (mask >> 8) & 0xF,
        (mask >> 4) & 0xF,
        mask & 0xF,
    )


def build_mask(
    nibbles,
) -> int:

    a, b, c, d = nibbles

    return (
        (a << 12)
        | (b << 8)
        | (c << 4)
        | d
    )


# =====================================================================
# P-box mask propagation
# =====================================================================

def permute_mask_forward(
    mask: int,
) -> int:
    """
    Propagate a linear mask forward through the Heys P-box.

    If bit at input position i is moved to PBOX[i],
    the corresponding linear mask bit follows the same mapping.
    """

    out = 0

    for input_position in range(
        1,
        17,
    ):

        output_position = (
            PBOX[
                input_position - 1
            ]
        )

        input_shift = (
            16 - input_position
        )

        output_shift = (
            16 - output_position
        )

        bit = (
            mask >> input_shift
        ) & 1

        out |= (
            bit << output_shift
        )

    return out


def permute_mask_inverse(
    mask: int,
) -> int:

    out = 0

    for input_position in range(
        1,
        17,
    ):

        output_position = (
            PBOX[
                input_position - 1
            ]
        )

        input_shift = (
            16 - input_position
        )

        output_shift = (
            16 - output_position
        )

        bit = (
            mask >> output_shift
        ) & 1

        out |= (
            bit << input_shift
        )

    return out


# =====================================================================
# S-box Linear Approximation Table
# =====================================================================

def build_sbox_lat() -> np.ndarray:
    """
    Normalized S-box correlation table:

        C[a,b]
        =
        (1/16) sum_x
        (-1)^(a·x xor b·S(x))

    Values lie in [-1, 1].
    """

    lat = np.zeros(
        (16, 16),
        dtype=np.float64,
    )

    for a in range(16):

        for b in range(16):

            total = 0

            for x in range(16):

                exponent = (
                    PARITY4[
                        a & x
                    ]
                    ^
                    PARITY4[
                        b & int(
                            SBOX[x]
                        )
                    ]
                )

                total += (
                    1
                    if exponent == 0
                    else -1
                )

            lat[a, b] = (
                total / 16.0
            )

    return lat


SBOX_LAT = build_sbox_lat()


# =====================================================================
# Reverse linear-trail search
# =====================================================================

@lru_cache(maxsize=None)
def local_reverse_options(
    output_mask: int,
):
    """
    For an S-box output mask b, return the strongest input masks a
    according to |LAT[a,b]|.
    """

    options = []

    for input_mask in range(16):

        corr = float(
            SBOX_LAT[
                input_mask,
                output_mask,
            ]
        )

        if abs(corr) > 0:

            options.append(
                (
                    input_mask,
                    corr,
                )
            )

    options.sort(
        key=lambda item:
            abs(item[1]),
        reverse=True,
    )

    return tuple(
        options[
            :TOP_LOCAL_TRANSITIONS
        ]
    )


@lru_cache(maxsize=None)
def reverse_sbox_layer(
    output_mask_16: int,
):
    """
    Given a 16-bit S-layer output mask, return strong possible
    S-layer input masks and their full-layer correlations.
    """

    output_nibbles = (
        mask_nibbles(
            output_mask_16
        )
    )

    option_lists = [
        local_reverse_options(
            b
        )
        for b in output_nibbles
    ]

    layer_options = []

    for combination in product(
        *option_lists
    ):

        input_nibbles = [
            item[0]
            for item in combination
        ]

        correlation = 1.0

        for _, corr in combination:
            correlation *= corr

        input_mask_16 = (
            build_mask(
                input_nibbles
            )
        )

        layer_options.append(
            (
                input_mask_16,
                correlation,
            )
        )

    layer_options.sort(
        key=lambda item:
            abs(item[1]),
        reverse=True,
    )

    return tuple(
        layer_options
    )


def best_three_round_trail(
    final_u4_mask: int,
) -> tuple[int, float]:
    """
    Search backward through rounds 1-3.

    final_u4_mask:
        linear mask on U4, the state just before the fourth S-box layer.

    Returns:
        strongest plaintext mask and predicted single-trail correlation.

    Key XORs only affect correlation sign, not its magnitude.
    """

    beam = [
        (
            1.0,
            final_u4_mask,
        )
    ]

    # Reverse rounds 3 -> 2 -> 1
    for _ in range(3):

        expanded = []

        for accumulated_corr, next_mask in beam:

            # next round input mask = PBOX(S-box output mask)
            sbox_output_mask = (
                permute_mask_inverse(
                    next_mask
                )
            )

            for (
                current_input_mask,
                layer_corr,
            ) in reverse_sbox_layer(
                sbox_output_mask
            ):

                new_corr = (
                    accumulated_corr
                    * layer_corr
                )

                expanded.append(
                    (
                        new_corr,
                        current_input_mask,
                    )
                )

        expanded.sort(
            key=lambda item:
                abs(item[0]),
            reverse=True,
        )

        beam = expanded[
            :BEAM_WIDTH
        ]

    best_corr, plaintext_mask = (
        beam[0]
    )

    return (
        plaintext_mask,
        best_corr,
    )


# =====================================================================
# Build approximation bank
# =====================================================================

def build_approximation_bank():
    """
    Approximation 0 is the canonical Heys relation already used
    in EXP01-07:

        P5 xor P7 xor P8
        xor U4,6 xor U4,8 xor U4,14 xor U4,16

    With MSB-first bit numbering:

        plaintext mask = 0x0B00
        U4 mask        = 0x0505

    Its theoretical bias magnitude is 1/32, hence correlation
    magnitude = 1/16 = 0.0625.

    Additional approximations are discovered structurally using
    the S-box LAT and P-box only.
    """

    bank = [
        {
            "approximation_id":
                "A0_canonical",

            "plaintext_mask":
                0x0B00,

            "u4_mask":
                0x0505,

            "mask_nibble2":
                0x5,

            "mask_nibble4":
                0x5,

            "predicted_abs_correlation":
                1.0 / 16.0,

            "source":
                "canonical",
        }
    ]

    candidates = []

    # Require both attacked nibbles to participate.
    #
    # This avoids relations that depend on only one 4-bit subkey
    # and therefore cannot discriminate the full 8-bit target.
    for mask2 in range(
        1,
        16,
    ):

        for mask4 in range(
            1,
            16,
        ):

            u4_mask = (
                (mask2 << 8)
                | mask4
            )

            plaintext_mask, trail_corr = (
                best_three_round_trail(
                    u4_mask
                )
            )

            candidates.append(
                {
                    "plaintext_mask":
                        plaintext_mask,

                    "u4_mask":
                        u4_mask,

                    "mask_nibble2":
                        mask2,

                    "mask_nibble4":
                        mask4,

                    "predicted_abs_correlation":
                        abs(
                            trail_corr
                        ),

                    "source":
                        "LAT_beam_search",
                }
            )

    candidates.sort(
        key=lambda row:
            row[
                "predicted_abs_correlation"
            ],
        reverse=True,
    )

    canonical_pair = (
        0x0B00,
        0x0505,
    )

    seen = {
        canonical_pair
    }

    for row in candidates:

        relation = (
            row[
                "plaintext_mask"
            ],
            row[
                "u4_mask"
            ],
        )

        if relation in seen:
            continue

        row = dict(row)

        row[
            "approximation_id"
        ] = (
            f"A{len(bank)}"
        )

        bank.append(
            row
        )

        seen.add(
            relation
        )

        if len(bank) >= N_APPROXIMATIONS:
            break

    if len(bank) < N_APPROXIMATIONS:

        raise RuntimeError(
            "Could not construct enough "
            "linear approximations."
        )

    return bank


# =====================================================================
# Precompute relation signs
# =====================================================================

def precompute_relation_signs(
    approximation_bank,
    plaintexts: np.ndarray,
    ciphertexts: np.ndarray,
) -> np.ndarray:
    """
    Output shape:

        [approximation, candidate, sample]

    Value:

        +1 if relation == 0
        -1 if relation == 1

    Then empirical correlation is simply the sample mean.
    """

    n_samples = len(
        plaintexts
    )

    n_approx = len(
        approximation_bank
    )

    signs = np.empty(
        (
            n_approx,
            256,
            n_samples,
        ),
        dtype=np.int8,
    )

    c2 = (
        (
            ciphertexts
            >> 8
        )
        & 0xF
    ).astype(
        np.uint8
    )

    c4 = (
        ciphertexts
        & 0xF
    ).astype(
        np.uint8
    )

    # Partial inverse S-box values for all 16 nibble guesses
    u2_all = np.empty(
        (16, n_samples),
        dtype=np.uint8,
    )

    u4_all = np.empty(
        (16, n_samples),
        dtype=np.uint8,
    )

    for k in range(16):

        u2_all[k] = (
            INV_SBOX[
                c2 ^ k
            ]
        )

        u4_all[k] = (
            INV_SBOX[
                c4 ^ k
            ]
        )

    for approx_index, approx in enumerate(
        approximation_bank
    ):

        plaintext_mask = int(
            approx[
                "plaintext_mask"
            ]
        )

        mask2 = int(
            approx[
                "mask_nibble2"
            ]
        )

        mask4 = int(
            approx[
                "mask_nibble4"
            ]
        )

        plaintext_parity = (
            PARITY16[
                (
                    plaintexts
                    & plaintext_mask
                ).astype(
                    np.uint16
                )
            ]
        )

        p2 = (
            PARITY4[
                u2_all
                & mask2
            ]
        )

        p4 = (
            PARITY4[
                u4_all
                & mask4
            ]
        )

        # Shape:
        #
        # [k2, k4, sample]
        candidate_parity = (
            p2[
                :,
                None,
                :
            ]
            ^
            p4[
                None,
                :,
                :
            ]
        )

        candidate_parity = (
            candidate_parity
            .reshape(
                256,
                n_samples,
            )
        )

        relation = (
            candidate_parity
            ^
            plaintext_parity[
                None,
                :
            ]
        )

        signs[
            approx_index
        ] = (
            1
            - 2
            * relation.astype(
                np.int8
            )
        )

    return signs


# =====================================================================
# Marking scores
# =====================================================================

def stable_log_cosh(
    x: np.ndarray,
) -> np.ndarray:

    return (
        np.logaddexp(
            x,
            -x,
        )
        - math.log(2.0)
    )


def compute_scores(
    correlations: np.ndarray,
    validation_size: int,
    predicted_abs_correlations: np.ndarray,
):
    """
    correlations shape:

        [L, 256]

    Under a wrong-key null model:

        correlation_hat * sqrt(M)
        approximately follows N(0,1).

    Four marking statistics are returned.
    """

    z = (
        correlations
        * math.sqrt(
            validation_size
        )
    )

    # -------------------------------------------------------------
    # M1: Single absolute bias/correlation
    #
    # Use canonical approximation A0 only.
    # -------------------------------------------------------------

    score_single = np.abs(
        z[0]
    )

    # -------------------------------------------------------------
    # M2: Multi-correlation energy
    #
    # Under independent null components:
    #
    #     sum z_l^2 ~ chi-square(L)
    # -------------------------------------------------------------

    score_energy = np.sum(
        z**2,
        axis=0,
    )

    # -------------------------------------------------------------
    # M3: Sign-agnostic approximate LLR
    #
    # H0:
    #     z ~ N(0,1)
    #
    # H1:
    #     z ~ 0.5 N(+mu,1) + 0.5 N(-mu,1)
    #
    # where
    #
    #     mu_l = sqrt(M) |C_l|
    #
    # Sign is marginalized because key XORs may flip trail signs.
    # -------------------------------------------------------------

    mu = (
        math.sqrt(
            validation_size
        )
        * predicted_abs_correlations
    )

    x = (
        mu[
            :,
            None
        ]
        * z
    )

    score_llr = np.sum(
        stable_log_cosh(
            x
        )
        - 0.5
        * (
            mu[
                :,
                None
            ] ** 2
        ),
        axis=0,
    )

    # -------------------------------------------------------------
    # M4: Consensus
    #
    # Count how many approximations independently exceed |z| >= 2.
    # -------------------------------------------------------------

    score_consensus = np.sum(
        np.abs(z)
        >= CONSENSUS_COMPONENT_Z,
        axis=0,
    )

    return {
        "M1_single":
            score_single,

        "M2_energy":
            score_energy,

        "M3_llr":
            score_llr,

        "M4_consensus":
            score_consensus,
    }


# =====================================================================
# Threshold calibration
# =====================================================================

def consensus_threshold(
    n_approximations: int,
    alpha: float,
) -> int:
    """
    Find the smallest number of component passes q such that

        P[Binomial(L,p0) >= q] <= alpha

    under a Gaussian null model.
    """

    p0 = (
        2.0
        * (
            1.0
            - norm.cdf(
                CONSENSUS_COMPONENT_Z
            )
        )
    )

    for q in range(
        0,
        n_approximations + 1,
    ):

        probability = (
            binom.sf(
                q - 1,
                n_approximations,
                p0,
            )
        )

        if probability <= alpha:
            return q

    return (
        n_approximations
        + 1
    )


def calibrate_thresholds(
    validation_size: int,
    predicted_abs_correlations: np.ndarray,
    rng_null: np.random.Generator,
):
    """
    Calibrate each marking statistic to nominal wrong-key FPR alpha.
    """

    l_count = len(
        predicted_abs_correlations
    )

    thresholds = {}

    # -------------------------------------------------------------
    # Single statistic
    # -------------------------------------------------------------

    thresholds[
        "M1_single"
    ] = {
        alpha:
            float(
                norm.ppf(
                    1.0
                    - alpha / 2.0
                )
            )
        for alpha in NOMINAL_FPRS
    }

    # -------------------------------------------------------------
    # Chi-square energy
    # -------------------------------------------------------------

    thresholds[
        "M2_energy"
    ] = {
        alpha:
            float(
                chi2.ppf(
                    1.0 - alpha,
                    df=l_count,
                )
            )
        for alpha in NOMINAL_FPRS
    }

    # -------------------------------------------------------------
    # LLR
    #
    # Numerical calibration under independent Gaussian H0.
    # -------------------------------------------------------------

    z_null = (
        rng_null.standard_normal(
            (
                N_NULL_CALIBRATION,
                l_count,
            )
        )
    )

    mu = (
        math.sqrt(
            validation_size
        )
        * predicted_abs_correlations
    )

    llr_null = np.sum(
        stable_log_cosh(
            z_null
            * mu[
                None,
                :
            ]
        )
        - 0.5
        * (
            mu[
                None,
                :
            ] ** 2
        ),
        axis=1,
    )

    thresholds[
        "M3_llr"
    ] = {
        alpha:
            float(
                np.quantile(
                    llr_null,
                    1.0 - alpha,
                )
            )
        for alpha in NOMINAL_FPRS
    }

    # -------------------------------------------------------------
    # Consensus
    # -------------------------------------------------------------

    thresholds[
        "M4_consensus"
    ] = {
        alpha:
            int(
                consensus_threshold(
                    l_count,
                    alpha,
                )
            )
        for alpha in NOMINAL_FPRS
    }

    return thresholds


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
    # Load EXP05 key distribution
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
        .sort_values(
            "candidate"
        )
        .reset_index(
            drop=True
        )
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
        ][
            "candidate"
        ]
    )

    p_true_initial = float(
        base_probabilities[
            true_index
        ]
    )

    # =================================================================
    # Step 2
    # Build structural approximation bank
    # =================================================================

    print()
    print(
        "Building linear-approximation bank..."
    )

    approximation_bank = (
        build_approximation_bank()
    )

    bank_df = pd.DataFrame(
        approximation_bank
    )

    bank_df[
        "plaintext_mask_hex"
    ] = [
        f"0x{x:04X}"
        for x in bank_df[
            "plaintext_mask"
        ]
    ]

    bank_df[
        "u4_mask_hex"
    ] = [
        f"0x{x:04X}"
        for x in bank_df[
            "u4_mask"
        ]
    ]

    bank_df.to_csv(
        OUTPUT_BANK_CSV,
        index=False,
    )

    print()
    print(
        "[Approximation bank]"
    )

    for row in approximation_bank:

        print(
            f"{row['approximation_id']:12s} | "
            f"P mask=0x{row['plaintext_mask']:04X} | "
            f"U4 mask=0x{row['u4_mask']:04X} | "
            f"|C|pred={row['predicted_abs_correlation']:.6f} | "
            f"{row['source']}"
        )

    predicted_abs_correlations = np.asarray(
        [
            row[
                "predicted_abs_correlation"
            ]
            for row in approximation_bank
        ],
        dtype=np.float64,
    )

    # =================================================================
    # Step 3
    # Rebuild data pool and take held-out samples
    # =================================================================

    plaintexts, ciphertexts = (
        generate_known_pairs(
            N_REFERENCE_PAIRS,
            SEED_DATA,
        )
    )

    heldout_p = (
        plaintexts[
            N_EXP05_DATA:
        ]
    )

    heldout_c = (
        ciphertexts[
            N_EXP05_DATA:
        ]
    )

    if len(
        heldout_p
    ) != N_HELDOUT:

        raise RuntimeError(
            "Unexpected held-out pool size."
        )

    # =================================================================
    # Step 4
    # Precompute relation signs
    # =================================================================

    print()
    print(
        "Precomputing all approximation/candidate/sample relations..."
    )

    relation_signs = (
        precompute_relation_signs(
            approximation_bank,
            heldout_p,
            heldout_c,
        )
    )

    # shape:
    #
    # L x 256 x 1808

    print(
        f"Relation tensor shape: "
        f"{relation_signs.shape}"
    )

    # =================================================================
    # Step 5
    # Threshold calibration
    # =================================================================

    rng_null = np.random.default_rng(
        SEED_NULL
    )

    thresholds_by_m = {}

    for m in VALIDATION_SIZES:

        thresholds_by_m[m] = (
            calibrate_thresholds(
                validation_size=m,
                predicted_abs_correlations=
                    predicted_abs_correlations,
                rng_null=rng_null,
            )
        )

    # =================================================================
    # Step 6
    # Monte-Carlo robustness benchmark
    # =================================================================

    rng = np.random.default_rng(
        SEED_MONTE_CARLO
    )

    methods = [
        "M1_single",
        "M2_energy",
        "M3_llr",
        "M4_consensus",
    ]

    accumulators = {}

    for m in VALIDATION_SIZES:

        for method in methods:

            for alpha in NOMINAL_FPRS:

                accumulators[
                    (
                        m,
                        method,
                        alpha,
                    )
                ] = {
                    "true_mark_count":
                        0,

                    "false_positive_sum":
                        0,

                    "marked_count_sum":
                        0,

                    "true_given_marked_sum":
                        0.0,

                    "effective_true_probability_sum":
                        0.0,

                    "marked_mass_sum":
                        0.0,
                }

    # -------------------------------------------------------------
    # One random permutation per trial.
    #
    # Prefixes of that permutation give random subsets for every M.
    # This is much faster than independently re-slicing the full
    # tensor for every validation size.
    # -------------------------------------------------------------

    for trial in range(
        N_TRIALS
    ):

        if (
            trial % 20 == 0
        ):

            print(
                f"Monte Carlo trial "
                f"{trial}/{N_TRIALS}"
            )

        permutation = (
            rng.permutation(
                N_HELDOUT
            )
        )

        reordered = (
            relation_signs[
                :,
                :,
                permutation,
            ]
        )

        cumulative = np.cumsum(
            reordered,
            axis=2,
            dtype=np.int32,
        )

        for m in VALIDATION_SIZES:

            correlation_matrix = (
                cumulative[
                    :,
                    :,
                    m - 1,
                ]
                / float(m)
            )

            scores = (
                compute_scores(
                    correlations=
                        correlation_matrix,

                    validation_size=m,

                    predicted_abs_correlations=
                        predicted_abs_correlations,
                )
            )

            thresholds = (
                thresholds_by_m[m]
            )

            for method in methods:

                score_vector = (
                    scores[
                        method
                    ]
                )

                for alpha in NOMINAL_FPRS:

                    threshold = (
                        thresholds[
                            method
                        ][
                            alpha
                        ]
                    )

                    marked = (
                        score_vector
                        >= threshold
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

                    accumulator = (
                        accumulators[
                            (
                                m,
                                method,
                                alpha,
                            )
                        ]
                    )

                    accumulator[
                        "true_mark_count"
                    ] += int(
                        true_marked
                    )

                    accumulator[
                        "false_positive_sum"
                    ] += (
                        false_positive_count
                    )

                    accumulator[
                        "marked_count_sum"
                    ] += (
                        n_marked
                    )

                    accumulator[
                        "marked_mass_sum"
                    ] += (
                        marked_mass
                    )

                    if true_marked:

                        accumulator[
                            "true_given_marked_sum"
                        ] += (
                            true_given_marked
                        )

                    # Includes verifier failures automatically:
                    #
                    # true not marked -> 0
                    accumulator[
                        "effective_true_probability_sum"
                    ] += (
                        true_given_marked
                    )

    # =================================================================
    # Step 7
    # Aggregate
    # =================================================================

    rows = []

    for (
        m,
        method,
        alpha,
    ), accumulator in accumulators.items():

        retention = (
            accumulator[
                "true_mark_count"
            ]
            / N_TRIALS
        )

        mean_fp = (
            accumulator[
                "false_positive_sum"
            ]
            / N_TRIALS
        )

        mean_marked = (
            accumulator[
                "marked_count_sum"
            ]
            / N_TRIALS
        )

        mean_marked_mass = (
            accumulator[
                "marked_mass_sum"
            ]
            / N_TRIALS
        )

        if (
            accumulator[
                "true_mark_count"
            ] > 0
        ):

            conditional_true = (
                accumulator[
                    "true_given_marked_sum"
                ]
                / accumulator[
                    "true_mark_count"
                ]
            )

        else:

            conditional_true = 0.0

        effective_true = (
            accumulator[
                "effective_true_probability_sum"
            ]
            / N_TRIALS
        )

        rows.append(
            {
                "validation_size":
                    m,

                "method":
                    method,

                "nominal_false_positive_rate":
                    alpha,

                "threshold":
                    thresholds_by_m[
                        m
                    ][
                        method
                    ][
                        alpha
                    ],

                "n_trials":
                    N_TRIALS,

                "true_key_retention_probability":
                    retention,

                "mean_false_positive_count":
                    mean_fp,

                "mean_marked_candidates":
                    mean_marked,

                "mean_marked_probability_mass":
                    mean_marked_mass,

                "mean_true_probability_given_marked":
                    conditional_true,

                "effective_true_probability":
                    effective_true,
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        OUTPUT_BENCHMARK_CSV,
        index=False,
    )

    # =================================================================
    # Step 8
    # Find best robust operating point for every method and M
    # =================================================================

    best_rows = []

    for m in VALIDATION_SIZES:

        for method in methods:

            subset = result_df[
                (
                    result_df[
                        "validation_size"
                    ]
                    == m
                )
                &
                (
                    result_df[
                        "method"
                    ]
                    == method
                )
                &
                (
                    result_df[
                        "true_key_retention_probability"
                    ]
                    >= ROBUST_RETENTION
                )
            ].copy()

            if len(
                subset
            ) == 0:

                continue

            best = (
                subset
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

            best_rows.append(
                best.to_dict()
            )

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        OUTPUT_BEST_CSV,
        index=False,
    )

    # =================================================================
    # Step 9
    # Overall best robust configuration per method
    # =================================================================

    overall_best = {}

    for method in methods:

        subset = best_df[
            best_df[
                "method"
            ]
            == method
        ]

        if len(
            subset
        ) == 0:

            continue

        best = (
            subset
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

        overall_best[
            method
        ] = {
            "validation_size":
                int(
                    best[
                        "validation_size"
                    ]
                ),

            "nominal_false_positive_rate":
                float(
                    best[
                        "nominal_false_positive_rate"
                    ]
                ),

            "true_key_retention_probability":
                float(
                    best[
                        "true_key_retention_probability"
                    ]
                ),

            "mean_false_positive_count":
                float(
                    best[
                        "mean_false_positive_count"
                    ]
                ),

            "mean_true_probability_given_marked":
                float(
                    best[
                        "mean_true_probability_given_marked"
                    ]
                ),

            "effective_true_probability":
                float(
                    best[
                        "effective_true_probability"
                    ]
                ),
        }

    # =================================================================
    # Step 10
    # Summary JSON
    # =================================================================

    summary = {
        "experiment":
            "exp08_marking_strategy_benchmark",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "initial_true_key_probability":
            p_true_initial,

        "n_linear_approximations":
            N_APPROXIMATIONS,

        "validation_sizes":
            VALIDATION_SIZES,

        "nominal_false_positive_rates":
            NOMINAL_FPRS,

        "trials_per_configuration":
            N_TRIALS,

        "methods": {
            "M1_single":
                (
                    "Canonical single linear "
                    "correlation magnitude."
                ),

            "M2_energy":
                (
                    "Sum of squared standardized "
                    "correlations; chi-square-style."
                ),

            "M3_llr":
                (
                    "Sign-marginalized approximate "
                    "log-likelihood ratio."
                ),

            "M4_consensus":
                (
                    "Number of approximations with "
                    f"|z| >= {CONSENSUS_COMPONENT_Z}."
                ),
        },

        "overall_best_robust_configuration":
            overall_best,

        "important_limitations": [
            (
                "Additional approximations are selected using "
                "single-trail LAT beam search. Linear-hull effects "
                "and key-dependent trail interference are not "
                "fully modeled."
            ),
            (
                "Chi-square and LLR threshold calibration use "
                "approximately independent Gaussian null components. "
                "The actual linear tests may be statistically correlated."
            ),
            (
                "All verifier scores are still evaluated classically. "
                "Reversible quantum-oracle implementation cost is "
                "not included in EXP08."
            ),
        ],
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
    # Effective true-key probability at robust operating point
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for method in methods:

        part = best_df[
            best_df[
                "method"
            ]
            == method
        ]

        if len(
            part
        ) == 0:
            continue

        ax.plot(
            part[
                "validation_size"
            ],
            part[
                "effective_true_probability"
            ],
            marker="o",
            label=method,
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

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xlabel(
        "Validation sample count M"
    )

    ax.set_ylabel(
        "Best effective true-key probability"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "EXP08: Robust Marking Performance"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_EFFECTIVE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # False positives at robust operating point
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for method in methods:

        part = best_df[
            best_df[
                "method"
            ]
            == method
        ]

        if len(
            part
        ) == 0:
            continue

        ax.plot(
            part[
                "validation_size"
            ],
            part[
                "mean_false_positive_count"
            ],
            marker="o",
            label=method,
        )

    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xlabel(
        "Validation sample count M"
    )

    ax.set_ylabel(
        "Mean false-positive keys"
    )

    ax.set_title(
        "EXP08: False Positives at >=90% True-Key Retention"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_FP,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 3
    # Retention vs false positives, using M=1024
    # =================================================================

    selected_m = max(
        VALIDATION_SIZES
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for method in methods:

        part = result_df[
            (
                result_df[
                    "validation_size"
                ]
                == selected_m
            )
            &
            (
                result_df[
                    "method"
                ]
                == method
            )
        ]

        ax.plot(
            part[
                "mean_false_positive_count"
            ],
            part[
                "true_key_retention_probability"
            ],
            marker="o",
            label=method,
        )

    ax.axhline(
        ROBUST_RETENTION,
        linestyle="--",
        linewidth=1.0,
        label="90% retention",
    )

    ax.set_xlabel(
        "Mean false-positive keys"
    )

    ax.set_ylabel(
        "True-key retention probability"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        f"EXP08: Marking Tradeoff at M={selected_m}"
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
    # Console summary
    # =================================================================

    print()
    print("=" * 78)
    print(
        "EXP08: Marking Strategy Benchmark"
    )
    print("=" * 78)

    print()
    print(
        f"True key                     : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"Initial P(true)              : "
        f"{p_true_initial:.6f} "
        f"({100 * p_true_initial:.3f}%)"
    )

    print(
        f"Linear approximations        : "
        f"{N_APPROXIMATIONS}"
    )

    print(
        f"Random trials/configuration  : "
        f"{N_TRIALS}"
    )

    print()
    print(
        "[Best robust configuration per method]"
    )

    for method, values in overall_best.items():

        print()

        print(
            method
        )

        print(
            f"  M                          : "
            f"{values['validation_size']}"
        )

        print(
            f"  nominal FPR                : "
            f"{values['nominal_false_positive_rate']:.3f}"
        )

        print(
            f"  true-key retention         : "
            f"{values['true_key_retention_probability']:.4f}"
        )

        print(
            f"  mean false positives       : "
            f"{values['mean_false_positive_count']:.2f}"
        )

        print(
            f"  mean P(true | marked)      : "
            f"{values['mean_true_probability_given_marked']:.4f}"
        )

        print(
            f"  effective P(true)          : "
            f"{values['effective_true_probability']:.4f}"
        )

    print()
    print("[Output files]")

    print(
        f"Approximation bank           : "
        f"{OUTPUT_BANK_CSV}"
    )

    print(
        f"Full benchmark               : "
        f"{OUTPUT_BENCHMARK_CSV}"
    )

    print(
        f"Best robust points           : "
        f"{OUTPUT_BEST_CSV}"
    )

    print(
        f"Summary                      : "
        f"{OUTPUT_JSON}"
    )

    print(
        f"Effective-probability figure : "
        f"{FIG_EFFECTIVE}"
    )

    print(
        f"False-positive figure        : "
        f"{FIG_FP}"
    )

    print(
        f"Tradeoff figure              : "
        f"{FIG_TRADEOFF}"
    )

    print()
    print(
        "[PASS] EXP08 completed successfully."
    )


if __name__ == "__main__":
    main()