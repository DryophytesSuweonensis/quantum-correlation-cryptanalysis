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

# Reuse the exact EXP08 cipher / LAT / relation implementation.
import exp08_marking_strategy_benchmark as exp08


EXP05_CSV = (
    RESULTS_DIR
    / "exp05_coherent_probabilities.csv"
)

EXP08_BANK_CSV = (
    RESULTS_DIR
    / "exp08_approximation_bank.csv"
)

EXP08_BEST_CSV = (
    RESULTS_DIR
    / "exp08_best_robust_points.csv"
)


# =====================================================================
# EXP09 output files
# =====================================================================

OUTPUT_THRESHOLDS_CSV = (
    RESULTS_DIR
    / "exp09_empirical_thresholds.csv"
)

OUTPUT_COVARIANCE_CSV = (
    RESULTS_DIR
    / "exp09_null_covariance.csv"
)

OUTPUT_BENCHMARK_CSV = (
    RESULTS_DIR
    / "exp09_marking_benchmark.csv"
)

OUTPUT_BEST_CSV = (
    RESULTS_DIR
    / "exp09_best_robust_points.csv"
)

OUTPUT_COMPARISON_CSV = (
    RESULTS_DIR
    / "exp09_vs_exp08_m1024.csv"
)

OUTPUT_SUMMARY_JSON = (
    RESULTS_DIR
    / "exp09_summary.json"
)


FIG_CORRELATION = (
    FIGURES_DIR
    / "exp09_null_correlation_matrix.png"
)

FIG_EFFECTIVE = (
    FIGURES_DIR
    / "exp09_effective_probability.png"
)

FIG_FALSE_POSITIVE = (
    FIGURES_DIR
    / "exp09_false_positives.png"
)

FIG_TRADEOFF = (
    FIGURES_DIR
    / "exp09_retention_selectivity.png"
)


# =====================================================================
# Configuration
# =====================================================================

SEED_DATA = 20261022

SEED_CALIBRATION_DATA = 20261030
SEED_CALIBRATION_SUBSET = 20261031
SEED_EVALUATION = 20261101

N_REFERENCE_PAIRS = 10_000

# EXP05 used the first 8192 pairs.
N_EXP05_DATA = 8_192

N_HELDOUT = (
    N_REFERENCE_PAIRS
    - N_EXP05_DATA
)

# Fresh plaintexts that were not used anywhere in EXP01-08.
N_CALIBRATION_PAIRS = 8_192

# Number of random calibration subsets.
#
# First half:
#   estimate empirical null mean/covariance.
#
# Second half:
#   calibrate empirical score thresholds.
#
N_CALIBRATION_TRIALS = 240

# Independent evaluation on the original EXP08 held-out pool.
N_EVALUATION_TRIALS = 200


VALIDATION_SIZES = [
    32,
    64,
    128,
    256,
    512,
    1024,
]


NOMINAL_FPRS = [
    0.50,
    0.25,
    0.10,
    0.05,
    0.02,
    0.01,
]


ROBUST_RETENTION = 0.90


# Same component threshold used by EXP08 M4.
CONSENSUS_COMPONENT_Z = 2.0


# Numerical stabilization for covariance inversion.
COVARIANCE_RIDGE = 1e-6


METHODS = [
    "M1_single",
    "M2_energy",
    "M3_llr",
    "M4_consensus",
    "M5_mahalanobis",
]


# =====================================================================
# Utility functions
# =====================================================================

def stable_log_cosh(
    x: np.ndarray,
) -> np.ndarray:
    """
    Numerically stable:

        log(cosh(x))

    using:

        log(exp(x) + exp(-x)) - log(2)
    """

    return (
        np.logaddexp(
            x,
            -x,
        )
        - math.log(2.0)
    )


def load_approximation_bank():
    """
    Load the exact approximation bank produced by EXP08.

    This is important:
    EXP09 must not search for new approximations using evaluation data.
    """

    if not EXP08_BANK_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP08_BANK_CSV}\n\n"
            "Run EXP08 first."
        )

    df = pd.read_csv(
        EXP08_BANK_CSV
    )

    bank = []

    for _, row in df.iterrows():

        bank.append(
            {
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
        )

    return bank


def generate_fresh_calibration_pairs(
    reference_plaintexts: np.ndarray,
):
    """
    Generate plaintext/ciphertext pairs that are completely disjoint
    from the 10,000 pairs used by EXP01-08.

    This prevents threshold calibration from reusing evaluation data.
    """

    all_plaintexts = np.arange(
        2**16,
        dtype=np.uint16,
    )

    unused_plaintexts = np.setdiff1d(
        all_plaintexts,
        reference_plaintexts,
        assume_unique=False,
    )

    if (
        len(unused_plaintexts)
        < N_CALIBRATION_PAIRS
    ):

        raise RuntimeError(
            "Not enough unused plaintexts "
            "for calibration."
        )

    rng = np.random.default_rng(
        SEED_CALIBRATION_DATA
    )

    calibration_plaintexts = (
        rng.choice(
            unused_plaintexts,
            size=N_CALIBRATION_PAIRS,
            replace=False,
        )
        .astype(
            np.uint16
        )
    )

    calibration_ciphertexts = (
        np.asarray(
            [
                exp08.encrypt_block(
                    int(p),
                    exp08.ROUND_KEYS,
                )
                for p in calibration_plaintexts
            ],
            dtype=np.uint16,
        )
    )

    return (
        calibration_plaintexts,
        calibration_ciphertexts,
    )


# =====================================================================
# Score functions
# =====================================================================

def compute_scores_from_z(
    z: np.ndarray,
    validation_size: int,
    predicted_abs_correlations: np.ndarray,
    null_mean: np.ndarray,
    covariance_inverse: np.ndarray,
):
    """
    Parameters
    ----------
    z:
        Shape [N, L]

        Standardized empirical correlations:

            z_l = sqrt(M) * C_hat_l

    Returns
    -------
    dict:
        One score vector for each marking method.
    """

    # -------------------------------------------------------------
    # M1: single canonical approximation
    # -------------------------------------------------------------

    score_single = np.abs(
        z[:, 0]
    )

    # -------------------------------------------------------------
    # M2: naive independent-energy statistic
    #
    # EXP08 assumed these components approximately behave as
    # independent N(0,1) wrong-key statistics.
    # -------------------------------------------------------------

    score_energy = np.sum(
        z**2,
        axis=1,
    )

    # -------------------------------------------------------------
    # M3: sign-marginalized approximate LLR
    #
    # Same score as EXP08.
    #
    # Difference:
    # threshold will now be calibrated empirically instead of
    # assuming an ideal Gaussian null.
    # -------------------------------------------------------------

    mu = (
        math.sqrt(
            validation_size
        )
        * predicted_abs_correlations
    )

    x = (
        z
        * mu[
            None,
            :
        ]
    )

    score_llr = np.sum(
        stable_log_cosh(
            x
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

    # -------------------------------------------------------------
    # M4: simple consensus
    # -------------------------------------------------------------

    score_consensus = np.sum(
        np.abs(z)
        >= CONSENSUS_COMPONENT_Z,
        axis=1,
    ).astype(
        np.float64
    )

    # -------------------------------------------------------------
    # M5: covariance-aware Mahalanobis energy
    #
    #       (z - mu0)^T Sigma^{-1} (z - mu0)
    #
    # If the eight linear statistics are correlated,
    # M5 avoids counting redundant information multiple times.
    # -------------------------------------------------------------

    centered = (
        z
        - null_mean[
            None,
            :
        ]
    )

    score_mahalanobis = np.einsum(
        "bi,ij,bj->b",
        centered,
        covariance_inverse,
        centered,
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

        "M5_mahalanobis":
            score_mahalanobis,
    }


# =====================================================================
# Empirical threshold calibration
# =====================================================================

def empirical_threshold(
    scores: np.ndarray,
    alpha: float,
    discrete: bool = False,
):
    """
    Find an empirical upper-tail threshold for:

        mark if score >= threshold

    The threshold is chosen conservatively such that the observed
    calibration false-positive rate does not exceed alpha.

    Returns
    -------
    threshold
    achieved_fpr
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    q = float(
        np.quantile(
            scores,
            1.0 - alpha,
            method="higher",
        )
    )

    achieved = float(
        np.mean(
            scores >= q
        )
    )

    if achieved > alpha:

        if discrete:

            threshold = (
                math.floor(q)
                + 1.0
            )

        else:

            threshold = float(
                np.nextafter(
                    q,
                    np.inf,
                )
            )

        achieved = float(
            np.mean(
                scores
                >= threshold
            )
        )

    else:

        threshold = q

    return (
        threshold,
        achieved,
    )


# =====================================================================
# Covariance diagnostics
# =====================================================================

def estimate_null_model(
    null_z_fit: np.ndarray,
):
    """
    Estimate mean and covariance of wrong-key z-vectors.
    """

    null_mean = np.mean(
        null_z_fit,
        axis=0,
    )

    covariance = np.cov(
        null_z_fit,
        rowvar=False,
        ddof=1,
    )

    l_count = (
        covariance.shape[0]
    )

    average_variance = float(
        np.trace(
            covariance
        )
        / l_count
    )

    ridge = (
        COVARIANCE_RIDGE
        * max(
            average_variance,
            1e-12,
        )
    )

    covariance_regularized = (
        covariance
        + ridge
        * np.eye(
            l_count
        )
    )

    covariance_inverse = (
        np.linalg.pinv(
            covariance_regularized
        )
    )

    standard_deviation = np.sqrt(
        np.diag(
            covariance
        )
    )

    denominator = np.outer(
        standard_deviation,
        standard_deviation,
    )

    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(
            covariance
        ),
        where=(
            denominator > 0
        ),
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance_regularized
    )

    condition_number = float(
        np.linalg.cond(
            covariance_regularized
        )
    )

    # Participation-ratio effective rank.
    #
    # 8 means approximately 8 independent dimensions.
    # Smaller means substantial redundancy.
    effective_rank = float(
        (
            np.sum(
                eigenvalues
            )
            ** 2
        )
        /
        np.sum(
            eigenvalues**2
        )
    )

    off_diagonal = (
        correlation
        - np.eye(
            l_count
        )
    )

    max_abs_offdiag = float(
        np.max(
            np.abs(
                off_diagonal
            )
        )
    )

    return {
        "mean":
            null_mean,

        "covariance":
            covariance,

        "covariance_regularized":
            covariance_regularized,

        "inverse":
            covariance_inverse,

        "correlation":
            correlation,

        "eigenvalues":
            eigenvalues,

        "condition_number":
            condition_number,

        "effective_rank":
            effective_rank,

        "max_abs_offdiagonal_correlation":
            max_abs_offdiag,
    }


# =====================================================================
# Calibration sample generation
# =====================================================================

def collect_calibration_null_vectors(
    relation_signs: np.ndarray,
    true_index: int,
):
    """
    Produce independent groups for:

        1. covariance fitting
        2. empirical threshold fitting

    Both use fresh P/C data not used in EXP01-08.
    """

    rng = np.random.default_rng(
        SEED_CALIBRATION_SUBSET
    )

    max_m = max(
        VALIDATION_SIZES
    )

    wrong_mask = np.ones(
        256,
        dtype=bool,
    )

    wrong_mask[
        true_index
    ] = False

    n_fit_trials = (
        N_CALIBRATION_TRIALS
        // 2
    )

    fit_vectors = {
        m: []
        for m in VALIDATION_SIZES
    }

    threshold_vectors = {
        m: []
        for m in VALIDATION_SIZES
    }

    for trial in range(
        N_CALIBRATION_TRIALS
    ):

        if (
            trial % 20 == 0
        ):

            print(
                f"Calibration trial "
                f"{trial}/"
                f"{N_CALIBRATION_TRIALS}"
            )

        indices = rng.choice(
            relation_signs.shape[2],
            size=max_m,
            replace=False,
        )

        selected = (
            relation_signs[
                :,
                :,
                indices,
            ]
        )

        cumulative = np.cumsum(
            selected,
            axis=2,
            dtype=np.int32,
        )

        if trial < n_fit_trials:

            destination = (
                fit_vectors
            )

        else:

            destination = (
                threshold_vectors
            )

        for m in VALIDATION_SIZES:

            correlations = (
                cumulative[
                    :,
                    :,
                    m - 1,
                ]
                / float(m)
            )

            # [L, 256]
            #
            # -> [256, L]
            z = (
                correlations.T
                * math.sqrt(
                    m
                )
            )

            # Benchmark-only null:
            # remove known true key.
            #
            # This is used to diagnose the statistical model.
            # It is not itself a realizable attack oracle.
            wrong_z = (
                z[
                    wrong_mask
                ]
            )

            destination[
                m
            ].append(
                wrong_z
            )

    for m in VALIDATION_SIZES:

        fit_vectors[
            m
        ] = np.concatenate(
            fit_vectors[
                m
            ],
            axis=0,
        )

        threshold_vectors[
            m
        ] = np.concatenate(
            threshold_vectors[
                m
            ],
            axis=0,
        )

    return (
        fit_vectors,
        threshold_vectors,
    )


# =====================================================================
# Main experiment
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
    # Load EXP05 state probabilities
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
            dtype=np.float64,
            copy=True,
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

    if len(
        true_indices
    ) != 1:

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
    # Load exact EXP08 approximation bank
    # =================================================================

    approximation_bank = (
        load_approximation_bank()
    )

    l_count = len(
        approximation_bank
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

    approximation_labels = [
        row[
            "approximation_id"
        ]
        for row in approximation_bank
    ]

    print()
    print(
        f"Loaded {l_count} "
        f"EXP08 linear approximations."
    )

    # =================================================================
    # Step 3
    # Reconstruct original EXP01-08 data
    # =================================================================

    reference_p, reference_c = (
        exp08.generate_known_pairs(
            N_REFERENCE_PAIRS,
            SEED_DATA,
        )
    )

    heldout_p = (
        reference_p[
            N_EXP05_DATA:
        ]
    )

    heldout_c = (
        reference_c[
            N_EXP05_DATA:
        ]
    )

    if len(
        heldout_p
    ) != N_HELDOUT:

        raise RuntimeError(
            "Unexpected evaluation held-out size."
        )

    # =================================================================
    # Step 4
    # Generate fresh calibration data
    # =================================================================

    print()
    print(
        "Generating fresh calibration "
        "plaintext/ciphertext pairs..."
    )

    (
        calibration_p,
        calibration_c,
    ) = (
        generate_fresh_calibration_pairs(
            reference_p
        )
    )

    # Confirm no overlap.
    overlap = np.intersect1d(
        reference_p,
        calibration_p,
    )

    if len(
        overlap
    ) != 0:

        raise RuntimeError(
            "Calibration plaintext pool "
            "overlaps reference data."
        )

    print(
        f"Fresh calibration pairs: "
        f"{len(calibration_p)}"
    )

    # =================================================================
    # Step 5
    # Precompute relation signs
    # =================================================================

    print()
    print(
        "Precomputing calibration "
        "relation tensor..."
    )

    calibration_signs = (
        exp08.precompute_relation_signs(
            approximation_bank,
            calibration_p,
            calibration_c,
        )
    )

    print(
        f"Calibration tensor: "
        f"{calibration_signs.shape}"
    )

    print()
    print(
        "Precomputing evaluation "
        "relation tensor..."
    )

    evaluation_signs = (
        exp08.precompute_relation_signs(
            approximation_bank,
            heldout_p,
            heldout_c,
        )
    )

    print(
        f"Evaluation tensor: "
        f"{evaluation_signs.shape}"
    )

    # =================================================================
    # Step 6
    # Collect fresh wrong-key null z-vectors
    # =================================================================

    print()
    print(
        "Collecting empirical wrong-key "
        "null distributions..."
    )

    (
        null_fit_vectors,
        null_threshold_vectors,
    ) = (
        collect_calibration_null_vectors(
            calibration_signs,
            true_index,
        )
    )

    # =================================================================
    # Step 7
    # Fit covariance models
    # =================================================================

    null_models = {}

    covariance_rows = []

    for m in VALIDATION_SIZES:

        model = estimate_null_model(
            null_fit_vectors[
                m
            ]
        )

        null_models[
            m
        ] = model

        covariance = (
            model[
                "covariance"
            ]
        )

        correlation = (
            model[
                "correlation"
            ]
        )

        for i in range(
            l_count
        ):

            for j in range(
                l_count
            ):

                covariance_rows.append(
                    {
                        "validation_size":
                            m,

                        "row_approximation":
                            approximation_labels[
                                i
                            ],

                        "column_approximation":
                            approximation_labels[
                                j
                            ],

                        "covariance":
                            float(
                                covariance[
                                    i,
                                    j
                                ]
                            ),

                        "correlation":
                            float(
                                correlation[
                                    i,
                                    j
                                ]
                            ),
                    }
                )

    covariance_df = pd.DataFrame(
        covariance_rows
    )

    covariance_df.to_csv(
        OUTPUT_COVARIANCE_CSV,
        index=False,
    )

    # =================================================================
    # Step 8
    # Empirical threshold calibration
    # =================================================================

    threshold_rows = []

    thresholds = {}

    for m in VALIDATION_SIZES:

        model = (
            null_models[
                m
            ]
        )

        threshold_z = (
            null_threshold_vectors[
                m
            ]
        )

        null_scores = (
            compute_scores_from_z(
                z=threshold_z,

                validation_size=m,

                predicted_abs_correlations=
                    predicted_abs_correlations,

                null_mean=
                    model[
                        "mean"
                    ],

                covariance_inverse=
                    model[
                        "inverse"
                    ],
            )
        )

        thresholds[
            m
        ] = {}

        for method in METHODS:

            thresholds[
                m
            ][
                method
            ] = {}

            for alpha in NOMINAL_FPRS:

                (
                    threshold,
                    achieved_fpr,
                ) = empirical_threshold(
                    null_scores[
                        method
                    ],

                    alpha,

                    discrete=(
                        method
                        == "M4_consensus"
                    ),
                )

                thresholds[
                    m
                ][
                    method
                ][
                    alpha
                ] = threshold

                threshold_rows.append(
                    {
                        "validation_size":
                            m,

                        "method":
                            method,

                        "nominal_false_positive_rate":
                            alpha,

                        "empirical_threshold":
                            threshold,

                        "calibration_achieved_fpr":
                            achieved_fpr,
                    }
                )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df.to_csv(
        OUTPUT_THRESHOLDS_CSV,
        index=False,
    )

    # =================================================================
    # Step 9
    # Evaluation on original held-out data
    # =================================================================

    print()
    print(
        "Running independent evaluation..."
    )

    rng_eval = np.random.default_rng(
        SEED_EVALUATION
    )

    accumulators = {}

    for m in VALIDATION_SIZES:

        for method in METHODS:

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

                    "marked_mass_sum":
                        0.0,

                    "true_given_marked_sum":
                        0.0,

                    "effective_true_probability_sum":
                        0.0,
                }

    max_m = max(
        VALIDATION_SIZES
    )

    for trial in range(
        N_EVALUATION_TRIALS
    ):

        if (
            trial % 20 == 0
        ):

            print(
                f"Evaluation trial "
                f"{trial}/"
                f"{N_EVALUATION_TRIALS}"
            )

        indices = rng_eval.choice(
            N_HELDOUT,
            size=max_m,
            replace=False,
        )

        selected = (
            evaluation_signs[
                :,
                :,
                indices,
            ]
        )

        cumulative = np.cumsum(
            selected,
            axis=2,
            dtype=np.int32,
        )

        for m in VALIDATION_SIZES:

            correlations = (
                cumulative[
                    :,
                    :,
                    m - 1,
                ]
                / float(m)
            )

            z = (
                correlations.T
                * math.sqrt(
                    m
                )
            )

            model = (
                null_models[
                    m
                ]
            )

            scores = (
                compute_scores_from_z(
                    z=z,

                    validation_size=m,

                    predicted_abs_correlations=
                        predicted_abs_correlations,

                    null_mean=
                        model[
                            "mean"
                        ],

                    covariance_inverse=
                        model[
                            "inverse"
                        ],
                )
            )

            for method in METHODS:

                score_vector = (
                    scores[
                        method
                    ]
                )

                for alpha in NOMINAL_FPRS:

                    threshold = (
                        thresholds[
                            m
                        ][
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

                    acc = (
                        accumulators[
                            (
                                m,
                                method,
                                alpha,
                            )
                        ]
                    )

                    acc[
                        "true_mark_count"
                    ] += int(
                        true_marked
                    )

                    acc[
                        "false_positive_sum"
                    ] += (
                        false_positive_count
                    )

                    acc[
                        "marked_count_sum"
                    ] += (
                        n_marked
                    )

                    acc[
                        "marked_mass_sum"
                    ] += (
                        marked_mass
                    )

                    if true_marked:

                        acc[
                            "true_given_marked_sum"
                        ] += (
                            true_given_marked
                        )

                    # 0 automatically if true key was rejected.
                    acc[
                        "effective_true_probability_sum"
                    ] += (
                        true_given_marked
                    )

    # =================================================================
    # Step 10
    # Aggregate evaluation results
    # =================================================================

    benchmark_rows = []

    threshold_lookup = (
        threshold_df.set_index(
            [
                "validation_size",
                "method",
                "nominal_false_positive_rate",
            ]
        )
    )

    for (
        m,
        method,
        alpha,
    ), acc in accumulators.items():

        true_count = (
            acc[
                "true_mark_count"
            ]
        )

        retention = (
            true_count
            / N_EVALUATION_TRIALS
        )

        mean_false_positive = (
            acc[
                "false_positive_sum"
            ]
            / N_EVALUATION_TRIALS
        )

        mean_marked = (
            acc[
                "marked_count_sum"
            ]
            / N_EVALUATION_TRIALS
        )

        mean_marked_mass = (
            acc[
                "marked_mass_sum"
            ]
            / N_EVALUATION_TRIALS
        )

        if true_count > 0:

            mean_true_given_marked = (
                acc[
                    "true_given_marked_sum"
                ]
                / true_count
            )

        else:

            mean_true_given_marked = 0.0

        effective_true_probability = (
            acc[
                "effective_true_probability_sum"
            ]
            / N_EVALUATION_TRIALS
        )

        threshold_row = (
            threshold_lookup.loc[
                (
                    m,
                    method,
                    alpha,
                )
            ]
        )

        benchmark_rows.append(
            {
                "validation_size":
                    m,

                "method":
                    method,

                "nominal_false_positive_rate":
                    alpha,

                "empirical_threshold":
                    float(
                        threshold_row[
                            "empirical_threshold"
                        ]
                    ),

                "calibration_achieved_fpr":
                    float(
                        threshold_row[
                            "calibration_achieved_fpr"
                        ]
                    ),

                "n_trials":
                    N_EVALUATION_TRIALS,

                "true_key_retention_probability":
                    retention,

                "mean_false_positive_count":
                    mean_false_positive,

                "mean_marked_candidates":
                    mean_marked,

                "mean_marked_probability_mass":
                    mean_marked_mass,

                "mean_true_probability_given_marked":
                    mean_true_given_marked,

                "effective_true_probability":
                    effective_true_probability,
            }
        )

    benchmark_df = pd.DataFrame(
        benchmark_rows
    )

    benchmark_df.to_csv(
        OUTPUT_BENCHMARK_CSV,
        index=False,
    )

    # =================================================================
    # Step 11
    # Best robust operating points
    # =================================================================

    best_rows = []

    for m in VALIDATION_SIZES:

        for method in METHODS:

            subset = benchmark_df[
                (
                    benchmark_df[
                        "validation_size"
                    ]
                    == m
                )
                &
                (
                    benchmark_df[
                        "method"
                    ]
                    == method
                )
                &
                (
                    benchmark_df[
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
    # Step 12
    # EXP08 vs EXP09 comparison at M=1024
    # =================================================================

    comparison_df = None

    if EXP08_BEST_CSV.exists():

        exp08_best = pd.read_csv(
            EXP08_BEST_CSV
        )

        exp08_m = exp08_best[
            exp08_best[
                "validation_size"
            ]
            == max(
                VALIDATION_SIZES
            )
        ].copy()

        exp09_m = best_df[
            best_df[
                "validation_size"
            ]
            == max(
                VALIDATION_SIZES
            )
        ].copy()

        exp08_m = exp08_m[
            [
                "method",
                "true_key_retention_probability",
                "mean_false_positive_count",
                "mean_true_probability_given_marked",
                "effective_true_probability",
            ]
        ].rename(
            columns={
                "true_key_retention_probability":
                    "exp08_retention",

                "mean_false_positive_count":
                    "exp08_false_positives",

                "mean_true_probability_given_marked":
                    "exp08_true_given_marked",

                "effective_true_probability":
                    "exp08_effective_true_probability",
            }
        )

        exp09_m = exp09_m[
            [
                "method",
                "true_key_retention_probability",
                "mean_false_positive_count",
                "mean_true_probability_given_marked",
                "effective_true_probability",
            ]
        ].rename(
            columns={
                "true_key_retention_probability":
                    "exp09_retention",

                "mean_false_positive_count":
                    "exp09_false_positives",

                "mean_true_probability_given_marked":
                    "exp09_true_given_marked",

                "effective_true_probability":
                    "exp09_effective_true_probability",
            }
        )

        comparison_df = pd.merge(
            exp08_m,
            exp09_m,
            on="method",
            how="outer",
        )

        comparison_df.to_csv(
            OUTPUT_COMPARISON_CSV,
            index=False,
        )

    # =================================================================
    # Step 13
    # Null covariance diagnostics
    # =================================================================

    diagnostic_rows = []

    for m in VALIDATION_SIZES:

        model = (
            null_models[
                m
            ]
        )

        diagnostic_rows.append(
            {
                "validation_size":
                    m,

                "condition_number":
                    model[
                        "condition_number"
                    ],

                "effective_rank":
                    model[
                        "effective_rank"
                    ],

                "max_abs_offdiagonal_correlation":
                    model[
                        "max_abs_offdiagonal_correlation"
                    ],
            }
        )

    diagnostics_df = pd.DataFrame(
        diagnostic_rows
    )

    # =================================================================
    # Step 14
    # Plot null correlation matrix at M=1024
    # =================================================================

    selected_m = max(
        VALIDATION_SIZES
    )

    correlation_matrix = (
        null_models[
            selected_m
        ][
            "correlation"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        correlation_matrix,
        vmin=-1.0,
        vmax=1.0,
    )

    ax.set_xticks(
        range(
            l_count
        )
    )

    ax.set_yticks(
        range(
            l_count
        )
    )

    ax.set_xticklabels(
        approximation_labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        approximation_labels
    )

    for i in range(
        l_count
    ):

        for j in range(
            l_count
        ):

            ax.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title(
        f"EXP09: Empirical Null Correlation Matrix "
        f"(M={selected_m})"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    fig.tight_layout()

    fig.savefig(
        FIG_CORRELATION,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # Step 15
    # Plot effective true-key probability
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for method in METHODS:

        part = best_df[
            best_df[
                "method"
            ]
            == method
        ].sort_values(
            "validation_size"
        )

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
        "EXP09: Empirically Calibrated Marking"
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

    plt.close(
        fig
    )

    # =================================================================
    # Step 16
    # Plot robust false-positive count
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for method in METHODS:

        part = best_df[
            best_df[
                "method"
            ]
            == method
        ].sort_values(
            "validation_size"
        )

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
        "EXP09: False Positives at >=90% True-Key Retention"
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

    plt.close(
        fig
    )

    # =================================================================
    # Step 17
    # Retention/selectivity tradeoff at M=1024
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for method in METHODS:

        part = benchmark_df[
            (
                benchmark_df[
                    "validation_size"
                ]
                == selected_m
            )
            &
            (
                benchmark_df[
                    "method"
                ]
                == method
            )
        ].sort_values(
            "mean_false_positive_count"
        )

        if len(
            part
        ) == 0:

            continue

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
        f"EXP09: Empirical Marking Tradeoff "
        f"at M={selected_m}"
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

    plt.close(
        fig
    )

    # =================================================================
    # Step 18
    # Overall best robust point per method
    # =================================================================

    overall_best = {}

    for method in METHODS:

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

            "calibration_achieved_fpr":
                float(
                    best[
                        "calibration_achieved_fpr"
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
    # Step 19
    # Summary JSON
    # =================================================================

    diagnostic_summary = {}

    for m in VALIDATION_SIZES:

        model = (
            null_models[
                m
            ]
        )

        diagnostic_summary[
            str(m)
        ] = {
            "condition_number":
                float(
                    model[
                        "condition_number"
                    ]
                ),

            "effective_rank":
                float(
                    model[
                        "effective_rank"
                    ]
                ),

            "max_abs_offdiagonal_correlation":
                float(
                    model[
                        "max_abs_offdiagonal_correlation"
                    ]
                ),

            "eigenvalues":
                [
                    float(x)
                    for x in model[
                        "eigenvalues"
                    ]
                ],
        }

    summary = {
        "experiment":
            "exp09_empirical_calibration",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "initial_true_key_probability":
            p_true_initial,

        "n_approximations":
            l_count,

        "fresh_calibration_pairs":
            N_CALIBRATION_PAIRS,

        "calibration_trials":
            N_CALIBRATION_TRIALS,

        "evaluation_trials":
            N_EVALUATION_TRIALS,

        "validation_sizes":
            VALIDATION_SIZES,

        "nominal_false_positive_rates":
            NOMINAL_FPRS,

        "robust_retention_requirement":
            ROBUST_RETENTION,

        "methods": {
            "M1_single":
                (
                    "Single canonical linear correlation."
                ),

            "M2_energy":
                (
                    "Naive sum of squared standardized "
                    "linear correlations."
                ),

            "M3_llr":
                (
                    "EXP08 sign-marginalized approximate "
                    "log-likelihood ratio with empirical "
                    "null threshold calibration."
                ),

            "M4_consensus":
                (
                    "Count of individually significant "
                    "linear correlations."
                ),

            "M5_mahalanobis":
                (
                    "Covariance-aware quadratic score "
                    "using empirically estimated wrong-key "
                    "mean and covariance."
                ),
        },

        "null_diagnostics":
            diagnostic_summary,

        "overall_best_robust_configuration":
            overall_best,

        "interpretation_rules": {
            "if_M2_and_M5_are_similar":
                (
                    "Cross-correlation among approximations "
                    "is probably not the primary reason for "
                    "EXP08 performance."
                ),

            "if_M5_substantially_outperforms_M2":
                (
                    "Approximation covariance matters and "
                    "the independence-style energy score "
                    "double-counts redundant information."
                ),

            "if_M3_remains_strong_after_empirical_calibration":
                (
                    "The EXP08 LLR improvement is not merely "
                    "an artifact of theoretical Gaussian "
                    "threshold calibration."
                ),

            "if_M3_collapses":
                (
                    "The EXP08 LLR result was likely sensitive "
                    "to its assumed null distribution or "
                    "model mismatch."
                ),
        },

        "limitations": [
            (
                "Wrong-key null fitting excludes the known true "
                "key. This is appropriate for benchmark diagnosis "
                "but is not itself an attack-side oracle."
            ),
            (
                "M5 Mahalanobis is a statistical verifier score, "
                "not yet a reversible quantum circuit."
            ),
            (
                "Quantum gate, ancilla, arithmetic, and "
                "uncomputation costs are intentionally deferred "
                "to the next experiment."
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
    # Console report
    # =================================================================

    print()
    print("=" * 80)
    print(
        "EXP09: Empirical Calibration of Multi-Correlation Marking"
    )
    print("=" * 80)

    print()
    print(
        f"True key                      : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"Initial P(true)               : "
        f"{p_true_initial:.6f} "
        f"({100 * p_true_initial:.3f}%)"
    )

    print(
        f"Linear approximations         : "
        f"{l_count}"
    )

    print(
        f"Fresh calibration P/C pairs   : "
        f"{N_CALIBRATION_PAIRS}"
    )

    print()

    print(
        "[Null-correlation diagnostics]"
    )

    for _, row in diagnostics_df.iterrows():

        print(
            f"M={int(row['validation_size']):4d} | "
            f"effective rank="
            f"{row['effective_rank']:.3f} | "
            f"max |offdiag corr|="
            f"{row['max_abs_offdiagonal_correlation']:.3f} | "
            f"cond(Sigma)="
            f"{row['condition_number']:.3f}"
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
            f"  M                         : "
            f"{values['validation_size']}"
        )

        print(
            f"  nominal FPR               : "
            f"{values['nominal_false_positive_rate']:.3f}"
        )

        print(
            f"  empirical calibration FPR : "
            f"{values['calibration_achieved_fpr']:.4f}"
        )

        print(
            f"  true-key retention        : "
            f"{values['true_key_retention_probability']:.4f}"
        )

        print(
            f"  mean false positives      : "
            f"{values['mean_false_positive_count']:.2f}"
        )

        print(
            f"  mean P(true | marked)     : "
            f"{values['mean_true_probability_given_marked']:.4f}"
        )

        print(
            f"  effective P(true)         : "
            f"{values['effective_true_probability']:.4f}"
        )

    if comparison_df is not None:

        print()
        print(
            "[EXP08 vs EXP09 at M=1024]"
        )

        print(
            comparison_df.to_string(
                index=False
            )
        )

    print()
    print(
        "[Output files]"
    )

    print(
        f"Thresholds                    : "
        f"{OUTPUT_THRESHOLDS_CSV}"
    )

    print(
        f"Covariance                    : "
        f"{OUTPUT_COVARIANCE_CSV}"
    )

    print(
        f"Benchmark                     : "
        f"{OUTPUT_BENCHMARK_CSV}"
    )

    print(
        f"Best robust points            : "
        f"{OUTPUT_BEST_CSV}"
    )

    if comparison_df is not None:

        print(
            f"EXP08 comparison              : "
            f"{OUTPUT_COMPARISON_CSV}"
        )

    print(
        f"Summary                       : "
        f"{OUTPUT_SUMMARY_JSON}"
    )

    print(
        f"Correlation matrix figure     : "
        f"{FIG_CORRELATION}"
    )

    print(
        f"Effective probability figure  : "
        f"{FIG_EFFECTIVE}"
    )

    print(
        f"False-positive figure         : "
        f"{FIG_FALSE_POSITIVE}"
    )

    print(
        f"Tradeoff figure               : "
        f"{FIG_TRADEOFF}"
    )

    print()
    print(
        "[PASS] EXP09 completed successfully."
    )


if __name__ == "__main__":
    main()