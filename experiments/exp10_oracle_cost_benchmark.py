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

EXP08_BANK_CSV = (
    RESULTS_DIR
    / "exp08_approximation_bank.csv"
)

EXP09_BEST_CSV = (
    RESULTS_DIR
    / "exp09_best_robust_points.csv"
)

EXP09_BENCHMARK_CSV = (
    RESULTS_DIR
    / "exp09_marking_benchmark.csv"
)


OUTPUT_RESOURCES_CSV = (
    RESULTS_DIR
    / "exp10_oracle_resource_estimates.csv"
)

OUTPUT_BASELINE_CSV = (
    RESULTS_DIR
    / "exp10_baseline_comparison.csv"
)

OUTPUT_SUMMARY_JSON = (
    RESULTS_DIR
    / "exp10_summary.json"
)


FIG_TOTAL_COST = (
    FIGURES_DIR
    / "exp10_oracle_cost.png"
)

FIG_COST_QUALITY = (
    FIGURES_DIR
    / "exp10_cost_vs_marking_quality.png"
)

FIG_ANCILLA = (
    FIGURES_DIR
    / "exp10_ancilla_qubits.png"
)

FIG_SENSITIVITY = (
    FIGURES_DIR
    / "exp10_cost_sensitivity.png"
)


# =====================================================================
# Configuration
# =====================================================================

ROBUST_RETENTION = 0.90

# Fixed-point precision used for nonlinear / coefficient arithmetic.
FIXED_POINT_BITS = 16

# EXP09 uses 8 approximations.
EXPECTED_APPROXIMATIONS = 8


# ---------------------------------------------------------------------
# Cost-model scenarios
#
# "TE" = Toffoli-equivalent proxy unit.
#
# These are NOT synthesized physical gate counts.
#
# The purpose is to see whether conclusions survive reasonable changes
# in assumptions about reversible lookup implementation.
# ---------------------------------------------------------------------

COST_SCENARIOS = {
    "optimistic": {
        "sbox_lookup_te":
            8,

        "qrom_entry_te":
            1,
    },

    "baseline": {
        "sbox_lookup_te":
            16,

        "qrom_entry_te":
            2,
    },

    "pessimistic": {
        "sbox_lookup_te":
            32,

        "qrom_entry_te":
            4,
    },
}


METHODS = [
    "M1_single",
    "M2_energy",
    "M3_llr",
    "M4_consensus",
    "M5_mahalanobis",
]


# =====================================================================
# Primitive reversible arithmetic cost model
# =====================================================================

def unsigned_width(
    maximum_value: int,
) -> int:

    if maximum_value <= 0:
        return 1

    return max(
        1,
        math.ceil(
            math.log2(
                maximum_value + 1
            )
        ),
    )


def ripple_add_te(
    width: int,
) -> int:
    """
    Proxy for an in-place ripple-carry addition.
    """

    return max(
        1,
        2 * width - 1,
    )


def controlled_increment_te(
    width: int,
) -> int:
    """
    Proxy for conditionally incrementing a b-bit counter.
    """

    return max(
        1,
        2 * width - 1,
    )


def compare_constant_te(
    width: int,
) -> int:
    """
    Proxy for reversible comparison with a classical constant.
    """

    return max(
        1,
        2 * width - 1,
    )


def absolute_value_te(
    width: int,
) -> int:
    """
    Signed absolute-value proxy.
    """

    return max(
        1,
        2 * width,
    )


def schoolbook_square_te(
    width: int,
) -> int:
    """
    Schoolbook multiplication-style square proxy.
    """

    return (
        width
        * width
    )


def schoolbook_multiply_te(
    width_a: int,
    width_b: int,
) -> int:

    return (
        width_a
        * width_b
    )


def constant_multiply_te(
    value_width: int,
    coefficient_bits: int,
) -> int:
    """
    Fixed-point constant multiplication proxy.

    Deliberately conservative.
    """

    return (
        value_width
        * coefficient_bits
    )


# =====================================================================
# Load EXP09 operating points
# =====================================================================

def load_best_robust_points():
    """
    Select one best robust configuration per marking method.

    Primary criterion:
        maximum effective true-key probability

    Secondary criterion:
        fewer false positives
    """

    if not EXP09_BEST_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP09_BEST_CSV}\n\n"
            "Run EXP09 first."
        )

    df = pd.read_csv(
        EXP09_BEST_CSV
    )

    df = df[
        df[
            "true_key_retention_probability"
        ]
        >= ROBUST_RETENTION
    ].copy()

    selected = []

    for method in METHODS:

        part = df[
            df[
                "method"
            ]
            == method
        ].copy()

        if len(part) == 0:
            continue

        best = (
            part
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

        selected.append(
            best.to_dict()
        )

    return pd.DataFrame(
        selected
    )


# =====================================================================
# Approximation-bank information
# =====================================================================

def load_approximation_bank():

    if not EXP08_BANK_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP08_BANK_CSV}\n\n"
            "Run EXP08 first."
        )

    df = pd.read_csv(
        EXP08_BANK_CSV
    )

    if len(df) != EXPECTED_APPROXIMATIONS:

        print(
            "[WARNING] Expected "
            f"{EXPECTED_APPROXIMATIONS} approximations "
            f"but found {len(df)}."
        )

    return df


# =====================================================================
# Relation-evaluation stage
# =====================================================================

def relation_stage_cost(
    validation_size: int,
    approximation_bank: pd.DataFrame,
    n_approximations_used: int,
    scenario: dict,
):
    """
    Cost of constructing empirical correlation counters.

    Assumption
    ----------
    For each validation P/C pair:

    1. Candidate 8-bit subkey is split into two 4-bit guesses.
    2. Two inverse S-box values are computed.
    3. The same two partial-decryption values are reused by all
       linear approximations.
    4. Each approximation updates one reversible counter.
    5. Temporary S-box and relation workspaces are uncomputed.

    Therefore one forward score-computation pass requires:

        2 inverse S-box computations
        +
        2 inverse S-box uncomputations

    per sample.

    The complete phase oracle later doubles this whole computation
    because score computation must itself be uncomputed.
    """

    m = validation_size

    count_width = unsigned_width(
        m
    )

    sbox_lookup_te = int(
        scenario[
            "sbox_lookup_te"
        ]
    )

    # -------------------------------------------------------------
    # Reversible S-box lookup
    # -------------------------------------------------------------

    sbox_calls_compute = (
        4 * m
    )

    sbox_te_compute = (
        sbox_calls_compute
        * sbox_lookup_te
    )

    # -------------------------------------------------------------
    # Correlation counters
    # -------------------------------------------------------------

    counter_updates_compute = (
        m
        * n_approximations_used
    )

    counter_te_compute = (
        counter_updates_compute
        * controlled_increment_te(
            count_width
        )
    )

    # -------------------------------------------------------------
    # Clifford parity cost
    #
    # Each relation ancilla receives the selected U4 bits.
    # It is then uncomputed after updating its counter.
    # -------------------------------------------------------------

    selected_bank = (
        approximation_bank
        .iloc[
            :n_approximations_used
        ]
    )

    mask_weight_sum = 0

    for _, row in selected_bank.iterrows():

        mask2 = int(
            row[
                "mask_nibble2"
            ]
        )

        mask4 = int(
            row[
                "mask_nibble4"
            ]
        )

        mask_weight_sum += (
            mask2.bit_count()
            +
            mask4.bit_count()
        )

    parity_cnot_compute = (
        2
        * m
        * mask_weight_sum
    )

    total_te_compute = (
        sbox_te_compute
        +
        counter_te_compute
    )

    return {
        "count_width":
            count_width,

        "sbox_calls_compute":
            sbox_calls_compute,

        "sbox_te_compute":
            sbox_te_compute,

        "counter_updates_compute":
            counter_updates_compute,

        "counter_te_compute":
            counter_te_compute,

        "parity_cnot_compute":
            parity_cnot_compute,

        "relation_te_compute":
            total_te_compute,

        "mask_weight_sum":
            mask_weight_sum,
    }


# =====================================================================
# Method-specific score costs
# =====================================================================

def score_cost_m1(
    m: int,
):
    """
    M1:
        |single correlation| >= threshold

    Since M is fixed, this can be implemented directly as two
    comparisons against the correlation counter:

        count <= lower
        OR
        count >= upper

    No squaring or nonlinear arithmetic is required.
    """

    count_width = unsigned_width(
        m
    )

    comparisons = 2

    predicate_te = (
        comparisons
        * compare_constant_te(
            count_width
        )
    )

    return {
        "score_te_compute":
            predicate_te,

        "score_cnot_compute":
            1,

        "score_accumulator_width":
            1,

        "temporary_qubits":
            2,

        "description":
            (
                "Two constant comparisons on one "
                "correlation counter."
            ),
    }


def score_cost_m2(
    m: int,
    l_count: int,
):
    """
    M2:
        sum_l z_l^2 >= threshold

    Multiplicative scaling by M is classical and absorbed into
    the threshold, so integer centered counts can be squared.
    """

    count_width = unsigned_width(
        m
    )

    deviation_width = (
        count_width
        + 1
    )

    square_width = (
        2
        * deviation_width
    )

    accumulator_width = (
        square_width
        + unsigned_width(
            l_count
        )
        + 1
    )

    center_cost = (
        l_count
        * ripple_add_te(
            deviation_width
        )
    )

    square_cost = (
        l_count
        * schoolbook_square_te(
            deviation_width
        )
    )

    accumulation_cost = (
        l_count
        * ripple_add_te(
            accumulator_width
        )
    )

    final_compare_cost = (
        compare_constant_te(
            accumulator_width
        )
    )

    total = (
        center_cost
        + square_cost
        + accumulation_cost
        + final_compare_cost
    )

    return {
        "score_te_compute":
            total,

        "score_cnot_compute":
            0,

        "score_accumulator_width":
            accumulator_width,

        "temporary_qubits":
            (
                square_width
                + accumulator_width
                + deviation_width
                + 1
            ),

        "description":
            (
                "Centered counters, L integer squares, "
                "accumulation, and one threshold comparison."
            ),
    }


def score_cost_m3(
    m: int,
    l_count: int,
    scenario: dict,
):
    """
    M3:
        sign-marginalized LLR

    Rather than evaluating log(cosh(.)) arithmetically inside the
    quantum circuit, exploit the fact that each empirical counter
    only has M+1 possible values.

    Because log(cosh(.)) is even, use absolute deviation from M/2
    and a symmetry-aware qROM table.

    Number of distinct |deviation| inputs:
        M/2 + 1

    Each approximation can therefore use a precomputed fixed-point
    lookup table.

    This is not free, but is much more plausible than implementing
    transcendental functions gate-by-gate.
    """

    count_width = unsigned_width(
        m
    )

    deviation_width = (
        count_width
        + 1
    )

    qrom_entries = (
        m // 2
        + 1
    )

    qrom_entry_te = int(
        scenario[
            "qrom_entry_te"
        ]
    )

    accumulator_width = (
        FIXED_POINT_BITS
        + unsigned_width(
            l_count
        )
        + 2
    )

    abs_cost = (
        l_count
        * absolute_value_te(
            deviation_width
        )
    )

    qrom_cost = (
        l_count
        * qrom_entries
        * qrom_entry_te
    )

    accumulation_cost = (
        l_count
        * ripple_add_te(
            accumulator_width
        )
    )

    final_compare_cost = (
        compare_constant_te(
            accumulator_width
        )
    )

    total = (
        abs_cost
        + qrom_cost
        + accumulation_cost
        + final_compare_cost
    )

    return {
        "score_te_compute":
            total,

        "score_cnot_compute":
            0,

        "score_accumulator_width":
            accumulator_width,

        "temporary_qubits":
            (
                deviation_width
                + FIXED_POINT_BITS
                + accumulator_width
                + 1
            ),

        "qrom_entries_per_approximation":
            qrom_entries,

        "description":
            (
                "Symmetry-aware fixed-point qROM "
                "implementation of LLR contributions."
            ),
    }


def score_cost_m4(
    m: int,
    l_count: int,
):
    """
    M4:
        number of component tests passing >= q

    Each component needs two comparisons because:

        |correlation_l| >= threshold

    is a two-sided test.
    """

    count_width = unsigned_width(
        m
    )

    component_comparisons = (
        2
        * l_count
    )

    comparison_cost = (
        component_comparisons
        * compare_constant_te(
            count_width
        )
    )

    popcount_width = unsigned_width(
        l_count
    )

    # Sequentially accumulate L Boolean pass flags.
    popcount_cost = (
        l_count
        * ripple_add_te(
            popcount_width
        )
    )

    final_compare_cost = (
        compare_constant_te(
            popcount_width
        )
    )

    total = (
        comparison_cost
        + popcount_cost
        + final_compare_cost
    )

    return {
        "score_te_compute":
            total,

        "score_cnot_compute":
            l_count,

        "score_accumulator_width":
            popcount_width,

        "temporary_qubits":
            (
                l_count
                + popcount_width
                + 1
            ),

        "description":
            (
                "Per-approximation two-sided tests "
                "plus Boolean popcount."
            ),
    }


def score_cost_m5(
    m: int,
    l_count: int,
):
    """
    M5:
        covariance-aware quadratic form

            d^T Sigma^{-1} d

    We model the symmetric quadratic form directly.

    Number of unique products:

        L(L+1)/2

    Every product then receives a fixed-point coefficient.
    """

    count_width = unsigned_width(
        m
    )

    deviation_width = (
        count_width
        + 1
    )

    n_products = (
        l_count
        * (
            l_count + 1
        )
        // 2
    )

    product_width = (
        2
        * deviation_width
    )

    accumulator_width = (
        product_width
        + FIXED_POINT_BITS
        + unsigned_width(
            n_products
        )
        + 2
    )

    centering_cost = (
        l_count
        * ripple_add_te(
            deviation_width
        )
    )

    variable_product_cost = (
        n_products
        * schoolbook_multiply_te(
            deviation_width,
            deviation_width,
        )
    )

    coefficient_cost = (
        n_products
        * constant_multiply_te(
            product_width,
            FIXED_POINT_BITS,
        )
    )

    accumulation_cost = (
        n_products
        * ripple_add_te(
            accumulator_width
        )
    )

    final_compare_cost = (
        compare_constant_te(
            accumulator_width
        )
    )

    total = (
        centering_cost
        + variable_product_cost
        + coefficient_cost
        + accumulation_cost
        + final_compare_cost
    )

    return {
        "score_te_compute":
            total,

        "score_cnot_compute":
            0,

        "score_accumulator_width":
            accumulator_width,

        "temporary_qubits":
            (
                l_count
                * deviation_width
                + product_width
                + accumulator_width
                + 1
            ),

        "quadratic_products":
            n_products,

        "description":
            (
                "Full covariance-aware fixed-point "
                "quadratic form."
            ),
    }


# =====================================================================
# Complete oracle resource estimate
# =====================================================================

def method_resource_estimate(
    method: str,
    operating_point: pd.Series,
    approximation_bank: pd.DataFrame,
    scenario_name: str,
    scenario: dict,
):

    m = int(
        operating_point[
            "validation_size"
        ]
    )

    l_total = len(
        approximation_bank
    )

    # -------------------------------------------------------------
    # Number of linear approximations actually required.
    # -------------------------------------------------------------

    if method == "M1_single":

        l_used = 1

    else:

        l_used = l_total

    relation = relation_stage_cost(
        validation_size=m,
        approximation_bank=approximation_bank,
        n_approximations_used=l_used,
        scenario=scenario,
    )

    # -------------------------------------------------------------
    # Score / threshold logic
    # -------------------------------------------------------------

    if method == "M1_single":

        score = score_cost_m1(
            m
        )

    elif method == "M2_energy":

        score = score_cost_m2(
            m,
            l_used,
        )

    elif method == "M3_llr":

        score = score_cost_m3(
            m,
            l_used,
            scenario,
        )

    elif method == "M4_consensus":

        score = score_cost_m4(
            m,
            l_used,
        )

    elif method == "M5_mahalanobis":

        score = score_cost_m5(
            m,
            l_used,
        )

    else:

        raise ValueError(
            f"Unknown method: {method}"
        )

    # -------------------------------------------------------------
    # One forward score computation
    # -------------------------------------------------------------

    compute_te = (
        relation[
            "relation_te_compute"
        ]
        +
        score[
            "score_te_compute"
        ]
    )

    compute_cnot = (
        relation[
            "parity_cnot_compute"
        ]
        +
        score[
            "score_cnot_compute"
        ]
    )

    # -------------------------------------------------------------
    # Reversible phase oracle
    #
    # |k>|0> -> (-1)^V(k)|k>|0>
    #
    # requires:
    #
    #   compute
    #   phase flip
    #   uncompute
    #
    # Therefore arithmetic cost is approximately doubled.
    # -------------------------------------------------------------

    oracle_te = (
        2
        * compute_te
    )

    oracle_cnot = (
        2
        * compute_cnot
    )

    # -------------------------------------------------------------
    # Logical qubit estimate
    #
    # Validation P/C values are compiled as classical constants,
    # so no coherent data-register is counted here.
    # -------------------------------------------------------------

    count_width = (
        relation[
            "count_width"
        ]
    )

    key_qubits = 8

    counter_qubits = (
        l_used
        * count_width
    )

    sbox_workspace_qubits = 8

    relation_ancilla_qubits = 1

    predicate_qubit = 1

    temporary_qubits = int(
        score[
            "temporary_qubits"
        ]
    )

    total_logical_qubits = (
        key_qubits
        + counter_qubits
        + sbox_workspace_qubits
        + relation_ancilla_qubits
        + predicate_qubit
        + temporary_qubits
    )

    effective_probability = float(
        operating_point[
            "effective_true_probability"
        ]
    )

    false_positives = float(
        operating_point[
            "mean_false_positive_count"
        ]
    )

    retention = float(
        operating_point[
            "true_key_retention_probability"
        ]
    )

    true_given_marked = float(
        operating_point[
            "mean_true_probability_given_marked"
        ]
    )

    marked_mass = float(
        operating_point[
            "mean_marked_probability_mass"
        ]
    )

    # -------------------------------------------------------------
    # Simple quality / cost metrics.
    #
    # These are NOT end-to-end quantum runtime metrics.
    # -------------------------------------------------------------

    quality_per_million_te = (
        effective_probability
        * 1_000_000.0
        / oracle_te
    )

    rejected_wrong_keys = (
        255.0
        - false_positives
    )

    wrong_keys_rejected_per_million_te = (
        rejected_wrong_keys
        * 1_000_000.0
        / oracle_te
    )

    te_per_effective_probability = (
        oracle_te
        / max(
            effective_probability,
            1e-15,
        )
    )

    result = {
        "scenario":
            scenario_name,

        "method":
            method,

        "validation_size":
            m,

        "approximations_used":
            l_used,

        "true_key_retention_probability":
            retention,

        "mean_false_positive_count":
            false_positives,

        "mean_true_probability_given_marked":
            true_given_marked,

        "mean_marked_probability_mass":
            marked_mass,

        "effective_true_probability":
            effective_probability,

        "count_width":
            count_width,

        "sbox_calls_forward_compute":
            relation[
                "sbox_calls_compute"
            ],

        "counter_updates_forward_compute":
            relation[
                "counter_updates_compute"
            ],

        "relation_te_forward_compute":
            relation[
                "relation_te_compute"
            ],

        "score_te_forward_compute":
            score[
                "score_te_compute"
            ],

        "total_te_forward_compute":
            compute_te,

        "phase_oracle_te_proxy":
            oracle_te,

        "phase_oracle_cnot_proxy":
            oracle_cnot,

        "logical_qubits_proxy":
            total_logical_qubits,

        "counter_qubits":
            counter_qubits,

        "score_temporary_qubits":
            temporary_qubits,

        "quality_per_million_te":
            quality_per_million_te,

        "wrong_keys_rejected_per_million_te":
            wrong_keys_rejected_per_million_te,

        "te_per_effective_probability":
            te_per_effective_probability,

        "score_description":
            score[
                "description"
            ],
    }

    if (
        "qrom_entries_per_approximation"
        in score
    ):

        result[
            "qrom_entries_per_approximation"
        ] = score[
            "qrom_entries_per_approximation"
        ]

    else:

        result[
            "qrom_entries_per_approximation"
        ] = 0

    if (
        "quadratic_products"
        in score
    ):

        result[
            "quadratic_products"
        ] = score[
            "quadratic_products"
        ]

    else:

        result[
            "quadratic_products"
        ] = 0

    return result


# =====================================================================
# Plot helpers
# =====================================================================

def method_label(
    method: str,
) -> str:

    mapping = {
        "M1_single":
            "M1 single",

        "M2_energy":
            "M2 energy",

        "M3_llr":
            "M3 LLR",

        "M4_consensus":
            "M4 consensus",

        "M5_mahalanobis":
            "M5 Mahalanobis",
    }

    return mapping.get(
        method,
        method,
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
    # Load EXP09 marking-performance points
    # =================================================================

    best_points = (
        load_best_robust_points()
    )

    approximation_bank = (
        load_approximation_bank()
    )

    print()
    print("=" * 80)
    print(
        "EXP10: Reversible Marking Oracle Cost Benchmark"
    )
    print("=" * 80)

    print()
    print(
        "[Selected EXP09 robust operating points]"
    )

    for _, row in best_points.iterrows():

        print(
            f"{row['method']:16s} | "
            f"M={int(row['validation_size']):4d} | "
            f"retention="
            f"{row['true_key_retention_probability']:.3f} | "
            f"FP="
            f"{row['mean_false_positive_count']:.3f} | "
            f"P_eff="
            f"{row['effective_true_probability']:.3f}"
        )

    # =================================================================
    # Step 2
    # Resource estimates under all cost-model scenarios
    # =================================================================

    resource_rows = []

    for scenario_name, scenario in (
        COST_SCENARIOS.items()
    ):

        for _, point in (
            best_points.iterrows()
        ):

            method = str(
                point[
                    "method"
                ]
            )

            estimate = (
                method_resource_estimate(
                    method=method,
                    operating_point=point,
                    approximation_bank=
                        approximation_bank,
                    scenario_name=
                        scenario_name,
                    scenario=scenario,
                )
            )

            resource_rows.append(
                estimate
            )

    resources_df = pd.DataFrame(
        resource_rows
    )

    resources_df.to_csv(
        OUTPUT_RESOURCES_CSV,
        index=False,
    )

    # =================================================================
    # Step 3
    # Baseline scenario comparison
    # =================================================================

    baseline_df = (
        resources_df[
            resources_df[
                "scenario"
            ]
            == "baseline"
        ]
        .copy()
        .sort_values(
            "phase_oracle_te_proxy"
        )
    )

    minimum_cost = float(
        baseline_df[
            "phase_oracle_te_proxy"
        ]
        .min()
    )

    baseline_df[
        "cost_relative_to_cheapest"
    ] = (
        baseline_df[
            "phase_oracle_te_proxy"
        ]
        / minimum_cost
    )

    m2_cost_series = (
        baseline_df.loc[
            baseline_df[
                "method"
            ]
            == "M2_energy",
            "phase_oracle_te_proxy",
        ]
    )

    if len(
        m2_cost_series
    ) == 1:

        m2_cost = float(
            m2_cost_series.iloc[0]
        )

        baseline_df[
            "cost_relative_to_M2"
        ] = (
            baseline_df[
                "phase_oracle_te_proxy"
            ]
            / m2_cost
        )

    else:

        baseline_df[
            "cost_relative_to_M2"
        ] = np.nan

    baseline_df.to_csv(
        OUTPUT_BASELINE_CSV,
        index=False,
    )

    # =================================================================
    # Step 4
    # Figure: total phase-oracle cost
    # =================================================================

    plot_df = (
        baseline_df
        .sort_values(
            "phase_oracle_te_proxy"
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    labels = [
        method_label(x)
        for x in plot_df[
            "method"
        ]
    ]

    ax.bar(
        labels,
        plot_df[
            "phase_oracle_te_proxy"
        ],
    )

    ax.set_yscale(
        "log"
    )

    ax.set_ylabel(
        "Phase-oracle cost proxy "
        "(Toffoli-equivalent units, log scale)"
    )

    ax.set_title(
        "EXP10: Reversible Marking Oracle Cost"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_TOTAL_COST,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # Step 5
    # Figure: cost vs marking quality
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for _, row in (
        baseline_df.iterrows()
    ):

        x = float(
            row[
                "phase_oracle_te_proxy"
            ]
        )

        y = float(
            row[
                "effective_true_probability"
            ]
        )

        ax.scatter(
            x,
            y,
            s=70,
        )

        ax.annotate(
            method_label(
                str(
                    row[
                        "method"
                    ]
                )
            ),
            (
                x,
                y,
            ),
            xytext=(
                6,
                5,
            ),
            textcoords="offset points",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Phase-oracle cost proxy "
        "(Toffoli-equivalent units, log scale)"
    )

    ax.set_ylabel(
        "Effective true-key probability"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "EXP10: Marking Quality versus Oracle Cost"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_COST_QUALITY,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # Step 6
    # Figure: logical-qubit proxy
    # =================================================================

    ancilla_df = (
        baseline_df
        .sort_values(
            "logical_qubits_proxy"
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    labels = [
        method_label(x)
        for x in ancilla_df[
            "method"
        ]
    ]

    ax.bar(
        labels,
        ancilla_df[
            "logical_qubits_proxy"
        ],
    )

    ax.set_ylabel(
        "Logical-qubit proxy"
    )

    ax.set_title(
        "EXP10: Oracle Workspace Requirement"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_ANCILLA,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # Step 7
    # Figure: sensitivity to lookup assumptions
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    scenario_order = [
        "optimistic",
        "baseline",
        "pessimistic",
    ]

    x = np.arange(
        len(
            scenario_order
        )
    )

    for method in METHODS:

        part = (
            resources_df[
                resources_df[
                    "method"
                ]
                == method
            ]
            .set_index(
                "scenario"
            )
        )

        if len(
            part
        ) == 0:

            continue

        values = [
            float(
                part.loc[
                    scenario,
                    "phase_oracle_te_proxy",
                ]
            )
            for scenario
            in scenario_order
        ]

        ax.plot(
            x,
            values,
            marker="o",
            label=method_label(
                method
            ),
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        scenario_order
    )

    ax.set_yscale(
        "log"
    )

    ax.set_ylabel(
        "Phase-oracle cost proxy "
        "(log scale)"
    )

    ax.set_xlabel(
        "Reversible lookup cost assumption"
    )

    ax.set_title(
        "EXP10: Oracle-Cost Sensitivity"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_SENSITIVITY,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # Step 8
    # Ranking tables
    # =================================================================

    quality_ranking = (
        baseline_df[
            [
                "method",
                "validation_size",
                "effective_true_probability",
                "mean_false_positive_count",
                "phase_oracle_te_proxy",
                "logical_qubits_proxy",
                "quality_per_million_te",
                "wrong_keys_rejected_per_million_te",
                "cost_relative_to_cheapest",
                "cost_relative_to_M2",
            ]
        ]
        .sort_values(
            "effective_true_probability",
            ascending=False,
        )
    )

    cost_efficiency_ranking = (
        baseline_df[
            [
                "method",
                "effective_true_probability",
                "phase_oracle_te_proxy",
                "quality_per_million_te",
                "wrong_keys_rejected_per_million_te",
            ]
        ]
        .sort_values(
            "quality_per_million_te",
            ascending=False,
        )
    )

    # =================================================================
    # Step 9
    # Automatic interpretation
    # =================================================================

    method_lookup = (
        baseline_df
        .set_index(
            "method"
        )
    )

    automatic_findings = []

    if (
        "M2_energy"
        in method_lookup.index
        and
        "M3_llr"
        in method_lookup.index
    ):

        m2 = (
            method_lookup.loc[
                "M2_energy"
            ]
        )

        m3 = (
            method_lookup.loc[
                "M3_llr"
            ]
        )

        quality_gain = (
            float(
                m3[
                    "effective_true_probability"
                ]
            )
            -
            float(
                m2[
                    "effective_true_probability"
                ]
            )
        )

        cost_ratio = (
            float(
                m3[
                    "phase_oracle_te_proxy"
                ]
            )
            /
            float(
                m2[
                    "phase_oracle_te_proxy"
                ]
            )
        )

        automatic_findings.append(
            {
                "comparison":
                    "M3_llr_vs_M2_energy",

                "M3_minus_M2_effective_probability":
                    quality_gain,

                "M3_over_M2_oracle_cost":
                    cost_ratio,
            }
        )

    # =================================================================
    # Step 10
    # Summary JSON
    # =================================================================

    summary = {
        "experiment":
            "exp10_oracle_cost_benchmark",

        "cost_model":
            (
                "Logical reversible-operation proxy. "
                "Values are not synthesized physical "
                "Toffoli/T counts."
            ),

        "important_assumptions": [
            (
                "Validation plaintext/ciphertext pairs are "
                "compiled into the circuit as classical constants. "
                "qRAM or coherent external-memory access cost is "
                "not included."
            ),
            (
                "Two inverse 4-bit S-box evaluations are shared "
                "across all linear approximations for each sample."
            ),
            (
                "The complete marking phase oracle includes "
                "compute and uncompute, so forward arithmetic "
                "cost is doubled."
            ),
            (
                "M3 LLR uses a symmetry-aware fixed-point qROM "
                "lookup instead of reversible transcendental "
                "function evaluation."
            ),
            (
                "M5 uses a direct symmetric fixed-point quadratic "
                "form for its covariance-aware statistic."
            ),
        ],

        "cost_scenarios":
            COST_SCENARIOS,

        "fixed_point_bits":
            FIXED_POINT_BITS,

        "baseline_results":
            baseline_df[
                [
                    "method",
                    "validation_size",
                    "effective_true_probability",
                    "mean_false_positive_count",
                    "phase_oracle_te_proxy",
                    "phase_oracle_cnot_proxy",
                    "logical_qubits_proxy",
                    "quality_per_million_te",
                    "cost_relative_to_cheapest",
                    "cost_relative_to_M2",
                ]
            ].to_dict(
                orient="records"
            ),

        "automatic_findings":
            automatic_findings,

        "next_step_rule": {
            "if_M3_cost_close_to_M2":
                (
                    "M3 remains the preferred verifier because "
                    "its EXP09 selectivity improvement is obtained "
                    "without a large oracle-cost penalty."
                ),

            "if_M3_much_more_expensive":
                (
                    "M2 becomes the practical candidate because "
                    "its EXP09 marking quality is close to M3 "
                    "with a simpler score circuit."
                ),

            "if_relation_stage_dominates":
                (
                    "Future optimization should focus on shared "
                    "partial-decryption and counter accumulation "
                    "rather than score arithmetic."
                ),
        },
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
    # Console output
    # =================================================================

    print()
    print(
        "[Baseline reversible-oracle estimates]"
    )

    print()

    display_columns = [
        "method",
        "validation_size",
        "effective_true_probability",
        "mean_false_positive_count",
        "phase_oracle_te_proxy",
        "logical_qubits_proxy",
        "quality_per_million_te",
        "cost_relative_to_M2",
    ]

    print(
        baseline_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "[Ranking by marking quality]"
    )

    print(
        quality_ranking.to_string(
            index=False
        )
    )

    print()
    print(
        "[Ranking by effective probability per cost]"
    )

    print(
        cost_efficiency_ranking.to_string(
            index=False
        )
    )

    if (
        "M2_energy"
        in method_lookup.index
        and
        "M3_llr"
        in method_lookup.index
    ):

        m2 = (
            method_lookup.loc[
                "M2_energy"
            ]
        )

        m3 = (
            method_lookup.loc[
                "M3_llr"
            ]
        )

        print()
        print(
            "[M2 versus M3]"
        )

        print(
            f"M2 effective P(true) : "
            f"{m2['effective_true_probability']:.4f}"
        )

        print(
            f"M3 effective P(true) : "
            f"{m3['effective_true_probability']:.4f}"
        )

        print(
            f"M2 oracle-cost proxy : "
            f"{m2['phase_oracle_te_proxy']:.0f}"
        )

        print(
            f"M3 oracle-cost proxy : "
            f"{m3['phase_oracle_te_proxy']:.0f}"
        )

        print(
            f"M3 / M2 cost ratio   : "
            f"{m3['phase_oracle_te_proxy'] / m2['phase_oracle_te_proxy']:.3f}"
        )

        print(
            f"M3 - M2 P_eff        : "
            f"{m3['effective_true_probability'] - m2['effective_true_probability']:.4f}"
        )

    print()
    print(
        "[Important warning]"
    )

    print(
        "These are relative reversible-computation estimates, "
        "not fault-tolerant physical gate counts."
    )

    print(
        "qRAM/data-loading cost, surface-code overhead, routing, "
        "magic-state factories, and hardware error correction "
        "are not included."
    )

    print()
    print(
        "[Output files]"
    )

    print(
        f"All resource estimates : "
        f"{OUTPUT_RESOURCES_CSV}"
    )

    print(
        f"Baseline comparison     : "
        f"{OUTPUT_BASELINE_CSV}"
    )

    print(
        f"Summary                 : "
        f"{OUTPUT_SUMMARY_JSON}"
    )

    print(
        f"Oracle-cost figure      : "
        f"{FIG_TOTAL_COST}"
    )

    print(
        f"Cost-quality figure     : "
        f"{FIG_COST_QUALITY}"
    )

    print(
        f"Ancilla figure          : "
        f"{FIG_ANCILLA}"
    )

    print(
        f"Sensitivity figure      : "
        f"{FIG_SENSITIVITY}"
    )

    print()
    print(
        "[PASS] EXP10 completed successfully."
    )


if __name__ == "__main__":
    main()