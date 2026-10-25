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


EXP05_CSV = (
    RESULTS_DIR
    / "exp05_coherent_probabilities.csv"
)

EXP09_BEST_CSV = (
    RESULTS_DIR
    / "exp09_best_robust_points.csv"
)

EXP10_BASELINE_CSV = (
    RESULTS_DIR
    / "exp10_baseline_comparison.csv"
)

EXP12_AA_CSV = (
    RESULTS_DIR
    / "exp12_sector_amplification_cost.csv"
)

EXP12_RAW_CSV = (
    RESULTS_DIR
    / "exp12_raw_transform_cost.csv"
)


# =====================================================================
# Outputs
# =====================================================================

OUTPUT_SWEEP_CSV = (
    RESULTS_DIR
    / "exp13_end_to_end_sweep.csv"
)

OUTPUT_FINAL_CSV = (
    RESULTS_DIR
    / "exp13_final_quantum_benchmark.csv"
)

OUTPUT_CLASSICAL_CSV = (
    RESULTS_DIR
    / "exp13_classical_work_proxy.csv"
)

OUTPUT_SUMMARY_JSON = (
    RESULTS_DIR
    / "exp13_final_summary.json"
)


FIG_FINAL_COST = (
    FIGURES_DIR
    / "exp13_final_quantum_cost.png"
)

FIG_SCENARIO = (
    FIGURES_DIR
    / "exp13_scenario_sensitivity.png"
)

FIG_BREAKDOWN = (
    FIGURES_DIR
    / "exp13_cost_breakdown.png"
)

FIG_SUCCESS = (
    FIGURES_DIR
    / "exp13_success_probability.png"
)

FIG_ITERATIONS = (
    FIGURES_DIR
    / "exp13_optimal_iterations.png"
)


# =====================================================================
# Configuration
# =====================================================================

METHODS = [
    "M2_energy",
    "M3_llr",
]

SCENARIOS = [
    "optimistic",
    "baseline",
    "pessimistic",
]


KEY_QUBITS = 8
INDEX_QUBITS = 13

N_KEY_CANDIDATES = 256

N_CORRELATION_PAIRS = 8_192
N_CLASSICAL_BASELINE_PAIRS = 10_000

N_APPROXIMATIONS = 8


# Maximum outer QAA iterations.
#
# Large enough to include the raw-state regime where initial joint
# good probability is extremely small.
MAX_OUTER_QAA_ITERATIONS = 300


# Used only for comparison against the previously chosen near-pure
# correlation-state preparation.
FIXED_SECTOR_TARGET = 0.99


# =====================================================================
# Reversible reflection / control proxy
# =====================================================================

def multi_controlled_phase_te(
    n_controls: int,
) -> int:

    if n_controls <= 1:
        return 0

    return max(
        1,
        2 * n_controls - 3,
    )


# Outer good predicate:
#
#     data == 0^13
#     AND
#     V(key) == 1
#
# One verifier predicate bit + 13 zero-data controls.
JOINT_GOOD_CONTROL_TE = (
    multi_controlled_phase_te(
        INDEX_QUBITS + 1
    )
)


# Reflection around the initial key/index all-zero state.
OUTER_ZERO_REFLECTION_TE = (
    multi_controlled_phase_te(
        KEY_QUBITS + INDEX_QUBITS
    )
)


# =====================================================================
# Probability helpers
# =====================================================================

def qaa_probability(
    initial_good_probability: float,
    iterations: int,
) -> float:

    p = float(
        np.clip(
            initial_good_probability,
            0.0,
            1.0,
        )
    )

    if p <= 0.0:
        return 0.0

    if p >= 1.0:
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


# =====================================================================
# Load EXP05
# =====================================================================

def load_exp05_true_probability():

    if not EXP05_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP05_CSV}\n\n"
            "Run EXP05 first."
        )

    df = pd.read_csv(
        EXP05_CSV
    )

    df = (
        df
        .sort_values(
            "candidate"
        )
        .reset_index(
            drop=True
        )
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

    true_candidate = int(
        row[
            "candidate"
        ]
    )

    p_true_given_sector = float(
        row[
            "coherent_probability"
        ]
    )

    return (
        true_candidate,
        p_true_given_sector,
    )


# =====================================================================
# Load EXP09 marking performance
# =====================================================================

def load_marking_points():

    if not EXP09_BEST_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP09_BEST_CSV}\n\n"
            "Run EXP09 first."
        )

    df = pd.read_csv(
        EXP09_BEST_CSV
    )

    rows = []

    for method in METHODS:

        part = df[
            df[
                "method"
            ]
            == method
        ].copy()

        if len(
            part
        ) == 0:

            raise RuntimeError(
                f"No EXP09 result for {method}."
            )

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

        rows.append(
            best.to_dict()
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Load EXP10 oracle costs
# =====================================================================

def load_oracle_costs():

    if not EXP10_BASELINE_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP10_BASELINE_CSV}\n\n"
            "Run EXP10 first."
        )

    df = pd.read_csv(
        EXP10_BASELINE_CSV
    )

    df = df[
        df[
            "method"
        ].isin(
            METHODS
        )
    ].copy()

    if len(df) != len(METHODS):

        raise RuntimeError(
            "EXP10 baseline is missing M2 or M3."
        )

    return df


# =====================================================================
# Load EXP12
# =====================================================================

def load_exp12():

    if not EXP12_AA_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP12_AA_CSV}\n\n"
            "Run EXP12 first."
        )

    if not EXP12_RAW_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP12_RAW_CSV}\n\n"
            "Run EXP12 first."
        )

    aa_df = pd.read_csv(
        EXP12_AA_CSV
    )

    raw_df = pd.read_csv(
        EXP12_RAW_CSV
    )

    return (
        aa_df,
        raw_df,
    )


# =====================================================================
# Joint outer-good probability
# =====================================================================

def marked_mass_inside_sector(
    p_true_given_sector: float,
    p_true_given_marked: float,
) -> float:
    """
    EXP09 gives:

        P(true | marked)

    while EXP05 gives:

        P(true | zero-Walsh sector)

    Therefore representative marked mass inside the sector is

        P(marked | sector)
        approximately
        P(true | sector) / P(true | marked).

    This is the same representative approximation used in EXP11.
    """

    if p_true_given_marked <= 0.0:
        return 1.0

    value = (
        p_true_given_sector
        / p_true_given_marked
    )

    return float(
        np.clip(
            value,
            p_true_given_sector,
            1.0,
        )
    )


def expected_true_success_outer_qaa(
    sector_probability: float,
    marked_mass_sector: float,
    retention_probability: float,
    true_probability_given_marked: float,
    outer_iterations: int,
) -> float:
    """
    Outer good predicate:

        G =
        (data == 0^13)
        AND
        verifier(key)

    Initial good probability:

        p_G
        =
        P(sector)
        *
        P(marked | sector)

    Outer QAA amplifies the JOINT good sector.

    If the true key is retained by the verifier, its relative
    probability inside the marked sector is assumed unchanged.

    EXP09 retention is then included as an outer statistical factor.
    """

    initial_joint_good = (
        sector_probability
        * marked_mass_sector
    )

    amplified_joint_good = (
        qaa_probability(
            initial_joint_good,
            outer_iterations,
        )
    )

    expected_true_probability = (
        retention_probability
        * true_probability_given_marked
        * amplified_joint_good
    )

    return float(
        np.clip(
            expected_true_probability,
            0.0,
            1.0,
        )
    )


# =====================================================================
# End-to-end outer QAA cost
# =====================================================================

def outer_qaa_cost(
    state_prep_te: float,
    marking_oracle_te: float,
    outer_iterations: int,
):
    """
    Outer QAA:

        A
        [S_joint A^-1 S0 A]^t

    Hence:

        A-family calls       = 1 + 2t
        marking-oracle calls = t
        zero reflections     = t

    The EXP10 marking oracle computes and uncomputes V(k).
    We add the small extra multi-controlled phase required to
    combine V(k) with data == 0^13.
    """

    t = int(
        outer_iterations
    )

    state_prep_calls = (
        1 + 2 * t
    )

    marking_calls = t

    reflection_calls = t

    joint_marking_oracle_te = (
        marking_oracle_te
        + JOINT_GOOD_CONTROL_TE
    )

    state_prep_cost = (
        state_prep_calls
        * state_prep_te
    )

    marking_cost = (
        marking_calls
        * joint_marking_oracle_te
    )

    reflection_cost = (
        reflection_calls
        * OUTER_ZERO_REFLECTION_TE
    )

    total = (
        state_prep_cost
        + marking_cost
        + reflection_cost
    )

    return {
        "state_prep_calls":
            state_prep_calls,

        "marking_oracle_calls":
            marking_calls,

        "outer_reflection_calls":
            reflection_calls,

        "joint_marking_oracle_te":
            joint_marking_oracle_te,

        "state_prep_cost_component":
            state_prep_cost,

        "marking_cost_component":
            marking_cost,

        "reflection_cost_component":
            reflection_cost,

        "total_te_proxy":
            total,
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
    # Load all previous experiments
    # =================================================================

    (
        true_candidate,
        p_true_given_sector,
    ) = (
        load_exp05_true_probability()
    )

    marking_df = (
        load_marking_points()
    )

    oracle_df = (
        load_oracle_costs()
    )

    (
        exp12_df,
        raw_df,
    ) = (
        load_exp12()
    )

    combined = pd.merge(
        marking_df,
        oracle_df[
            [
                "method",
                "phase_oracle_te_proxy",
                "logical_qubits_proxy",
            ]
        ],
        on="method",
        how="inner",
    )

    method_lookup = (
        combined
        .set_index(
            "method"
        )
    )

    # =================================================================
    # Step 2
    # Build method parameters
    # =================================================================

    method_parameters = {}

    for method in METHODS:

        row = (
            method_lookup.loc[
                method
            ]
        )

        retention = float(
            row[
                "true_key_retention_probability"
            ]
        )

        true_given_marked = float(
            row[
                "mean_true_probability_given_marked"
            ]
        )

        marked_mass = (
            marked_mass_inside_sector(
                p_true_given_sector=
                    p_true_given_sector,

                p_true_given_marked=
                    true_given_marked,
            )
        )

        method_parameters[
            method
        ] = {
            "retention":
                retention,

            "true_given_marked":
                true_given_marked,

            "marked_mass_sector":
                marked_mass,

            "oracle_te":
                float(
                    row[
                        "phase_oracle_te_proxy"
                    ]
                ),

            "oracle_logical_qubits":
                int(
                    row[
                        "logical_qubits_proxy"
                    ]
                ),

            "validation_size":
                int(
                    row[
                        "validation_size"
                    ]
                ),

            "mean_false_positives":
                float(
                    row[
                        "mean_false_positive_count"
                    ]
                ),
        }

    # =================================================================
    # Step 3
    # Full joint inner-AA / outer-QAA sweep
    # =================================================================

    sweep_rows = []

    for scenario in SCENARIOS:

        sector_part = (
            exp12_df[
                exp12_df[
                    "scenario"
                ]
                == scenario
            ]
            .sort_values(
                "inner_aa_iterations"
            )
        )

        for method in METHODS:

            params = (
                method_parameters[
                    method
                ]
            )

            for _, state_row in (
                sector_part.iterrows()
            ):

                inner_t = int(
                    state_row[
                        "inner_aa_iterations"
                    ]
                )

                sector_probability = float(
                    state_row[
                        "zero_sector_probability"
                    ]
                )

                state_prep_te = float(
                    state_row[
                        "state_prep_te"
                    ]
                )

                raw_family_calls_per_A = int(
                    state_row[
                        "raw_transform_family_calls"
                    ]
                )

                for outer_t in range(
                    1,
                    MAX_OUTER_QAA_ITERATIONS + 1
                ):

                    true_success = (
                        expected_true_success_outer_qaa(
                            sector_probability=
                                sector_probability,

                            marked_mass_sector=
                                params[
                                    "marked_mass_sector"
                                ],

                            retention_probability=
                                params[
                                    "retention"
                                ],

                            true_probability_given_marked=
                                params[
                                    "true_given_marked"
                                ],

                            outer_iterations=
                                outer_t,
                        )
                    )

                    resources = (
                        outer_qaa_cost(
                            state_prep_te=
                                state_prep_te,

                            marking_oracle_te=
                                params[
                                    "oracle_te"
                                ],

                            outer_iterations=
                                outer_t,
                        )
                    )

                    expected_te_per_true = (
                        resources[
                            "total_te_proxy"
                        ]
                        / max(
                            true_success,
                            1e-15,
                        )
                    )

                    total_raw_transform_calls = (
                        resources[
                            "state_prep_calls"
                        ]
                        * raw_family_calls_per_A
                    )

                    total_qrom_calls = (
                        2
                        * total_raw_transform_calls
                    )

                    marking_relation_updates = (
                        resources[
                            "marking_oracle_calls"
                        ]
                        * params[
                            "validation_size"
                        ]
                        * N_APPROXIMATIONS
                    )

                    sweep_rows.append(
                        {
                            "scenario":
                                scenario,

                            "method":
                                method,

                            "inner_aa_iterations":
                                inner_t,

                            "outer_qaa_iterations":
                                outer_t,

                            "sector_probability":
                                sector_probability,

                            "state_prep_te":
                                state_prep_te,

                            "retention_probability":
                                params[
                                    "retention"
                                ],

                            "true_probability_given_marked":
                                params[
                                    "true_given_marked"
                                ],

                            "marked_mass_inside_sector":
                                params[
                                    "marked_mass_sector"
                                ],

                            "expected_true_key_probability":
                                true_success,

                            "state_prep_calls":
                                resources[
                                    "state_prep_calls"
                                ],

                            "marking_oracle_calls":
                                resources[
                                    "marking_oracle_calls"
                                ],

                            "outer_reflection_calls":
                                resources[
                                    "outer_reflection_calls"
                                ],

                            "state_prep_cost_component":
                                resources[
                                    "state_prep_cost_component"
                                ],

                            "marking_cost_component":
                                resources[
                                    "marking_cost_component"
                                ],

                            "reflection_cost_component":
                                resources[
                                    "reflection_cost_component"
                                ],

                            "total_te_proxy":
                                resources[
                                    "total_te_proxy"
                                ],

                            "expected_te_per_true_key":
                                expected_te_per_true,

                            "raw_transform_family_calls_per_A":
                                raw_family_calls_per_A,

                            "total_raw_transform_calls":
                                total_raw_transform_calls,

                            "total_qrom_calls":
                                total_qrom_calls,

                            "marking_relation_updates":
                                marking_relation_updates,
                        }
                    )

    sweep_df = pd.DataFrame(
        sweep_rows
    )

    sweep_df.to_csv(
        OUTPUT_SWEEP_CSV,
        index=False,
    )

    # =================================================================
    # Step 4
    # Quantum baseline strategies
    # =================================================================

    final_rows = []

    for scenario in SCENARIOS:

        sector_part = (
            exp12_df[
                exp12_df[
                    "scenario"
                ]
                == scenario
            ]
            .sort_values(
                "inner_aa_iterations"
            )
        )

        raw_row = (
            sector_part[
                sector_part[
                    "inner_aa_iterations"
                ]
                == 0
            ]
            .iloc[0]
        )

        # -------------------------------------------------------------
        # Strategy A:
        # raw transform + literal postselection
        # -------------------------------------------------------------

        raw_sector_probability = float(
            raw_row[
                "zero_sector_probability"
            ]
        )

        raw_state_cost = float(
            raw_row[
                "state_prep_te"
            ]
        )

        raw_true_probability = (
            raw_sector_probability
            * p_true_given_sector
        )

        raw_expected_cost = (
            raw_state_cost
            / raw_true_probability
        )

        final_rows.append(
            {
                "scenario":
                    scenario,

                "strategy":
                    "raw_postselection",

                "method":
                    "none",

                "inner_aa_iterations":
                    0,

                "outer_qaa_iterations":
                    0,

                "sector_probability":
                    raw_sector_probability,

                "expected_true_key_probability":
                    raw_true_probability,

                "expected_te_per_true_key":
                    raw_expected_cost,

                "state_prep_te":
                    raw_state_cost,

                "total_te_per_execution":
                    raw_state_cost,

                "state_prep_calls":
                    1,

                "marking_oracle_calls":
                    0,

                "state_prep_cost_component":
                    raw_state_cost,

                "marking_cost_component":
                    0.0,

                "reflection_cost_component":
                    0.0,
            }
        )

        # -------------------------------------------------------------
        # Strategy B:
        # optimally prepared correlation sector, then direct sampling
        # -------------------------------------------------------------

        direct_rows = []

        for _, row in (
            sector_part.iterrows()
        ):

            sector_probability = float(
                row[
                    "zero_sector_probability"
                ]
            )

            state_prep_te = float(
                row[
                    "state_prep_te"
                ]
            )

            true_probability = (
                sector_probability
                * p_true_given_sector
            )

            cost_per_true = (
                state_prep_te
                / max(
                    true_probability,
                    1e-15,
                )
            )

            direct_rows.append(
                {
                    "inner_t":
                        int(
                            row[
                                "inner_aa_iterations"
                            ]
                        ),

                    "sector_probability":
                        sector_probability,

                    "state_prep_te":
                        state_prep_te,

                    "true_probability":
                        true_probability,

                    "cost_per_true":
                        cost_per_true,
                }
            )

        direct_df = pd.DataFrame(
            direct_rows
        )

        best_direct = (
            direct_df
            .sort_values(
                "cost_per_true"
            )
            .iloc[0]
        )

        final_rows.append(
            {
                "scenario":
                    scenario,

                "strategy":
                    "prepared_direct_sampling",

                "method":
                    "none",

                "inner_aa_iterations":
                    int(
                        best_direct[
                            "inner_t"
                        ]
                    ),

                "outer_qaa_iterations":
                    0,

                "sector_probability":
                    float(
                        best_direct[
                            "sector_probability"
                        ]
                    ),

                "expected_true_key_probability":
                    float(
                        best_direct[
                            "true_probability"
                        ]
                    ),

                "expected_te_per_true_key":
                    float(
                        best_direct[
                            "cost_per_true"
                        ]
                    ),

                "state_prep_te":
                    float(
                        best_direct[
                            "state_prep_te"
                        ]
                    ),

                "total_te_per_execution":
                    float(
                        best_direct[
                            "state_prep_te"
                        ]
                    ),

                "state_prep_calls":
                    1,

                "marking_oracle_calls":
                    0,

                "state_prep_cost_component":
                    float(
                        best_direct[
                            "state_prep_te"
                        ]
                    ),

                "marking_cost_component":
                    0.0,

                "reflection_cost_component":
                    0.0,
            }
        )

        # -------------------------------------------------------------
        # Strategy C/D:
        # globally optimized nested quantum attack
        # -------------------------------------------------------------

        for method in METHODS:

            part = sweep_df[
                (
                    sweep_df[
                        "scenario"
                    ]
                    == scenario
                )
                &
                (
                    sweep_df[
                        "method"
                    ]
                    == method
                )
            ]

            best = (
                part
                .sort_values(
                    "expected_te_per_true_key"
                )
                .iloc[0]
            )

            final_rows.append(
                {
                    "scenario":
                        scenario,

                    "strategy":
                        f"{method}_joint_QAA",

                    "method":
                        method,

                    "inner_aa_iterations":
                        int(
                            best[
                                "inner_aa_iterations"
                            ]
                        ),

                    "outer_qaa_iterations":
                        int(
                            best[
                                "outer_qaa_iterations"
                            ]
                        ),

                    "sector_probability":
                        float(
                            best[
                                "sector_probability"
                            ]
                        ),

                    "expected_true_key_probability":
                        float(
                            best[
                                "expected_true_key_probability"
                            ]
                        ),

                    "expected_te_per_true_key":
                        float(
                            best[
                                "expected_te_per_true_key"
                            ]
                        ),

                    "state_prep_te":
                        float(
                            best[
                                "state_prep_te"
                            ]
                        ),

                    "total_te_per_execution":
                        float(
                            best[
                                "total_te_proxy"
                            ]
                        ),

                    "state_prep_calls":
                        int(
                            best[
                                "state_prep_calls"
                            ]
                        ),

                    "marking_oracle_calls":
                        int(
                            best[
                                "marking_oracle_calls"
                            ]
                        ),

                    "state_prep_cost_component":
                        float(
                            best[
                                "state_prep_cost_component"
                            ]
                        ),

                    "marking_cost_component":
                        float(
                            best[
                                "marking_cost_component"
                            ]
                        ),

                    "reflection_cost_component":
                        float(
                            best[
                                "reflection_cost_component"
                            ]
                        ),
                }
            )

        # -------------------------------------------------------------
        # Fixed-99% correlation-state preparation comparison
        # -------------------------------------------------------------

        fixed_rows = sector_part[
            sector_part[
                "zero_sector_probability"
            ]
            >= FIXED_SECTOR_TARGET
        ]

        if len(
            fixed_rows
        ) > 0:

            fixed_state = (
                fixed_rows
                .sort_values(
                    "inner_aa_iterations"
                )
                .iloc[0]
            )

            fixed_inner_t = int(
                fixed_state[
                    "inner_aa_iterations"
                ]
            )

            for method in METHODS:

                fixed_part = sweep_df[
                    (
                        sweep_df[
                            "scenario"
                        ]
                        == scenario
                    )
                    &
                    (
                        sweep_df[
                            "method"
                        ]
                        == method
                    )
                    &
                    (
                        sweep_df[
                            "inner_aa_iterations"
                        ]
                        == fixed_inner_t
                    )
                ]

                best_fixed = (
                    fixed_part
                    .sort_values(
                        "expected_te_per_true_key"
                    )
                    .iloc[0]
                )

                final_rows.append(
                    {
                        "scenario":
                            scenario,

                        "strategy":
                            f"{method}_fixed99_QAA",

                        "method":
                            method,

                        "inner_aa_iterations":
                            int(
                                best_fixed[
                                    "inner_aa_iterations"
                                ]
                            ),

                        "outer_qaa_iterations":
                            int(
                                best_fixed[
                                    "outer_qaa_iterations"
                                ]
                            ),

                        "sector_probability":
                            float(
                                best_fixed[
                                    "sector_probability"
                                ]
                            ),

                        "expected_true_key_probability":
                            float(
                                best_fixed[
                                    "expected_true_key_probability"
                                ]
                            ),

                        "expected_te_per_true_key":
                            float(
                                best_fixed[
                                    "expected_te_per_true_key"
                                ]
                            ),

                        "state_prep_te":
                            float(
                                best_fixed[
                                    "state_prep_te"
                                ]
                            ),

                        "total_te_per_execution":
                            float(
                                best_fixed[
                                    "total_te_proxy"
                                ]
                            ),

                        "state_prep_calls":
                            int(
                                best_fixed[
                                    "state_prep_calls"
                                ]
                            ),

                        "marking_oracle_calls":
                            int(
                                best_fixed[
                                    "marking_oracle_calls"
                                ]
                            ),

                        "state_prep_cost_component":
                            float(
                                best_fixed[
                                    "state_prep_cost_component"
                                ]
                            ),

                        "marking_cost_component":
                            float(
                                best_fixed[
                                    "marking_cost_component"
                                ]
                            ),

                        "reflection_cost_component":
                            float(
                                best_fixed[
                                    "reflection_cost_component"
                                ]
                            ),
                    }
                )

    final_df = pd.DataFrame(
        final_rows
    )

    final_df.to_csv(
        OUTPUT_FINAL_CSV,
        index=False,
    )

    # =================================================================
    # Step 5
    # Logical-qubit proxy
    # =================================================================

    raw_qubit_lookup = (
        raw_df
        .set_index(
            "scenario"
        )
    )

    for method in METHODS:

        marking_qubits = int(
            method_parameters[
                method
            ][
                "oracle_logical_qubits"
            ]
        )

        # EXP10 marking cost already includes 8 key qubits,
        # but not the EXP05 13-qubit Walsh index register.
        combined_qubits = (
            marking_qubits
            + INDEX_QUBITS
            + 1
        )

        method_parameters[
            method
        ][
            "combined_logical_qubits_proxy"
        ] = combined_qubits

    # =================================================================
    # Step 6
    # Classical work proxy
    # =================================================================

    classical_candidate_sample_evals = (
        N_KEY_CANDIDATES
        * N_CLASSICAL_BASELINE_PAIRS
    )

    classical_rows = [
        {
            "strategy":
                "EXP01 classical linear key ranking",

            "known_plaintext_ciphertext_pairs":
                N_CLASSICAL_BASELINE_PAIRS,

            "key_candidates":
                N_KEY_CANDIDATES,

            "candidate_sample_relation_evaluations":
                classical_candidate_sample_evals,

            "cost_unit":
                (
                    "classical candidate-sample "
                    "relation evaluations"
                ),

            "directly_comparable_to_TE":
                False,

            "note":
                (
                    "This work unit is intentionally kept "
                    "separate from quantum TE proxies."
                ),
        }
    ]

    # Add baseline quantum coherent-work summaries.
    for method in METHODS:

        row = final_df[
            (
                final_df[
                    "scenario"
                ]
                == "baseline"
            )
            &
            (
                final_df[
                    "strategy"
                ]
                == f"{method}_joint_QAA"
            )
        ].iloc[0]

        inner_t = int(
            row[
                "inner_aa_iterations"
            ]
        )

        outer_t = int(
            row[
                "outer_qaa_iterations"
            ]
        )

        state_prep_calls = (
            1
            + 2 * outer_t
        )

        raw_transform_calls_per_A = (
            1
            + 2 * inner_t
        )

        total_raw_transform_calls = (
            state_prep_calls
            * raw_transform_calls_per_A
        )

        total_qrom_calls = (
            2
            * total_raw_transform_calls
        )

        validation_size = (
            method_parameters[
                method
            ][
                "validation_size"
            ]
        )

        marking_relation_updates = (
            outer_t
            * validation_size
            * N_APPROXIMATIONS
        )

        classical_rows.append(
            {
                "strategy":
                    f"EXP13 {method} optimized quantum attack",

                "known_plaintext_ciphertext_pairs":
                    (
                        N_CORRELATION_PAIRS
                        + validation_size
                    ),

                "key_candidates":
                    N_KEY_CANDIDATES,

                "candidate_sample_relation_evaluations":
                    np.nan,

                "cost_unit":
                    (
                        "coherent QROM/oracle work; "
                        "see TE proxy separately"
                    ),

                "directly_comparable_to_TE":
                    False,

                "note":
                    (
                        f"Raw-transform calls={total_raw_transform_calls}, "
                        f"QROM calls={total_qrom_calls}, "
                        f"validation relation updates="
                        f"{marking_relation_updates}."
                    ),
            }
        )

    classical_df = pd.DataFrame(
        classical_rows
    )

    classical_df.to_csv(
        OUTPUT_CLASSICAL_CSV,
        index=False,
    )

    # =================================================================
    # Step 7
    # Figure: final baseline quantum cost
    # =================================================================

    main_strategies = [
        "raw_postselection",
        "prepared_direct_sampling",
        "M2_energy_joint_QAA",
        "M3_llr_joint_QAA",
    ]

    baseline_main = final_df[
        (
            final_df[
                "scenario"
            ]
            == "baseline"
        )
        &
        (
            final_df[
                "strategy"
            ].isin(
                main_strategies
            )
        )
    ].copy()

    baseline_main[
        "strategy"
    ] = pd.Categorical(
        baseline_main[
            "strategy"
        ],
        categories=
            main_strategies,
        ordered=True,
    )

    baseline_main = (
        baseline_main
        .sort_values(
            "strategy"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        baseline_main[
            "strategy"
        ].astype(str),
        baseline_main[
            "expected_te_per_true_key"
        ],
    )

    ax.set_yscale(
        "log"
    )

    ax.set_ylabel(
        "Expected TE proxy per true-key output"
    )

    ax.set_title(
        "EXP13: Final End-to-End Quantum Cost"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_FINAL_COST,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 8
    # Figure: QROM scenario sensitivity
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    strategies = [
        "raw_postselection",
        "prepared_direct_sampling",
        "M2_energy_joint_QAA",
        "M3_llr_joint_QAA",
    ]

    for strategy in strategies:

        part = []

        for scenario in SCENARIOS:

            row = final_df[
                (
                    final_df[
                        "scenario"
                    ]
                    == scenario
                )
                &
                (
                    final_df[
                        "strategy"
                    ]
                    == strategy
                )
            ]

            if len(
                row
            ) == 0:

                continue

            part.append(
                float(
                    row.iloc[0][
                        "expected_te_per_true_key"
                    ]
                )
            )

        if len(
            part
        ) == len(
            SCENARIOS
        ):

            ax.plot(
                SCENARIOS,
                part,
                marker="o",
                label=strategy,
            )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "QROM cost assumption"
    )

    ax.set_ylabel(
        "Expected TE proxy per true-key output"
    )

    ax.set_title(
        "EXP13: QROM-Cost Sensitivity"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_SCENARIO,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 9
    # Figure: baseline cost breakdown
    # =================================================================

    qaa_baseline = final_df[
        (
            final_df[
                "scenario"
            ]
            == "baseline"
        )
        &
        (
            final_df[
                "strategy"
            ].isin(
                [
                    "M2_energy_joint_QAA",
                    "M3_llr_joint_QAA",
                ]
            )
        )
    ].copy()

    x = np.arange(
        len(
            qaa_baseline
        )
    )

    state_cost = (
        qaa_baseline[
            "state_prep_cost_component"
        ].to_numpy()
    )

    mark_cost = (
        qaa_baseline[
            "marking_cost_component"
        ].to_numpy()
    )

    reflection_cost = (
        qaa_baseline[
            "reflection_cost_component"
        ].to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x,
        state_cost,
        label="State preparation",
    )

    ax.bar(
        x,
        mark_cost,
        bottom=state_cost,
        label="Marking oracle",
    )

    ax.bar(
        x,
        reflection_cost,
        bottom=(
            state_cost
            + mark_cost
        ),
        label="Outer reflection",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        qaa_baseline[
            "method"
        ]
    )

    ax.set_ylabel(
        "TE proxy per QAA execution"
    )

    ax.set_title(
        "EXP13: Cost Breakdown at Joint Optimum"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_BREAKDOWN,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 10
    # Figure: success probability
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        baseline_main[
            "strategy"
        ].astype(str),
        baseline_main[
            "expected_true_key_probability"
        ],
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_ylabel(
        "True-key probability per execution"
    )

    ax.set_title(
        "EXP13: True-Key Probability per Execution"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        FIG_SUCCESS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 11
    # Figure: jointly optimized inner/outer iteration counts
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = np.arange(
        len(
            qaa_baseline
        )
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        qaa_baseline[
            "inner_aa_iterations"
        ],
        width,
        label="Inner sector-AA iterations",
    )

    ax.bar(
        x + width / 2,
        qaa_baseline[
            "outer_qaa_iterations"
        ],
        width,
        label="Outer QAA iterations",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        qaa_baseline[
            "method"
        ]
    )

    ax.set_ylabel(
        "Iteration count"
    )

    ax.set_title(
        "EXP13: Jointly Cost-Optimal AA Iterations"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_ITERATIONS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Step 12
    # Final numerical summary
    # =================================================================

    final_baseline = final_df[
        final_df[
            "scenario"
        ]
        == "baseline"
    ].copy()

    global_qaa = final_baseline[
        final_baseline[
            "strategy"
        ].isin(
            [
                "M2_energy_joint_QAA",
                "M3_llr_joint_QAA",
            ]
        )
    ]

    best_quantum = (
        global_qaa
        .sort_values(
            "expected_te_per_true_key"
        )
        .iloc[0]
    )

    direct_row = final_baseline[
        final_baseline[
            "strategy"
        ]
        == "prepared_direct_sampling"
    ].iloc[0]

    raw_row = final_baseline[
        final_baseline[
            "strategy"
        ]
        == "raw_postselection"
    ].iloc[0]

    fixed99_comparison = {}

    for method in METHODS:

        global_row = final_baseline[
            final_baseline[
                "strategy"
            ]
            == f"{method}_joint_QAA"
        ]

        fixed_row = final_baseline[
            final_baseline[
                "strategy"
            ]
            == f"{method}_fixed99_QAA"
        ]

        if (
            len(global_row) == 1
            and len(fixed_row) == 1
        ):

            global_cost = float(
                global_row.iloc[0][
                    "expected_te_per_true_key"
                ]
            )

            fixed_cost = float(
                fixed_row.iloc[0][
                    "expected_te_per_true_key"
                ]
            )

            fixed99_comparison[
                method
            ] = {
                "global_joint_optimum_te_per_true":
                    global_cost,

                "fixed_99_percent_te_per_true":
                    fixed_cost,

                "global_over_fixed99":
                    (
                        global_cost
                        / fixed_cost
                    ),

                "cost_reduction_from_joint_optimization":
                    (
                        1.0
                        -
                        global_cost
                        / fixed_cost
                    ),
            }

    summary = {
        "experiment":
            "exp13_final_end_to_end_benchmark",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "P_true_given_zero_walsh_sector":
            p_true_given_sector,

        "outer_good_predicate":
            (
                "(data register == 0^13) AND verifier(key)"
            ),

        "baseline_best_quantum_strategy": {
            "strategy":
                str(
                    best_quantum[
                        "strategy"
                    ]
                ),

            "method":
                str(
                    best_quantum[
                        "method"
                    ]
                ),

            "inner_aa_iterations":
                int(
                    best_quantum[
                        "inner_aa_iterations"
                    ]
                ),

            "outer_qaa_iterations":
                int(
                    best_quantum[
                        "outer_qaa_iterations"
                    ]
                ),

            "sector_probability":
                float(
                    best_quantum[
                        "sector_probability"
                    ]
                ),

            "true_key_probability_per_execution":
                float(
                    best_quantum[
                        "expected_true_key_probability"
                    ]
                ),

            "TE_per_execution":
                float(
                    best_quantum[
                        "total_te_per_execution"
                    ]
                ),

            "expected_TE_per_true_key":
                float(
                    best_quantum[
                        "expected_te_per_true_key"
                    ]
                ),
        },

        "baseline_raw_postselection": {
            "expected_TE_per_true_key":
                float(
                    raw_row[
                        "expected_te_per_true_key"
                    ]
                ),
        },

        "baseline_prepared_direct_sampling": {
            "inner_aa_iterations":
                int(
                    direct_row[
                        "inner_aa_iterations"
                    ]
                ),

            "sector_probability":
                float(
                    direct_row[
                        "sector_probability"
                    ]
                ),

            "expected_TE_per_true_key":
                float(
                    direct_row[
                        "expected_te_per_true_key"
                    ]
                ),
        },

        "best_QAA_over_prepared_direct_cost_ratio":
            (
                float(
                    best_quantum[
                        "expected_te_per_true_key"
                    ]
                )
                /
                float(
                    direct_row[
                        "expected_te_per_true_key"
                    ]
                )
            ),

        "best_QAA_over_raw_postselection_cost_ratio":
            (
                float(
                    best_quantum[
                        "expected_te_per_true_key"
                    ]
                )
                /
                float(
                    raw_row[
                        "expected_te_per_true_key"
                    ]
                )
            ),

        "fixed99_vs_joint_optimization":
            fixed99_comparison,

        "classical_baseline": {
            "known_PC_pairs":
                N_CLASSICAL_BASELINE_PAIRS,

            "candidate_sample_relation_evaluations":
                classical_candidate_sample_evals,

            "direct_TE_comparison_valid":
                False,
        },

        "final_interpretation_rules": [
            (
                "The quantum strategies are compared in a common "
                "logical Toffoli-equivalent proxy inherited from "
                "EXP10 and EXP12."
            ),
            (
                "The classical baseline is NOT converted into the "
                "same TE unit. Therefore EXP13 does not claim a "
                "classical-versus-quantum runtime speedup."
            ),
            (
                "Inner sector amplification and outer key-marking "
                "amplitude amplification are jointly optimized."
            ),
            (
                "The outer good predicate explicitly includes "
                "data=0^13, so residual non-zero-Walsh amplitude "
                "cannot be incorrectly counted as useful key signal."
            ),
            (
                "If the joint optimum uses substantially fewer than "
                "39 inner-AA iterations, then near-perfect correlation "
                "sector preparation is not globally cost-optimal."
            ),
            (
                "The dominant resource should be identified from the "
                "state-preparation versus marking-oracle cost breakdown."
            ),
        ],

        "limitations": [
            (
                "QROM and reversible-arithmetic costs remain analytical "
                "logical resource models rather than fault-tolerant "
                "hardware syntheses."
            ),
            (
                "No surface-code, routing, magic-state factory, clock "
                "rate, or hardware error model is included."
            ),
            (
                "EXP09 marking retention and conditional marked-state "
                "probabilities are statistical averages over validation "
                "subsets."
            ),
            (
                "The classical and quantum resource units remain "
                "different, preventing a defensible absolute speedup "
                "claim."
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
    print("=" * 84)
    print(
        "EXP13: Final End-to-End Quantum Correlation-Attack Benchmark"
    )
    print("=" * 84)

    print()
    print(
        f"True key                           : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"P(true | zero-Walsh sector)        : "
        f"{p_true_given_sector:.6f} "
        f"({100 * p_true_given_sector:.3f}%)"
    )

    print()
    print(
        "[Marking parameters]"
    )

    for method in METHODS:

        p = (
            method_parameters[
                method
            ]
        )

        print()

        print(
            method
        )

        print(
            f"  retention                       : "
            f"{p['retention']:.4f}"
        )

        print(
            f"  P(true | marked)                : "
            f"{p['true_given_marked']:.4f}"
        )

        print(
            f"  P(marked | zero sector)         : "
            f"{p['marked_mass_sector']:.6f}"
        )

        print(
            f"  marking oracle TE               : "
            f"{p['oracle_te']:.0f}"
        )

    print()
    print(
        "[Final baseline-scenario strategies]"
    )

    display_columns = [
        "strategy",
        "inner_aa_iterations",
        "outer_qaa_iterations",
        "sector_probability",
        "expected_true_key_probability",
        "total_te_per_execution",
        "expected_te_per_true_key",
    ]

    print(
        final_baseline[
            display_columns
        ]
        .sort_values(
            "expected_te_per_true_key"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "[Best final quantum strategy]"
    )

    print(
        f"Strategy                           : "
        f"{best_quantum['strategy']}"
    )

    print(
        f"Inner AA iterations                : "
        f"{int(best_quantum['inner_aa_iterations'])}"
    )

    print(
        f"Outer QAA iterations               : "
        f"{int(best_quantum['outer_qaa_iterations'])}"
    )

    print(
        f"Sector probability                 : "
        f"{best_quantum['sector_probability']:.6f}"
    )

    print(
        f"P(true) per execution              : "
        f"{best_quantum['expected_true_key_probability']:.6f}"
    )

    print(
        f"TE per execution                   : "
        f"{best_quantum['total_te_per_execution']:.0f}"
    )

    print(
        f"Expected TE per true key           : "
        f"{best_quantum['expected_te_per_true_key']:.0f}"
    )

    print()
    print(
        "[Quantum-only improvement ratios]"
    )

    print(
        f"Best QAA / prepared direct         : "
        f"{summary['best_QAA_over_prepared_direct_cost_ratio']:.4f}"
    )

    print(
        f"Best QAA / raw postselection       : "
        f"{summary['best_QAA_over_raw_postselection_cost_ratio']:.4f}"
    )

    print()
    print(
        "[Classical baseline]"
    )

    print(
        f"Known P/C pairs                    : "
        f"{N_CLASSICAL_BASELINE_PAIRS}"
    )

    print(
        f"Candidate-sample evaluations       : "
        f"{classical_candidate_sample_evals:,}"
    )

    print()
    print(
        "IMPORTANT: classical candidate-sample evaluations and "
        "quantum TE proxies are different units."
    )

    print(
        "EXP13 therefore does NOT report an unsupported numerical "
        "classical-versus-quantum speedup."
    )

    print()
    print(
        "[Output files]"
    )

    print(
        f"Full inner/outer sweep             : "
        f"{OUTPUT_SWEEP_CSV}"
    )

    print(
        f"Final quantum benchmark            : "
        f"{OUTPUT_FINAL_CSV}"
    )

    print(
        f"Classical work proxy               : "
        f"{OUTPUT_CLASSICAL_CSV}"
    )

    print(
        f"Final summary                      : "
        f"{OUTPUT_SUMMARY_JSON}"
    )

    print(
        f"Final quantum-cost figure          : "
        f"{FIG_FINAL_COST}"
    )

    print(
        f"Scenario-sensitivity figure        : "
        f"{FIG_SCENARIO}"
    )

    print(
        f"Cost-breakdown figure              : "
        f"{FIG_BREAKDOWN}"
    )

    print(
        f"Success-probability figure         : "
        f"{FIG_SUCCESS}"
    )

    print(
        f"Optimal-iterations figure          : "
        f"{FIG_ITERATIONS}"
    )

    print()
    print(
        "[PASS] EXP13 completed successfully."
    )


if __name__ == "__main__":
    main()