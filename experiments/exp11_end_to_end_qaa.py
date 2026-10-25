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


# =====================================================================
# Output files
# =====================================================================

OUTPUT_SWEEP_CSV = (
    RESULTS_DIR
    / "exp11_qaa_end_to_end_sweep.csv"
)

OUTPUT_BEST_CSV = (
    RESULTS_DIR
    / "exp11_best_qaa_points.csv"
)

OUTPUT_BREAK_EVEN_CSV = (
    RESULTS_DIR
    / "exp11_break_even.csv"
)

OUTPUT_WORK_CSV = (
    RESULTS_DIR
    / "exp11_work_proxies.csv"
)

OUTPUT_SUMMARY_JSON = (
    RESULTS_DIR
    / "exp11_summary.json"
)


FIG_SUCCESS = (
    FIGURES_DIR
    / "exp11_qaa_success_vs_iterations.png"
)

FIG_COST_PREP = (
    FIGURES_DIR
    / "exp11_cost_vs_state_prep.png"
)

FIG_OPTIMAL_T = (
    FIGURES_DIR
    / "exp11_optimal_iterations.png"
)

FIG_ORACLE_CALLS = (
    FIGURES_DIR
    / "exp11_qaa_resource_calls.png"
)


# =====================================================================
# Configuration
# =====================================================================

# Main practical candidates from EXP09/10.
METHODS = [
    "M2_energy",
    "M3_llr",
]

# Maximum QAA iterations to inspect.
MAX_QAA_ITERATIONS = 30

# Cost of the reflection S_0 around the reference |0...0> state.
#
# This is deliberately small relative to ~5e5 oracle proxy units.
# It has negligible effect on the present conclusions.
STATE_REFLECTION_TE = 15

MEASUREMENT_TE = 0


# ---------------------------------------------------------------------
# State-preparation cost sweep
#
# We do NOT yet know the true coherent cost C_A of preparing the
# EXP05 correlation state.
#
# Therefore C_A is swept relative to the baseline M3 oracle cost.
#
# Example:
#
#   multiplier = 0.25
#
# means:
#
#   C_A = 0.25 * C_oracle(M3)
# ---------------------------------------------------------------------

STATE_PREP_MULTIPLIERS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
    0.25,
    0.50,
    1.00,
    2.00,
    5.00,
]


# Dense grid used to estimate the QAA/direct-sampling break-even point.
BREAK_EVEN_MULTIPLIERS = np.geomspace(
    1e-3,
    10.0,
    600,
)


# =====================================================================
# Existing data-budget information
# =====================================================================

N_KEY_CANDIDATES = 256

# EXP01 classical attack.
N_CLASSICAL_BASELINE_PAIRS = 10_000

# EXP05 correlation-state construction.
N_CORRELATION_STATE_PAIRS = 8_192


# =====================================================================
# Utilities
# =====================================================================

def qaa_probability(
    initial_good_probability: float,
    iterations: int,
) -> float:
    """
    Standard amplitude-amplification probability:

        p_t = sin^2((2t+1) theta)

    with

        sin^2(theta) = p_good.
    """

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


def load_initial_true_probability():
    """
    Read EXP05 correlation-state true-key probability.
    """

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

    p_true = float(
        row[
            "coherent_probability"
        ]
    )

    return (
        true_candidate,
        p_true,
    )


def load_exp09_points():
    """
    Load the best robust M2/M3 marking operating points.
    """

    if not EXP09_BEST_CSV.exists():

        raise FileNotFoundError(
            f"Missing:\n{EXP09_BEST_CSV}\n\n"
            "Run EXP09 first."
        )

    df = pd.read_csv(
        EXP09_BEST_CSV
    )

    selected = []

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

        selected.append(
            best.to_dict()
        )

    return pd.DataFrame(
        selected
    )


def load_exp10_oracle_costs():
    """
    Load baseline reversible phase-oracle cost proxies.
    """

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
            "Could not find both M2 and M3 "
            "in EXP10 baseline results."
        )

    return df


# =====================================================================
# Representative marked-sector estimate
# =====================================================================

def estimate_marked_mass_when_true_retained(
    p_true_initial: float,
    p_true_given_marked: float,
) -> float:
    """
    If the true key is marked,

        P(true | marked)
        =
        P(true) / P(marked)

    therefore approximately

        P(marked)
        =
        P(true) / P(true | marked).

    EXP09 reports an average conditional probability, so this is
    an approximate representative marked-sector mass rather than
    an exact trial-by-trial quantity.
    """

    if p_true_given_marked <= 0.0:

        return 1.0

    p_marked = (
        p_true_initial
        / p_true_given_marked
    )

    return float(
        np.clip(
            p_marked,
            p_true_initial,
            1.0,
        )
    )


# =====================================================================
# QAA cost
# =====================================================================

def qaa_total_cost(
    iterations: int,
    state_prep_te: float,
    marking_oracle_te: float,
) -> float:
    """
    Standard QAA structure:

        A
        [ S_f, A^-1, S_0, A ]^t
        measure

    where

        A     : correlation-state preparation
        S_f   : marking phase oracle
        S_0   : reflection about |0...0>

    Thus:

        state-preparation-family calls = 1 + 2t
        marking-oracle calls           = t
        S0 reflections                 = t
    """

    t = int(
        iterations
    )

    state_prep_calls = (
        1 + 2 * t
    )

    oracle_calls = t

    reflection_calls = t

    return (
        state_prep_calls
        * state_prep_te
        +
        oracle_calls
        * marking_oracle_te
        +
        reflection_calls
        * STATE_REFLECTION_TE
        +
        MEASUREMENT_TE
    )


def expected_true_success_after_qaa(
    initial_marked_mass: float,
    retention_probability: float,
    true_given_marked: float,
    iterations: int,
) -> float:
    """
    Conservative benchmark model.

    If the verifier retains the true key:

        P_true_after_QAA
        =
        P(true | marked)
        *
        P(marked after QAA)

    EXP09 verifier retention is then included:

        expected P_true
        approximately
        P_retention
        *
        P(true | marked)
        *
        P_marked_QAA

    Trials where the verifier rejects the true key are assigned
    zero success here.

    This is conservative and avoids pretending that a failed
    verifier can still be rescued by QAA.
    """

    amplified_marked_probability = (
        qaa_probability(
            initial_marked_mass,
            iterations,
        )
    )

    probability = (
        retention_probability
        * true_given_marked
        * amplified_marked_probability
    )

    return float(
        np.clip(
            probability,
            0.0,
            1.0,
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

    # =================================================================
    # Step 1
    # Load previous experiment results
    # =================================================================

    (
        true_candidate,
        p_true_initial,
    ) = (
        load_initial_true_probability()
    )

    exp09_df = (
        load_exp09_points()
    )

    exp10_df = (
        load_exp10_oracle_costs()
    )

    combined = pd.merge(
        exp09_df,
        exp10_df[
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

    m3_oracle_cost = float(
        method_lookup.loc[
            "M3_llr",
            "phase_oracle_te_proxy",
        ]
    )

    # =================================================================
    # Step 2
    # Derive representative marked-sector probabilities
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

        oracle_cost = float(
            row[
                "phase_oracle_te_proxy"
            ]
        )

        p_marked = (
            estimate_marked_mass_when_true_retained(
                p_true_initial=
                    p_true_initial,

                p_true_given_marked=
                    true_given_marked,
            )
        )

        theta = math.asin(
            math.sqrt(
                p_marked
            )
        )

        method_parameters[
            method
        ] = {
            "retention":
                retention,

            "true_given_marked":
                true_given_marked,

            "marked_mass":
                p_marked,

            "theta":
                theta,

            "oracle_cost":
                oracle_cost,

            "logical_qubits":
                int(
                    row[
                        "logical_qubits_proxy"
                    ]
                ),
        }

    # =================================================================
    # Step 3
    # QAA success curve independent of state-preparation cost
    # =================================================================

    success_rows = []

    for method in METHODS:

        params = (
            method_parameters[
                method
            ]
        )

        for t in range(
            MAX_QAA_ITERATIONS + 1
        ):

            marked_probability = (
                qaa_probability(
                    params[
                        "marked_mass"
                    ],
                    t,
                )
            )

            # For t=0 no verifier/QAA is actually required.
            #
            # Direct correlation-state measurement therefore has
            # probability P_true_initial.
            if t == 0:

                expected_true_probability = (
                    p_true_initial
                )

            else:

                expected_true_probability = (
                    expected_true_success_after_qaa(
                        initial_marked_mass=
                            params[
                                "marked_mass"
                            ],

                        retention_probability=
                            params[
                                "retention"
                            ],

                        true_given_marked=
                            params[
                                "true_given_marked"
                            ],

                        iterations=t,
                    )
                )

            success_rows.append(
                {
                    "method":
                        method,

                    "iterations":
                        t,

                    "initial_marked_mass":
                        params[
                            "marked_mass"
                        ],

                    "amplified_marked_probability":
                        marked_probability,

                    "expected_true_key_probability":
                        expected_true_probability,
                }
            )

    success_df = pd.DataFrame(
        success_rows
    )

    # =================================================================
    # Step 4
    # End-to-end cost sweep
    # =================================================================

    sweep_rows = []

    for prep_multiplier in (
        STATE_PREP_MULTIPLIERS
    ):

        state_prep_te = (
            prep_multiplier
            * m3_oracle_cost
        )

        # -------------------------------------------------------------
        # Direct repeated measurement of the EXP05 state
        # -------------------------------------------------------------

        if state_prep_te > 0:

            direct_cost_per_true_hit = (
                state_prep_te
                / p_true_initial
            )

        else:

            direct_cost_per_true_hit = 0.0

        for method in METHODS:

            params = (
                method_parameters[
                    method
                ]
            )

            for t in range(
                1,
                MAX_QAA_ITERATIONS + 1
            ):

                true_success = (
                    expected_true_success_after_qaa(
                        initial_marked_mass=
                            params[
                                "marked_mass"
                            ],

                        retention_probability=
                            params[
                                "retention"
                            ],

                        true_given_marked=
                            params[
                                "true_given_marked"
                            ],

                        iterations=t,
                    )
                )

                total_cost = (
                    qaa_total_cost(
                        iterations=t,

                        state_prep_te=
                            state_prep_te,

                        marking_oracle_te=
                            params[
                                "oracle_cost"
                            ],
                    )
                )

                if true_success > 0:

                    expected_cost_per_true_hit = (
                        total_cost
                        / true_success
                    )

                else:

                    expected_cost_per_true_hit = (
                        np.inf
                    )

                state_prep_calls = (
                    1 + 2 * t
                )

                oracle_calls = t

                reflection_calls = t

                if (
                    direct_cost_per_true_hit
                    > 0
                ):

                    relative_to_direct = (
                        expected_cost_per_true_hit
                        / direct_cost_per_true_hit
                    )

                else:

                    relative_to_direct = np.nan

                sweep_rows.append(
                    {
                        "state_prep_multiplier":
                            prep_multiplier,

                        "state_prep_te":
                            state_prep_te,

                        "method":
                            method,

                        "iterations":
                            t,

                        "state_prep_calls":
                            state_prep_calls,

                        "marking_oracle_calls":
                            oracle_calls,

                        "reflection_calls":
                            reflection_calls,

                        "marking_oracle_te":
                            params[
                                "oracle_cost"
                            ],

                        "true_key_retention_probability":
                            params[
                                "retention"
                            ],

                        "true_probability_given_marked":
                            params[
                                "true_given_marked"
                            ],

                        "initial_marked_mass":
                            params[
                                "marked_mass"
                            ],

                        "expected_true_key_probability":
                            true_success,

                        "total_te_proxy":
                            total_cost,

                        "expected_te_per_true_key":
                            expected_cost_per_true_hit,

                        "direct_sampling_te_per_true_key":
                            direct_cost_per_true_hit,

                        "qaa_over_direct_cost_ratio":
                            relative_to_direct,
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
    # Step 5
    # Cost-optimal QAA point for every preparation-cost assumption
    # =================================================================

    best_rows = []

    for prep_multiplier in (
        STATE_PREP_MULTIPLIERS
    ):

        for method in METHODS:

            part = sweep_df[
                (
                    sweep_df[
                        "state_prep_multiplier"
                    ]
                    == prep_multiplier
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
    # Step 6
    # Break-even state-preparation cost
    #
    # Find the smallest C_A for which optimized QAA becomes cheaper
    # than direct repeated measurement.
    # =================================================================

    break_even_rows = []

    for method in METHODS:

        params = (
            method_parameters[
                method
            ]
        )

        first_better = None

        for multiplier in (
            BREAK_EVEN_MULTIPLIERS
        ):

            state_prep_te = (
                multiplier
                * m3_oracle_cost
            )

            direct_cost = (
                state_prep_te
                / p_true_initial
            )

            best_qaa_cost = np.inf
            best_t = None
            best_success = None

            for t in range(
                1,
                MAX_QAA_ITERATIONS + 1
            ):

                true_success = (
                    expected_true_success_after_qaa(
                        initial_marked_mass=
                            params[
                                "marked_mass"
                            ],

                        retention_probability=
                            params[
                                "retention"
                            ],

                        true_given_marked=
                            params[
                                "true_given_marked"
                            ],

                        iterations=t,
                    )
                )

                total_cost = (
                    qaa_total_cost(
                        iterations=t,

                        state_prep_te=
                            state_prep_te,

                        marking_oracle_te=
                            params[
                                "oracle_cost"
                            ],
                    )
                )

                cost_per_success = (
                    total_cost
                    / max(
                        true_success,
                        1e-15,
                    )
                )

                if (
                    cost_per_success
                    < best_qaa_cost
                ):

                    best_qaa_cost = (
                        cost_per_success
                    )

                    best_t = t

                    best_success = (
                        true_success
                    )

            if (
                best_qaa_cost
                <= direct_cost
            ):

                first_better = {
                    "method":
                        method,

                    "break_even_state_prep_multiplier":
                        float(
                            multiplier
                        ),

                    "break_even_state_prep_te":
                        float(
                            state_prep_te
                        ),

                    "marking_oracle_te":
                        float(
                            params[
                                "oracle_cost"
                            ]
                        ),

                    "state_prep_over_method_oracle":
                        float(
                            state_prep_te
                            / params[
                                "oracle_cost"
                            ]
                        ),

                    "cost_optimal_iterations":
                        int(
                            best_t
                        ),

                    "expected_true_key_probability":
                        float(
                            best_success
                        ),

                    "qaa_expected_te_per_true_key":
                        float(
                            best_qaa_cost
                        ),

                    "direct_expected_te_per_true_key":
                        float(
                            direct_cost
                        ),
                }

                break

        if first_better is None:

            first_better = {
                "method":
                    method,

                "break_even_state_prep_multiplier":
                    np.nan,

                "break_even_state_prep_te":
                    np.nan,

                "marking_oracle_te":
                    float(
                        params[
                            "oracle_cost"
                        ]
                    ),

                "state_prep_over_method_oracle":
                    np.nan,

                "cost_optimal_iterations":
                    np.nan,

                "expected_true_key_probability":
                    np.nan,

                "qaa_expected_te_per_true_key":
                    np.nan,

                "direct_expected_te_per_true_key":
                    np.nan,
            }

        break_even_rows.append(
            first_better
        )

    break_even_df = pd.DataFrame(
        break_even_rows
    )

    break_even_df.to_csv(
        OUTPUT_BREAK_EVEN_CSV,
        index=False,
    )

    # =================================================================
    # Step 7
    # Primitive-work / data-budget proxies
    # =================================================================

    work_rows = []

    # Classical EXP01
    classical_candidate_sample_evals = (
        N_KEY_CANDIDATES
        * N_CLASSICAL_BASELINE_PAIRS
    )

    work_rows.append(
        {
            "strategy":
                "Classical EXP01 linear attack",

            "known_pc_pairs":
                N_CLASSICAL_BASELINE_PAIRS,

            "key_candidates":
                N_KEY_CANDIDATES,

            "candidate_sample_evaluations":
                classical_candidate_sample_evals,

            "relation_updates":
                classical_candidate_sample_evals,

            "note":
                (
                    "Classical candidate loop; "
                    "not measured in Toffoli-equivalent units."
                ),
        }
    )

    # Quantum verifier work per oracle call
    for method in METHODS:

        row = (
            method_lookup.loc[
                method
            ]
        )

        m = int(
            row[
                "validation_size"
            ]
        )

        if method == "M2_energy":
            l_count = 8

        elif method == "M3_llr":
            l_count = 8

        else:
            l_count = 8

        work_rows.append(
            {
                "strategy":
                    f"Quantum {method} one marking-oracle call",

                "known_pc_pairs":
                    (
                        N_CORRELATION_STATE_PAIRS
                        + m
                    ),

                "key_candidates":
                    N_KEY_CANDIDATES,

                "candidate_sample_evaluations":
                    m,

                "relation_updates":
                    (
                        m
                        * l_count
                    ),

                "note":
                    (
                        "Candidate keys processed coherently in "
                        "superposition; reversible gate overhead "
                        "is reported separately by EXP10."
                    ),
            }
        )

    work_df = pd.DataFrame(
        work_rows
    )

    work_df.to_csv(
        OUTPUT_WORK_CSV,
        index=False,
    )

    # =================================================================
    # Figure 1
    # QAA success versus number of iterations
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for method in METHODS:

        part = success_df[
            success_df[
                "method"
            ]
            == method
        ]

        ax.plot(
            part[
                "iterations"
            ],
            part[
                "expected_true_key_probability"
            ],
            marker="o",
            label=method,
        )

    ax.axhline(
        p_true_initial,
        linestyle="--",
        linewidth=1.0,
        label=(
            f"Direct EXP05 P(true) "
            f"= {100 * p_true_initial:.2f}%"
        ),
    )

    ax.set_xlabel(
        "QAA iterations"
    )

    ax.set_ylabel(
        "Expected true-key probability"
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "EXP11: True-Key Probability after QAA"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_SUCCESS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 2
    # Optimized QAA versus direct sampling as C_A varies
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    positive_multipliers = [
        x
        for x in STATE_PREP_MULTIPLIERS
        if x > 0
    ]

    direct_values = []

    for multiplier in (
        positive_multipliers
    ):

        state_prep_te = (
            multiplier
            * m3_oracle_cost
        )

        direct_values.append(
            state_prep_te
            / p_true_initial
        )

    ax.plot(
        positive_multipliers,
        direct_values,
        marker="o",
        linestyle="--",
        label="Direct correlation-state sampling",
    )

    for method in METHODS:

        part = best_df[
            (
                best_df[
                    "method"
                ]
                == method
            )
            &
            (
                best_df[
                    "state_prep_multiplier"
                ]
                > 0
            )
        ].sort_values(
            "state_prep_multiplier"
        )

        ax.plot(
            part[
                "state_prep_multiplier"
            ],
            part[
                "expected_te_per_true_key"
            ],
            marker="o",
            label=f"{method} + QAA",
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "State-preparation cost / M3 oracle cost"
    )

    ax.set_ylabel(
        "Expected TE proxy per true-key output"
    )

    ax.set_title(
        "EXP11: End-to-End Cost versus State-Preparation Cost"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_COST_PREP,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 3
    # Cost-optimal iteration count
    # =================================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for method in METHODS:

        part = best_df[
            (
                best_df[
                    "method"
                ]
                == method
            )
            &
            (
                best_df[
                    "state_prep_multiplier"
                ]
                > 0
            )
        ].sort_values(
            "state_prep_multiplier"
        )

        ax.plot(
            part[
                "state_prep_multiplier"
            ],
            part[
                "iterations"
            ],
            marker="o",
            label=method,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "State-preparation cost / M3 oracle cost"
    )

    ax.set_ylabel(
        "Cost-optimal QAA iterations"
    )

    ax.set_title(
        "EXP11: Cost-Optimal QAA Iteration Count"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_OPTIMAL_T,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Figure 4
    # Number of expensive resource-family calls at C_A = M3 oracle
    # =================================================================

    representative_multiplier = 1.0

    representative = best_df[
        best_df[
            "state_prep_multiplier"
        ]
        == representative_multiplier
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = np.arange(
        len(
            representative
        )
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        representative[
            "state_prep_calls"
        ],
        width,
        label="State-preparation-family calls",
    )

    ax.bar(
        x + width / 2,
        representative[
            "marking_oracle_calls"
        ],
        width,
        label="Marking-oracle calls",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        representative[
            "method"
        ]
    )

    ax.set_ylabel(
        "Calls per QAA execution"
    )

    ax.set_title(
        "EXP11: Cost-Optimal Resource Calls "
        "(C_A = M3 Oracle Cost)"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_ORACLE_CALLS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # Summary
    # =================================================================

    maximum_success_summary = {}

    for method in METHODS:

        params = (
            method_parameters[
                method
            ]
        )

        rows = []

        for t in range(
            1,
            MAX_QAA_ITERATIONS + 1
        ):

            p_success = (
                expected_true_success_after_qaa(
                    initial_marked_mass=
                        params[
                            "marked_mass"
                        ],

                    retention_probability=
                        params[
                            "retention"
                        ],

                    true_given_marked=
                        params[
                            "true_given_marked"
                        ],

                    iterations=t,
                )
            )

            rows.append(
                (
                    t,
                    p_success,
                )
            )

        t_max, p_max = max(
            rows,
            key=lambda x:
                x[1],
        )

        maximum_success_summary[
            method
        ] = {
            "representative_initial_marked_mass":
                float(
                    params[
                        "marked_mass"
                    ]
                ),

            "marking_oracle_te":
                float(
                    params[
                        "oracle_cost"
                    ]
                ),

            "retention_probability":
                float(
                    params[
                        "retention"
                    ]
                ),

            "true_probability_given_marked":
                float(
                    params[
                        "true_given_marked"
                    ]
                ),

            "maximum_expected_true_probability":
                float(
                    p_max
                ),

            "iteration_for_maximum_probability":
                int(
                    t_max
                ),
        }

    summary = {
        "experiment":
            "exp11_end_to_end_qaa",

        "true_candidate":
            true_candidate,

        "true_candidate_hex":
            f"{true_candidate:02X}",

        "initial_true_key_probability":
            p_true_initial,

        "methods":
            METHODS,

        "maximum_success":
            maximum_success_summary,

        "state_prep_cost_sweep_relative_to_M3_oracle":
            STATE_PREP_MULTIPLIERS,

        "break_even":
            break_even_df.to_dict(
                orient="records"
            ),

        "important_interpretation": [
            (
                "EXP11 does not assign a fictitious exact gate cost "
                "to coherent correlation-state preparation. Instead "
                "that unknown cost is swept explicitly."
            ),
            (
                "The outer QAA cost includes one initial A call and "
                "two A-family calls per Grover iteration, together "
                "with one EXP10 marking phase-oracle call per iteration."
            ),
            (
                "EXP09 verifier-retention failures are treated "
                "conservatively as zero true-key success."
            ),
            (
                "The marked-sector probability is estimated from "
                "P(true)/P(true|marked), using the representative "
                "EXP09 robust operating point."
            ),
            (
                "Classical and quantum primitive-work counts are "
                "reported separately because Toffoli-equivalent "
                "logical cost is not directly comparable with a "
                "classical CPU operation count."
            ),
        ],

        "limitations": [
            (
                "The coherent correlation-state preparation unitary A "
                "has not yet been synthesized gate-by-gate."
            ),
            (
                "qRAM or coherent external-memory access remains "
                "outside the EXP10/EXP11 cost model."
            ),
            (
                "Surface-code overhead, routing, T-state factories, "
                "hardware noise, and physical runtime are not included."
            ),
            (
                "EXP09 statistics are averaged over random validation "
                "subsets; EXP11 therefore represents a statistical "
                "end-to-end benchmark rather than one fixed compiled "
                "oracle instance."
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
    # Console output
    # =================================================================

    print()
    print("=" * 80)
    print(
        "EXP11: End-to-End QAA Cost Benchmark"
    )
    print("=" * 80)

    print()
    print(
        f"True key                     : "
        f"0x{true_candidate:02X}"
    )

    print(
        f"Initial EXP05 P(true)        : "
        f"{p_true_initial:.6f} "
        f"({100 * p_true_initial:.3f}%)"
    )

    print()
    print(
        "[Representative verifier parameters]"
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
            f"  marking oracle TE          : "
            f"{p['oracle_cost']:.0f}"
        )

        print(
            f"  true-key retention         : "
            f"{p['retention']:.4f}"
        )

        print(
            f"  P(true | marked)           : "
            f"{p['true_given_marked']:.4f}"
        )

        print(
            f"  estimated P(marked)        : "
            f"{p['marked_mass']:.6f}"
        )

    print()
    print(
        "[Maximum QAA success]"
    )

    for method, values in (
        maximum_success_summary.items()
    ):

        print()

        print(
            method
        )

        print(
            f"  max expected P(true)       : "
            f"{values['maximum_expected_true_probability']:.4f}"
        )

        print(
            f"  iteration at max P(true)   : "
            f"{values['iteration_for_maximum_probability']}"
        )

    print()
    print(
        "[QAA/direct break-even]"
    )

    print(
        break_even_df.to_string(
            index=False
        )
    )

    print()
    print(
        "[Cost-optimal points]"
    )

    display_columns = [
        "state_prep_multiplier",
        "method",
        "iterations",
        "expected_true_key_probability",
        "state_prep_calls",
        "marking_oracle_calls",
        "expected_te_per_true_key",
        "qaa_over_direct_cost_ratio",
    ]

    print(
        best_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "[Important]"
    )

    print(
        "The exact coherent state-preparation cost is still unknown."
    )

    print(
        "Therefore the most important EXP11 output is the "
        "break-even preparation-cost ratio, not one absolute "
        "end-to-end gate-count number."
    )

    print()
    print(
        "[Output files]"
    )

    print(
        f"Full QAA sweep          : "
        f"{OUTPUT_SWEEP_CSV}"
    )

    print(
        f"Cost-optimal QAA points : "
        f"{OUTPUT_BEST_CSV}"
    )

    print(
        f"Break-even analysis     : "
        f"{OUTPUT_BREAK_EVEN_CSV}"
    )

    print(
        f"Work proxies            : "
        f"{OUTPUT_WORK_CSV}"
    )

    print(
        f"Summary                 : "
        f"{OUTPUT_SUMMARY_JSON}"
    )

    print(
        f"Success figure          : "
        f"{FIG_SUCCESS}"
    )

    print(
        f"Prep-cost figure        : "
        f"{FIG_COST_PREP}"
    )

    print(
        f"Optimal-t figure        : "
        f"{FIG_OPTIMAL_T}"
    )

    print(
        f"Resource-call figure    : "
        f"{FIG_ORACLE_CALLS}"
    )

    print()
    print(
        "[PASS] EXP11 completed successfully."
    )


if __name__ == "__main__":
    main()