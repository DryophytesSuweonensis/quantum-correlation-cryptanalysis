from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(SRC_DIR))

from heys_spn import encrypt_block, target_last_round_subkey
from linear_attack import run_linear_attack


SEED = 20261022
N_KNOWN_PAIRS = 10_000

ROUND_KEYS = (
    0x3A94,  # K1
    0xD63F,  # K2
    0xB127,  # K3
    0x8E5C,  # K4
    0xA2B4,  # K5
)


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    true_k2, true_k4 = target_last_round_subkey(ROUND_KEYS)
    true_candidate = (true_k2 << 4) | true_k4

    print("=" * 72)
    print("Phase 1: Classical Linear Cryptanalysis Baseline")
    print("=" * 72)

    print(f"Seed               : {SEED}")
    print(f"Known P/C pairs    : {N_KNOWN_PAIRS}")
    print(f"K5                  : 0x{ROUND_KEYS[4]:04X}")
    print(
        f"Target subkey       : "
        f"[K5,5..8 = 0x{true_k2:X}, "
        f"K5,13..16 = 0x{true_k4:X}]"
    )
    print(f"8-bit candidate     : 0x{true_candidate:02X}")
    print()

    plaintexts, ciphertexts = generate_known_pairs(
        n_pairs=N_KNOWN_PAIRS,
        seed=SEED,
    )

    df = run_linear_attack(
        plaintexts=plaintexts,
        ciphertexts=ciphertexts,
        true_k2=true_k2,
        true_k4=true_k4,
    )

    output_csv = RESULTS_DIR / "candidate_correlations.csv"
    df.to_csv(output_csv, index=False)

    true_row = df.loc[df["is_true_key"]].iloc[0]

    print("Top 10 candidates")
    print("-" * 72)

    columns = [
        "rank",
        "candidate_hex",
        "bias",
        "abs_bias",
        "is_true_key",
    ]

    print(
        df.loc[:9, columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("True-key result")
    print("-" * 72)
    print(f"Rank                : {int(true_row['rank'])}")
    print(f"Candidate           : 0x{true_row['candidate_hex']}")
    print(f"Bias                : {true_row['bias']:.6f}")
    print(f"|Bias|              : {true_row['abs_bias']:.6f}")
    print(f"Theoretical |bias|  : {1 / 32:.6f}")
    print()

    print("CSV saved to:")
    print(output_csv)

    if int(true_row["rank"]) == 1:
        print()
        print("[PASS] Correct target subkey is ranked #1.")
    else:
        print()
        print(
            "[WARNING] Correct target subkey is not ranked #1. "
            "Increase sample count or inspect the fixed-key instance."
        )


if __name__ == "__main__":
    main()
