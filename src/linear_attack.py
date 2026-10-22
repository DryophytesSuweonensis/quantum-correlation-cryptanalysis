from __future__ import annotations

import numpy as np
import pandas as pd

from heys_spn import INV_SBOX, bit_at


INV_SBOX_ARRAY = np.asarray(INV_SBOX, dtype=np.uint8)


def plaintext_linear_parity(plaintext: int) -> int:
    """
    Compute

        P5 xor P7 xor P8

    using Heys' bit numbering:
        bit 1 = MSB
        bit 16 = LSB.
    """
    return (
        bit_at(plaintext, 5)
        ^ bit_at(plaintext, 7)
        ^ bit_at(plaintext, 8)
    )


def u4_target_parity(nibble: np.ndarray) -> np.ndarray:
    """
    For one last-round S-box input nibble, extract the bit parity

        bit_2 xor bit_4

    corresponding to:
        U4,6 xor U4,8
    or:
        U4,14 xor U4,16.

    Within a 4-bit nibble:
        bit_1 = binary bit 3
        bit_2 = binary bit 2
        bit_3 = binary bit 1
        bit_4 = binary bit 0
    """
    return ((nibble >> 2) & 1) ^ (nibble & 1)


def run_linear_attack(
    plaintexts: np.ndarray,
    ciphertexts: np.ndarray,
    true_k2: int | None = None,
    true_k4: int | None = None,
) -> pd.DataFrame:
    """
    Perform the 8-bit partial last-round key recovery from Heys'
    linear cryptanalysis example.

    Candidate key:

        k2 = K5,5..8
        k4 = K5,13..16

    For each candidate evaluate

        P5 xor P7 xor P8
        xor U4,6 xor U4,8
        xor U4,14 xor U4,16 = 0

    over all known plaintext/ciphertext pairs.
    """
    plaintexts = np.asarray(plaintexts, dtype=np.uint16)
    ciphertexts = np.asarray(ciphertexts, dtype=np.uint16)

    if plaintexts.shape != ciphertexts.shape:
        raise ValueError("plaintexts and ciphertexts must have same shape.")

    n_samples = len(plaintexts)

    if n_samples == 0:
        raise ValueError("At least one plaintext/ciphertext pair is required.")

    p_parity = np.fromiter(
        (plaintext_linear_parity(int(p)) for p in plaintexts),
        dtype=np.uint8,
        count=n_samples,
    )

    c2 = ((ciphertexts >> 8) & 0xF).astype(np.uint8)
    c4 = (ciphertexts & 0xF).astype(np.uint8)

    rows = []

    for k2 in range(16):
        u2 = INV_SBOX_ARRAY[c2 ^ k2]
        u2_parity = u4_target_parity(u2)

        for k4 in range(16):
            u4 = INV_SBOX_ARRAY[c4 ^ k4]
            u4_parity = u4_target_parity(u4)

            relation = p_parity ^ u2_parity ^ u4_parity

            count_zero = int(np.count_nonzero(relation == 0))
            probability_zero = count_zero / n_samples

            bias = probability_zero - 0.5
            abs_bias = abs(bias)

            candidate = (k2 << 4) | k4

            is_true = False
            if true_k2 is not None and true_k4 is not None:
                is_true = (k2 == true_k2) and (k4 == true_k4)

            rows.append(
                {
                    "candidate": candidate,
                    "candidate_hex": f"{candidate:02X}",
                    "k5_bits_5_8": k2,
                    "k5_bits_5_8_hex": f"{k2:X}",
                    "k5_bits_13_16": k4,
                    "k5_bits_13_16_hex": f"{k4:X}",
                    "count_zero": count_zero,
                    "n_samples": n_samples,
                    "probability_zero": probability_zero,
                    "bias": bias,
                    "abs_bias": abs_bias,
                    "is_true_key": is_true,
                }
            )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="abs_bias",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    return df
