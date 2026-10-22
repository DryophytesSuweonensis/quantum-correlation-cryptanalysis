from __future__ import annotations

from typing import Sequence


# Howard Heys SPN S-box
SBOX = (
    0xE, 0x4, 0xD, 0x1,
    0x2, 0xF, 0xB, 0x8,
    0x3, 0xA, 0x6, 0xC,
    0x5, 0x9, 0x0, 0x7,
)

INV_SBOX = [0] * 16
for x, y in enumerate(SBOX):
    INV_SBOX[y] = x
INV_SBOX = tuple(INV_SBOX)


# Input bit position -> output bit position.
# Bit position 1 is the MSB, 16 is the LSB.
PBOX = (
    1, 5, 9, 13,
    2, 6, 10, 14,
    3, 7, 11, 15,
    4, 8, 12, 16,
)


def bit_at(x: int, position: int) -> int:
    """
    Return one bit from a 16-bit word.

    Parameters
    ----------
    x : int
        16-bit integer.
    position : int
        Bit position in [1, 16], where 1 denotes the MSB.

    Returns
    -------
    int
        0 or 1.
    """
    if not 1 <= position <= 16:
        raise ValueError("position must be between 1 and 16.")

    return (x >> (16 - position)) & 1


def substitute_word(x: int) -> int:
    """Apply the 4-bit S-box independently to four nibbles."""
    y = 0

    for shift in (12, 8, 4, 0):
        nibble = (x >> shift) & 0xF
        y |= SBOX[nibble] << shift

    return y


def inverse_substitute_word(x: int) -> int:
    """Apply the inverse S-box independently to four nibbles."""
    y = 0

    for shift in (12, 8, 4, 0):
        nibble = (x >> shift) & 0xF
        y |= INV_SBOX[nibble] << shift

    return y


def permute_word(x: int) -> int:
    """
    Apply Heys' 16-bit bit permutation.

    PBOX[i-1] gives the output location of input bit i.
    """
    y = 0

    for input_pos, output_pos in enumerate(PBOX, start=1):
        bit = bit_at(x, input_pos)
        y |= bit << (16 - output_pos)

    return y


def inverse_permute_word(x: int) -> int:
    """Apply inverse of Heys' bit permutation."""
    y = 0

    for input_pos, output_pos in enumerate(PBOX, start=1):
        bit = bit_at(x, output_pos)
        y |= bit << (16 - input_pos)

    return y


def encrypt_block(
    plaintext: int,
    round_keys: Sequence[int],
) -> int:
    """
    Encrypt one 16-bit block using the 4-round Heys SPN.

    Five independent 16-bit round keys are required:

        K1, K2, K3, K4, K5

    Rounds 1-3:
        XOR key -> S-box -> permutation

    Round 4:
        XOR K4 -> S-box -> XOR K5
    """
    if len(round_keys) != 5:
        raise ValueError("Exactly five 16-bit round keys are required.")

    if not 0 <= plaintext <= 0xFFFF:
        raise ValueError("plaintext must be a 16-bit integer.")

    state = plaintext

    # Rounds 1, 2, 3
    for r in range(3):
        state ^= round_keys[r]
        state = substitute_word(state)
        state = permute_word(state)

    # Round 4
    state ^= round_keys[3]
    state = substitute_word(state)

    # Final whitening key
    state ^= round_keys[4]

    return state & 0xFFFF


def decrypt_block(
    ciphertext: int,
    round_keys: Sequence[int],
) -> int:
    """Decrypt one 16-bit Heys SPN block."""
    if len(round_keys) != 5:
        raise ValueError("Exactly five 16-bit round keys are required.")

    state = ciphertext

    # Undo final whitening and final substitution
    state ^= round_keys[4]
    state = inverse_substitute_word(state)
    state ^= round_keys[3]

    # Undo rounds 3 -> 1
    for r in (2, 1, 0):
        state = inverse_permute_word(state)
        state = inverse_substitute_word(state)
        state ^= round_keys[r]

    return state & 0xFFFF


def target_last_round_subkey(round_keys: Sequence[int]) -> tuple[int, int]:
    """
    Return the two 4-bit K5 nibbles attacked in Heys' tutorial:

        K5,5..8   -> second nibble
        K5,13..16 -> fourth nibble
    """
    k5 = round_keys[4]

    k_second = (k5 >> 8) & 0xF
    k_fourth = k5 & 0xF

    return k_second, k_fourth
