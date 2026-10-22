from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heys_spn import (
    SBOX,
    INV_SBOX,
    encrypt_block,
    decrypt_block,
    permute_word,
    inverse_permute_word,
    target_last_round_subkey,
)


ROUND_KEYS = (
    0x3A94,
    0xD63F,
    0xB127,
    0x8E5C,
    0xA2B4,
)


def test_sbox_inverse():
    for x in range(16):
        assert INV_SBOX[SBOX[x]] == x


@pytest.mark.parametrize(
    "value",
    [
        0x0000,
        0x0001,
        0x1234,
        0xABCD,
        0xBEEF,
        0xFFFF,
    ],
)
def test_permutation_inverse(value):
    assert inverse_permute_word(permute_word(value)) == value


@pytest.mark.parametrize(
    "plaintext",
    [
        0x0000,
        0x0001,
        0x1234,
        0xABCD,
        0xBEEF,
        0xFFFF,
    ],
)
def test_encrypt_decrypt_roundtrip(plaintext):
    ciphertext = encrypt_block(plaintext, ROUND_KEYS)
    recovered = decrypt_block(ciphertext, ROUND_KEYS)

    assert recovered == plaintext


def test_regression_vector():
    plaintext = 0x1234
    expected_ciphertext = 0x03CC

    assert encrypt_block(plaintext, ROUND_KEYS) == expected_ciphertext


def test_target_subkey():
    k2, k4 = target_last_round_subkey(ROUND_KEYS)

    assert k2 == 0x2
    assert k4 == 0x4
