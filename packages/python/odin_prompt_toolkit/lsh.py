"""Robust, deterministic SimHash (random-hyperplane LSH) utilities.

This module provides locality-sensitive hashing for similarity search on
high-dimensional vectors (e.g., embeddings).

- Uses normalized vectors (direction only)
- Deterministic per-family, per-bit, per-dimension sign via SplitMix64-based hash
- Supports multiple independent hash families for robustness

Ported from research/src/signature_cli/core/lsh.py (canonical Python implementation).
"""

import math
import string
from dataclasses import dataclass, field


@dataclass
class LSHFamily:
    """Result of LSH hashing for one family."""

    family: int
    bits: int
    signature: str  # hex string (bits/4 hex chars)
    bands: list[str] = field(default_factory=list)  # contiguous slices of hex


_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """SplitMix64 hash function for deterministic random generation."""
    z = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z & _MASK64


def _sign_for(family: int, bit: int, dim: int) -> int:
    """Get deterministic +1/-1 sign from (family, bit, dim).

    Args:
        family: Hash family index
        bit: Bit index within the signature
        dim: Dimension index of the vector

    Returns:
        +1 or -1 deterministically based on inputs
    """
    # Combine into 64-bit seed and mix
    seed = (family << 48) ^ (bit << 24) ^ dim
    h = _splitmix64(seed)
    # Use lowest bit for sign
    return 1 if (h & 1) == 1 else -1


def _simhash_lsh_multi_python(
    normalized_vector: list[float],
    families: int = 3,
    bits: int = 256,
    bands: int = 16,
) -> list[LSHFamily]:
    """Compute SimHash LSH signatures for a normalized vector.

    Uses random hyperplane LSH with deterministic hyperplanes generated
    via SplitMix64. Multiple independent hash families can be computed
    for improved recall in similarity search.

    Args:
        normalized_vector: Input vector (should be L2-normalized for best results)
        families: Number of independent hash families to compute (default: 3)
        bits: Number of bits per signature (default: 256)
        bands: Number of bands to split signature into for LSH indexing (default: 16)

    Returns:
        List of LSHFamily results, one per family
    """
    families = max(1, families)
    bits = max(64, bits)
    bands = max(1, bands)

    results: list[LSHFamily] = []

    for f in range(families):
        bool_bits: list[bool] = []

        for b in range(bits):
            dot = 0.0
            for j, val in enumerate(normalized_vector):
                s = _sign_for(f, b, j)
                dot += val * s
            bool_bits.append(dot > 0)

        # Pack into hex string
        hex_chars: list[str] = []
        for i in range(0, len(bool_bits), 4):
            n = (
                (8 if bool_bits[i] else 0)
                + (4 if bool_bits[i + 1] else 0)
                + (2 if bool_bits[i + 2] else 0)
                + (1 if bool_bits[i + 3] else 0)
            )
            hex_chars.append(format(n, "x"))

        signature = "".join(hex_chars)

        # Split into bands (contiguous slices)
        band_len = len(signature) // bands if bands > 0 else len(signature)
        band_len = max(1, band_len)

        band_arr: list[str] = []
        for i in range(0, len(signature), band_len):
            band_arr.append(signature[i : i + band_len])
            if len(band_arr) == bands:
                break

        results.append(
            LSHFamily(
                family=f,
                bits=bits,
                signature=signature,
                bands=band_arr,
            )
        )

    return results


_valid_hex = string.ascii_lowercase[:6] + string.digits


def _clean(val: str) -> str:
    return "".join(c for c in val.lower() if c in _valid_hex)


def _hamming_distance_hex_python(a: str, b: str) -> int:
    """Compute Hamming distance between two hex strings.

    Each hex character represents 4 bits, so the distance is computed
    by XORing corresponding nibbles and counting set bits.

    Args:
        a: First hex string
        b: Second hex string

    Returns:
        Hamming distance in bits
    """
    # Clean strings to lowercase hex only
    x = _clean(a)
    y = _clean(b)

    min_len = min(len(x), len(y))
    dist = 0

    for i in range(min_len):
        n1 = int(x[i], 16)
        n2 = int(y[i], 16)
        xor = n1 ^ n2
        # Count bits in 4-bit value
        dist += ((xor >> 3) & 1) + ((xor >> 2) & 1) + ((xor >> 1) & 1) + (xor & 1)

    # Extra nibbles count as differing bits (4 bits per nibble)
    dist += abs(len(x) - len(y)) * 4

    return dist


def _cosine_from_hamming_python(distance_bits: int, total_bits: int) -> float:
    """Estimate cosine similarity from Hamming distance.

    For random hyperplane LSH, the probability that two vectors have
    different signs for a random hyperplane is proportional to the
    angle between them. This function converts Hamming distance back
    to an estimated cosine similarity.

    Args:
        distance_bits: Hamming distance in bits
        total_bits: Total number of bits in the signature

    Returns:
        Estimated cosine similarity in range [-1, 1]
    """
    if total_bits <= 0:
        return 0.0

    p_diff = distance_bits / total_bits
    return math.cos(math.pi * p_diff)


def _normalize_vector_python(vector: list[float]) -> list[float]:
    """L2-normalize a vector (pure Python implementation).

    Args:
        vector: Input vector

    Returns:
        L2-normalized vector (unit length)
    """
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0:
        return vector
    return [x / magnitude for x in vector]


# Transparent native acceleration
# Try to use native implementation, fall back to pure Python
from odin_prompt_toolkit._accel import NATIVE_AVAILABLE

if NATIVE_AVAILABLE:
    from odin_prompt_toolkit._accel import (
        cosine_from_hamming,
        hamming_distance_hex,
        normalize_vector,
        simhash_lsh_multi,
    )
else:
    # Use pure Python implementations
    simhash_lsh_multi = _simhash_lsh_multi_python
    normalize_vector = _normalize_vector_python
    hamming_distance_hex = _hamming_distance_hex_python
    cosine_from_hamming = _cosine_from_hamming_python
