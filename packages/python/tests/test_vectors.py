"""Test Python implementation against canonical test vectors."""

import json
from pathlib import Path

import pytest

from odin_sig import (
    cosine_from_hamming,
    hamming_distance_hex,
    normalize_vector,
    simhash_lsh_multi,
)
from odin_sig.lsh import _sign_for, _splitmix64

# Path to test vectors directory
VECTORS_DIR = Path(__file__).parent.parent.parent.parent / "spec" / "test-vectors"


def load_vectors(filename: str) -> dict:
    """Load test vectors from JSON file."""
    with open(VECTORS_DIR / filename) as f:
        return json.load(f)


class TestSplitMix64:
    """Test SplitMix64 PRNG against canonical vectors."""

    def test_splitmix64_vectors(self):
        vectors = load_vectors("splitmix64.json")

        for case in vectors["vectors"]:
            input_val = case["input"]
            expected = case["output"]
            actual = format(_splitmix64(input_val), "016X")

            assert actual == expected, f"SplitMix64({input_val}): expected {expected}, got {actual}"


class TestSignFor:
    """Test sign_for function against canonical vectors."""

    def test_sign_for_vectors(self):
        vectors = load_vectors("sign_for.json")

        for case in vectors["vectors"]:
            family = case["family"]
            bit = case["bit"]
            dim = case["dim"]
            expected = case["sign"]

            actual = _sign_for(family, bit, dim)

            assert actual == expected, (
                f"sign_for({family}, {bit}, {dim}): expected {expected}, got {actual}"
            )


class TestSimHash:
    """Test SimHash LSH against canonical vectors."""

    def test_simhash_vectors(self):
        vectors = load_vectors("simhash.json")

        for case in vectors["vectors"]:
            name = case["name"]
            input_vec = case["input"]
            config = case["config"]
            expected_families = case["expected"]

            # Generate signatures
            families = simhash_lsh_multi(
                input_vec,
                families=config["families"],
                bits=config["bits"],
                bands=config["bands"],
            )

            # Check each family
            for i, (actual, expected) in enumerate(zip(families, expected_families)):
                assert actual.family == expected["family"], (
                    f"{name} family {i}: family index mismatch"
                )
                assert actual.bits == expected["bits"], f"{name} family {i}: bits mismatch"
                assert actual.signature == expected["signature"], (
                    f"{name} family {i}: signature mismatch"
                )
                assert actual.bands == expected["bands"], f"{name} family {i}: bands mismatch"


class TestHammingDistance:
    """Test Hamming distance calculation against canonical vectors."""

    def test_hamming_vectors(self):
        vectors = load_vectors("hamming.json")

        for case in vectors["vectors"]:
            a = case["a"]
            b = case["b"]
            expected = case["distance"]
            description = case["description"]

            actual = hamming_distance_hex(a, b)

            assert actual == expected, (
                f"{description}: hamming_distance_hex('{a}', '{b}') = {actual}, expected {expected}"
            )


class TestCosineFromHamming:
    """Test cosine similarity estimation against canonical vectors."""

    def test_cosine_vectors(self):
        vectors = load_vectors("cosine.json")

        for case in vectors["vectors"]:
            distance = case["distance"]
            total_bits = case["total_bits"]
            expected = case["cosine_similarity"]

            actual = cosine_from_hamming(distance, total_bits)

            # Allow small floating-point differences
            assert abs(actual - expected) < 1e-10, (
                f"cosine_from_hamming({distance}, {total_bits}) = {actual}, expected {expected}"
            )


class TestSHA256:
    """Test SHA256 canonical format against vectors."""

    def test_sha256_vectors(self):
        from odin_sig.types import compute_embedding_sha256

        vectors = load_vectors("sha256.json")

        for case in vectors["vectors"]:
            input_vec = case["input"]
            expected_json = case["expected_json"]
            expected_hash = case["expected_sha256"]
            description = case["description"]

            # Compute hash
            actual_hash = compute_embedding_sha256(input_vec)

            assert actual_hash == expected_hash, (
                f"{description}: hash mismatch\n"
                f"Expected JSON: {expected_json}\n"
                f"Expected hash: {expected_hash}\n"
                f"Actual hash: {actual_hash}"
            )


class TestSignatureFormat:
    """Test signature string parsing against canonical vectors."""

    def test_signature_format_vectors(self):
        from odin_sig.types import parse_signature_string

        vectors = load_vectors("signature_format.json")

        for case in vectors["vectors"]:
            input_str = case["input"]
            description = case["description"]

            if case["valid"]:
                # Should parse successfully
                expected_version = case["expected_version"]
                expected_sig = case["expected_signature"]

                result = parse_signature_string(input_str)

                assert result.version.value == expected_version, f"{description}: version mismatch"
                assert result.signature == expected_sig, f"{description}: signature mismatch"
            else:
                # Should raise ValueError
                with pytest.raises(ValueError):
                    parse_signature_string(input_str)
