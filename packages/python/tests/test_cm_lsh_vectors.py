"""Test CM-LSH implementation against canonical test vectors."""

import json
from pathlib import Path

import numpy as np

from odin_prompt_toolkit.cm_lsh import create_default_cm_lsh

# Path to test vectors directory
VECTORS_DIR = Path(__file__).parent.parent.parent.parent / "spec" / "test-vectors"


def load_vectors(filename: str) -> dict:
    """Load test vectors from JSON file."""
    with open(VECTORS_DIR / filename) as f:
        return json.load(f)


class TestCMLSHVectors:
    """Test CM-LSH implementation against canonical Rust vectors."""

    def test_cm_lsh_hash_vectors(self):
        """Test CM-LSH hash generation against Rust implementation.

        Note: Due to floating-point precision differences (Python f64 vs Rust f32),
        exact bit-for-bit matches are not expected. We verify:
        1. Correct structure (512 bits, 64 bands)
        2. LSH-TS compatibility (first 256 bits should match within tolerance)
        3. Overall similarity to expected results
        """
        vectors = load_vectors("cm_lsh.json")

        for case in vectors["hash_vectors"]:
            name = case["name"]
            input_vec = case["input"]
            expected = case["expected"]

            # Create CM-LSH hasher
            cm_lsh = create_default_cm_lsh(len(input_vec), family=0)

            # Generate hash
            hash_result = cm_lsh.hash(input_vec)

            # Check structure
            assert hash_result.bits == expected["bits"], f"{name}: bits mismatch"
            assert len(hash_result.bands) == len(expected["bands"]), f"{name}: bands count mismatch"

            # Check LSH-TS portion has reasonable similarity (allow some bit differences)
            from odin_prompt_toolkit import hamming_distance_hex

            lsh_ts_actual = hash_result.lsh_ts_compat()
            lsh_ts_expected = expected["lsh_ts_compat"]
            hamming_dist = hamming_distance_hex(lsh_ts_actual, lsh_ts_expected)

            # Allow up to 7% bit difference due to floating-point precision
            # (Python f64 vs Rust f32 can cause small differences in dot products)
            max_allowed_diff = int(256 * 0.07)
            assert hamming_dist <= max_allowed_diff, (
                f"{name}: LSH-TS compatibility has too many bit differences\n"
                f"Hamming distance: {hamming_dist} (max allowed: {max_allowed_diff})\n"
                f"Expected: {lsh_ts_expected}\n"
                f"Actual:   {lsh_ts_actual}"
            )

    def test_cm_lsh_similarity_vectors(self):
        """Test CM-LSH similarity computation against Rust implementation.

        Note: Due to floating-point precision differences, we allow up to 1%
        relative difference in similarity scores.
        """
        vectors = load_vectors("cm_lsh.json")

        # Create hasher
        cm_lsh = create_default_cm_lsh(384, family=0)

        for case in vectors["similarity_vectors"]:
            name = case["name"]
            e1 = case["embedding1"]
            e2 = case["embedding2"]
            expected_sim = case["similarity"]

            # Compute similarity
            h1 = cm_lsh.hash(e1)
            h2 = cm_lsh.hash(e2)
            actual_sim = cm_lsh.sim(h1, h2)

            # Allow 1% relative difference or 0.01 absolute difference (whichever is larger)
            relative_diff = abs(actual_sim - expected_sim) / max(abs(expected_sim), 0.01)
            absolute_diff = abs(actual_sim - expected_sim)

            assert relative_diff < 0.01 or absolute_diff < 0.01, (
                f"{name}: similarity mismatch\n"
                f"Expected: {expected_sim:.6f}\n"
                f"Actual:   {actual_sim:.6f}\n"
                f"Relative diff: {relative_diff:.2%}\n"
                f"Absolute diff: {absolute_diff:.6f}"
            )

    def test_cm_lsh_self_similarity(self):
        """Test that identical vectors have similarity ~1.0."""
        cm_lsh = create_default_cm_lsh(384, family=0)

        vector = list(np.random.randn(384))
        h1 = cm_lsh.hash(vector)
        h2 = cm_lsh.hash(vector)

        similarity = cm_lsh.sim(h1, h2)
        assert similarity > 0.99, f"Self-similarity should be ~1.0, got {similarity}"

    def test_cm_lsh_lsh_ts_compatibility(self):
        """Test that first 256 bits match standalone LSH-TS."""
        cm_lsh = create_default_cm_lsh(384, family=0)

        vector = list(np.random.randn(384))

        # Verify LSH-TS compatibility
        assert cm_lsh.verify_lsh_ts(vector), "CM-LSH should be LSH-TS compatible"
