#!/usr/bin/env python3
"""Similarity comparison example.

This example demonstrates:
- Comparing multiple vectors using LSH signatures
- Computing Hamming distance between signatures
- Estimating cosine similarity from Hamming distance

Run with: python python/examples/similarity_comparison.py
"""

from signature_sdk import (
    cosine_from_hamming,
    hamming_distance_hex,
    normalize_vector,
    simhash_lsh_multi,
)


def main():
    print("=== LSH Similarity Comparison ===\n")

    # Three example vectors (unnormalized)
    vector_a = [1.0, 1.0, 1.0, 1.0]  # Original
    vector_b = [1.0, 0.9, 1.1, 1.0]  # Similar (small perturbation)
    vector_c = [-1.0, -1.0, -1.0, -1.0]  # Opposite direction

    print("Input vectors:")
    print(f"  A: {vector_a}")
    print(f"  B: {vector_b} (similar to A)")
    print(f"  C: {vector_c} (opposite to A)\n")

    # Normalize all vectors
    norm_a = normalize_vector(vector_a)
    norm_b = normalize_vector(vector_b)
    norm_c = normalize_vector(vector_c)

    # Generate signatures
    sig_a = simhash_lsh_multi(norm_a)
    sig_b = simhash_lsh_multi(norm_b)
    sig_c = simhash_lsh_multi(norm_c)

    print("Signatures (first family only):")
    print(f"  A: {sig_a[0].signature[:16]}")
    print(f"  B: {sig_b[0].signature[:16]}")
    print(f"  C: {sig_c[0].signature[:16]}")
    print("     (showing first 16 hex chars of 64)\n")

    # Compare all pairs
    print("Pairwise comparisons:\n")

    compare_signatures("A vs B", sig_a[0].signature, sig_b[0].signature)
    compare_signatures("A vs C", sig_a[0].signature, sig_c[0].signature)
    compare_signatures("B vs C", sig_b[0].signature, sig_c[0].signature)

    print("\n✓ Comparison complete!")
    print("\nInterpretation:")
    print("  - Similarity > 0.9: Very similar")
    print("  - Similarity 0.7-0.9: Moderately similar")
    print("  - Similarity < 0.5: Dissimilar")


def compare_signatures(label: str, sig1: str, sig2: str) -> None:
    """Compare two signatures and print results."""
    hamming = hamming_distance_hex(sig1, sig2)
    similarity = cosine_from_hamming(hamming, 256)

    print(f"{label}:")
    print(f"  Hamming distance: {hamming}/256 bits differ")
    print(f"  Estimated cosine similarity: {similarity:.4f}")
    print()


if __name__ == "__main__":
    main()
