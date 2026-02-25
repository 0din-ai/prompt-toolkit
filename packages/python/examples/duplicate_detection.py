#!/usr/bin/env python3
"""Duplicate detection using LSH bands.

This example demonstrates:
- Batch signature generation for multiple vectors
- Using LSH bands for efficient candidate generation
- Finding near-duplicates in a collection

Run with: python python/examples/duplicate_detection.py
"""

from collections import defaultdict
from odin_sig import (
    cosine_from_hamming,
    hamming_distance_hex,
    normalize_vector,
    simhash_lsh_multi,
)


def main():
    print("=== Duplicate Detection with LSH ===\n")

    # Example vectors representing different documents
    # Vectors 0, 1, 2 are similar (duplicates)
    # Vectors 3, 4 are different
    vectors = [
        [1.0, 1.0, 1.0, 1.0],  # Doc 0
        [1.0, 0.95, 1.05, 1.0],  # Doc 1 (near-duplicate of 0)
        [0.98, 1.02, 1.0, 1.01],  # Doc 2 (near-duplicate of 0, 1)
        [0.0, 1.0, 0.0, 1.0],  # Doc 3 (different)
        [-1.0, -1.0, -1.0, -1.0],  # Doc 4 (opposite, different)
    ]

    print(f"Processing {len(vectors)} documents...\n")

    # Normalize and generate signatures
    signatures = [simhash_lsh_multi(normalize_vector(v)) for v in vectors]

    # Build band index for candidate generation
    # Map: (band_index, band_value) -> [doc_ids]
    band_index = defaultdict(list)

    for doc_id, sig in enumerate(signatures):
        # Use first family only for this example
        family = sig[0]

        for band_idx, band_value in enumerate(family.bands):
            key = (band_idx, band_value)
            band_index[key].append(doc_id)

    # Find candidate pairs (documents that share at least one band)
    candidates = set()

    for docs in band_index.values():
        if len(docs) > 1:
            # Multiple documents match this band
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    pair = tuple(sorted([docs[i], docs[j]]))
                    candidates.add(pair)

    print(f"Found {len(candidates)} candidate pairs from band matching\n")

    # Verify candidates with full Hamming distance
    threshold = 0.85  # Cosine similarity threshold for duplicates
    duplicates = []

    for id1, id2 in candidates:
        sig1 = signatures[id1][0].signature
        sig2 = signatures[id2][0].signature

        hamming = hamming_distance_hex(sig1, sig2)
        similarity = cosine_from_hamming(hamming, 256)

        if similarity >= threshold:
            duplicates.append((id1, id2, similarity))

    # Sort by similarity (descending)
    duplicates.sort(key=lambda x: x[2], reverse=True)

    print(f"Detected duplicates (similarity >= {threshold}):\n")
    for id1, id2, sim in duplicates:
        print(f"  Doc {id1} <-> Doc {id2}: {sim:.4f}")

    print("\n✓ Duplicate detection complete!")
    print("\nKey insight:")
    print("  - Band matching reduces comparisons from O(n²) to O(n)")
    print("  - Only candidate pairs need full Hamming distance computation")
    print("  - Tune bands/bits ratio for precision/recall tradeoff")


if __name__ == "__main__":
    main()
