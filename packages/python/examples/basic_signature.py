#!/usr/bin/env python3
"""Basic LSH signature generation example.

This example demonstrates:
- Generating an LSH signature from a normalized vector
- Default configuration (3 families, 256 bits, 16 bands)
- Formatting and parsing signature strings

Run with: python python/examples/basic_signature.py
"""

from odin_prompt_toolkit import simhash_lsh_multi


def main():
    print("=== Basic LSH Signature Generation ===\n")

    # Example normalized vector (4 dimensions for clarity)
    # In practice, this would come from an embedding model (384 or 1536 dims)
    normalized_vector = [0.5, 0.5, 0.5, 0.5]

    print(f"Input vector: {normalized_vector}")
    print(f"Vector dimensions: {len(normalized_vector)}\n")

    # Generate LSH signatures with default configuration
    families_count = 3
    bits = 256
    bands = 16

    print("Configuration:")
    print(f"  Families: {families_count}")
    print(f"  Bits per signature: {bits}")
    print(f"  Bands: {bands}\n")

    families = simhash_lsh_multi(normalized_vector, families=families_count, bits=bits, bands=bands)

    # Display results for each family
    for family in families:
        print(f"Family {family.family}:")
        print(f"  Signature (hex): {family.signature}")
        print(f"  Signature length: {len(family.signature)} hex chars = {family.bits} bits")
        print(f"  Number of bands: {len(family.bands)}")
        print(f"  Band 0: {family.bands[0]} (first {len(family.bands[0])} hex chars)")
        print()

    # Format as 0din signature string (V1 format)
    primary_sig = families[0].signature
    signature_string = f"0din-v1:{primary_sig}"

    print("Formatted signature string:")
    print(f"  {signature_string}")
    print()

    # In a real application, you would:
    # 1. Store this signature in a database with the original text
    # 2. Use bands for efficient similarity search (LSH indexing)
    # 3. Compare signatures using hamming distance

    print("✓ Signature generation complete!")


if __name__ == "__main__":
    main()
