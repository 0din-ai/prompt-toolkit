#!/usr/bin/env python3
"""Confidence Matrix LSH (CM-LSH) example.

This example demonstrates:
- Enhanced LSH with confidence matrix
- Dual hash structure (512-bit signature + 512-bit confidence)
- Backward compatibility with standard LSH (first 256 bits)
- Calibrated similarity estimation

Run with: python python/examples/cm_lsh_example.py
"""

try:
    from odin_prompt_toolkit.cm_lsh import create_default_cm_lsh
    from odin_prompt_toolkit import normalize_vector
except ImportError:
    print("This example requires the CM-LSH implementation.")
    print("Install with: pip install -e '.[cm-lsh]'")
    exit(1)


def main():
    print("=== Confidence Matrix LSH (CM-LSH) ===\n")

    # Example vectors
    vector_a = [1.0, 1.0, 1.0, 1.0]
    vector_b = [1.0, 0.9, 1.1, 1.0]  # Similar to A
    vector_c = [-1.0, -1.0, -1.0, -1.0]  # Opposite to A

    print("Input vectors:")
    print(f"  A: {vector_a}")
    print(f"  B: {vector_b} (similar to A)")
    print(f"  C: {vector_c} (opposite to A)\n")

    # Normalize vectors
    norm_a = normalize_vector(vector_a)
    norm_b = normalize_vector(vector_b)
    norm_c = normalize_vector(vector_c)

    # Create CM-LSH hasher with default configuration
    # This uses identity ITQ (no learned rotation) for simplicity
    # Family 0 for deterministic results
    hasher = create_default_cm_lsh(len(norm_a), family=0)

    print("CM-LSH Configuration:")
    print("  Total bits: 512 (256 LSH-TS + 256 ITQ)")
    print("  First 256 bits: LSH-TS compatible")
    print("  Confidence matrix: Alpha-weighted agreement\n")

    # Generate dual hashes
    hash_a = hasher.hash(norm_a)
    hash_b = hasher.hash(norm_b)
    hash_c = hasher.hash(norm_c)

    print("Dual hashes (showing first 32 hex chars of 128):")
    print(f"  A: hash={hash_a.hash_a[:32]} conf={hash_a.hash_b[:32]}")
    print(f"  B: hash={hash_b.hash_a[:32]} conf={hash_b.hash_b[:32]}")
    print(f"  C: hash={hash_c.hash_a[:32]} conf={hash_c.hash_b[:32]}")
    print()

    # Demonstrate LSH-TS backward compatibility
    print("LSH-TS compatibility (first 256 bits):")
    print(f"  A: {hash_a.lsh_ts_compat()[:16]}")
    print(f"  B: {hash_b.lsh_ts_compat()[:16]}")
    print(f"  C: {hash_c.lsh_ts_compat()[:16]}")
    print("     (showing first 16 hex chars of 64)\n")

    # Compute calibrated similarities
    print("Calibrated similarities:\n")

    sim_ab = hasher.sim(hash_a, hash_b)
    sim_ac = hasher.sim(hash_a, hash_c)
    sim_bc = hasher.sim(hash_b, hash_c)

    print(f"  A vs B: {sim_ab:.4f}")
    print(f"  A vs C: {sim_ac:.4f}")
    print(f"  B vs C: {sim_bc:.4f}")

    print("\n✓ CM-LSH example complete!")
    print("\nKey advantages of CM-LSH:")
    print("  - Confidence matrix weights reliable bits higher")
    print("  - Isotonic calibration improves similarity estimates")
    print("  - Dual hash (LSH + ITQ) for better quantization")
    print("  - Backward compatible with standard LSH")


if __name__ == "__main__":
    main()
