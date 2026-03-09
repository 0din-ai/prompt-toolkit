#!/usr/bin/env python3
"""Benchmark native vs pure Python LSH signature generation."""

import time
import sys


# Test with and without native
def benchmark(use_native: bool, iterations: int = 100):
    """Benchmark signature generation."""
    if use_native:
        # Import native version
        import signature_sdk_native as impl

        name = "Native (Rust)"
    else:
        # Import pure Python version by going directly to the _python functions
        sys.path.insert(0, "../python")
        from signature_sdk.lsh import (
            _simhash_lsh_multi_python as simhash_lsh_multi,
            _normalize_vector_python as normalize_vector,
        )

        impl_dict = {
            "simhash_lsh_multi": simhash_lsh_multi,
            "normalize_vector": normalize_vector,
        }

        class Impl:
            pass

        impl = Impl()
        for k, v in impl_dict.items():
            setattr(impl, k, v)
        name = "Pure Python"

    # Create test vector (384 dimensions, typical for V1 embeddings)
    test_vector = [float(i) / 384.0 for i in range(384)]
    normalized = impl.normalize_vector(test_vector)

    # Warm up
    for _ in range(10):
        impl.simhash_lsh_multi(normalized, families=3, bits=256, bands=16)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        impl.simhash_lsh_multi(normalized, families=3, bits=256, bands=16)
    elapsed = time.perf_counter() - start

    sigs_per_sec = iterations / elapsed
    ms_per_sig = (elapsed / iterations) * 1000

    print(f"{name}:")
    print(f"  Total time: {elapsed:.3f}s for {iterations} iterations")
    print(f"  Throughput: {sigs_per_sec:.1f} sigs/sec")
    print(f"  Latency: {ms_per_sig:.2f} ms/sig")

    return sigs_per_sec


if __name__ == "__main__":
    print(
        "Benchmarking LSH signature generation (384-dim vectors, 3 families × 256 bits)\n"
    )

    iterations = 1000

    # Benchmark pure Python
    python_speed = benchmark(use_native=False, iterations=iterations)
    print()

    # Benchmark native
    native_speed = benchmark(use_native=True, iterations=iterations)
    print()

    # Calculate speedup
    speedup = native_speed / python_speed
    print(f"🚀 Speedup: {speedup:.1f}× faster with native Rust extension")
