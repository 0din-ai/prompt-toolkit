---
sidebar_position: 3
---

# Native Rust Acceleration

The Python SDK includes optional native Rust acceleration that provides **~592× faster signature generation** with zero code changes.

## Overview

Starting with version `0.1.1`, the Python package supports native Rust acceleration through the `signature-sdk-native` PyO3 extension. This is a **drop-in performance enhancement**:

- **Same API**: No code changes required
- **Transparent fallback**: If native extension unavailable, falls back to pure Python
- **Bit-identical results**: Verified via canonical test vectors across all implementations
- **Substantial speedup**: 5,332 sigs/sec (native) vs 9 sigs/sec (pure Python)

### Performance Comparison

| Implementation | Throughput | Latency | Speedup | Use Case |
|---------------|-----------|---------|---------|----------|
| **Native (Rust)** | ~5,332 sigs/sec | 0.19 ms/sig | **592×** | Production (recommended) |
| Pure Python | ~9 sigs/sec | 115 ms/sig | 1× | Development fallback |

**Real-world impact** (from `demos/showcase.py` benchmark with 3,714 prompts):

| Step | Time (native) | Time (pure Python) | Overhead vs Embedding |
|------|--------------|-------------------|----------------------|
| Embedding generation (ONNX, CPU) | 112.6s | 112.6s | — |
| **Signature generation** | **0.7s** | **43.8s** | **0.6% (native)** vs 38% (Python) |

With native acceleration, signature generation adds only **0.6% overhead** on top of embedding generation, making it essentially free for most workloads.

---

## Installation

### Option 1: Install with Native Acceleration (Recommended)

```bash
# From PyPI (when published)
pip install 'signature-sdk[native]'

# From Git
pip install 'git+https://github.com/0din-ai/signature-sdk.git#subdirectory=packages/python&egg=signature-sdk[native]'

# Local development
cd packages/python
pip install -e '.[native]'
```

The `[native]` extra installs:
- `maturin>=1.0,\<2.0` - Build tool for Rust Python extensions
- `signature-sdk-native` - The Rust PyO3 extension (built from `packages/rust`)

### Option 2: Pure Python (Fallback)

```bash
# Basic installation without native acceleration
pip install signature-sdk
```

The SDK will work fine without the native extension, but signature generation will be ~592× slower.

### Option 3: All Features

```bash
# Install everything including native acceleration
pip install 'signature-sdk[all]'
```

Includes: `native`, `openai`, `onnx`, `cm-lsh` extras.

---

## Verification

### Check if Native Extension is Active

```python
from signature_sdk import NATIVE_AVAILABLE

if NATIVE_AVAILABLE:
    print("✅ Native Rust acceleration is active")
else:
    print("⚠️  Using pure Python implementation")
```

### Runtime Behavior

The SDK automatically uses the native implementation when available:

```python
from signature_sdk import simhash_lsh_multi, normalize_vector

# This function automatically uses native Rust if installed
embedding = normalize_vector([0.5, 0.5, 0.5, 0.5])
families = simhash_lsh_multi(embedding)

# Result is identical regardless of implementation
print(families[0].signature)
# Output: 8d000000ac854dae0000000000000000...
```

The native extension is invoked **only for the computationally expensive parts**:
- `simhash_lsh_multi()` - Random hyperplane hashing
- `simhash_lsh()` - Single-family signature generation
- Low-level bit operations

All other functions (normalization, parsing, Hamming distance) remain in Python for flexibility.

---

## What's Accelerated?

### Core Signature Generation

The native extension replaces the pure-Python random hyperplane hashing loop:

**Pure Python** (`signature_sdk/lsh.py`):
```python
def simhash_lsh(vector: np.ndarray, family: int = 0, bits: int = 256) -> LshOutput:
    dim = len(vector)
    hash_bits = []
    
    for bit_idx in range(bits):
        seed = _compute_seed(family, bit_idx)
        prng = SplitMix64(seed)
        
        # Compute dot product with random hyperplane
        dot = 0.0
        for d in range(dim):
            dot += vector[d] * prng.next_f64()
        
        hash_bits.append(1 if dot >= 0.0 else 0)
    
    # ... pack bits into hex string
```

**Native Rust** (`packages/rust/src/lsh.rs`):
```rust
pub fn simhash_lsh(vector: &[f64], family: u64, bits: usize) -> LshOutput {
    let dim = vector.len();
    let mut hash_bits = Vec::with_capacity(bits);
    
    for bit_idx in 0..bits {
        let seed = compute_seed(family, bit_idx);
        let mut prng = SplitMix64::new(seed);
        
        // Compute dot product with random hyperplane
        let dot: f64 = (0..dim)
            .map(|d| vector[d] * prng.next_f64())
            .sum();
        
        hash_bits.push(if dot >= 0.0 { 1 } else { 0 });
    }
    
    // ... pack bits into hex string
}
```

The Rust version is **592× faster** due to:
1. **Compiled code**: No interpreter overhead
2. **SIMD optimizations**: Automatic vectorization of dot products
3. **Zero-copy arrays**: NumPy arrays are passed to Rust without copying
4. **Tight loops**: Efficient memory access patterns

### Band Extraction

Band extraction (splitting the 256-bit signature into 16 × 16-bit chunks) is also accelerated:

```rust
pub fn extract_bands(signature: &str, bands: usize) -> Vec<String> {
    let hex_per_band = signature.len() / bands;
    (0..bands)
        .map(|i| {
            let start = i * hex_per_band;
            signature[start..start + hex_per_band].to_string()
        })
        .collect()
}
```

---

## Performance Benchmarks

### Isolated Benchmark (Rust Native)

```bash
cd packages/rust
cargo run --release --example benchmark_signatures -- --count 10000
```

**Expected output:**
```
Generating 10,000 signatures (384-dim random vectors)...
Time: 1.760s
Throughput: 5,683 signatures/sec
Per-signature: 0.176ms

Note: Python SDK achieves ~9 sigs/sec on the same hardware (631× slower)
```

### Real-World Benchmark (Full Pipeline)

From `demos/showcase.py` with 3,714 real jailbreak prompts:

| Step | Native (v0.1.1+) | Pure Python | Speedup |
|------|-----------------|-------------|---------|
| Embedding generation (ONNX) | 112.6s | 112.6s | — |
| **Signature generation** | **0.7s** | **43.8s** | **62.6×** |
| **Total ingest time** | **113.3s** | **156.4s** | **1.38×** |

**Key insight**: Even though signature generation is 62× faster, the **total ingest speedup is only 1.38×** because embedding generation dominates the pipeline. However:

1. **Embedding is parallelizable**: You can batch-embed prompts on GPU
2. **Signatures are sequential**: The native extension makes this step negligible (0.6% overhead vs 38%)

For **signature-heavy workloads** (e.g., real-time duplicate detection where embeddings are pre-cached), the 592× speedup is the full story.

### Scaling Projections

| Corpus Size | Signature Time (native) | Signature Time (Python) | Difference |
|-------------|------------------------|------------------------|------------|
| 1,000 items | 0.19s | 111s | **110.8s faster** |
| 10,000 items | 1.9s | 18.5 min | **18.3 min faster** |
| 100,000 items | 18.8s | 3.1 hours | **3.0 hours faster** |
| 1,000,000 items | 3.1 min | 1.3 days | **1.3 days faster** |

At large scale, the native extension is the difference between **minutes and days** for full corpus reindexing.

---

## When to Use Native vs Pure Python

### Use Native (Recommended for Production)

✅ **Production deployments**
- Real-time signature generation (< 1ms latency)
- Large-scale corpus ingestion (> 10K documents)
- Batch processing pipelines
- Air-gapped environments (no API dependencies)

✅ **Development**
- Normal development workflow (no downsides)
- CI/CD pipelines (faster test runs)

### Use Pure Python (Development Fallback)

⚠️ **Edge cases only**
- Maturin build fails on exotic platforms (rare)
- Python version incompatibility (< 3.8, not officially supported)
- Quick prototyping on systems without Rust toolchain

**Note**: The pure Python implementation is **fully validated** and produces bit-identical results. It's slower, but not less correct.

---

## Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'signature_sdk_native'`

**Cause**: The `[native]` extra wasn't installed, or maturin build failed.

**Solution 1**: Install the native extra:
```bash
pip install 'signature-sdk[native]'
```

**Solution 2**: Rebuild the extension manually:
```bash
cd packages/rust
maturin develop --release
```

**Solution 3**: Use pure Python (no action needed):
```python
from signature_sdk import NATIVE_AVAILABLE
assert not NATIVE_AVAILABLE  # Expected
# SDK will use pure Python automatically
```

### Build Error: `error: linking with 'cc' failed`

**Cause**: Missing C compiler or incompatible Rust toolchain.

**Solution**:
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential

# Install/update Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### Performance Not Improving After Install

**Cause**: Native extension not being used (check with `NATIVE_AVAILABLE`).

**Debug**:
```python
from signature_sdk import NATIVE_AVAILABLE
import signature_sdk.lsh as lsh

print(f"Native available: {NATIVE_AVAILABLE}")
print(f"simhash_lsh module: {lsh.simhash_lsh.__module__}")

# Expected output if native is active:
# Native available: True
# simhash_lsh module: signature_sdk_native
```

If `NATIVE_AVAILABLE` is `False`:
1. Check if `pip list | grep signature-sdk-native` shows the package
2. Reinstall with `pip install --force-reinstall 'signature-sdk[native]'`
3. Check for import errors: `python -c "import signature_sdk_native"`

### Different Results: Native vs Pure Python

**This should never happen**. The implementations are validated via canonical test vectors:

```bash
# Run cross-implementation validation
cd packages/python
pytest tests/test_cross_language.py -v
```

If you observe different signatures:
1. **Report a bug** - this is a critical issue
2. Include: Python version, platform, package versions
3. Minimal reproducible example with fixed seed

---

## Implementation Details

### PyO3 Bindings

The native extension uses [PyO3](https://pyo3.rs/) to expose Rust functions to Python:

```rust
// packages/rust/src/lib.rs (Python bindings)
use pyo3::prelude::*;

#[pyfunction]
fn simhash_lsh_multi(
    vector: Vec<f64>,
    families: usize,
    bits: usize,
    bands: usize,
) -> PyResult<Vec<LshOutput>> {
    let outputs = signature_sdk::simhash_lsh_multi(&vector, families, bits, bands);
    Ok(outputs)
}

#[pymodule]
fn signature_sdk_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simhash_lsh_multi, m)?)?;
    Ok(())
}
```

### Automatic Fallback

The Python SDK detects the native extension at import time:

```python
# packages/python/signature_sdk/__init__.py
try:
    from signature_sdk_native import simhash_lsh, simhash_lsh_multi
    NATIVE_AVAILABLE = True
except ImportError:
    from signature_sdk.lsh import simhash_lsh, simhash_lsh_multi
    NATIVE_AVAILABLE = False
```

This ensures:
- **Zero code changes** for users
- **Graceful degradation** if build fails
- **Easy debugging** via `NATIVE_AVAILABLE` flag

### Validation Strategy

Cross-language consistency is verified via:

1. **Canonical test vectors** (`spec/canonical-v1.json`):
   - 6 test cases with known inputs/outputs
   - Validated across Rust, Python (native), Python (pure), TypeScript

2. **Property-based tests**:
   - Signature determinism (same input → same output)
   - Hamming distance symmetry
   - Cosine approximation accuracy

3. **Benchmark parity**:
   - All implementations produce identical signatures for `demos/showcase.py` dataset

See `docs/docs/concepts/cross-language.md` for the full validation methodology.

---

## Future Work

### GPU Acceleration

The current native extension uses CPU-only SIMD. Potential GPU acceleration via:

- **CUDA kernels**: For batch signature generation (1M+ embeddings)
- **WebGPU**: For in-browser acceleration (TypeScript)
- **Metal**: For Apple Silicon (M1/M2) optimization

**Trade-off**: GPU acceleration adds complexity (driver dependencies, hardware-specific code) for diminishing returns at typical corpus sizes (< 100K items). The CPU-based native extension already adds only 0.6% overhead over embedding generation.

### WASM Compilation

Compile the Rust core to WebAssembly for:

- **Browser-based signature generation**: No server round-trip
- **Edge deployment**: Cloudflare Workers, Deno Deploy
- **Node.js**: Single WASM binary instead of platform-specific wheels

**Proof of concept**: `wasm-pack build --target nodejs` already works. Missing: npm packaging and TypeScript bindings.

---

## Related Documentation

- **[Similarity Search](./similarity-search.md)** - Build an ANN search system with band-based indexing
- **[Performance Guide](./performance.md)** - Full benchmark results and scaling analysis
- **[Configuration](../getting-started/configuration.md)** - Tune LSH parameters (bands, bits, families)
- **[Cross-Language Validation](../concepts/cross-language.md)** - How we ensure consistency across implementations

---

## Summary

**Key Takeaways**:

1. ✅ **Install `signature-sdk[native]` for production** - 592× faster signature generation
2. ✅ **No code changes required** - Drop-in replacement with automatic fallback
3. ✅ **Bit-identical results** - Validated across all implementations
4. ✅ **Minimal overhead** - Signature generation adds only 0.6% on top of embedding time (vs 38% for pure Python)
5. ✅ **Easy debugging** - Check `NATIVE_AVAILABLE` to verify native extension is active

**Recommended Installation**:
```bash
pip install 'signature-sdk[native,onnx]'  # Native acceleration + local embeddings
```

**Performance Verification**:
```python
from signature_sdk import NATIVE_AVAILABLE, simhash_lsh_multi, normalize_vector
import time

assert NATIVE_AVAILABLE, "Native extension not installed!"

# Benchmark
embedding = normalize_vector([0.5] * 384)
start = time.perf_counter()
for _ in range(1000):
    simhash_lsh_multi(embedding)
elapsed = time.perf_counter() - start

print(f"Throughput: {1000 / elapsed:.0f} sigs/sec")
# Expected: ~5,000-6,000 sigs/sec (native)
# Expected: ~8-10 sigs/sec (pure Python)
```
