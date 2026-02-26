# odin-sig-native

Native Rust acceleration for the Python odin-sig SDK.

This package provides PyO3-based native extensions that accelerate the performance-critical LSH signature generation functions by ~627× compared to pure Python.

## What it accelerates

- `simhash_lsh_multi()` — SimHash LSH signature generation (99% of CPU time)
- `normalize_vector()` — L2 vector normalization
- `hamming_distance_hex()` — Hamming distance between hex signatures
- `cosine_from_hamming()` — Cosine similarity estimation
- `compute_embedding_sha256()` — Canonical embedding hash

## Installation

### From source (development)

```bash
cd packages/python-native

# Install maturin if needed
pip install maturin

# Build and install in development mode
maturin develop --release
```

This builds the native extension and installs it into your current Python environment.

### As a wheel

```bash
# Build wheel
maturin build --release

# Install the wheel
pip install target/wheels/odin_sig_native-*.whl
```

## Usage

The native extension is **transparent** — once installed, the Python SDK automatically uses it:

```python
from odin_sig import simhash_lsh_multi, NATIVE_AVAILABLE

# Check if native acceleration is active
print(f"Native acceleration: {NATIVE_AVAILABLE}")

# This will use Rust if available, pure Python otherwise
signatures = simhash_lsh_multi(normalized_vector)
```

## Requirements

- **Rust**: 1.70 or newer
- **Python**: 3.10 or newer
- **PyO3**: 0.23+ (automatically handled by maturin)

## Development

```bash
# Run tests against the native extension
cd ../python
maturin develop --release -m ../python-native/Cargo.toml
python -m pytest tests/test_vectors.py -v

# Benchmark
python demos/benchmark_native.py
```

## Architecture

This crate is a thin PyO3 wrapper around the core `odin-sig` Rust library. It contains **zero LSH logic** — all computation is delegated to `odin-sig` via path dependency.

```
odin-sig-python (this crate)
    ├── PyO3 type conversions
    ├── Python module setup
    └── depends on → odin-sig (../rust)
                       └── Core LSH implementation
```
