# odin-prompt-toolkit-native

Native Rust acceleration for the Python odin-prompt-toolkit SDK.

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
pip install target/wheels/odin_prompt_toolkit_native-*.whl
```

## Usage

The native extension is **transparent** — once installed, the Python SDK automatically uses it:

```python
from odin_prompt_toolkit import simhash_lsh_multi, NATIVE_AVAILABLE

# Check if native acceleration is active
print(f"Native acceleration: {NATIVE_AVAILABLE}")

# This will use Rust if available, pure Python otherwise
signatures = simhash_lsh_multi(normalized_vector)
```

## Requirements

- **Rust**: 1.70 or newer
- **Python**: 3.10 or newer
- **PyO3**: 0.23+ (automatically handled by maturin)

## Performance

Benchmark results (384-dim vectors, 3 families × 256 bits):

| Implementation | Throughput | Latency | Speedup |
|---------------|-----------|---------|---------|
| **Native (Rust)** | 5,685 sigs/sec | 0.18 ms/sig | **653×** |
| Pure Python | 8.7 sigs/sec | 115 ms/sig | 1× (baseline) |

## Development

```bash
# Build and test
cd packages/python-native
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release

# Run benchmark
python3 benchmark.py

# Run tests (from Python SDK directory)
cd ../python
source ../python-native/.venv/bin/activate
pip install pytest
python -m pytest tests/test_vectors.py -v
```

## Architecture

This crate is a thin PyO3 wrapper around the core `odin-prompt-toolkit` Rust library. It contains **zero LSH logic** — all computation is delegated to `odin-prompt-toolkit` via path dependency.

```
odin-prompt-toolkit-python (this crate)
    ├── PyO3 type conversions
    ├── Python module setup
    └── depends on → odin-prompt-toolkit (../rust)
                       └── Core LSH implementation
```
