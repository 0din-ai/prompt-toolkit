# odin-prompt-toolkit (Python)

Multi-language SDK for LSH (Locality-Sensitive Hashing) signature generation for AI prompt similarity detection.

This is the Python implementation of the odin-prompt-toolkit algorithm, also available in [Rust](../rust) and [TypeScript](../typescript).

## Installation

`0din-prompt-toolkit` ships as **two packages** so you only ship compiled code when you want it:

| Package | What it is | When you get it |
|---------|-----------|-----------------|
| `0din-prompt-toolkit` | Pure-Python core. One universal (`py3-none-any`) wheel — installs on any OS/arch/Python, **no compiler**. | Always (base install) |
| `0din-prompt-toolkit-native` | Optional Rust accelerator (PyO3). **Prebuilt** wheels for Linux/macOS/Windows × CPython 3.10–3.13. | Only with the `[native]` extra |

```bash
# Pure Python — works everywhere, slower signature generation
pip install 0din-prompt-toolkit

# With native Rust acceleration — recommended for production
pip install "0din-prompt-toolkit[native]"

# Optional features
pip install "0din-prompt-toolkit[onnx]"        # local ONNX embeddings + SusFactor
pip install "0din-prompt-toolkit[openai]"      # OpenAI embeddings
pip install "0din-prompt-toolkit[cm-lsh]"      # Confidence Matrix LSH
pip install "0din-prompt-toolkit[threatfeed]"  # 0DIN threat feed
pip install "0din-prompt-toolkit[all]"         # everything, including native
```

From git (development):

```bash
pip install "0din-prompt-toolkit[native] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"
```

## Native vs pure Python

Both paths produce **bit-identical signatures** (verified across all implementations via canonical test vectors). They differ only in **how they install** and **how fast** signature generation runs.

- **Pure Python** (`0din-prompt-toolkit`) — a single universal wheel with no compiled code. Installs anywhere with zero build tools. Signature generation runs in a Python loop.
- **Native** (`[native]` → `0din-prompt-toolkit-native`) — a prebuilt compiled Rust extension. `pip` downloads a wheel matching your platform, so **no Rust toolchain or compiler is required** on Linux/macOS/Windows with CPython 3.10–3.13. On any platform without a matching wheel, the base package still installs and **automatically falls back to pure Python** — nothing breaks, it's just slower.

The accelerator is **transparent** — the same API uses native automatically when it's present:

```python
from odin_prompt_toolkit import NATIVE_AVAILABLE
print("native" if NATIVE_AVAILABLE else "pure Python")
```

Force pure Python even when native is installed: `export ODIN_PROMPT_TOOLKIT_NO_NATIVE=1`.

### Speed tradeoff

Native replaces the hot signature-generation loop with compiled, SIMD-optimized Rust:

| | Native (Rust) | Pure Python |
|---|---|---|
| Throughput | ~5,300 sigs/sec | ~85 sigs/sec (384-dim) … ~9 sigs/sec (1024-dim) |
| Per signature | ~0.2 ms | ~12 ms … ~115 ms |
| vs native | — | **~60×–600× slower** |

The multiplier depends on **embedding dimension**: pure Python loops over every dimension, so the gap widens with larger vectors (~63× at 384-dim, ~590× at 1024-dim). Native throughput is roughly constant.

**What it means end-to-end:** in a real pipeline, embedding generation usually dominates. On the 3,714-prompt benchmark (local ONNX, CPU) embedding took 112.6 s; adding signature generation cost **0.7 s with native (+0.6%)** vs **43.8 s pure Python (+38%)**. When embeddings are pre-computed or cached (e.g. real-time dedup), signature generation *is* the cost — and the full speedup applies.

**Rule of thumb:** use `[native]` in production; pure Python is a correct, always-available fallback for prototyping or unusual platforms.

## Quick Start

### Basic LSH Signatures

```python
from odin_prompt_toolkit import simhash_lsh_multi, normalize_vector

# Your embedding vector (must be L2-normalized)
vector = [0.5, 0.5, 0.5, 0.5]
normalized = normalize_vector(vector)

# Generate LSH signatures (3 families, 256 bits, 16 bands)
families = simhash_lsh_multi(normalized)

print(f"Signature: {families[0].signature}")
print(f"Bands: {families[0].bands}")
```

### Similarity Comparison

```python
from odin_prompt_toolkit import simhash_lsh_multi, hamming_distance_hex, cosine_from_hamming

# Generate signatures for two vectors
families1 = simhash_lsh_multi(vector1)
families2 = simhash_lsh_multi(vector2)

# Compute Hamming distance
distance = hamming_distance_hex(families1[0].signature, families2[0].signature)

# Estimate cosine similarity
similarity = cosine_from_hamming(distance, 256)
print(f"Estimated cosine similarity: {similarity:.3f}")
```

### Confidence Matrix LSH (CM-LSH)

```python
from odin_prompt_toolkit.cm_lsh import create_default_cm_lsh

# Create CM-LSH hasher (1024 dimensions)
cm_lsh = create_default_cm_lsh(1024, family=0)

# Generate 512-bit signature with confidence matrix
hash1 = cm_lsh.hash(embedding1)
hash2 = cm_lsh.hash(embedding2)

# Compute calibrated similarity
similarity = cm_lsh.sim(hash1, hash2)
print(f"CM-LSH similarity: {similarity:.3f}")

# Check for duplicates
is_duplicate = cm_lsh.is_dup(hash1, hash2, threshold=0.85)
```

## Signature Versions

- **V0**: OpenAI text-embedding-3-large (1536 dimensions, API-based)
- **V1**: 0din-jailbreak-embeddings-small ONNX (1024 dimensions, local)
- **Latest**: Resolves to V1

**Important**: V0 and V1 signatures are **not comparable** due to different embedding spaces.

## Algorithm

SimHash via Random Hyperplane LSH (Charikar 2002):
- Deterministic hyperplanes via SplitMix64 PRNG
- Default: 3 families × 256 bits × 16 bands
- Hex-encoded signatures (64 hex chars = 256 bits)
- Hamming distance → cosine similarity via `cos(π × d/n)`

See the [specification](../../spec/SPEC.md) for complete algorithm details.

## Development

### Setup

```bash
cd python
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy odin_prompt_toolkit/
```

### Formatting

```bash
black odin_prompt_toolkit/ tests/
ruff check odin_prompt_toolkit/ tests/
```

## License

Apache License 2.0
