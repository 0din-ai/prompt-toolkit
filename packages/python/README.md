# 0din-sig (Python)

Multi-language SDK for LSH (Locality-Sensitive Hashing) signature generation for AI prompt similarity detection.

This is the Python implementation of the 0din-sig algorithm, also available in [Rust](../rust) and [TypeScript](../typescript).

## Installation

### From Git (Development)

```bash
# Basic installation
pip install git+https://github.com/0din/sig-sdk.git#subdirectory=python

# With OpenAI support
pip install "0din-sig[openai] @ git+https://github.com/0din/sig-sdk.git#subdirectory=python"

# With ONNX support (local embeddings)
pip install "0din-sig[onnx] @ git+https://github.com/0din/sig-sdk.git#subdirectory=python"

# With CM-LSH (Confidence Matrix LSH)
pip install "0din-sig[cm-lsh] @ git+https://github.com/0din/sig-sdk.git#subdirectory=python"

# All features
pip install "0din-sig[all] @ git+https://github.com/0din/sig-sdk.git#subdirectory=python"
```

## Quick Start

### Basic LSH Signatures

```python
from odin_sig import simhash_lsh_multi, normalize_vector

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
from odin_sig import simhash_lsh_multi, hamming_distance_hex, cosine_from_hamming

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
from odin_sig.cm_lsh import create_default_cm_lsh

# Create CM-LSH hasher (384 dimensions)
cm_lsh = create_default_cm_lsh(384, family=0)

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
- **V1**: multilingual-e5-small ONNX (384 dimensions, local)
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
mypy odin_sig/
```

### Formatting

```bash
black odin_sig/ tests/
ruff check odin_sig/ tests/
```

## License

MIT
