# Installing odin-prompt-toolkit (Python)

## Requirements
- Python >= 3.10

## Install from wheel

```bash
pip install odin_prompt_toolkit-<VERSION>-py3-none-any.whl
```

Replace `<VERSION>` with the version number (e.g., `0.1.1`).

## Install with optional features

```bash
# OpenAI embeddings (API-based)
pip install "odin_prompt_toolkit-<VERSION>-py3-none-any.whl[openai]"

# ONNX embeddings (local, no API key needed)
pip install "odin_prompt_toolkit-<VERSION>-py3-none-any.whl[onnx]"

# Confidence Matrix LSH (higher accuracy)
pip install "odin_prompt_toolkit-<VERSION>-py3-none-any.whl[cm-lsh]"

# All features
pip install "odin_prompt_toolkit-<VERSION>-py3-none-any.whl[all]"
```

## Quick Start

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

## Signature Versions

- **V0**: OpenAI text-embedding-3-large (1536 dimensions, API-based)
- **V1**: 0din-jailbreak-embeddings-small ONNX (1024 dimensions, local)
- **Latest**: Resolves to V1

**Important**: V0 and V1 signatures are **not comparable** due to different embedding spaces.

## Documentation

For complete documentation, see:
- [Python SDK README](README.md)
- [Algorithm Specification](../../spec/SPEC.md)
- [Signature Versioning](../../spec/VERSIONING.md)
