---
sidebar_position: 3
---

# Configuration

Configure embedding providers, LSH parameters, and advanced options.

## LSH Configuration

Customize the hash generation parameters:

- **families**: Number of independent hash families (default: 3)
- **bits**: Bits per signature (default: 256)
- **bands**: Number of bands for indexing (default: 16)

See [Quick Start](./quick-start#configuration-options) for examples.

## Embedding Providers

### OpenAI (V0)

Requires API key:

```bash
export OPENAI_API_KEY=your-key-here
```

### ONNX (V1)

No configuration required. Model downloads automatically on first use to:
- `~/.cache/odin-sig/models/v1/` (Linux/macOS)
- `%LOCALAPPDATA%\odin-sig\models\v1\` (Windows)

## Next Steps

- [LSH Overview](../concepts/lsh-overview) — Understand the algorithm
- [API Reference](../api/core-functions) — Complete API documentation
