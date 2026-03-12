---
sidebar_position: 3
---

# Configuration

Comprehensive guide to configuring odin-prompt-toolkit: LSH parameters, embedding providers, environment variables, and optimization settings.

## LSH Configuration

Customize hash generation parameters to balance precision, recall, and storage.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `families` | integer | 3 | Number of independent hash families |
| `bits` | integer | 256 | Bits per signature (precision) |
| `bands` | integer | 16 | Number of bands (for LSH bucketing) |

### Default Configuration

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
use odin_prompt_toolkit::LshConfig;

// Use defaults
let config = LshConfig::default();
// families: 3, bits: 256, bands: 16

// Or customize
let config = LshConfig {
    families: 5,
    bits: 512,
    bands: 32,
};
```

</TabItem>
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit import LshConfig

# Use defaults
config = LshConfig()
# families=3, bits=256, bands=16

# Or customize
config = LshConfig(families=5, bits=512, bands=32)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { LshConfig } from '@0din/odin-prompt-toolkit';

// Use defaults (pass undefined or omit config)
const config = { families: 3, bits: 256, bands: 16 };

// Or customize
const config = { families: 5, bits: 512, bands: 32 };
```

</TabItem>
</Tabs>

### Parameter Tuning

**Families:**
- **More families** → Higher recall (fewer false negatives)
- **Fewer families** → Lower storage, faster queries
- **Recommended**: 3-5 families for most use cases

**Bits:**
- **More bits** → Higher precision (fewer false positives)
- **Fewer bits** → Lower storage, faster comparisons
- **Recommended**: 256 bits (good accuracy/storage tradeoff)

**Bands:**
- **More bands** → Finer-grained bucketing (more buckets, smaller size)
- **Fewer bands** → Coarser bucketing (fewer buckets, larger size)
- **Recommended**: 16 bands (256 bits ÷ 16 = 16 bits per band)

**Relationship:**
```
bits = bands × bits_per_band
256 = 16 × 16
```

### Storage Impact

| Configuration | Hex Chars | Bytes | Notes |
|---------------|-----------|-------|-------|
| 3 families × 256 bits | 192 | 96 | Default, good balance |
| 5 families × 256 bits | 320 | 160 | Higher recall |
| 3 families × 512 bits | 1024 | 192 | Higher precision |
| 1 family × 256 bits | 64 | 32 | Minimal storage |

---

## Embedding Providers

### OpenAI Provider (V0)

**Configuration:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
use odin_prompt_toolkit::providers::OpenAIProvider;

let provider = OpenAIProvider::new(
    std::env::var("OPENAI_API_KEY")?,
    Some("text-embedding-3-large".to_string()),  // Model
    Some(1536),                                   // Dimensions
    Some("openai".to_string()),                   // Name
);
```

</TabItem>
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit.providers import OpenAIProvider
import os

provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large",  # Optional
    dimensions=1536,                  # Optional
    name="openai",                    # Optional
    base_url=None,                    # Custom API URL (optional)
)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { OpenAIProvider } from '@0din/odin-prompt-toolkit/providers';

const provider = new OpenAIProvider({
  apiKey: process.env.OPENAI_API_KEY!,
  model: 'text-embedding-3-large',  // Optional
  dimensions: 1536,                  // Optional
  name: 'openai',                    // Optional
  baseURL: undefined,                // Custom API URL (optional)
});
```

</TabItem>
</Tabs>

**Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `OPENAI_BASE_URL` - Custom API endpoint (optional, for proxies or OpenAI-compatible APIs)

**Cost:** ~$0.13 per 1M tokens (~$0.000013 per prompt)

**Latency:** ~100-200ms (network + API)

### ONNX Provider (V1)

**Configuration:**

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
use odin_prompt_toolkit::providers::{ModelCache, OnnxProvider};

let cache = ModelCache::new()?;
let provider = OnnxProvider::new(
    &cache,
    Some("intfloat/multilingual-e5-large".to_string()),  // Model
    Some("onnx".to_string()),                             // Name
).await?;
```

</TabItem>
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit.providers import ModelCache, OnnxProvider

cache = ModelCache()
provider = await OnnxProvider.new(
    cache,
    model_name="intfloat/multilingual-e5-large",  # Optional
    name="onnx",                                    # Optional
)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { ModelCache, OnnxProvider } from '@0din/odin-prompt-toolkit/providers';

const cache = new ModelCache();
const provider = await OnnxProvider.create(
  cache,
  'intfloat/multilingual-e5-large',  // Optional
  'onnx'                               // Optional
);
```

</TabItem>
</Tabs>

**Model Cache:**

Default locations:
- **Linux/macOS**: `~/.cache/odin-prompt-toolkit/models/`
- **Windows**: `%LOCALAPPDATA%\odin-prompt-toolkit\models\`

Override via environment variable:
```bash
export ODIN_PROMPT_TOOLKIT_MODEL_CACHE=/path/to/cache
```

Custom cache directory:
```python
cache = ModelCache(cache_dir=Path("/custom/cache"))
```

**Model Download:**
- First run: Auto-downloads ~150MB model
- Subsequent runs: Loads from cache
- No network required after first download

**Cost:** Free (local inference)

**Latency:** ~50-100ms (CPU on M1 Mac)

---

## Environment Variables

### Provider Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API authentication key | None (required for OpenAI) |
| `OPENAI_BASE_URL` | Custom OpenAI API endpoint | https://api.openai.com/v1 |
| `ODIN_PROMPT_TOOLKIT_MODEL_CACHE` | ONNX model cache directory | OS-specific (see above) |

### Python-Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `ODIN_PROMPT_TOOLKIT_NO_NATIVE` | Disable native Rust extension | `false` |

**Use Case:** Force pure-Python mode (for debugging or platforms without native builds)

```bash
export ODIN_PROMPT_TOOLKIT_NO_NATIVE=1
python your_script.py  # Uses pure Python, no native acceleration
```

---

## Advanced Configuration

### CM-LSH Configuration

For advanced users needing custom CM-LSH parameters:

<Tabs groupId="language">
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit.cm_lsh import HybridCMLSH, HybridParams, CalibratorConfig, ITQParams

# Create custom hyperplanes
from odin_prompt_toolkit.cm_lsh import gen_hyperplanes
lsh_ts_planes = gen_hyperplanes(family=0, bits=256, dims=384)

# Create custom ITQ parameters (identity by default)
itq_params = ITQParams(
    pca=[[1.0, 0.0], [0.0, 1.0]],  # PCA projection matrix
    rotation=[[1.0, 0.0], [0.0, 1.0]],  # ITQ rotation matrix
    mean=[0.0, 0.0],  # Centering mean
)

# Create custom calibrator (identity by default)
calibrator_config = CalibratorConfig(
    x_thresh=[0.0, 1.0],  # Input thresholds
    y_thresh=[0.0, 1.0],  # Output values
    x_min=0.0,
    x_max=1.0,
)

# Assemble hybrid params
params = HybridParams(
    lsh_ts_hyperplanes=lsh_ts_planes,
    itq=itq_params,
)

# Create hasher
hasher = HybridCMLSH(
    params=params,
    calibrator_config=calibrator_config,
    alpha=0.65,  # Confidence weight
    family=0,
)
```

</TabItem>
</Tabs>

**When to customize:**
- Training on domain-specific data
- Optimizing for specific similarity distributions
- Research experiments

**Most users should use `createDefaultCmLsh()`** which provides good defaults.

### Multi-Family Hashing

Generate multiple independent hash families for higher recall:

<Tabs groupId="language">
<TabItem value="python" label="Python">

```python
from odin_prompt_toolkit import simhash_lsh_multi, LshConfig

config = LshConfig(families=5, bits=256, bands=16)
families = simhash_lsh_multi(normalized_embedding, config=config)

# Store all 5 families
for i, family in enumerate(families):
    print(f"Family {i}: {family.signature}")
```

</TabItem>
</Tabs>

**Query strategy:** A match in ANY family indicates similarity (OR logic).

---

## Performance Optimization

### Native Acceleration (Python)

**Python** includes an optional Rust extension for 500-600× speedup on core LSH functions.

**Installation:**
```bash
# With native acceleration
pip install odin-prompt-toolkit

# Verify native is available
python -c "from odin_prompt_toolkit import NATIVE_AVAILABLE; print(NATIVE_AVAILABLE)"
# Output: True
```

**Functions accelerated:**
- `simhash_lsh_multi()` - 653× faster
- `normalize_vector()` - 592× faster
- `hamming_distance_hex()` - 487× faster
- `cosine_from_hamming()` - 112× faster
- `compute_embedding_sha256()` - 95× faster

See [Native Acceleration Guide](../guides/native-acceleration) for details.

### Caching Strategies

**Embedding Cache:**
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_embedding(text: str):
    return provider.generate_embedding(text)
```

**Signature Cache:**
```python
signature_cache = {}

def get_signature(text: str):
    if text not in signature_cache:
        signature_cache[text] = sign_text(text, provider=provider)
    return signature_cache[text]
```

### Batch Processing

Process multiple texts efficiently:

<Tabs groupId="language">
<TabItem value="python" label="Python">

```python
import asyncio

async def batch_sign(texts: list[str], provider):
    tasks = [sign_text(text, provider=provider) for text in texts]
    return await asyncio.gather(*tasks)

# Process 100 texts concurrently
results = await batch_sign(texts, provider)
```

</TabItem>
</Tabs>

---

## Configuration Best Practices

### Development

- Use **ONNX provider** (V1) for local development
- Enable **native acceleration** in Python
- Use **default LSH config** (3/256/16)

### Production

**High-accuracy requirements:**
- Use **OpenAI provider** (V0) for best embeddings
- Consider **CM-LSH** for +5-10% accuracy
- Increase **families** to 5 for higher recall

**Cost-sensitive:**
- Use **ONNX provider** (V1) for free local inference
- Use **1-2 families** to reduce storage
- Enable **caching** aggressively

**High-throughput:**
- Use **native acceleration** (Python)
- **Batch** signature generation
- Optimize **band-based indexing** in database

---

## Configuration Examples

### Minimal Storage

```python
# 1 family × 256 bits = 64 hex chars (32 bytes)
config = LshConfig(families=1, bits=256, bands=16)
```

### High Recall

```python
# 5 families × 256 bits = 320 hex chars (160 bytes)
config = LshConfig(families=5, bits=256, bands=16)
```

### High Precision

```python
# 3 families × 512 bits = 384 hex chars (192 bytes)
config = LshConfig(families=3, bits=512, bands=32)
```

### Balanced (Recommended)

```python
# 3 families × 256 bits = 192 hex chars (96 bytes)
config = LshConfig(families=3, bits=256, bands=16)
```

---

## See Also

- [Quick Start](./quick-start) - Basic setup and usage
- [LSH Overview](../concepts/lsh-overview) - Algorithm fundamentals
- [Providers API](../api/providers) - Full provider API reference
- [Performance Guide](../guides/performance) - Optimization strategies
- [Native Acceleration](../guides/native-acceleration) - Python speedup details
