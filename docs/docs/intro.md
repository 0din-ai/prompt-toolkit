---
sidebar_position: 1
slug: /
---

# Introduction to 0din-sig

**0din-sig** is a multi-language SDK for generating LSH (Locality-Sensitive Hashing) signatures from AI prompt embeddings. It provides fast, deterministic similarity detection across three languages: Rust, Python, and TypeScript.

## What is 0din-sig?

0din-sig implements **SimHash** via random hyperplane LSH ([Charikar 2002](https://dl.acm.org/doi/10.1145/509907.509965)), a proven algorithm for approximate nearest neighbor search. It converts high-dimensional embedding vectors (384-1536 dimensions) into compact 256-bit binary signatures that preserve cosine similarity.

### Key Features

- 🚀 **Fast** — 256-bit signatures enable O(1) similarity lookups in hash tables
- 🔒 **Deterministic** — Same input always produces the same signature
- 🌍 **Cross-language** — Identical signatures from Rust, Python, and TypeScript
- 📦 **No API required** — Local ONNX embeddings (V1) or OpenAI API (V0)
- 🎯 **Accurate** — Preserves cosine similarity via random hyperplane projections
- 🧪 **Battle-tested** — 61 tests validating 124 test cases across 3 languages

## Why Use LSH Signatures?

Traditional approaches to finding similar prompts require computing pairwise cosine similarities between all embeddings — **O(n²) comparisons** for a dataset of size n. With millions of prompts, this becomes prohibitively expensive.

LSH signatures enable **O(n) duplicate detection** through band-based candidate generation:

1. **Hash** each embedding into a 256-bit signature
2. **Split** signatures into 16 bands (16 hex chars each)
3. **Index** documents by band values in hash tables
4. **Query** candidates from matching bands (not all documents!)
5. **Verify** candidates with full Hamming distance

This reduces comparisons from **millions → hundreds** while maintaining high recall.

## Use Cases

- **Duplicate Detection** — Find near-duplicate AI prompts in large datasets
- **Similarity Search** — Build approximate nearest neighbor (ANN) systems
- **Content Deduplication** — Identify similar user inputs or generated content
- **Prompt Clustering** — Group similar prompts without expensive pairwise comparisons
- **Change Detection** — Monitor when prompt embeddings drift over time

## Quick Example

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use odin_sig::{sign_text, SignatureVersion};
use odin_sig::providers::{ModelCache, OnnxProvider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize local ONNX provider (no API key needed)
    let cache = ModelCache::new()?;
    let provider = OnnxProvider::new(&cache, None, None).await?;
    
    // Generate signature from text in one call (uses latest model: V1)
    let result = sign_text(
        "How do I reset my password?",
        &provider,
        SignatureVersion::Latest,
        None,
    ).await?;
    
    println!("{}", result.to_signature_string());
    // Output: 0din-v1:8d000000ac854dae...
    
    Ok(())
}
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
import asyncio
from odin_sig import sign_text, SignatureVersion
from odin_sig.providers import ModelCache, OnnxProvider

async def main():
    # Initialize local ONNX provider (no API key needed)
    cache = ModelCache()
    provider = await OnnxProvider.new(cache)
    
    # Generate signature from text in one call (uses latest model: V1)
    result = await sign_text(
        "How do I reset my password?",
        provider,
    )
    
    print(result.signature_string)
    # Output: 0din-v1:8d000000ac854dae...
    
    await provider.close()

asyncio.run(main())
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { signText, SignatureVersion, getSignatureString } from '@0din/sig';
import { ModelCache, OnnxProvider } from '@0din/sig/providers';

async function main() {
  // Initialize local ONNX provider (no API key needed)
  const cache = new ModelCache();
  const provider = await OnnxProvider.create(cache);
  
  // Generate signature from text in one call (uses latest model: V1)
  const result = await signText(
    "How do I reset my password?",
    provider,
  );
  
  console.log(getSignatureString(result));
  // Output: 0din-v1:8d000000ac854dae...
  
  await provider.close();
}

main();
```

  </TabItem>
</Tabs>

The `sign_text()` / `signText()` function is the **recommended high-level API** that handles the entire pipeline: embedding generation → normalization → LSH hashing → signature formatting.

For advanced use cases requiring manual embedding management, see the [Core Functions API](./api/core-functions).

## How It Works

0din-sig uses a deterministic LSH algorithm:

1. **Normalize** embedding to unit length (L2 norm)
2. **Generate** 256 random hyperplanes (deterministic via SplitMix64 PRNG)
3. **Project** normalized embedding onto each hyperplane (dot product)
4. **Quantize** projections to bits: `bit = 1 if dot > 0 else 0`
5. **Pack** 256 bits into a 64-character hex string
6. **Split** into 16 bands for LSH indexing

The hyperplanes are seeded by `(family << 48) ^ (bit << 24) ^ dimension`, ensuring the same hyperplanes are generated across all languages.

## Signature Versions

| Version | Provider | Model | Dimensions | Use Case |
|---------|----------|-------|------------|----------|
| **V0** | OpenAI | text-embedding-3-large | 1536 | API-based, production embeddings |
| **V1** | ONNX | multilingual-e5-small | 384 | Local, API-free, lower latency |

**Important:** V0 and V1 signatures use different embedding spaces and are **not comparable**.

## Next Steps

- **[Installation](./getting-started/installation)** — Install for Rust, Python, or TypeScript
- **[Quick Start](./getting-started/quick-start)** — Generate your first signature
- **[LSH Overview](./concepts/lsh-overview)** — Deep dive into the algorithm
- **[Guides](./guides/duplicate-detection)** — Build real-world applications

## Project Status

✅ **Production Ready** — All three language implementations validated with 109 passing tests

| Language | Package | Status | Tests |
|----------|---------|--------|-------|
| Rust | `odin-sig` | ✅ Ready | 50 passing |
| Python | `0din-sig` | ✅ Ready | 32 passing |
| TypeScript | `@0din/sig` | ✅ Ready | 27 passing |

See the [Validation Report](https://github.com/0din/sig-sdk/blob/main/VALIDATION.md) for detailed cross-language validation results.
