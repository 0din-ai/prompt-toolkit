---
sidebar_position: 2
---

# Quick Start

Generate your first LSH signature in less than 5 minutes.

## Prerequisites

- **Installation**: Follow the [Installation Guide](./installation) first
- **Basic understanding**: Familiarity with text embeddings

## Your First Signature (Recommended)

The fastest way to generate a signature is using the high-level `sign_text()` function with a local ONNX provider:

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use signature_sdk::{sign_text, SignatureVersion};
use signature_sdk::providers::{ModelCache, OnnxProvider};

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
    
    // Print formatted signature
    println!("{}", result.to_signature_string());
    // Output: 0din-v1:8d000000ac854dae...
    
    println!("Provider: {}", result.provider);
    println!("Model: {}", result.model);
    println!("Dimensions: {}", result.dimensions);
    
    Ok(())
}
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
import asyncio
from signature_sdk import sign_text, SignatureVersion
from signature_sdk.providers import ModelCache, OnnxProvider

async def main():
    # Initialize local ONNX provider (no API key needed)
    cache = ModelCache()
    provider = await OnnxProvider.new(cache)
    
    # Generate signature from text in one call (uses latest model: V1)
    result = await sign_text(
        "How do I reset my password?",
        provider,
    )
    
    # Print formatted signature
    print(result.signature_string)
    # Output: 0din-v1:8d000000ac854dae...
    
    print(f"Provider: {result.provider}")
    print(f"Model: {result.model}")
    print(f"Dimensions: {result.dimensions}")
    
    await provider.close()

asyncio.run(main())
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { signText, SignatureVersion, getSignatureString } from '@0din/signature-sdk';
import { ModelCache, OnnxProvider } from '@0din/signature-sdk/providers';

async function main() {
  // Initialize local ONNX provider (no API key needed)
  const cache = new ModelCache();
  const provider = await OnnxProvider.create(cache);
  
  // Generate signature from text in one call (uses latest model: V1)
  const result = await signText(
    "How do I reset my password?",
    provider,
  );
  
  // Print formatted signature
  console.log(getSignatureString(result));
  // Output: 0din-v1:8d000000ac854dae...
  
  console.log(`Provider: ${result.provider}`);
  console.log(`Model: ${result.model}`);
  console.log(`Dimensions: ${result.dimensions}`);
  
  await provider.close();
}

main();
```

  </TabItem>
</Tabs>

:::tip Recommended Approach
The `sign_text()` / `signText()` function is the **recommended API** for most use cases. It handles:
- Embedding generation (via OpenAI API or local ONNX)
- Vector normalization
- LSH signature computation
- Signature formatting

All in a single async function call!
:::

## Using OpenAI Provider

For production use with OpenAI's text-embedding-3-large model:

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use signature_sdk::{sign_text, SignatureVersion};
use signature_sdk::providers::OpenAIProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAIProvider::new(
        std::env::var("OPENAI_API_KEY")?,
        None, // model (defaults to text-embedding-3-large)
        None, // dimensions (defaults to 1536)
        None, // name
    );
    
    let result = sign_text(
        "How do I reset my password?",
        &provider,
        SignatureVersion::V0,  // V0 for 1536-dim embeddings
        None,
    ).await?;
    
    println!("{}", result.to_signature_string());
    
    Ok(())
}
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
import asyncio
import os
from signature_sdk import sign_text, SignatureVersion
from signature_sdk.providers import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    
    result = await sign_text(
        "How do I reset my password?",
        provider,
        SignatureVersion.V0,  # V0 for 1536-dim embeddings
    )
    
    print(result.signature_string)
    await provider.close()

asyncio.run(main())
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { signText, SignatureVersion, getSignatureString } from '@0din/signature-sdk';
import { OpenAIProvider } from '@0din/signature-sdk/providers';

async function main() {
  const provider = new OpenAIProvider({
    apiKey: process.env.OPENAI_API_KEY!,
  });
  
  const result = await signText(
    "How do I reset my password?",
    provider,
    SignatureVersion.V0,  // V0 for 1536-dim embeddings
  );
  
  console.log(getSignatureString(result));
  await provider.close();
}

main();
```

  </TabItem>
</Tabs>

## Low-Level API (Advanced)

For advanced use cases where you already have embeddings or need fine-grained control, you can use the core LSH functions directly:

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use signature_sdk::{simhash_lsh_multi, normalize_vector, LshConfig};

fn main() {
    // Your pre-computed embedding
    let embedding = vec![0.5; 384];
    
    // Normalize to unit length
    let normalized = normalize_vector(&embedding);
    
    // Generate LSH signatures
    let families = simhash_lsh_multi(&normalized, &LshConfig::default());
    
    // Access the signature
    println!("Signature: {}", families[0].signature);
}
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
from signature_sdk import simhash_lsh_multi, normalize_vector

# Your pre-computed embedding
embedding = [0.5] * 384

# Normalize to unit length
normalized = normalize_vector(embedding)

# Generate LSH signatures
families = simhash_lsh_multi(normalized)

# Access the signature
print(f"Signature: {families[0].signature}")
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { simhashLshMulti, normalizeVector } from '@0din/signature-sdk';

// Your pre-computed embedding
const embedding = new Array(384).fill(0.5);

// Normalize to unit length
const normalized = normalizeVector(embedding);

// Generate LSH signatures
const families = simhashLshMulti(normalized);

// Access the signature
console.log(`Signature: ${families[0].signature}`);
```

  </TabItem>
</Tabs>

See the [Core Functions API](../api/core-functions) for detailed documentation of all low-level functions.

## Compare Two Prompts

Calculate similarity between two embeddings:

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use signature_sdk::{
    simhash_lsh_multi, normalize_vector, hamming_distance_hex, 
    cosine_from_hamming, LshConfig
};

fn main() {
    let embedding1 = vec![1.0, 1.0, 1.0, 1.0];
    let embedding2 = vec![1.0, 0.9, 1.1, 1.0];  // Similar to embedding1
    
    let norm1 = normalize_vector(&embedding1);
    let norm2 = normalize_vector(&embedding2);
    
    let sig1 = simhash_lsh_multi(&norm1, &LshConfig::default());
    let sig2 = simhash_lsh_multi(&norm2, &LshConfig::default());
    
    // Compute Hamming distance
    let hamming = hamming_distance_hex(&sig1[0].signature, &sig2[0].signature);
    
    // Estimate cosine similarity
    let similarity = cosine_from_hamming(hamming, 256);
    
    println!("Hamming distance: {}/256 bits", hamming);
    println!("Estimated cosine similarity: {:.4}", similarity);
    // Output:
    // Hamming distance: 56/256 bits
    // Estimated cosine similarity: 0.7730
}
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
from signature_sdk import (
    simhash_lsh_multi, normalize_vector, 
    hamming_distance_hex, cosine_from_hamming
)

embedding1 = [1.0, 1.0, 1.0, 1.0]
embedding2 = [1.0, 0.9, 1.1, 1.0]  # Similar to embedding1

norm1 = normalize_vector(embedding1)
norm2 = normalize_vector(embedding2)

sig1 = simhash_lsh_multi(norm1)
sig2 = simhash_lsh_multi(norm2)

# Compute Hamming distance
hamming = hamming_distance_hex(sig1[0].signature, sig2[0].signature)

# Estimate cosine similarity
similarity = cosine_from_hamming(hamming, 256)

print(f"Hamming distance: {hamming}/256 bits")
print(f"Estimated cosine similarity: {similarity:.4f}")
# Output:
# Hamming distance: 56/256 bits
# Estimated cosine similarity: 0.7730
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import {
  simhashLshMulti, normalizeVector,
  hammingDistanceHex, cosineFromHamming
} from '@0din/signature-sdk';

const embedding1 = [1.0, 1.0, 1.0, 1.0];
const embedding2 = [1.0, 0.9, 1.1, 1.0];  // Similar to embedding1

const norm1 = normalizeVector(embedding1);
const norm2 = normalizeVector(embedding2);

const sig1 = simhashLshMulti(norm1);
const sig2 = simhashLshMulti(norm2);

// Compute Hamming distance
const hamming = hammingDistanceHex(sig1[0].signature, sig2[0].signature);

// Estimate cosine similarity
const similarity = cosineFromHamming(hamming, 256);

console.log(`Hamming distance: ${hamming}/256 bits`);
console.log(`Estimated cosine similarity: ${similarity.toFixed(4)}`);
// Output:
// Hamming distance: 56/256 bits
// Estimated cosine similarity: 0.7730
```

  </TabItem>
</Tabs>

## Understanding the Output

### Signature Structure

```
8d000000ac854dae91814006c580080a101141b001f30360003854003aba581a
│                                                              │
└──────────────────── 64 hex characters ──────────────────────┘
                     (256 bits / 4 = 64)
```

Each signature contains:
- **256 bits** of information
- **64 hex characters** (4 bits per character)
- **16 bands** of 4 characters each (for LSH indexing)

### Multiple Families

The default configuration generates **3 independent hash families**:

```rust
let families = simhash_lsh_multi(&normalized, &LshConfig::default());
println!("Family 0: {}", families[0].signature);
println!("Family 1: {}", families[1].signature);
println!("Family 2: {}", families[2].signature);
```

Multiple families improve recall in similarity search by providing different "views" of the same embedding.

### Bands

Each signature is split into **16 bands** for efficient indexing:

```rust
let family = &families[0];
println!("Band 0: {}", family.bands[0]);  // First 4 hex chars
println!("Band 1: {}", family.bands[1]);  // Next 4 hex chars
// ... 16 bands total
```

Bands enable O(n) candidate generation: if two documents share **any** band value, they're candidates for full comparison.

## Signature Format

Signatures can be formatted as strings for storage:

```rust
let signature_string = format!("0din-v1:{}", families[0].signature);
println!("{}", signature_string);
// Output: 0din-v1:8d000000ac854dae91814006c580080a101141b001f30360003854003aba581a
```

Format: `0din-v{version}:<hex_signature>`

- **v0**: OpenAI embeddings (1536 dimensions)
- **v1**: ONNX embeddings (384 dimensions)

:::warning Version Compatibility
V0 and V1 signatures are **not comparable** because they use different embedding spaces. Always compare signatures with the same version.
:::

## Configuration Options

Customize LSH parameters:

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
let config = LshConfig {
    families: 5,    // Generate 5 hash families (default: 3)
    bits: 512,      // Use 512 bits per signature (default: 256)
    bands: 32,      // Split into 32 bands (default: 16)
};

let families = simhash_lsh_multi(&normalized, &config);
```

  </TabItem>
  <TabItem value="python" label="Python">

```python
families = simhash_lsh_multi(
    normalized,
    families=5,  # Generate 5 hash families (default: 3)
    bits=512,    # Use 512 bits per signature (default: 256)
    bands=32     # Split into 32 bands (default: 16)
)
```

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
const families = simhashLshMulti(normalized, {
  families: 5,  // Generate 5 hash families (default: 3)
  bits: 512,    // Use 512 bits per signature (default: 256)
  bands: 32     // Split into 32 bands (default: 16)
});
```

  </TabItem>
</Tabs>

**Tuning guidelines:**
- **More families** → Higher recall, slower queries
- **More bits** → Better precision, larger storage
- **More bands** → More candidates, higher recall

## Next Steps

- **[Configuration Guide](./configuration)** — Learn about embedding providers and advanced options
- **[LSH Overview](../concepts/lsh-overview)** — Deep dive into how LSH works
- **[Duplicate Detection Guide](../guides/duplicate-detection)** — Build a real-world duplicate detector
- **[API Reference](../api/core-functions)** — Complete API documentation

## Common Patterns

### Store Signatures in Database

```python
# Generate signature
signature = simhash_lsh_multi(normalized)[0].signature
signature_string = f"0din-v1:{signature}"

# Store in database
db.execute(
    "INSERT INTO embeddings (text, signature) VALUES (?, ?)",
    (original_text, signature_string)
)
```

### Find Duplicates

```python
# Index by bands
for i, band in enumerate(families[0].bands):
    band_index[(i, band)].append(document_id)

# Query candidates
candidates = set()
for i, band in enumerate(query_bands):
    candidates.update(band_index.get((i, band), []))
```

See the [Duplicate Detection Guide](../guides/duplicate-detection) for a complete implementation.
