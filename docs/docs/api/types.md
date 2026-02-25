---
sidebar_position: 2
---

# Types

Core data types used across all implementations.

## LshConfig

Configuration for LSH hashing.

```rust
pub struct LshConfig {
    pub families: usize,  // Number of hash families (default: 3)
    pub bits: usize,      // Bits per signature (default: 256)
    pub bands: usize,     // Number of bands (default: 16)
}
```

## LshFamily

Result of LSH hashing for one family.

```rust
pub struct LshFamily {
    pub family: usize,     // Family index
    pub bits: usize,       // Number of bits
    pub signature: String, // Hex signature
    pub bands: Vec<String>, // Band slices
}
```

## SignatureVersion

```rust
pub enum SignatureVersion {
    V0,      // OpenAI embeddings (1536 dims)
    V1,      // ONNX embeddings (384 dims)
    Latest,  // Resolves to V1
}
```

See the [Quick Start](../getting-started/quick-start) for usage examples.
