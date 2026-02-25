---
sidebar_position: 1
---

# Core Functions

The main LSH functions available in all three languages.

## sign_text / signText (Recommended)

**High-level convenience function for generating signatures from text prompts.**

This is the recommended API for most use cases. It handles the entire pipeline: embedding generation → normalization → LSH hashing → signature formatting.

**Rust**:
```rust
pub async fn sign_text(
    text: &str,
    provider: &dyn EmbeddingProvider,
    version: SignatureVersion,  // Use SignatureVersion::Latest for the latest model
    config: Option<LshConfig>,
) -> Result<SignatureResult>
```

**Python**:
```python
async def sign_text(
    text: str,
    provider: EmbeddingProvider,
    version: SignatureVersion = SignatureVersion.LATEST,
    config: Optional[LshConfig] = None,
) -> SignatureResult
```

**TypeScript**:
```typescript
async function signText(
  text: string,
  provider: EmbeddingProvider,
  version: SignatureVersion = SignatureVersion.LATEST,
  config?: LshConfig
): Promise<SignatureResult>
```

**Parameters:**
- `text`: The input text prompt to sign
- `provider`: An embedding provider (OpenAIProvider, OnnxProvider, or custom)
- `version`: Signature version (defaults to LATEST, which resolves to V1). Options: V0, V1, or LATEST
- `config`: Optional LSH configuration (defaults to 3 families, 256 bits, 16 bands)

**Returns:**
- `SignatureResult`: Complete result including:
  - Formatted signature string (e.g., `"0din-v1:8d000000..."`)
  - Provider and model metadata
  - Embedding SHA256 hash
  - LSH families and bands
  - Timing information

**Example:**
```rust
// Use latest model (recommended)
let result = sign_text("How do I reset my password?", &provider, SignatureVersion::Latest, None).await?;
println!("{}", result.to_signature_string());

// Or omit version in Python/TypeScript (defaults to LATEST)
// Python: result = await sign_text("How do I reset my password?", provider)
// TypeScript: const result = await signText("How do I reset my password?", provider);
```

---

## simhash_lsh_multi (Low-Level)

Generate LSH signatures from a normalized embedding vector.

**Rust**:
```rust
pub fn simhash_lsh_multi(normalized_vector: &[f32], config: &LshConfig) -> Vec<LshFamily>
```

**Python**:
```python
def simhash_lsh_multi(
    normalized_vector: list[float],
    families: int = 3,
    bits: int = 256,
    bands: int = 16
) -> list[LSHFamily]
```

**TypeScript**:
```typescript
function simhashLshMulti(
  normalizedVector: number[],
  config?: LshConfig
): LSHFamily[]
```

## normalize_vector

Normalize a vector to unit length (L2 norm).

## hamming_distance_hex

Compute Hamming distance between two hex signatures.

## cosine_from_hamming

Estimate cosine similarity from Hamming distance.

See the [Quick Start](../getting-started/quick-start) for usage examples.
