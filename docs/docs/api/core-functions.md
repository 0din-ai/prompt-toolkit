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

---

## normalize_vector

Normalize a vector to unit length (L2 norm = 1).

Required preprocessing step before LSH hashing. Ensures cosine similarity can be estimated from Hamming distance.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn normalize_vector(vector: &[f32]) -> Vec<f32>
```

**Parameters:**
- `vector`: Input vector (any dimensionality)

**Returns:**
- Normalized vector where `||v|| = 1`

**Example:**
```rust
use odin_prompt_toolkit::normalize_vector;

let vector = vec![3.0, 4.0];  // magnitude = 5
let normalized = normalize_vector(&vector);
// Result: [0.6, 0.8]
```

</TabItem>
<TabItem value="python" label="Python">

```python
def normalize_vector(vector: list[float]) -> list[float]
```

**Parameters:**
- `vector`: Input vector (any dimensionality)

**Returns:**
- Normalized vector where `||v|| = 1`

**Example:**
```python
from odin_prompt_toolkit import normalize_vector

vector = [3.0, 4.0]  # magnitude = 5
normalized = normalize_vector(vector)
# Result: [0.6, 0.8]
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function normalizeVector(vector: number[]): number[]
```

**Parameters:**
- `vector`: Input vector (any dimensionality)

**Returns:**
- Normalized vector where `||v|| = 1`

**Example:**
```typescript
import { normalizeVector } from '@0din/odin-prompt-toolkit';

const vector = [3.0, 4.0];  // magnitude = 5
const normalized = normalizeVector(vector);
// Result: [0.6, 0.8]
```

</TabItem>
</Tabs>

---

## hamming_distance_hex

Compute Hamming distance (number of differing bits) between two hex-encoded signatures.

Used to measure similarity between LSH signatures. Lower distance = higher similarity.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn hamming_distance_hex(hex_a: &str, hex_b: &str) -> u32
```

**Parameters:**
- `hex_a`: First signature (hex string, e.g., `"8d000000..."`)
- `hex_b`: Second signature (hex string, same length as `hex_a`)

**Returns:**
- Number of differing bits (0 to `4 × hex_length`)

**Example:**
```rust
use odin_prompt_toolkit::hamming_distance_hex;

let sig_a = "8d000000ac854dae";
let sig_b = "8d000000ac854daf";
let distance = hamming_distance_hex(sig_a, sig_b);
// Result: 1 (last bit differs)
```

</TabItem>
<TabItem value="python" label="Python">

```python
def hamming_distance_hex(hex_a: str, hex_b: str) -> int
```

**Parameters:**
- `hex_a`: First signature (hex string, e.g., `"8d000000..."`)
- `hex_b`: Second signature (hex string, same length as `hex_a`)

**Returns:**
- Number of differing bits (0 to `4 × hex_length`)

**Example:**
```python
from odin_prompt_toolkit import hamming_distance_hex

sig_a = "8d000000ac854dae"
sig_b = "8d000000ac854daf"
distance = hamming_distance_hex(sig_a, sig_b)
# Result: 1 (last bit differs)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function hammingDistanceHex(hexA: string, hexB: string): number
```

**Parameters:**
- `hexA`: First signature (hex string, e.g., `"8d000000..."`)
- `hexB`: Second signature (hex string, same length as `hexA`)

**Returns:**
- Number of differing bits (0 to `4 × hex_length`)

**Example:**
```typescript
import { hammingDistanceHex } from '@0din/odin-prompt-toolkit';

const sigA = "8d000000ac854dae";
const sigB = "8d000000ac854daf";
const distance = hammingDistanceHex(sigA, sigB);
// Result: 1 (last bit differs)
```

</TabItem>
</Tabs>

---

## cosine_from_hamming

Estimate cosine similarity from Hamming distance using the formula: `cos(π × d/n)` where `d` is Hamming distance and `n` is the number of bits.

This is based on the Random Hyperplane LSH theoretical relationship between Hamming distance and cosine similarity.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn cosine_from_hamming(hamming_distance: u32, bits: u32) -> f64
```

**Parameters:**
- `hamming_distance`: Hamming distance between signatures
- `bits`: Total number of bits in each signature (e.g., 256)

**Returns:**
- Estimated cosine similarity in range `[-1.0, 1.0]`

**Example:**
```rust
use odin_prompt_toolkit::{hamming_distance_hex, cosine_from_hamming};

let distance = hamming_distance_hex("8d00...", "8d01...");
let similarity = cosine_from_hamming(distance, 256);
// e.g., similarity ≈ 0.95 for small distance
```

</TabItem>
<TabItem value="python" label="Python">

```python
def cosine_from_hamming(hamming_distance: int, bits: int) -> float
```

**Parameters:**
- `hamming_distance`: Hamming distance between signatures
- `bits`: Total number of bits in each signature (e.g., 256)

**Returns:**
- Estimated cosine similarity in range `[-1.0, 1.0]`

**Example:**
```python
from odin_prompt_toolkit import hamming_distance_hex, cosine_from_hamming

distance = hamming_distance_hex("8d00...", "8d01...")
similarity = cosine_from_hamming(distance, 256)
# e.g., similarity ≈ 0.95 for small distance
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function cosineFromHamming(hammingDistance: number, bits: number): number
```

**Parameters:**
- `hammingDistance`: Hamming distance between signatures
- `bits`: Total number of bits in each signature (e.g., 256)

**Returns:**
- Estimated cosine similarity in range `[-1.0, 1.0]`

**Example:**
```typescript
import { hammingDistanceHex, cosineFromHamming } from '@0din/odin-prompt-toolkit';

const distance = hammingDistanceHex("8d00...", "8d01...");
const similarity = cosineFromHamming(distance, 256);
// e.g., similarity ≈ 0.95 for small distance
```

</TabItem>
</Tabs>

:::info
The cosine estimation is most accurate for similarities > 0.5. For very dissimilar vectors (cosine < 0), LSH provides weaker guarantees.
:::

---

## compute_embedding_sha256

Compute the SHA256 hash of a normalized embedding vector in canonical JSON format.

Used to uniquely identify embeddings and verify cross-language compatibility. All three implementations produce identical SHA256 hashes for identical vectors.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn compute_embedding_sha256(normalized_embedding: &[f32]) -> String
```

**Parameters:**
- `normalized_embedding`: Normalized vector (L2 norm = 1)

**Returns:**
- Hex-encoded SHA256 hash (64 characters)

**Example:**
```rust
use odin_prompt_toolkit::{normalize_vector, compute_embedding_sha256};

let vector = vec![0.5, 0.5, 0.5, 0.5];
let normalized = normalize_vector(&vector);
let hash = compute_embedding_sha256(&normalized);
// Result: "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
```

</TabItem>
<TabItem value="python" label="Python">

```python
def compute_embedding_sha256(normalized_embedding: list[float]) -> str
```

**Parameters:**
- `normalized_embedding`: Normalized vector (L2 norm = 1)

**Returns:**
- Hex-encoded SHA256 hash (64 characters)

**Example:**
```python
from odin_prompt_toolkit import normalize_vector, compute_embedding_sha256

vector = [0.5, 0.5, 0.5, 0.5]
normalized = normalize_vector(vector)
hash = compute_embedding_sha256(normalized)
# Result: "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function computeEmbeddingSha256(normalizedEmbedding: number[]): string
```

**Parameters:**
- `normalizedEmbedding`: Normalized vector (L2 norm = 1)

**Returns:**
- Hex-encoded SHA256 hash (64 characters)

**Example:**
```typescript
import { normalizeVector, computeEmbeddingSha256 } from '@0din/odin-prompt-toolkit';

const vector = [0.5, 0.5, 0.5, 0.5];
const normalized = normalizeVector(vector);
const hash = computeEmbeddingSha256(normalized);
// Result: "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
```

</TabItem>
</Tabs>

:::tip Cross-Language Verification
All three implementations use the same canonical JSON format: `{"embedding":[0.5,0.5,0.5,0.5]}` with no spaces and consistent float precision. This ensures bit-identical SHA256 hashes across languages.
:::

---

See the [Quick Start](../getting-started/quick-start) for usage examples.
