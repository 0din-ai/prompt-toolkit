---
sidebar_position: 2
---

# Types

Core data types used across all implementations.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Configuration Types

### LshConfig

Configuration parameters for LSH hashing algorithm.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct LshConfig {
    pub families: usize,  // Number of hash families (default: 3)
    pub bits: usize,      // Bits per signature (default: 256)
    pub bands: usize,     // Number of bands (default: 16)
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class LshConfig:
    families: int = 3   # Number of hash families
    bits: int = 256     # Bits per signature
    bands: int = 16     # Number of bands
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface LshConfig {
  families: number;  // Number of hash families (default: 3)
  bits: number;      // Bits per signature (default: 256)
  bands: number;     // Number of bands (default: 16)
}
```

</TabItem>
</Tabs>

**Fields:**
- `families`: Number of independent hash families to generate (increases recall)
- `bits`: Number of bits per signature (increases precision, default 256 = 64 hex chars)
- `bands`: Number of bands to divide signature into (for LSH bucketing)

**Default:** `{ families: 3, bits: 256, bands: 16 }`

---

## Version & Algorithm Types

### SignatureVersion

Enumeration of supported signature versions. Each version corresponds to a specific embedding model.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub enum SignatureVersion {
    V0,      // OpenAI text-embedding-3-large (1536 dims)
    V1,      // multilingual-e5-large ONNX (1024 dims)
    Latest,  // Resolves to V1
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
class SignatureVersion(str, Enum):
    V0 = "v0"          # OpenAI text-embedding-3-large (1536 dims)
    V1 = "v1"          # multilingual-e5-large ONNX (1024 dims)
    LATEST = "latest"  # Resolves to V1
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
enum SignatureVersion {
  V0 = 'v0',        // OpenAI text-embedding-3-large (1536 dims)
  V1 = 'v1',        // multilingual-e5-large ONNX (1024 dims)
  LATEST = 'latest' // Resolves to V1
}
```

</TabItem>
</Tabs>

:::warning Version Compatibility
V0 and V1 signatures are **NOT comparable** because they use different embedding models and dimensionalities. Always compare signatures from the same version.
:::

### HashAlgorithm

Algorithm identifier for hash method.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub enum HashAlgorithm {
    Lsh,     // Generic LSH
    OpenAI,  // OpenAI embeddings (V0)
    Onnx,    // ONNX local embeddings (V1)
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
class HashAlgorithm(str, Enum):
    LSH = "lsh"       # Generic LSH
    OPENAI = "openai" # OpenAI embeddings (V0)
    ONNX = "onnx"     # ONNX local embeddings (V1)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
enum HashAlgorithm {
  LSH = 'lsh',       // Generic LSH
  OPENAI = 'openai', // OpenAI embeddings (V0)
  ONNX = 'onnx'      // ONNX local embeddings (V1)
}
```

</TabItem>
</Tabs>

---

## LSH Output Types

### LshFamily

Result of LSH hashing for a single hash family.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct LshFamily {
    pub family: usize,      // Family index (0, 1, 2 for default config)
    pub bits: usize,        // Number of bits (256 by default)
    pub signature: String,  // Hex-encoded signature (64 chars for 256 bits)
    pub bands: Vec<String>, // Band slices (16 by default, 4 hex chars each)
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class LSHFamily:
    family: int            # Family index
    bits: int              # Number of bits
    signature: str         # Hex-encoded signature
    bands: list[str]       # Band slices
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface LSHFamily {
  family: number;   // Family index
  bits: number;     // Number of bits
  signature: string; // Hex-encoded signature
  bands: string[];   // Band slices
}
```

</TabItem>
</Tabs>

**Example:**
```json
{
  "family": 0,
  "bits": 256,
  "signature": "8d000000ac854dae7f3b9c1e...",
  "bands": ["8d00", "0000", "ac85", ...]
}
```

### LshOutput

Complete LSH hashing result with all families.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct LshOutput {
    pub families: Vec<LshFamily>,
    pub config: LshConfig,
    pub normalized_embedding_sha256: String,
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class LshOutput:
    families: list[LSHFamily]
    config: LshConfig
    normalized_embedding_sha256: str
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface LshOutput {
  families: LSHFamily[];
  config: LshConfig;
  normalizedEmbeddingSha256: string;
}
```

</TabItem>
</Tabs>

---

## Signature Types

### ParsedSignature

Result of parsing a signature string like `"0din-v1:8d000000..."`.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct ParsedSignature {
    pub version: SignatureVersion,
    pub signature: String,  // Hex signature (family 0 only)
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class ParsedSignature:
    version: SignatureVersion
    signature: str  # Hex signature (family 0 only)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface ParsedSignature {
  version: SignatureVersion;
  signature: string;  // Hex signature (family 0 only)
}
```

</TabItem>
</Tabs>

### SignatureResult

Complete result from `sign_text()` / `signText()` including metadata.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct SignatureResult {
    pub signature_string: String,     // Formatted: "0din-v1:..."
    pub version: SignatureVersion,
    pub algorithm: HashAlgorithm,
    pub lsh: LshOutput,
    pub embedding_result: EmbeddingResult,
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class SignatureResult:
    signature_string: str              # Formatted: "0din-v1:..."
    version: SignatureVersion
    algorithm: HashAlgorithm
    lsh: LshOutput
    embedding_result: EmbeddingResult
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface SignatureResult {
  signatureString: string;      // Formatted: "0din-v1:..."
  version: SignatureVersion;
  algorithm: HashAlgorithm;
  lsh: LshOutput;
  embeddingResult: EmbeddingResult;
}
```

</TabItem>
</Tabs>

---

## Embedding Types

### EmbeddingResult

Result from embedding generation (from provider).

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct EmbeddingResult {
    pub embedding: Vec<f32>,
    pub normalized_embedding: Vec<f32>,
    pub normalized_embedding_sha256: String,
    pub model: String,
    pub dimensions: usize,
    pub token_count: Option<usize>,
    pub timing_ms: Option<f64>,
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class EmbeddingResult:
    embedding: list[float]
    normalized_embedding: list[float]
    normalized_embedding_sha256: str
    model: str
    dimensions: int
    token_count: Optional[int] = None
    timing_ms: Optional[float] = None
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface EmbeddingResult {
  embedding: number[];
  normalizedEmbedding: number[];
  normalizedEmbeddingSha256: string;
  model: string;
  dimensions: number;
  tokenCount?: number;
  timingMs?: number;
}
```

</TabItem>
</Tabs>

**Fields:**
- `embedding`: Raw embedding vector from provider
- `normalized_embedding`: L2-normalized embedding (unit length)
- `normalized_embedding_sha256`: SHA256 hash of normalized embedding
- `model`: Model identifier (e.g., `"text-embedding-3-large"`, `"Alibaba-NLP/gte-large-en-v1.5"`)
- `dimensions`: Embedding dimensionality
- `token_count`: Token count (if provider reports it)
- `timing_ms`: Embedding generation time in milliseconds

---

## Comparison Types

### PromptInfo

Metadata about a single prompt in a comparison.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct PromptInfo {
    pub text: String,
    pub signature: String,
    pub embedding_sha256: String,
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class PromptInfo:
    text: str
    signature: str
    embedding_sha256: str
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface PromptInfo {
  text: string;
  signature: string;
  embeddingSha256: string;
}
```

</TabItem>
</Tabs>

### QualityStats

Quality metrics comparing estimated vs actual cosine similarity.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct QualityStats {
    pub absolute_error: f64,
    pub signed_error: f64,
    pub squared_error: f64,
    pub quality_rating: String,
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class QualityStats:
    absolute_error: float
    signed_error: float
    squared_error: float
    quality_rating: str
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface QualityStats {
  absoluteError: number;
  signedError: number;
  squaredError: number;
  qualityRating: string;
}
```

</TabItem>
</Tabs>

**Quality Ratings:**
- `"excellent"`: Error < 0.01
- `"good"`: Error < 0.05
- `"acceptable"`: Error < 0.10
- `"poor"`: Error ≥ 0.10

### ComparisonResult

Complete comparison result between two prompts.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct ComparisonResult {
    pub prompt_a: PromptInfo,
    pub prompt_b: PromptInfo,
    pub hamming_distance: u32,
    pub cosine_similarity: f64,
    pub lsh_config: LshConfig,
    pub quality_stats: Option<QualityStats>,
    pub timing_ms: Option<f64>,
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class ComparisonResult:
    prompt_a: PromptInfo
    prompt_b: PromptInfo
    hamming_distance: int
    cosine_similarity: float
    lsh_config: LshConfig
    quality_stats: Optional[QualityStats] = None
    timing_ms: Optional[float] = None
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface ComparisonResult {
  promptA: PromptInfo;
  promptB: PromptInfo;
  hammingDistance: number;
  cosineSimilarity: number;
  lshConfig: LshConfig;
  qualityStats?: QualityStats;
  timingMs?: number;
}
```

</TabItem>
</Tabs>

---

## Hasher Types

### Hasher

Protocol/trait/interface for pluggable hash algorithms.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub trait Hasher {
    fn hash(&self, normalized_vector: &[f32]) -> LshOutput;
    fn algorithm(&self) -> HashAlgorithm;
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
class Hasher(Protocol):
    """Protocol for hash algorithm implementations."""
    
    def hash(self, normalized_vector: list[float]) -> LshOutput:
        """Generate LSH signature from normalized vector."""
        ...
    
    def algorithm(self) -> HashAlgorithm:
        """Return the algorithm identifier."""
        ...
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface Hasher {
  hash(normalizedVector: number[]): LshOutput;
  algorithm(): HashAlgorithm;
}
```

</TabItem>
</Tabs>

**Implementations:**
- `SimHashLsh`: Random Hyperplane LSH (default)
- `HybridCMLSH`: Confidence Matrix LSH (advanced, includes confidence scores)

---

## Error Types

### SigError

Base error type for all library operations. See [Error Handling](./errors.md) for details.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub enum SigError {
    Config(String),
    Provider(String),
    Model(String),
    Io(std::io::Error),
    Serialization(serde_json::Error),
    InvalidInput(String),
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
class SigError(Exception):
    """Base exception for signature-sdk operations."""

# Specialized exceptions:
class ConfigError(SigError): ...
class ProviderError(SigError): ...
class ModelError(SigError): ...
class InvalidInputError(SigError): ...
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
class SigError extends Error { }

// Specialized error classes:
class ConfigError extends SigError { }
class ProviderError extends SigError { }
class ModelError extends SigError { }
class InvalidInputError extends SigError { }
```

</TabItem>
</Tabs>

---

## See Also

- [Core Functions](./core-functions) - Functions that operate on these types
- [Providers](./providers) - Embedding provider interfaces
- [Error Handling](./errors) - Error types and handling patterns
- [Quick Start](../getting-started/quick-start) - Usage examples
