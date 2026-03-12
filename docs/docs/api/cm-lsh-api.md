---
sidebar_position: 4
---

# CM-LSH API Reference

API reference for Confidence Matrix LSH (CM-LSH), an advanced LSH variant that includes confidence scores for improved accuracy.

:::info
CM-LSH is available in all three languages (Rust, Python, TypeScript) and provides ~5-10% better similarity estimation than standard LSH.
:::

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Overview

CM-LSH (Confidence Matrix LSH) extends standard Random Hyperplane LSH with:
- **Dual hash structure**: `hashA` (direction bits) + `hashB` (confidence bits)
- **LSH-TS + ITQ**: Combined hyperplane projections for better distribution
- **Isotonic calibration**: Maps raw similarity to calibrated cosine estimates
- **Confidence weighting**: Higher weight for high-confidence bit agreements

See [CM-LSH Concepts](../concepts/cm-lsh) for algorithm details.

---

## HybridCMLSH

Main CM-LSH hasher class.

### Constructor

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
impl HybridCMLSH {
    pub fn new(
        params: HybridParams,
        calibrator_config: CalibratorConfig,
        alpha: f32,      // Confidence weight (default: 0.65)
        family: usize,   // Family index (default: 0)
    ) -> Self
}
```

**Example:**
```rust
use signature_sdk::cm_lsh::{HybridCMLSH, create_default_cm_lsh};

// Use default factory (recommended)
let hasher = create_default_cm_lsh(1024, 0);

// Or construct manually
let hasher = HybridCMLSH::new(
    params,
    calibrator_config,
    0.65,  // alpha
    0,     // family
);
```

</TabItem>
<TabItem value="python" label="Python">

```python
class HybridCMLSH:
    def __init__(
        self,
        params: HybridParams,
        calibrator_config: CalibratorConfig,
        alpha: float = 0.65,    # Confidence weight
        family: int = 0,        # Family index
    )
```

**Example:**
```python
from signature_sdk.cm_lsh import HybridCMLSH, create_default_cm_lsh

# Use default factory (recommended)
hasher = create_default_cm_lsh(dimensions=384, family=0)

# Or construct manually
hasher = HybridCMLSH(
    params=params,
    calibrator_config=calibrator_config,
    alpha=0.65,
    family=0,
)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
class HybridCMLSH {
  constructor(
    params: HybridParams,
    calibratorConfig: CalibratorConfig,
    alpha?: number,   // Confidence weight (default: 0.65)
    family?: number   // Family index (default: 0)
  )
}
```

**Example:**
```typescript
import { HybridCMLSH, createDefaultCmLsh } from '@0din/signature-sdk';

// Use default factory (recommended)
const hasher = createDefaultCmLsh(384, 0);

// Or construct manually
const hasher = new HybridCMLSH(
  params,
  calibratorConfig,
  0.65,  // alpha
  0      // family
);
```

</TabItem>
</Tabs>

**Parameters:**
- `params`: Hyperplane parameters (LSH-TS + ITQ projections)
- `calibrator_config`: Isotonic calibration configuration
- `alpha`: Confidence weight (0-1, default 0.65). Higher = more weight to confident bits
- `family`: Family index for multi-family hashing (default 0)

---

### Methods

#### hash()

Generate CM-LSH dual hash from embedding.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn hash(&self, embedding: &[f32]) -> DualHash
```

**Example:**
```rust
let embedding = vec![0.1, 0.2, 0.3, /* ... */];
let hash = hasher.hash(&embedding);

println!("Signature: {}", hash.hash_a);  // Direction bits (hex)
println!("Confidence: {}", hash.hash_b);  // Confidence bits (hex)
```

</TabItem>
<TabItem value="python" label="Python">

```python
def hash(self, embedding: list[float]) -> DualHash
```

**Example:**
```python
embedding = [0.1, 0.2, 0.3, ...]
hash = hasher.hash(embedding)

print(f"Signature: {hash.hash_a}")    # Direction bits (hex)
print(f"Confidence: {hash.hash_b}")    # Confidence bits (hex)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
hash(embedding: number[]): DualHash
```

**Example:**
```typescript
const embedding = [0.1, 0.2, 0.3, ...];
const hash = hasher.hash(embedding);

console.log(`Signature: ${hash.hashA}`);    // Direction bits (hex)
console.log(`Confidence: ${hash.hashB}`);    // Confidence bits (hex)
```

</TabItem>
</Tabs>

**Parameters:**
- `embedding`: Input embedding vector (will be L2-normalized internally)

**Returns:**
- `DualHash` with `hashA` (512-bit signature), `hashB` (512-bit confidence), and bands

---

#### sim()

Compute calibrated similarity between two dual hashes.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn sim(&self, h1: &DualHash, h2: &DualHash) -> f64
```

**Example:**
```rust
let hash1 = hasher.hash(&embedding1);
let hash2 = hasher.hash(&embedding2);
let similarity = hasher.sim(&hash1, &hash2);

println!("Similarity: {:.3}", similarity);  // e.g., 0.847
```

</TabItem>
<TabItem value="python" label="Python">

```python
def sim(self, h1: DualHash, h2: DualHash) -> float
```

**Example:**
```python
hash1 = hasher.hash(embedding1)
hash2 = hasher.hash(embedding2)
similarity = hasher.sim(hash1, hash2)

print(f"Similarity: {similarity:.3f}")  # e.g., 0.847
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
sim(h1: DualHash, h2: DualHash): number
```

**Example:**
```typescript
const hash1 = hasher.hash(embedding1);
const hash2 = hasher.hash(embedding2);
const similarity = hasher.sim(hash1, hash2);

console.log(`Similarity: ${similarity.toFixed(3)}`);  // e.g., 0.847
```

</TabItem>
</Tabs>

**Returns:**
- Calibrated cosine similarity estimate in range `[0.0, 1.0]`

**Algorithm:**
1. Compute bit agreement rates (overall and confident-only)
2. Weight: `alpha × confident_rate + (1-alpha) × overall_rate`
3. Apply isotonic calibration mapping

---

#### cmp()

Compare two embeddings directly (convenience method = hash + sim).

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn cmp(&self, e1: &[f32], e2: &[f32]) -> f64
```

**Example:**
```rust
let similarity = hasher.cmp(&embedding1, &embedding2);
```

</TabItem>
<TabItem value="python" label="Python">

```python
def cmp(self, e1: list[float], e2: list[float]) -> float
```

**Example:**
```python
similarity = hasher.cmp(embedding1, embedding2)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
cmp(e1: number[], e2: number[]): number
```

**Example:**
```typescript
const similarity = hasher.cmp(embedding1, embedding2);
```

</TabItem>
</Tabs>

**Equivalent to:**
```
sim(hash(e1), hash(e2))
```

---

#### isDup()

Check if two hashes represent duplicates (similarity above threshold).

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn is_dup(&self, h1: &DualHash, h2: &DualHash, threshold: f64) -> bool
```

**Example:**
```rust
let is_duplicate = hasher.is_dup(&hash1, &hash2, 0.85);
if is_duplicate {
    println!("Duplicate detected!");
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
def is_dup(self, h1: DualHash, h2: DualHash, threshold: float = 0.85) -> bool
```

**Example:**
```python
if hasher.is_dup(hash1, hash2, threshold=0.85):
    print("Duplicate detected!")
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
isDup(h1: DualHash, h2: DualHash, threshold?: number): boolean
```

**Example:**
```typescript
if (hasher.isDup(hash1, hash2, 0.85)) {
  console.log('Duplicate detected!');
}
```

</TabItem>
</Tabs>

**Parameters:**
- `h1`, `h2`: Dual hashes to compare
- `threshold`: Similarity threshold (default: 0.85)

**Returns:**
- `true` if `sim(h1, h2) >= threshold`

---

#### verifyLshTs()

Verify LSH-TS compatibility with standard LSH (debugging/validation).

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn verify_lsh_ts(&self, embedding: &[f32]) -> String
```

</TabItem>
<TabItem value="python" label="Python">

```python
def verify_lsh_ts(self, embedding: list[float]) -> str
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
verifyLshTs(embedding: number[]): string
```

</TabItem>
</Tabs>

**Returns:**
- Hex signature from LSH-TS component only (first 256 bits of `hashA`)

**Use Case:** Verify that LSH-TS produces compatible signatures with standard LSH

---

## Factory Functions

### createDefaultCmLsh / create_default_cm_lsh

Create a HybridCMLSH instance with default parameters (recommended).

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn create_default_cm_lsh(dimensions: usize, family: usize) -> HybridCMLSH
```

**Example:**
```rust
use signature_sdk::cm_lsh::create_default_cm_lsh;

// For 1024-dimensional embeddings (V1/ONNX)
let hasher = create_default_cm_lsh(1024, 0);

// For 1536-dimensional embeddings (V0/OpenAI)
let hasher = create_default_cm_lsh(1536, 0);
```

</TabItem>
<TabItem value="python" label="Python">

```python
def create_default_cm_lsh(dimensions: int, family: int = 0) -> HybridCMLSH
```

**Example:**
```python
from signature_sdk.cm_lsh import create_default_cm_lsh

# For 1024-dimensional embeddings (V1/ONNX)
hasher = create_default_cm_lsh(1024, family=0)

# For 1536-dimensional embeddings (V0/OpenAI)
hasher = create_default_cm_lsh(1536, family=0)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function createDefaultCmLsh(dimensions: number, family?: number): HybridCMLSH
```

**Example:**
```typescript
import { createDefaultCmLsh } from '@0din/signature-sdk';

// For 1024-dimensional embeddings (V1/ONNX)
const hasher = createDefaultCmLsh(384, 0);

// For 1536-dimensional embeddings (V0/OpenAI)
const hasher = createDefaultCmLsh(1536, 0);
```

</TabItem>
</Tabs>

**Parameters:**
- `dimensions`: Embedding dimensionality (384 or 1536)
- `family`: Family index (default: 0)

**Defaults:**
- **Hyperplanes**: 512 bits (256 LSH-TS + 256 ITQ)
- **Alpha**: 0.65 (65% weight to confident bits)
- **Calibration**: Identity function (x -> x)
- **ITQ**: Identity rotation (no dimension reduction)

:::tip
For production use, consider training custom ITQ parameters and isotonic calibration on your data distribution for optimal accuracy.
:::

---

### genHyperplanes / gen_hyperplanes

Generate deterministic random hyperplanes for a specific family.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub fn gen_hyperplanes(family: usize, bits: usize, dims: usize) -> Vec<Vec<f32>>
```

**Example:**
```rust
use signature_sdk::cm_lsh::gen_hyperplanes;

// Generate 512 hyperplanes for 1024-dim embeddings, family 0
let planes = gen_hyperplanes(0, 512, 384);
// Returns: Vec<Vec<f32>> of shape [512, 384]
```

</TabItem>
<TabItem value="python" label="Python">

```python
def gen_hyperplanes(family: int, bits: int, dims: int) -> list[list[float]]
```

**Example:**
```python
from signature_sdk.cm_lsh import gen_hyperplanes

# Generate 512 hyperplanes for 1024-dim embeddings, family 0
planes = gen_hyperplanes(family=0, bits=512, dims=384)
# Returns: list[list[float]] of shape [512, 384]
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
function genHyperplanes(family: number, bits: number, dims: number): Float32Array[]
```

**Example:**
```typescript
import { genHyperplanes } from '@0din/signature-sdk';

// Generate 512 hyperplanes for 1024-dim embeddings, family 0
const planes = genHyperplanes(0, 512, 384);
// Returns: Float32Array[] of length 512, each with 384 elements
```

</TabItem>
</Tabs>

**Parameters:**
- `family`: Family index (seeds PRNG)
- `bits`: Number of hyperplanes to generate
- `dims`: Embedding dimensionality

**Returns:**
- Matrix of random unit vectors (deterministic based on family seed)

**Use Case:** 
- Building custom `HybridParams`
- Implementing multi-family CM-LSH

---

## Type Definitions

### DualHash

CM-LSH dual hash result.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct DualHash {
    pub hash_a: String,    // Direction bits (hex, 128 chars = 512 bits)
    pub hash_b: String,    // Confidence bits (hex, 128 chars = 512 bits)
    pub bands: Vec<String>, // Band slices for LSH bucketing
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class DualHash:
    hash_a: str           # Direction bits (hex, 128 chars)
    hash_b: str           # Confidence bits (hex, 128 chars)
    bands: list[str]      # Band slices
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface DualHash {
  hashA: string;      // Direction bits (hex, 128 chars)
  hashB: string;      // Confidence bits (hex, 128 chars)
  bands: string[];    // Band slices
}
```

</TabItem>
</Tabs>

**Example:**
```json
{
  "hashA": "8d000000ac854dae...",  // 512 bits (128 hex chars)
  "hashB": "ff1234567890abcd...",  // 512 bits (128 hex chars)
  "bands": ["8d00", "0000", "ac85", ...]
}
```

---

### ITQParams

Iterative Quantization parameters for dimension reduction.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct ITQParams {
    pub pca: Vec<Vec<f32>>,       // PCA projection matrix
    pub rotation: Vec<Vec<f32>>,  // ITQ rotation matrix
    pub mean: Vec<f32>,           // Centering mean
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class ITQParams:
    pca: list[list[float]]       # PCA projection matrix
    rotation: list[list[float]]  # ITQ rotation matrix
    mean: list[float]            # Centering mean
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface ITQParams {
  pca: Float32Array[];       // PCA projection matrix
  rotation: Float32Array[];  // ITQ rotation matrix
  mean: Float32Array;        // Centering mean
}
```

</TabItem>
</Tabs>

---

### HybridParams

Combined hyperplane parameters (LSH-TS + ITQ).

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct HybridParams {
    pub lsh_ts_hyperplanes: Vec<Vec<f32>>,  // 256 LSH-TS planes
    pub itq: ITQParams,                     // ITQ parameters
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class HybridParams:
    lsh_ts_hyperplanes: list[list[float]]  # 256 LSH-TS planes
    itq: ITQParams                         # ITQ parameters
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface HybridParams {
  lshTsHyperplanes: Float32Array[];  // 256 LSH-TS planes
  itq: ITQParams;                    // ITQ parameters
}
```

</TabItem>
</Tabs>

---

### CalibratorConfig

Isotonic calibration configuration.

<Tabs groupId="language">
<TabItem value="rust" label="Rust">

```rust
pub struct CalibratorConfig {
    pub x_thresh: Vec<f64>,  // Input thresholds
    pub y_thresh: Vec<f64>,  // Output (calibrated) values
    pub x_min: f64,          // Minimum input
    pub x_max: f64,          // Maximum input
}
```

</TabItem>
<TabItem value="python" label="Python">

```python
@dataclass
class CalibratorConfig:
    x_thresh: list[float]  # Input thresholds
    y_thresh: list[float]  # Output (calibrated) values
    x_min: float           # Minimum input
    x_max: float           # Maximum input
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
interface CalibratorConfig {
  xThresh: number[];  // Input thresholds
  yThresh: number[];  // Output (calibrated) values
  xMin: number;       // Minimum input
  xMax: number;       // Maximum input
}
```

</TabItem>
</Tabs>

**Purpose:** Maps raw bit agreement rates to calibrated cosine similarity estimates using piecewise linear interpolation (isotonic regression).

---

## See Also

- [CM-LSH Concepts](../concepts/cm-lsh) - Algorithm explanation and use cases
- [Core Functions](./core-functions) - Standard LSH functions
- [Types](./types) - Data type reference
- [Performance Guide](../guides/performance) - CM-LSH vs standard LSH benchmarks
