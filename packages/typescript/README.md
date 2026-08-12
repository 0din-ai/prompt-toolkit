# @0din/odin-prompt-toolkitnature-sdk (TypeScript)

Multi-language SDK for LSH (Locality-Sensitive Hashing) signature generation for AI prompt similarity detection.

This is the TypeScript implementation of the odin-prompt-toolkit algorithm, also available in [Rust](../rust) and [Python](../python).

## Installation

### From Git (Development)

```bash
npm install git+https://github.com/0din-ai/odin-prompt-toolkit.git#main:typescript
# or
yarn add git+https://github.com/0din-ai/odin-prompt-toolkit.git#main:typescript
# or
pnpm add git+https://github.com/0din-ai/odin-prompt-toolkit.git#main:typescript
```

## Quick Start

### Basic LSH Signatures

```typescript
import { simhashLshMulti, normalizeVector } from '@0din/odin-prompt-toolkit';

// Your embedding vector (must be L2-normalized)
const vector = [0.5, 0.5, 0.5, 0.5];
const normalized = normalizeVector(vector);

// Generate LSH signatures (3 families, 256 bits, 16 bands)
const families = simhashLshMulti(normalized);

console.log(`Signature: ${families[0].signature}`);
console.log(`Bands: ${families[0].bands}`);
```

### Similarity Comparison

```typescript
import { 
  simhashLshMulti, 
  hammingDistanceHex, 
  cosineFromHamming 
} from '@0din/odin-prompt-toolkit';

// Generate signatures for two vectors
const families1 = simhashLshMulti(vector1);
const families2 = simhashLshMulti(vector2);

// Compute Hamming distance
const distance = hammingDistanceHex(
  families1[0].signature, 
  families2[0].signature
);

// Estimate cosine similarity
const similarity = cosineFromHamming(distance, 256);
console.log(`Estimated cosine similarity: ${similarity.toFixed(3)}`);
```

### Versioned Signatures

```typescript
import { 
  signatureString, 
  parseSignatureString,
  SignatureVersion 
} from '@0din/odin-prompt-toolkit';

// Format signature with version
const versionedSig = signatureString(SignatureVersion.V1, signature);
console.log(versionedSig); // "0din-v1:abcd1234..."

// Parse signature string
const parsed = parseSignatureString('0din-v1:abcd1234');
console.log(parsed.version); // 'v1'
console.log(parsed.signature); // 'abcd1234'
```

## API Reference

### Core Functions

#### `simhashLshMulti(vector, config?)`

Generate LSH signatures for a normalized vector.

**Parameters:**
- `vector: number[]` - L2-normalized embedding vector
- `config?: LshConfig` - Optional configuration
  - `families?: number` - Number of hash families (default: 3)
  - `bits?: number` - Bits per signature (default: 256)
  - `bands?: number` - Number of bands (default: 16)

**Returns:** `LSHFamily[]` - Array of signatures, one per family

#### `hammingDistanceHex(a, b)`

Compute Hamming distance between two hex signatures.

**Parameters:**
- `a: string` - First hex signature
- `b: string` - Second hex signature

**Returns:** `number` - Hamming distance in bits

#### `cosineFromHamming(distance, totalBits)`

Estimate cosine similarity from Hamming distance.

**Parameters:**
- `distance: number` - Hamming distance in bits
- `totalBits: number` - Total bits in signature

**Returns:** `number` - Estimated cosine similarity [-1, 1]

#### `normalizeVector(vector)`

L2-normalize a vector to unit length.

**Parameters:**
- `vector: number[]` - Input vector

**Returns:** `number[]` - Normalized vector

### Type Definitions

```typescript
interface LSHFamily {
  family: number;
  bits: number;
  signature: string; // hex string
  bands: string[]; // band slices
}

interface LshConfig {
  families?: number;
  bits?: number;
  bands?: number;
}

enum SignatureVersion {
  V0 = 'v0', // OpenAI (1536 dims)
  V1 = 'v1', // ONNX (1024 dims)
  LATEST = 'latest', // Resolves to V1
}
```

## Signature Versions

- **V0**: OpenAI text-embedding-3-large (1536 dimensions, API-based)
- **V1**: 0din-jailbreak-embeddings-small ONNX (1024 dimensions, local)
- **Latest**: Resolves to V1

**Important**: V0 and V1 signatures are **not comparable** due to different embedding spaces.

## Algorithm

SimHash via Random Hyperplane LSH (Charikar 2002):
- Deterministic hyperplanes via SplitMix64 PRNG
- Default: 3 families × 256 bits × 16 bands
- Hex-encoded signatures (64 hex chars = 256 bits)
- Hamming distance → cosine similarity via `cos(π × d/n)`

See the [specification](../../spec/SPEC.md) for complete algorithm details.

## Development

### Setup

```bash
cd typescript
npm install
```

### Build

```bash
npm run build
```

### Run Tests

```bash
npm test
```

### Linting & Formatting

```bash
npm run lint
npm run format
```

## License

Apache License 2.0
