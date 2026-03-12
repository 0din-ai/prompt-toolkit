---
sidebar_position: 1
---

# LSH Overview

Locality-Sensitive Hashing (LSH) is a technique for efficiently finding similar items in high-dimensional spaces.

## The Problem

Traditional similarity search requires comparing every pair of items:

```
For 1 million documents:
Comparisons needed = 1,000,000 × 999,999 / 2 = ~500 billion
```

This is computationally prohibitive.

## The Solution: LSH

LSH maps similar items to the same "buckets" with high probability:

1. **Hash** items into short signatures
2. **Index** by hash buckets
3. **Query** only items in matching buckets

Result: **O(n) instead of O(n²)**

## SimHash Algorithm

odin-prompt-toolkit uses SimHash via random hyperplane LSH ([Charikar 2002](https://dl.acm.org/doi/10.1145/509907.509965)):

```
Embedding Vector → Normalize → Generate Hyperplanes → 
Compute Dot Products → Sign Bits → Pack to Hex → 256-bit Signature
```

### Step 1: Normalize

Convert embedding to unit length:

```
normalized = embedding / ||embedding||
```

### Step 2: Random Hyperplanes

Generate 256 deterministic random hyperplanes using SplitMix64 PRNG with seed:

```
seed = (family << 48) ^ (bit << 24) ^ dimension
```

### Step 3: Project & Quantize

For each hyperplane, compute dot product and extract sign bit:

```
bit = 1 if dot(normalized, hyperplane) > 0 else 0
```

### Step 4: Pack to Hex

Pack 256 bits into 64 hex characters.

## Why It Works

**Key insight**: If two vectors are similar (high cosine similarity), they're likely to have the same sign when projected onto a random hyperplane.

**Probability**: `P(same_sign) = 1 - θ/π` where θ is the angle between vectors.

For cosine similarity 0.9 (θ ≈ 25.8°):
- `P(same_sign) ≈ 0.92`
- Expected Hamming distance: `256 × (1 - 0.92) ≈ 20 bits`

## Banding for Efficiency

Splitting signatures into bands enables efficient candidate generation:

```
Signature: 8d000000ac854dae91814006c580080a...
Bands:     [8d00] [0000] [00ac] [854d] ...
           band0  band1  band2  band3
```

**Algorithm**:
1. Index documents by band values
2. Query: Find all documents sharing **any** band
3. Verify candidates with full Hamming distance

**Trade-off**:
- More bands → Higher recall (catch more candidates)
- Fewer bands → Higher precision (fewer false positives)

## Next Steps

- [Signature Versions](./signature-versions) — V0 vs V1
- [Duplicate Detection Guide](../guides/duplicate-detection) — Build a detector
- [Algorithm Specification](../reference/spec) — Formal specification
