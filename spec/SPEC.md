# signature-sdk Algorithm Specification

This document provides the formal specification for the LSH (Locality-Sensitive Hashing) signature algorithm used across all three language implementations (Rust, Python, TypeScript).

## Version

Specification version: 1.0.0  
Last updated: 2026-02-24

## 1. SplitMix64 PRNG

A deterministic pseudorandom number generator used to produce random hyperplane signs.

### Algorithm

```
splitmix64(x: u64) -> u64:
  z = (x + 0x9E3779B97F4A7C15) mod 2^64
  z = ((z XOR (z >> 30)) * 0xBF58476D1CE4E5B9) mod 2^64
  z = ((z XOR (z >> 27)) * 0x94D049BB133111EB) mod 2^64
  return (z XOR (z >> 31)) mod 2^64
```

### Constants

- `0x9E3779B97F4A7C15` — Golden ratio constant
- `0xBF58476D1CE4E5B9` — Mixing constant 1
- `0x94D049BB133111EB` — Mixing constant 2

### Properties

- Deterministic: same input always produces same output
- 64-bit arithmetic with wraparound (mod 2^64)
- All operations are bitwise and arithmetic (no floating point)

## 2. Sign Generation

Deterministic generation of +1 or -1 signs for random hyperplane components.

### Algorithm

```
sign_for(family: usize, bit: usize, dim: usize) -> {+1, -1}:
  seed = (family << 48) XOR (bit << 24) XOR dim
  h = splitmix64(seed)
  return +1 if (h AND 1) == 1 else -1
```

### Parameters

- `family`: Hash family index (typically 0-2)
- `bit`: Bit index within signature (typically 0-255)
- `dim`: Dimension index of input vector (0 to vector length - 1)

### Properties

- Deterministic: same (family, bit, dim) tuple always produces same sign
- Each dimension of each bit of each family has an independent random sign
- No seed storage required — derived from parameters

## 3. SimHash LSH (Random Hyperplane LSH)

The core locality-sensitive hashing algorithm using random hyperplane projections.

### Algorithm

```
simhash_lsh_multi(normalized_vector: f64[], config: LshConfig) -> LshFamily[]:
  results = []
  
  for family in [0, config.families):
    bool_bits = []
    
    for bit in [0, config.bits):
      dot = 0.0
      for dim in [0, len(normalized_vector)):
        sign = sign_for(family, bit, dim)
        dot += normalized_vector[dim] * sign
      
      bool_bits[bit] = (dot > 0.0)
    
    signature = pack_to_hex(bool_bits)
    bands = split_into_bands(signature, config.bands)
    
    results.append(LshFamily {
      family: family,
      bits: config.bits,
      signature: signature,
      bands: bands
    })
  
  return results
```

### Default Configuration

```
LshConfig:
  families: 3    # Number of independent hash families
  bits: 256      # Bits per signature (64 hex chars)
  bands: 16      # Number of bands for LSH indexing
```

### Input Requirements

- `normalized_vector` MUST be L2-normalized (unit length)
- Vector can be any dimension (typically 1024 or 1536)
- All computations use `f64` precision for dot products

### Hex Packing

Bits are packed into hexadecimal with MSB-first within each nibble:

```
pack_to_hex(bits: bool[]) -> string:
  hex_chars = []
  for i in range(0, len(bits), 4):
    nibble = bits[i]*8 + bits[i+1]*4 + bits[i+2]*2 + bits[i+3]*1
    hex_chars.append(format(nibble, 'x'))  # lowercase hex
  return join(hex_chars)
```

Example:
- Bits `[true, false, true, false]` → nibble `8+0+2+0 = 10` → hex `'a'`
- 256 bits → 64 hex characters

### Band Splitting

The signature is split into contiguous bands for LSH indexing:

```
split_into_bands(signature: string, num_bands: usize) -> string[]:
  band_length = len(signature) / num_bands  # integer division
  bands = []
  for i in range(0, len(signature), band_length):
    bands.append(signature[i:i+band_length])
    if len(bands) == num_bands:
      break
  return bands
```

For 256 bits (64 hex chars) with 16 bands: each band is 4 hex chars.

## 4. Hamming Distance

Compute bit-level Hamming distance between two hex-encoded signatures.

### Algorithm

```
hamming_distance_hex(a: string, b: string) -> usize:
  # Clean both to lowercase hex only
  a_clean = filter(a, is_hex_digit).to_lowercase()
  b_clean = filter(b, is_hex_digit).to_lowercase()
  
  min_len = min(len(a_clean), len(b_clean))
  distance = 0
  
  for i in range(0, min_len):
    n1 = parse_hex(a_clean[i])
    n2 = parse_hex(b_clean[i])
    xor = n1 XOR n2
    distance += popcount(xor)  # count set bits in 4-bit value
  
  # Extra characters in longer string count as 4 differing bits each
  distance += abs(len(a_clean) - len(b_clean)) * 4
  
  return distance
```

### Popcount for 4-bit Values

```
popcount(n: u8) -> usize:  # n in [0, 15]
  count = 0
  if (n & 8) != 0: count += 1
  if (n & 4) != 0: count += 1
  if (n & 2) != 0: count += 1
  if (n & 1) != 0: count += 1
  return count
```

## 5. Cosine Similarity Estimation

Convert Hamming distance to estimated cosine similarity using the theoretical relationship for random hyperplane LSH.

### Algorithm

```
cosine_from_hamming(distance_bits: usize, total_bits: usize) -> f64:
  if total_bits == 0:
    return 0.0
  
  p_diff = distance_bits / total_bits
  return cos(PI * p_diff)
```

### Theoretical Basis

For random hyperplane LSH, the probability that two vectors have different signs for a random hyperplane is:

```
P(sign differs) = theta / PI
```

where `theta` is the angle between the vectors. Therefore:

```
hamming_rate = distance / total_bits ≈ theta / PI
cosine_similarity = cos(theta) ≈ cos(PI * hamming_rate)
```

## 6. Vector Normalization

L2 normalization to unit length (required before LSH).

### Algorithm

```
normalize_vector(vector: f32[]) -> f32[]:
  magnitude = sqrt(sum(x^2 for x in vector))
  
  if magnitude == 0.0:
    return vector  # zero vector stays zero
  
  return [x / magnitude for x in vector]
```

### Precision Note

- Input and output use `f32` (single precision) for memory efficiency
- LSH dot products use `f64` (double precision) for accuracy
- This matches OpenAI API responses and standard ML practice

## 7. Canonical Embedding SHA256

Deterministic SHA256 hash of normalized embeddings for deduplication.

### Algorithm

```
compute_embedding_sha256(normalized_embedding: f32[]) -> string:
  # 1. Quantize each value to 6 decimal places
  quantized = [round(x * 1e6) / 1e6 for x in normalized_embedding]
  
  # 2. Format as JSON array with Python-compatible formatting:
  #    - Space after comma
  #    - Whole numbers include ".0"
  parts = []
  for x in quantized:
    s = format_float(x)
    if is_whole_number(x) and !contains(s, '.'):
      s = s + ".0"
    parts.append(s)
  
  json_str = "[" + join(parts, ", ") + "]"
  
  # 3. Compute SHA256 of the JSON string
  return sha256_hex(json_str)
```

### Example

Input: `[0.1, 0.2, 0.3]`  
Quantized: `[0.1, 0.2, 0.3]`  
JSON: `"[0.1, 0.2, 0.3]"` (note space after commas)  
SHA256: `9a04781069052282acb2e95529c7f5bcd85149ab2ec559c550dce80b81ceb04e`

### Rationale

The 6-decimal quantization eliminates floating-point jitter from:
- OpenAI API non-determinism (different servers/GPUs)
- Cross-platform float representation differences
- Numerical precision variations in inference

This ensures the SHA256 is stable across runs and platforms.

## 8. CM-LSH (Confidence Matrix LSH)

Enhanced LSH with confidence weighting for more accurate similarity estimates.

### Overview

CM-LSH combines two quantization methods:
1. **LSH-TS**: Standard random hyperplane LSH (256 bits)
2. **ITQ**: Iterative Quantization via PCA + rotation (256 bits)

Total: 512-bit hash with 512-bit confidence matrix.

### Algorithm

```
cm_lsh_hash(embedding: f32[], params: HybridParams) -> DualHash:
  # 1. Normalize
  emb = normalize_vector(embedding)
  
  # 2. LSH-TS projection (256 bits)
  p1 = emb · params.lsh_ts_hyperplanes^T  # matrix multiply
  
  # 3. ITQ projection (256 bits)
  centered = emb - params.itq.mean
  pca_proj = centered · params.itq.pca^T
  p2 = pca_proj · params.itq.rotation^T
  
  # 4. Concatenate projections (512 values)
  proj = concatenate([p1, p2])
  
  # 5. Sign bits (hash_a)
  hash_a_bits = [p > 0 for p in proj]
  hash_a = pack_to_hex(hash_a_bits)  # 128 hex chars
  
  # 6. Confidence bits (hash_b)
  conf_threshold = percentile(abs(proj), 45)
  hash_b_bits = [abs(p) > conf_threshold for p in proj]
  hash_b = pack_to_hex(hash_b_bits)  # 128 hex chars
  
  # 7. Split into bands (64 bands for ANN search)
  bands = split_into_bands(hash_a, 64)
  
  return DualHash {
    hash_a: hash_a,
    hash_b: hash_b,
    bands: bands,
    bits: 512
  }
```

### Similarity Computation

```
cm_lsh_similarity(h1: DualHash, h2: DualHash, alpha: f64 = 0.65) -> f64:
  a1_bits = unpack_hex(h1.hash_a)
  a2_bits = unpack_hex(h2.hash_a)
  b1_bits = unpack_hex(h1.hash_b)
  b2_bits = unpack_hex(h2.hash_b)
  
  # Compute agreement and confidence overlap
  agree = [a1_bits[i] == a2_bits[i] for i in range(512)]
  both_confident = [b1_bits[i] AND b2_bits[i] for i in range(512)]
  
  # Weighted similarity
  if any(both_confident):
    confident_agree_rate = mean([agree[i] for i in confident_indices])
    overall_agree_rate = mean(agree)
    raw_sim = alpha * confident_agree_rate + (1 - alpha) * overall_agree_rate
  else:
    raw_sim = mean(agree)
  
  # Apply calibration (isotonic regression)
  return calibrator.predict(raw_sim)
```

### Default Parameters

For `create_default_cm_lsh(dimensions)`:
- LSH-TS hyperplanes: generated via `sign_for` (same as SimHash)
- ITQ mean: zeros
- ITQ PCA: identity (first 256 dims)
- ITQ rotation: identity
- Calibrator: linear (no adjustment)
- Alpha: 0.65

### LSH-TS Compatibility

The first 256 bits of `hash_a` are identical to SimHash family 0:

```
lsh_ts_compat(dual_hash: DualHash) -> string:
  return dual_hash.hash_a[0:64]  # first 64 hex chars
```

## 9. Signature Version Format

Signatures are encoded as version-prefixed strings.

### Format

```
0din-v{N}:<hex_signature>
```

### Examples

- V0: `0din-v0:a3f9c2e1b8d4f7a2...` (64 hex chars = 256 bits)
- V1: `0din-v1:7f2c8a9d3e1b5f4c...` (64 hex chars = 256 bits)

### Parsing

```
parse_signature_string(s: string) -> (version, signature):
  if !starts_with(s, "0din-"):
    error("Invalid signature format")
  
  parts = split(s, ':', 2)
  if len(parts) != 2:
    error("Invalid signature format")
  
  version_str = parts[0]  # "0din-v0" or "0din-v1"
  signature = parts[1]    # hex string
  
  if version_str == "0din-v0":
    return (V0, signature)
  elif version_str == "0din-v1":
    return (V1, signature)
  else:
    error("Unsupported signature version")
```

## 10. Reference Test Vectors

See `spec/test-vectors/` for JSON-encoded test vectors covering:

- SplitMix64 PRNG outputs
- `sign_for` outputs for various inputs
- Complete SimHash signatures for known vectors
- Hamming distances between known signatures
- SHA256 hashes for known embeddings
- Signature string parsing examples

All implementations MUST pass these test vectors.

## 11. Precision Notes

### Float Precision

- Embeddings: stored as `f32` (memory efficient, matches ML standards)
- LSH dot products: computed in `f64` (matches Python reference)
- CM-LSH projections: computed in `f64`

### Expected Differences

When comparing f32 vs f64 embeddings:
- Typical difference: ~1 bit per 256-bit signature (~0.39%)
- Cosine similarity impact: ~0.001 (negligible)
- This is acceptable for practical similarity matching

### Cross-Platform Consistency

- Integer operations (SplitMix64, sign_for): exact across all platforms
- Float operations: may differ at low precision due to IEEE 754 rounding
- SHA256: exact across all platforms (string-based input)

## 12. Implementation Requirements

### MUST Requirements

1. SplitMix64 MUST use exact constants and operations specified
2. `sign_for` MUST use exact seed formula
3. Hex packing MUST be lowercase with MSB-first within nibbles
4. SHA256 MUST use the canonical JSON format (6-decimal quantization, space after comma)
5. All test vectors MUST pass

### SHOULD Requirements

1. LSH dot products SHOULD use f64 precision
2. Input validation SHOULD check for normalized vectors
3. Errors SHOULD be clear and actionable

### MAY Requirements

1. Implementations MAY use SIMD for dot products
2. Implementations MAY cache hyperplane signs
3. Implementations MAY optimize for specific vector dimensions

## 13. References

- Charikar, M. (2002). "Similarity estimation techniques from rounding algorithms." STOC.
- Gong, Y., Lazebnik, S. (2011). "Iterative Quantization: A Procrustean Approach to Learning Binary Codes." CVPR.
- SplitMix64: https://prng.di.unimi.it/splitmix64.c
