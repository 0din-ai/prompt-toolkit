---
sidebar_position: 5
---

# Cross-Language Compatibility

All three implementations (Rust, Python, TypeScript) produce identical signatures from identical inputs.

## Validation

✅ **61 tests** across 3 languages validate:
- SplitMix64 PRNG consistency
- Hyperplane sign generation
- SimHash LSH algorithm
- Hamming distance calculation
- Cosine similarity estimation
- SHA256 canonical format

See the [Validation Report](https://github.com/0din/sig-sdk/blob/main/VALIDATION.md) for details.

## Deterministic Algorithm

The same input **always** produces the same signature because:

1. **Deterministic PRNG**: SplitMix64 with fixed seed formula
2. **No floating-point precision**: Sign bits only (`dot > 0`)
3. **Canonical normalization**: Consistent L2 norm calculation

## Test Vectors

All implementations are validated against canonical Rust test vectors in `/spec/test-vectors/`.
