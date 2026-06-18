---
sidebar_position: 1
---

# Algorithm Specification

The complete formal specification is available in the repository.

See [spec/SPEC.md](https://github.com/0din-ai/prompt-toolkit/blob/main/spec/SPEC.md) for:

- Formal algorithm definition
- Pseudocode for all operations
- Mathematical foundations
- Implementation requirements
- Test vector format

## Key Sections

1. **Overview** — Algorithm purpose and design goals
2. **Normalization** — L2 norm calculation
3. **SplitMix64 PRNG** — Deterministic random generation
4. **Hyperplane Generation** — Seed formula and sign extraction
5. **SimHash** — Bit packing and hex encoding
6. **Hamming Distance** — Bit difference calculation
7. **Cosine Estimation** — Similarity from Hamming distance
8. **SHA256 Canonical** — Deterministic embedding hashing
9. **Signature Format** — String representation
10. **Banding** — LSH indexing strategy
11. **CM-LSH** — Confidence Matrix extension
12. **Cross-Language** — Validation requirements
13. **Test Vectors** — Canonical test cases
