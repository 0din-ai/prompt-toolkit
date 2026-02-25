---
sidebar_position: 4
---

# Confidence Matrix LSH

CM-LSH enhances standard LSH with a confidence matrix that weights reliable bits higher.

## Features

- **Dual hash**: 512-bit signature (256 LSH + 256 ITQ)
- **Confidence weighting**: Alpha-weighted bit agreement
- **Calibrated similarity**: Isotonic regression for accurate estimates
- **Backward compatible**: First 256 bits match standard LSH

## Availability

- ✅ **Rust**: With `cm-lsh` feature flag
- ✅ **Python**: With `cm-lsh` optional dependency
- ✅ **TypeScript**: Available

See the [CM-LSH example](https://github.com/0din/sig-sdk/tree/main/packages/rust/examples) for usage.
