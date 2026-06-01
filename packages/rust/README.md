# odin-prompt-toolkit (Rust)

Multi-language SDK for LSH (Locality-Sensitive Hashing) signature generation for AI prompt similarity detection.

This is the **canonical Rust implementation** of the odin-prompt-toolkit algorithm, also available in [Python](../python) and [TypeScript](../typescript).

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
odin-prompt-toolkit = "0.1"

# Or with specific features
odin-prompt-toolkit = { version = "0.1", features = ["openai", "onnx", "cm-lsh"] }
```

### Feature Flags

| Feature | Description | Dependencies | Use Case |
|---------|-------------|--------------|----------|
| `openai` | OpenAI embedding provider | `reqwest` | Cloud-based embeddings (1536-dim) |
| `onnx` | ONNX embedding provider | `ort`, `ndarray`, `tokenizers`, `dirs` | Local embeddings (1024-dim) |
| `cm-lsh` | Confidence Matrix LSH | None | Higher accuracy LSH (experimental) |
| **default** | `openai + onnx` | — | Includes both providers |

**Examples:**

```toml
# Minimal (LSH only, no embedding providers)
odin-prompt-toolkit = { version = "0.1", default-features = false }

# OpenAI only
odin-prompt-toolkit = { version = "0.1", default-features = false, features = ["openai"] }

# Local ONNX only
odin-prompt-toolkit = { version = "0.1", default-features = false, features = ["onnx"] }

# Everything (including experimental CM-LSH)
odin-prompt-toolkit = { version = "0.1", features = ["cm-lsh"] }
```

---

## Quick Start

### Basic LSH Signatures

```rust
use odin_prompt_toolkit::{simhash_lsh_multi, normalize_vector};

fn main() {
    // Your embedding vector (must be L2-normalized)
    let vector = vec![0.5, 0.5, 0.5, 0.5];
    let normalized = normalize_vector(&vector);

    // Generate LSH signatures (3 families, 256 bits, 16 bands)
    let families = simhash_lsh_multi(&normalized, 3, 256, 16);

    println!("Signature: {}", families[0].signature);
    println!("Bands: {:?}", families[0].bands);
}
```

### Similarity Comparison

```rust
use odin_prompt_toolkit::{simhash_lsh_multi, hamming_distance_hex, cosine_from_hamming};

fn main() {
    let vector1 = vec![0.5, 0.5, 0.5, 0.5];
    let vector2 = vec![0.51, 0.49, 0.5, 0.5];

    // Generate signatures
    let families1 = simhash_lsh_multi(&vector1, 3, 256, 16);
    let families2 = simhash_lsh_multi(&vector2, 3, 256, 16);

    // Compute Hamming distance
    let distance = hamming_distance_hex(
        &families1[0].signature,
        &families2[0].signature,
    );

    // Estimate cosine similarity
    let similarity = cosine_from_hamming(distance, 256);
    println!("Estimated cosine similarity: {:.3}", similarity);
}
```

### High-Level API with Embeddings

```rust
use odin_prompt_toolkit::sign_text;
use odin_prompt_toolkit::types::SignatureVersion;
use odin_prompt_toolkit::providers::onnx::OnnxProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize ONNX provider (auto-downloads model)
    let provider = OnnxProvider::default()?;

    // Generate signature from text
    let result = sign_text(
        "Hello world",
        SignatureVersion::V1,
        &provider,
    ).await?;

    println!("Signature: {}", result.signature);
    // Output: "0din-v1:8d000000ac854dae..."

    Ok(())
}
```

### Versioned Signatures

```rust
use odin_prompt_toolkit::{signature_string, parse_signature_string};
use odin_prompt_toolkit::types::{SignatureVersion, LshOutput};

fn main() {
    let lsh_output = LshOutput {
        family: 0,
        bits: 256,
        signature: "8d000000ac854dae...".to_string(),
        bands: vec!["8d00".to_string(), "0000".to_string()],
    };

    // Format signature with version
    let versioned = signature_string(&lsh_output, SignatureVersion::V1);
    println!("{}", versioned); // "0din-v1:8d000000ac854dae..."

    // Parse signature string
    let parsed = parse_signature_string(&versioned)?;
    println!("Version: {:?}", parsed.version); // V1
    println!("Algorithm: {}", parsed.algorithm); // "lsh"
    println!("Signature: {}", parsed.signature); // "8d000000ac854dae..."
}
```

---

## API Overview

### Core Functions

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `simhash_lsh_multi` | Generate multi-family LSH signatures | `vector`, `families`, `bits`, `bands` | `Vec<LshOutput>` |
| `simhash_lsh` | Generate single LSH signature | `vector`, `family`, `bits` | `LshOutput` |
| `normalize_vector` | L2-normalize vector | `vector` | `Vec<f64>` |
| `hamming_distance_hex` | Compute Hamming distance | `sig_a`, `sig_b` | `u32` |
| `cosine_from_hamming` | Estimate cosine similarity | `distance`, `bits` | `f64` |
| `compute_embedding_sha256` | SHA256 hash of embedding | `embedding` | `String` |
| `signature_string` | Format versioned signature | `lsh_output`, `version` | `String` |
| `parse_signature_string` | Parse versioned signature | `signature` | `ParsedSignature` |
| `sign_text` | High-level: text → signature | `text`, `version`, `provider` | `SignatureResult` |

### Types

| Type | Description |
|------|-------------|
| `LshOutput` | Single LSH signature with bands |
| `LshConfig` | LSH configuration (families, bits, bands) |
| `SignatureVersion` | V0 (OpenAI), V1 (ONNX), Latest |
| `ParsedSignature` | Parsed signature components |
| `SignatureResult` | High-level signature result |
| `EmbeddingResult` | Embedding with metadata |
| `PromptInfo` | Prompt preview and length |
| `ComparisonResult` | Similarity comparison details |
| `QualityStats` | LSH approximation quality metrics |
| `SigError` | Error type hierarchy |

### Embedding Providers

| Provider | Feature Flag | Model | Dimensions | Use Case |
|----------|-------------|-------|------------|----------|
| `OnnxProvider` | `onnx` | 0din-jailbreak-embeddings-small | 1024 | Local/offline, cost-free |
| `OpenAIProvider` | `openai` | text-embedding-3-large | 1536 | Cloud-based, high quality |

### Hasher Abstraction

```rust
use odin_prompt_toolkit::hasher::Hasher;
use odin_prompt_toolkit::hashers::get_hasher;

// Get hasher by algorithm name
let hasher = get_hasher("lsh")?;

// Hash a vector
let vector = vec![0.5; 1024];
let hash_output = hasher.hash(&vector, 0); // family 0

println!("Signature: {}", hash_output.signature);
```

---

## Signature Versions

- **V0**: OpenAI text-embedding-3-large (1536 dimensions, API-based)
- **V1**: 0din-jailbreak-embeddings-small ONNX (1024 dimensions, local)
- **Latest**: Resolves to V1

**Important**: V0 and V1 signatures are **not comparable** due to different embedding spaces.

**Signature Format**: `0din-v{N}:{hex_signature}`

Example: `0din-v1:8d000000ac854dae0000000000000000...` (64 hex chars = 256 bits)

---

## Algorithm

SimHash via **Random Hyperplane LSH** (Charikar 2002):

1. **Deterministic PRNG**: SplitMix64 with seed = `family * 1e12 + bit_index`
2. **Random Hyperplanes**: Generate `bits` random unit vectors in `dim` dimensions
3. **Sign Computation**: For each hyperplane, compute `sign(dot(vector, hyperplane))`
4. **Bit Packing**: Pack signs into hex string (4 bits per hex char)
5. **Band Extraction**: Split signature into `bands` equal-length chunks for indexing

**Default Configuration**:
- **3 families**: Multiple independent hash functions
- **256 bits**: 64 hex characters per signature
- **16 bands**: 4 hex characters per band (16 bits)

**Hamming → Cosine Estimation**:
```
cosine_similarity ≈ cos(π × hamming_distance / total_bits)
```

See the [specification](../../spec/SPEC.md) for complete algorithm details.

---

## Examples

The `examples/` directory contains runnable examples:

```bash
# Generate canonical test vectors
cargo run --example generate_vectors

# Benchmark signature generation
cargo run --release --example benchmark_signatures -- --count 10000

# Compare two prompts
cargo run --example compare_prompts -- \
  "Hello world" \
  "Hi there"

# Full pipeline: text → embedding → signature
cargo run --features onnx --example sign_text_onnx
```

**Example Output** (benchmark):
```
Generating 10,000 signatures (1024-dim random vectors)...
Time: 1.760s
Throughput: 5,683 signatures/sec
Per-signature: 0.176ms
```

---

## Performance

### Signature Generation

| Implementation | Throughput | Latency | Speedup |
|---------------|-----------|---------|---------|
| **Rust (this crate)** | 5,683 sigs/sec | 0.176 ms/sig | **1×** (baseline) |
| Python (native) | 5,332 sigs/sec | 0.19 ms/sig | 0.94× (PyO3 overhead) |
| Python (pure) | 9 sigs/sec | 111 ms/sig | **631× slower** |
| TypeScript | 850 sigs/sec | 1.18 ms/sig | 6.7× slower |

**Benchmark command**:
```bash
cargo run --release --example benchmark_signatures -- --count 10000
```

### Real-World Performance

From `demos/RESULTS.md` with 3,714 prompts:

| Step | Time | Rate |
|------|------|------|
| Embedding (ONNX, CPU) | 112.6s | 33 prompts/sec |
| **Signature (Rust)** | **0.7s** | **5,332 sigs/sec** |
| Total pipeline | 113.3s | 33 prompts/sec |

**Key insight**: Signature generation adds only **0.6% overhead** on top of embedding generation.

See [Performance Guide](../../docs/docs/guides/performance.md) for full benchmarks.

---

## Development

### Setup

```bash
cd packages/rust
cargo build
```

### Run Tests

```bash
# All tests
cargo test

# With all features
cargo test --all-features

# Specific test
cargo test test_simhash_lsh_deterministic
```

**Test Coverage**: 50 tests across:
- LSH core (deterministic PRNG, signature generation)
- Hamming distance & cosine estimation
- Normalization & SHA256
- Providers (OpenAI, ONNX)
- Hasher abstraction
- Error handling

### Linting & Formatting

```bash
# Check formatting
cargo fmt --check

# Format code
cargo fmt

# Run clippy
cargo clippy --all-features -- -D warnings
```

### Documentation

```bash
# Generate API docs
cargo doc --all-features --no-deps --open

# Docs will open in browser at:
# file:///.../target/doc/odin_prompt_toolkit/index.html
```

### Feature Testing

```bash
# Test with only LSH (no providers)
cargo test --no-default-features

# Test with ONNX only
cargo test --no-default-features --features onnx

# Test with OpenAI only
cargo test --no-default-features --features openai

# Test with CM-LSH
cargo test --features cm-lsh
```

---

## Cross-Language Validation

This Rust implementation serves as the **canonical reference**. Python and TypeScript implementations are validated against Rust test vectors.

**Validation methodology**:
1. Generate canonical test vectors in Rust (`examples/generate_vectors.rs`)
2. Export to JSON (`spec/test-vectors/*.json`)
3. Import and validate in Python/TypeScript tests
4. Verify bit-identical signatures across all languages

**Test vectors validated**:
- SplitMix64 PRNG (7 cases)
- Hyperplane signs (72 cases)
- SimHash LSH (5 cases)
- Hamming distance (10 cases)
- Cosine estimation (8 cases)
- SHA256 (7 cases)
- Signature format parsing (7 cases)

**Total**: 109 tests passing across Rust (50), Python (32), TypeScript (27).

See [Cross-Language Validation](../../docs/docs/concepts/cross-language.md) for details.

---

## Error Handling

The crate uses `SigError` for all error cases:

```rust
use odin_prompt_toolkit::error::SigError;

match parse_signature_string("invalid") {
    Ok(parsed) => println!("Parsed: {:?}", parsed),
    Err(SigError::InvalidInput(msg)) => eprintln!("Invalid input: {}", msg),
    Err(SigError::ProviderError(msg)) => eprintln!("Provider failed: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}
```

**Error Types**:
- `InvalidInput` - Invalid input parameters or format
- `ProviderError` - Embedding provider failure (API, ONNX)
- `ConfigError` - Invalid configuration
- `IoError` - File/network I/O errors
- `ParseError` - Signature parsing errors

See [Error Handling Guide](../../docs/docs/api/errors.md) for patterns and best practices.

---

## Production Deployment

### Air-Gapped Environments

For deployments without internet access:

1. **Use ONNX provider** (no API calls):
   ```toml
   odin-prompt-toolkit = { version = "0.1", default-features = false, features = ["onnx"] }
   ```

2. **Pre-download ONNX model**:
   ```bash
   # Download to ~/.cache/odin-prompt-toolkit/models/v1/onnx/
   # Or set SIGNATURE_SDK_MODEL_DIR to custom location
   ```

3. **Bundle model with application**:
   ```rust
   use std::env;
   env::set_var("SIGNATURE_SDK_MODEL_DIR", "/app/models");
   ```

### Performance Tuning

**LSH Configuration**:
- **More bands** (16 → 32): Higher recall, more candidates
- **More bits** (256 → 512): Higher precision, larger storage
- **More families** (3 → 5): Better distribution, more storage

**Embedding Batch Processing**:
```rust
// Process embeddings in parallel
use rayon::prelude::*;

let signatures: Vec<_> = embeddings
    .par_iter()
    .map(|emb| simhash_lsh_multi(emb, 3, 256, 16))
    .collect();
```

### Monitoring

```rust
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<(), SigError> {
    tracing_subscriber::fmt::init();

    let provider = OnnxProvider::default()?;
    info!("ONNX provider initialized");

    let result = sign_text("Hello", SignatureVersion::V1, &provider).await?;
    info!("Generated signature: {}", result.signature);

    Ok(())
}
```

---

## Comparison with Other Libraries

| Feature | odin-prompt-toolkit (Rust) | Annoy | Faiss LSH | datasketch |
|---------|----------------|-------|-----------|------------|
| Language | Rust | C++ (Python bindings) | C++ (Python bindings) | Python |
| Deterministic | ✅ Yes | ❌ No | ⚠️ Depends | ⚠️ Depends |
| Cross-language | ✅ Rust/Python/TS | ❌ Python only | ❌ Python only | ❌ Python only |
| Versioned format | ✅ Yes | ❌ Binary index | ❌ Binary index | ❌ No |
| Zero dependencies | ✅ Yes (LSH core) | ❌ Needs build tools | ❌ Needs BLAS | ❌ Needs numpy |
| Embedding providers | ✅ Built-in | ❌ BYO | ❌ BYO | ❌ BYO |
| Test vectors | ✅ Canonical | ❌ None | ❌ None | ⚠️ Partial |

---

## Roadmap

### Short-Term (v0.2)
- [ ] Additional embedding providers (HuggingFace, Cohere)
- [ ] Python PyO3 bindings as separate crate
- [ ] TypeScript WASM compilation target
- [ ] Benchmarks vs Annoy/Faiss

### Medium-Term (v0.3)
- [ ] GPU acceleration (CUDA, Metal) for batch processing
- [ ] Compressed signature storage (binary format)
- [ ] CLI tool for signature generation
- [ ] gRPC/REST server (optional)

### Long-Term (v1.0)
- [ ] Adaptive LSH (dynamic band/bit tuning)
- [ ] Multi-modal embeddings (text + image)
- [ ] Distributed signature index
- [ ] Production-ready CM-LSH

---

## License

MIT License - see [LICENSE](../../LICENSE) for details.

---

## Related Documentation

- **[Docusaurus Site](../../docs)** - Full documentation and guides
- **[API Reference](../../docs/docs/api/core-functions.md)** - Complete function reference
- **[Performance Guide](../../docs/docs/guides/performance.md)** - Benchmarks and optimization
- **[Migration Guide](../../docs/docs/guides/migration.md)** - Migrate from heimdall/thor/research
- **[Specification](../../spec/SPEC.md)** - Algorithm specification
- **[Test Vectors](../../spec/test-vectors/)** - Canonical test vectors

---

## Support

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and use cases
- **Examples**: See `examples/` directory for runnable code

---

## Contributing

Contributions welcome! Please:

1. Run tests: `cargo test --all-features`
2. Run clippy: `cargo clippy --all-features -- -D warnings`
3. Format code: `cargo fmt`
4. Update documentation: `cargo doc`
5. Add test coverage for new features

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.
