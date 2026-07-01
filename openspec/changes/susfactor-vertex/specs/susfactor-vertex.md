# Spec: susfactor-vertex

## Functional Requirements

### FR-1: Shared logic (`susfactor::common`)
The following functions MUST be extracted from `classifier.rs` into a new `src/susfactor/common.rs` and used verbatim by both backends:
- `tokenize_full(tokenizer, text) -> Result<(Vec<i64>, Vec<i64>)>`
- `chunk_token_ids(ids) -> Vec<Vec<i64>>` (constants: `MAX_CONTENT_TOKENS=510`, `CHUNK_OVERLAP=50`, `CHUNK_STRIDE=460`)
- `suspicious_prob(logits: &[f32]) -> f32` (numerically-stable softmax, P(class 1))
- `label_for_score(score: f32, threshold: f32) -> &'static str`
- Per-chunk `SusFactorResult` assembly
- `ChunkedSusFactorResult` reduction (`is_suspicious = any chunk suspicious`)

### FR-2: `SusFactorProvider` trait
```rust
#[async_trait]
pub trait SusFactorProvider: Send + Sync {
    fn model(&self) -> &str;
    fn threshold(&self) -> f32;
    async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult>;
}
```

### FR-3: `OnnxSusFactor`
- Current `SusFactorClassifier` refactored to implement `SusFactorProvider` and call `common` functions
- Renamed to `OnnxSusFactor`; `SusFactorClassifier` kept as a type alias with a deprecation notice for one minor version
- All existing tests must pass unchanged after refactor

### FR-4: `VertexSusFactor`
- Fields: `client: reqwest::Client`, `endpoint_url: String`, `tokenizer: Arc<Tokenizer>`, `model_name: String`, `threshold: f32`, `auth: Arc<dyn gcp_auth::TokenProvider>`, `max_concurrent_chunks: usize`
- Constructor: load tokenizer via `ModelCache` (same as `OnnxSusFactor`), MUST NOT require `onnx/model.onnx`
- `classify()`: tokenize → chunk → fan-out rawPredict requests → parse logits → shared softmax/label/reduce
- Chunk requests dispatched concurrently, bounded by `max_concurrent_chunks` (default 4)
- `reqwest::Client` MUST have `connect_timeout` and total `timeout` set
- On timeout/transport error: return `SigError::Provider` with actionable message; no panic

### FR-5: Vertex wire contract
- Endpoint: `POST https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_id}:rawPredict`
- Request body: KServe V2 JSON with `inputs: [input_ids INT64, attention_mask INT64]`; `outputs` field omitted (get all)
- Response: locate output named `logits`; fall back to first output if absent; error if no outputs
- logits array must have ≥ 2 elements; otherwise `SigError::Model("Unexpected SusFactor output shape ...")`

### FR-6: `VertexAuth`
- Implemented via `gcp_auth = "0.12"`
- `Arc<dyn gcp_auth::TokenProvider>` stored in struct (initialized via `gcp_auth::provider().await?`)
- Call `provider.token(&["https://www.googleapis.com/auth/cloud-platform"]).await?` per request (library handles caching)
- Auth errors → `SigError::Provider` with actionable message

### FR-7: Cargo features
```toml
susfactor-vertex = ["dep:gcp_auth", "dep:tokenizers", "dep:reqwest", "reqwest?/json"]
```
- `tokenizers` dep MUST be widened so it is available to `susfactor-vertex` independently of `onnx`
- `susfactor-vertex` MUST NOT pull in `ort` or `ndarray`
- `cargo build --features susfactor-vertex --no-default-features` must succeed

### FR-8: Shadow mode
- A `ShadowSusFactor` struct (or function wrapper) that wraps an `OnnxSusFactor` and a `VertexSusFactor`
- Invokes both concurrently; returns ONNX result always
- Vertex failure MUST NOT affect the response
- MUST emit `ShadowDivergence { chunk_score_deltas: Vec<f32>, label_mismatch: bool, is_suspicious_mismatch: bool }` per request (caller logs/meters)
- No automatic fallback in `vertex`-only mode

### FR-9: No automatic fallback
`VertexSusFactor::classify()` MUST return an error if Vertex is unavailable. No silent fallback to ONNX.

## Non-Functional Requirements

### NFR-1: Parity
`OnnxSusFactor` MUST pass all existing SusFactor unit and golden-vector tests.

### NFR-2: Feature isolation
`cargo build --features susfactor-vertex --no-default-features` MUST compile without `ort`/`ndarray`.

### NFR-3: Error handling
All errors MUST surface as `SigError::Provider` or `SigError::Model` variants with actionable messages. No panics.

## Test Requirements

1. **Refactor parity**: `OnnxSusFactor` passes all existing tests + golden vectors
2. **Vertex protocol (mocked)**: `mockito` mock for fixed logits → assert `ChunkedSusFactorResult` matches shared softmax/label path; cover single chunk, multi-chunk, threshold boundary, HTTP/timeout error mapping
3. **Shadow divergence**: shadow wrapper returns ONNX result, reports correct score delta/label mismatch
4. **Feature isolation**: `cargo build --features susfactor-vertex --no-default-features`
