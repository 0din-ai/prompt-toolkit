# Design: susfactor-vertex

## File Layout

```
packages/rust/src/susfactor/
  mod.rs           — updated re-exports (provider trait, both backends, shadow, common fns)
  types.rs         — unchanged (SusFactorResult, ChunkedSusFactorResult, constants)
  common.rs        — NEW: shared tokenize_full, chunk_token_ids, suspicious_prob,
                         label_for_score, assemble_result, reduce_chunks
  provider.rs      — NEW: SusFactorProvider trait
  classifier.rs    — refactored to OnnxSusFactor; SusFactorClassifier type alias
  vertex.rs        — NEW: VertexSusFactor, VertexAuth (gcp_auth wrapper), wire types
  shadow.rs        — NEW: ShadowSusFactor, ShadowDivergence
```

## Trait Signature

```rust
// src/susfactor/provider.rs
#[async_trait]
pub trait SusFactorProvider: Send + Sync {
    fn model(&self) -> &str;
    fn threshold(&self) -> f32;
    async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult>;
}
```

## `VertexSusFactor` Key Fields

```rust
pub struct VertexSusFactor {
    client: reqwest::Client,
    endpoint_url: String,
    tokenizer: Arc<tokenizers::Tokenizer>,
    model_name: String,
    threshold: f32,
    auth: Arc<dyn gcp_auth::TokenProvider>,
    max_concurrent_chunks: usize,   // default: 4
}
```

## Wire Types (Triton KServe V2)

```rust
// Request
#[derive(Serialize)]
struct InferRequest {
    inputs: Vec<InferInput>,
}
#[derive(Serialize)]
struct InferInput {
    name: &'static str,
    shape: [usize; 2],
    datatype: &'static str,   // "INT64"
    data: Vec<i64>,
}

// Response
#[derive(Deserialize)]
struct InferResponse {
    outputs: Vec<InferOutput>,
}
#[derive(Deserialize)]
struct InferOutput {
    name: String,
    data: Vec<f32>,
}
```

## Feature Graph

```toml
[features]
default = ["openai", "onnx"]
openai = ["dep:reqwest", "reqwest?/json"]
onnx = ["dep:ort", "dep:ndarray", "dep:tokenizers", "dep:dirs", "dep:reqwest", "dep:futures-util", "reqwest?/stream"]
susfactor = ["onnx"]
susfactor-vertex = ["dep:gcp_auth", "dep:tokenizers", "dep:reqwest", "reqwest?/json"]
cm-lsh = []
threatfeed = ["dep:reqwest", "dep:dirs", "dep:chrono", "dep:urlencoding"]

[dependencies]
gcp_auth = { version = "0.12", optional = true }
tokenizers = { version = "0.20", features = ["http"], optional = true }
# ^ tokenizers moved from onnx-only to optional at top level
```

Key change: `tokenizers` is no longer gated exclusively by `onnx` — it becomes a standalone optional dep, gated by `onnx` OR `susfactor-vertex`.

## Shadow Mode Design

```rust
pub struct ShadowSusFactor {
    primary: OnnxSusFactor,
    shadow: VertexSusFactor,
}

pub struct ShadowDivergence {
    pub chunk_score_deltas: Vec<f32>,
    pub label_mismatch: bool,
    pub is_suspicious_mismatch: bool,
}

impl ShadowSusFactor {
    pub async fn classify_with_divergence(
        &self, text: &str
    ) -> Result<(ChunkedSusFactorResult, Option<ShadowDivergence>)>;
}
```

Both backends called concurrently via `tokio::join!`. Vertex failure → `divergence = None` (not an error to the caller). Primary (ONNX) result returned always.

`ShadowSusFactor` does NOT implement `SusFactorProvider` (its signature differs — returns divergence info). The host integrates it directly.

## Concurrency Model for `VertexSusFactor`

Use `futures::stream::iter(chunks).map(|chunk| self.classify_chunk(chunk)).buffer_unordered(self.max_concurrent_chunks).collect::<Vec<_>>().await` — requires `futures-util` already available under `onnx`. For `susfactor-vertex` feature without `onnx`, add `dep:futures-util` to `susfactor-vertex` deps.

Alternative: `tokio::task::JoinSet` (no extra dep). Prefer `JoinSet` to avoid the `futures-util` dep in feature-isolated build.

## Open Questions (for human)

These map to spec §9 and do NOT block the Rust implementation:
- **Q1 (Region)**: Which Vertex region? (`us-west1` vs `us-central1`) — needed for env config, not code
- **Q2 (Batching)**: One rawPredict per chunk (implement) or batch all chunks per request (optional optimization per spec §9.2) — spec says one-per-chunk is the baseline
- **Q3 (Acceptance bound)**: Shadow divergence pass criteria — needed for rollout gate, not code
- **Q4 (Auth crate)**: `gcp_auth` recommended; proceeding unless objection
- **Q5 (Tokenizer sourcing)**: Startup fetch vs image bundle — infra decision, code supports both
