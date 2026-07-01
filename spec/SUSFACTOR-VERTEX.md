# SusFactor Vertex AI Serving Specification

This document specifies the **Vertex AI serving backend** for the SusFactor
classifier: the `SusFactorProvider` abstraction, the two backends
(`OnnxSusFactor`, `VertexSusFactor`), the Vertex wire contract, the auth model,
and the shadow-mode migration strategy.

It is the implementation spec for the `susfactor-vertex` feature targeted at
prompt-toolkit **v0.8.0** and consumed by Heimdall.

For the existing SusFactor capability and its caller-facing contract, see
[`INTEGRATION.md` §2](INTEGRATION.md). For the in-pod ONNX classifier behavior
that this spec preserves byte-for-byte, see
`packages/rust/src/susfactor/classifier.rs`.

**Spec version**: 1.0.0
**Last updated**: 2026-06-30
**Status**: Proposed (v0.8.0)

---

## 1. Motivation and scope

### 1.1 Goal

Allow SusFactor classification to run against a remote **Vertex AI** endpoint
that serves the existing SusFactor ONNX graph, instead of loading the ~2 GB
model into the application pod. Selection is by configuration; the caller-facing
contract (`classify()` → `ChunkedSusFactorResult`) is unchanged.

### 1.2 Driver

**Operational simplicity** — stop shipping and loading the SusFactor model file
in pods (init-container download + `emptyDir`, OOM risk, model lifecycle). This
spec covers **SusFactor only**. The embedding/signature providers
(`OpenAIProvider`, `OnnxProvider`) are explicitly **out of scope**.

### 1.3 Non-goals

- No change to signature generation (V0/V1) or the LSH pipeline.
- No change to the `ChunkedSusFactorResult` / `SusFactorResult` wire shape.
- No new SusFactor scoring semantics: tokenization, chunking, softmax, and
  labeling are identical to the ONNX backend.

---

## 2. Design principle: only the matmul moves

SusFactor is a **classifier**, not an embedder. The ONNX graph emits raw
`logits[1, 2]` (float32); `softmax[1]` = P(suspicious). TEI `/v1/embeddings`
and the OpenAI-compatible path **cannot** serve this — they return embeddings,
not classifier logits.

Therefore the Vertex backend keeps **tokenization, chunking, softmax, and
labeling client-side in Rust** — byte-identical to the ONNX backend — and
delegates **only the ONNX graph execution** (`input_ids`, `attention_mask` →
`logits[1, 2]`) to Vertex.

```
ONNX backend (today):
  text → tokenize → chunk → [ORT in-process → logits] → softmax → label

Vertex backend (this spec):
  text → tokenize → chunk → [HTTP rawPredict → logits] → softmax → label
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            only this step changes
```

### 2.1 Shared logic requirement (MUST)

The following MUST be extracted into a shared module
(`susfactor::common`) and used **verbatim** by both backends, so the two cannot
diverge:

- `tokenize_full(text) -> (input_ids: Vec<i64>, attention_mask: Vec<i64>)`
- `chunk_token_ids(ids) -> Vec<Vec<i64>>` with constants
  `MAX_CONTENT_TOKENS = 510`, `CHUNK_OVERLAP = 50`, `CHUNK_STRIDE = 460`
- `suspicious_prob(logits) -> f32` (numerically-stable softmax, P(class 1))
- `label_for_score(score, threshold) -> &str` (`>=` is inclusive → suspicious)
- per-chunk `SusFactorResult` assembly and the `ChunkedSusFactorResult`
  reduction (`is_suspicious = any chunk suspicious`)

`OnnxSusFactor` MUST produce results identical to today's
`SusFactorClassifier` after this extraction (pure refactor; no behavior change).

---

## 3. `SusFactorProvider` trait

A trait abstraction analogous to `EmbeddingProvider` (`src/provider.rs`).

```rust
#[async_trait]
pub trait SusFactorProvider: Send + Sync {
    /// Canonical model identifier reported in results.
    fn model(&self) -> &str;

    /// Decision threshold used to derive labels.
    fn threshold(&self) -> f32;

    /// Classify a prompt of any length, returning one result per chunk.
    async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult>;
}
```

### 3.1 Requirements

1. Both `OnnxSusFactor` and `VertexSusFactor` MUST implement this trait.
2. `classify()` MUST return a `ChunkedSusFactorResult` with the same field
   semantics described in [`INTEGRATION.md` §2](INTEGRATION.md) regardless of
   backend.
3. The existing concrete `SusFactorClassifier` name SHOULD be retained as a
   type alias to `OnnxSusFactor` for one minor version to avoid breaking
   downstream imports; deprecation MAY follow.

---

## 4. Backends

### 4.1 `OnnxSusFactor`

The current `SusFactorClassifier` logic, refactored to implement
`SusFactorProvider` and to call the shared `common` functions. Behavior,
constructor signature, model files, and defaults are unchanged:

- `DEFAULT_MODEL = "0dinai/susfactor-e5-large"`
- `DEFAULT_ONNX_REPO = "0dinai/susfactor-e5-large-onnx"`
- `DEFAULT_THRESHOLD = 0.5`, `MAX_SEQUENCE_LENGTH = 512`

### 4.2 `VertexSusFactor`

```rust
pub struct VertexSusFactor {
    client: reqwest::Client,
    endpoint_url: String,            // full rawPredict URL (see §5.1)
    tokenizer: Arc<tokenizers::Tokenizer>,
    model_name: String,              // canonical id reported in results
    threshold: f32,
    auth: VertexAuth,                // §6
    max_concurrent_chunks: usize,    // bound on in-flight chunk requests
}
```

**Construction (MUST):**
- Load the tokenizer locally (from `ModelCache` / HF / local dir), exactly as
  `OnnxSusFactor` does. The pod keeps `tokenizer.json` (small); it MUST NOT
  require `onnx/model.onnx`.
- Accept the endpoint URL (or the components to build it), project, region,
  threshold, and canonical model id via the constructor.

**`classify()` flow (MUST):**
1. `tokenize_full(text)` (shared).
2. `chunk_token_ids(ids)` (shared).
3. For each chunk, build a Triton V2 `rawPredict` body (§5.2) and POST to
   `endpoint_url` with a bearer token from `auth` (§6).
4. Parse `logits[1, 2]` from the response (§5.3).
5. `suspicious_prob(logits)` → `label_for_score(score, threshold)` (shared).
6. Assemble per-chunk `SusFactorResult`; reduce to `ChunkedSusFactorResult`
   (shared).

**Concurrency (SHOULD):** chunk requests SHOULD be dispatched concurrently,
bounded by `max_concurrent_chunks` (default small, e.g. 4), to mirror the
fan-out of the ONNX backend without overwhelming the endpoint.

**Timeouts (MUST):** the `reqwest::Client` MUST set a per-request timeout
(connect + total). On timeout or transport error, `classify()` MUST return
`SigError::Provider` with an actionable message; it MUST NOT panic.

---

## 5. Vertex wire contract (Triton V2 `rawPredict`)

### 5.1 Serving container and endpoint

- Serving container: **NVIDIA Triton** with platform `onnxruntime_onnx`,
  serving the existing SusFactor ONNX graph.
- Endpoint (MUST): Vertex AI `rawPredict`:

  ```
  https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_id}:rawPredict
  ```

- Rationale: `rawPredict` is the lowest-overhead path for custom containers and
  passes the Triton V2 body through unmodified. The OpenAI-compatible
  `/v1/embeddings` path is NOT applicable (classifier logits, not embeddings).

### 5.2 Request body (Triton KServe V2 inference)

For a single chunk of `seq_len` tokens:

```json
{
  "inputs": [
    {
      "name": "input_ids",
      "shape": [1, <seq_len>],
      "datatype": "INT64",
      "data": [<input_ids...>]
    },
    {
      "name": "attention_mask",
      "shape": [1, <seq_len>],
      "datatype": "INT64",
      "data": [<attention_mask...>]
    }
  ]
}
```

Requirements:
- Input tensor names MUST be `input_ids` and `attention_mask` (matching the
  ONNX graph inputs).
- `datatype` MUST be `INT64`; values are the same `i64` token IDs / mask used
  by the ONNX backend.
- One request per chunk (default). Batching multiple chunks into a single
  `rawPredict` is an OPTIONAL optimization (see §9, open question 2) and MUST
  preserve per-chunk results if adopted.

### 5.3 Response body and logits extraction

```json
{
  "outputs": [
    {
      "name": "logits",
      "shape": [1, 2],
      "datatype": "FP32",
      "data": [<logit_0>, <logit_1>]
    }
  ]
}
```

Requirements:
- The implementation MUST locate the output named `logits`; if absent, it MUST
  fall back to the first output (matching the ONNX backend's behavior) and MUST
  error if no output is present.
- The flattened logits MUST contain at least 2 elements; otherwise return
  `SigError::Model("Unexpected SusFactor output shape ...")`.
- `suspicious_prob(&logits)` is applied client-side; the endpoint MUST NOT be
  relied upon to return a probability or label.

### 5.4 Triton model repository layout (infra)

The model MUST be present in GCS in Triton's expected structure:

```
gs://{models-bucket}/triton/susfactor_e5_large/
  config.pbtxt
  1/
    model.onnx
    model.onnx_data        # if the export uses external weights
```

`config.pbtxt` requirements:
- `platform: "onnxruntime_onnx"`
- input `input_ids` INT64, dims `[-1]` (variable), with batch dim
- input `attention_mask` INT64, dims `[-1]`, with batch dim
- output `logits` FP32, dims `[2]`, with batch dim
- dynamic batching enabled; CPU instance group
- The output tensor name in `config.pbtxt` MUST match the ONNX graph's output
  name (`logits`). Verify against the export before deploying.

---

## 6. Authentication (`VertexAuth`)

This is the first GCP auth code in the toolkit. `OpenAIProvider`'s static
bearer token is insufficient — Vertex requires refreshable OAuth2 access tokens.

Requirements:
1. In GKE (Workload Identity): obtain tokens via Application Default
   Credentials. A `gcp_auth`-style provider or the GKE metadata server
   (`http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token`,
   header `Metadata-Flavor: Google`) MUST be used.
2. In local development: ADC via `GOOGLE_APPLICATION_CREDENTIALS` or
   `gcloud auth application-default login`.
3. Tokens MUST be cached and refreshed before expiry (refresh with a safety
   margin, e.g. 60s). Each request attaches `Authorization: Bearer <token>`.
4. Auth scope MUST be `https://www.googleapis.com/auth/cloud-platform`.
5. Auth errors MUST surface as actionable `SigError::Provider` messages.

The auth implementation SHOULD live behind the `susfactor-vertex` feature and
SHOULD be small and Vertex-scoped (no general-purpose GCP client).

---

## 7. Cargo features and dependencies

```toml
[features]
# existing: default = ["openai", "onnx"], susfactor = ["onnx"]
susfactor-vertex = ["dep:gcp_auth", "dep:reqwest", "dep:tokenizers", "reqwest?/json"]
```

Requirements:
- `susfactor-vertex` MUST NOT pull in `ort` or `ndarray` (no in-pod inference).
- `tokenizers` MUST be available to `susfactor-vertex` independently of `onnx`
  (it is currently gated only behind `onnx`); the gating MUST be widened.
- `gcp_auth` (or an agreed equivalent such as `yup-oauth2`) is the only new
  dependency. (See §9, open question 4.)
- `async-trait`, `reqwest`, `mockito` (dev) are already present.

---

## 8. Migration and rollout: shadow mode

The migration MUST support a **shadow** mode that runs both backends and
records divergence, so the Vertex backend can be validated against the ONNX
reference on live traffic before cutover.

### 8.1 Backend selection

The host (Heimdall) selects the backend; the SDK provides all three building
blocks (`OnnxSusFactor`, `VertexSusFactor`, and a `Shadow` wrapper or the
primitives to build one). Selection values:

| Value    | Behavior                                                        |
|----------|-----------------------------------------------------------------|
| `onnx`   | `OnnxSusFactor` only (current behavior)                         |
| `vertex` | `VertexSusFactor` only                                          |
| `shadow` | Both; **primary = ONNX**, Vertex called for comparison only     |

### 8.2 Shadow semantics (MUST)

1. In `shadow`, both backends are invoked concurrently for each request.
2. The **primary (ONNX)** result is returned to the caller. A failure or
   timeout on the **shadow (Vertex)** side MUST NOT affect the response.
3. The implementation MUST emit a structured divergence signal per request:
   - per-chunk score delta (`|onnx.score − vertex.score|`)
   - label mismatch (boolean)
   - `is_suspicious` mismatch (boolean)
   These are logged/metered by the host (e.g. OpenTelemetry).

### 8.3 Cutover

1. Dev: run `shadow`; collect divergence over representative traffic until the
   acceptance bound (§9, open question 3) is met.
2. Stage: `shadow` → `vertex`.
3. Prod: `shadow` → `vertex`.
4. Post-cutover (infra): remove the SusFactor model from the init-container
   download and shrink pod memory accordingly. Note the embedding model still
   loads in-pod until signatures migrate separately.

### 8.4 No automatic ONNX fallback

`vertex` mode MUST NOT silently fall back to in-pod ONNX, because that would
require shipping the model and defeat the driver. If Vertex is unavailable,
`classify()` returns an error and the host maps it to the existing
`503 SUSFACTOR_UNAVAILABLE` behavior.

---

## 9. Open questions (require human decision)

1. **Region**: cluster is `us-west1`; deferred plan used `us-central1`. Choose
   the Vertex region (latency vs availability/cost; cross-region adds egress).
2. **Batching**: one `rawPredict` per chunk, or batch all chunks of a prompt
   into a single request. Batching reduces round-trips for long prompts but
   complicates per-chunk result mapping.
3. **Acceptance bound** for shadow divergence: define exact pass criteria
   (max label-mismatch rate ≈ 0; max `|score delta|`). Both backends run ONNX
   Runtime, so numerics should align closely.
4. **Auth crate**: `gcp_auth` vs `yup-oauth2` — pick one (first GCP auth dep).
5. **Tokenizer sourcing on pods**: keep fetching `tokenizer.json` at startup,
   or bundle it into the image to remove a startup network dependency.

---

## 10. Test requirements

1. **Refactor parity**: `OnnxSusFactor` MUST pass all existing SusFactor unit
   and golden-vector tests unchanged (see `spec/test-vectors/susfactor_vectors.json`).
2. **Vertex protocol (mocked)**: using `mockito`, assert that for fixed logits
   returned by a mock endpoint, `VertexSusFactor.classify()` yields a
   `ChunkedSusFactorResult` identical to what the shared softmax/label/reduce
   path produces for the same logits. Cover: single chunk, multi-chunk (long
   prompt), `>=` threshold boundary, and HTTP/timeout error mapping.
3. **Shadow divergence**: unit-test the shadow wrapper returns the primary
   (ONNX) result and reports the correct score delta / label mismatch.
4. **Feature isolation**: `cargo build --features susfactor-vertex
   --no-default-features` MUST compile without `ort`/`ndarray`.

---

## 11. Implementation checklist

- [ ] Extract `susfactor::common` (tokenize, chunk, softmax, label, reduce).
- [ ] Define `SusFactorProvider` trait (`src/susfactor/provider.rs`).
- [ ] Refactor current classifier into `OnnxSusFactor` impl; keep alias.
- [ ] Implement `VertexAuth` (ADC / metadata, cached refresh).
- [ ] Implement `VertexSusFactor` (`src/susfactor/vertex.rs`): rawPredict body,
      logits parse, bounded-concurrency chunk fan-out, timeouts.
- [ ] Add `susfactor-vertex` feature; widen `tokenizers` gating.
- [ ] Tests per §10.
- [ ] Update `mod.rs` re-exports and `INTEGRATION.md` §2 implementation table.
- [ ] CHANGELOG; bump to v0.8.0; tag.

---

## 12. References

- SusFactor classifier (ONNX reference): `packages/rust/src/susfactor/classifier.rs`
- SusFactor types: `packages/rust/src/susfactor/types.rs`
- Caller-facing contract: [`INTEGRATION.md` §2](INTEGRATION.md)
- Embedding provider trait (pattern precedent): `packages/rust/src/provider.rs`
- Heimdall host integration: `github.com/0din-ai/heimdall`
  (`crates/heimdall-server/src/service.rs`, `config.rs`)
- Vertex AI `rawPredict`: Google Cloud AI Platform prediction API
- Triton KServe V2 inference protocol
