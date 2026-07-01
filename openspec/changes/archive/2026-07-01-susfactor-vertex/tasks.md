# Tasks: susfactor-vertex

These tasks follow a strict TDD sequence. Each task must have a failing test before implementation.

## Task 1: Extract `susfactor::common`

**Files**: `packages/rust/src/susfactor/common.rs` (new), `packages/rust/src/susfactor/classifier.rs` (refactor)

**Work**:
1. Create `common.rs` with public functions: `tokenize_full`, `chunk_token_ids`, `suspicious_prob`, `label_for_score`, `assemble_chunk_result`, `reduce_to_chunked_result`
2. Refactor `classifier.rs` to call `common::*` instead of inline impls
3. Ensure all existing tests pass (no behavior change)

**Test**: All existing SusFactor unit tests + golden vector tests pass. Run: `cargo test --features susfactor -p odin-prompt-toolkit`

**Done when**: `cargo test --features susfactor` is green; `common.rs` exports all 6 shared functions.

---

## Task 2: Define `SusFactorProvider` trait + rename `SusFactorClassifier` → `OnnxSusFactor`

**Files**: `packages/rust/src/susfactor/provider.rs` (new), `packages/rust/src/susfactor/classifier.rs`, `packages/rust/src/susfactor/mod.rs`, `packages/rust/src/lib.rs`

**Work**:
1. Create `provider.rs` with `SusFactorProvider` trait
2. Rename `SusFactorClassifier` struct to `OnnxSusFactor`; implement `SusFactorProvider`
3. Add `pub type SusFactorClassifier = OnnxSusFactor;` alias with `#[deprecated]` note
4. Update `mod.rs` re-exports; update `lib.rs` re-exports
5. Ensure all existing tests compile and pass

**Test**: Existing tests compile with renamed type; `OnnxSusFactor` implements `SusFactorProvider` (compile-time check via a trait object test).

**Done when**: `cargo test --features susfactor` green; `OnnxSusFactor` implements `SusFactorProvider`.

---

## Task 3: Update Cargo.toml — `susfactor-vertex` feature + widen `tokenizers` gating

**Files**: `packages/rust/Cargo.toml`

**Work**:
1. Add `gcp_auth = { version = "0.12", optional = true }` to `[dependencies]`
2. Change `tokenizers` from `onnx`-only to standalone optional (gated by `onnx` OR `susfactor-vertex`)
3. Add feature: `susfactor-vertex = ["dep:gcp_auth", "dep:tokenizers", "dep:reqwest", "reqwest?/json"]`

**Test**: `cargo build --features susfactor-vertex --no-default-features` compiles (no `ort`/`ndarray`).

**Done when**: Feature-isolated build succeeds; existing `cargo test --features susfactor` still green.

---

## Task 4: Implement `VertexSusFactor`

**Files**: `packages/rust/src/susfactor/vertex.rs` (new), `packages/rust/src/susfactor/mod.rs`

**Work**:
1. Define wire types (`InferRequest`, `InferInput`, `InferResponse`, `InferOutput`) with `serde` derives
2. Implement `VertexSusFactor` struct + constructor (loads tokenizer via `ModelCache`)
3. Implement `classify()` with: tokenize → chunk → bounded-concurrency rawPredict fan-out (via `tokio::task::JoinSet`) → logits parse → shared softmax/label/reduce
4. Timeout: connect 5s, total 30s (configurable via constructor)
5. Error handling: HTTP errors → `SigError::Provider`; shape errors → `SigError::Model`
6. Add to `mod.rs` under `#[cfg(feature = "susfactor-vertex")]`

**Tests** (all mocked via `mockito`):
- Single chunk: mock returns `{"outputs":[{"name":"logits","shape":[1,2],"datatype":"FP32","data":[−1.5,2.3]}]}`; assert score ≈ `softmax([−1.5,2.3])[1]`
- Multi-chunk long prompt: mock returns logits for 2+ chunks; assert `ChunkedSusFactorResult.chunks.len() == N`
- Threshold boundary: score == threshold → label is "suspicious" (inclusive)
- HTTP 500 error → `SigError::Provider`
- Timeout → `SigError::Provider`
- Missing `logits` output → falls back to first output
- Zero outputs → `SigError::Model`

**Done when**: All 7 mock tests pass; `cargo build --features susfactor-vertex --no-default-features` clean.

---

## Task 5: Implement `ShadowSusFactor`

**Files**: `packages/rust/src/susfactor/shadow.rs` (new), `packages/rust/src/susfactor/mod.rs`

**Work**:
1. Define `ShadowDivergence { chunk_score_deltas: Vec<f32>, label_mismatch: bool, is_suspicious_mismatch: bool }`
2. Implement `ShadowSusFactor { primary: OnnxSusFactor, shadow: VertexSusFactor }` with `classify_with_divergence()`
3. Use `tokio::join!` for concurrent invocation
4. Vertex failure → return `(onnx_result, None)` (no error propagation)
5. Add to `mod.rs` under `#[cfg(all(feature = "susfactor", feature = "susfactor-vertex"))]`

**Tests**:
- Shadow returns primary (ONNX) result when Vertex succeeds
- Shadow returns primary result when Vertex fails (mock returns 500)
- Divergence fields correct: score delta = |onnx.score − vertex.score| per chunk; label_mismatch and is_suspicious_mismatch flags correct

**Done when**: 3 shadow tests pass; existing tests still green.

---

## Task 6: Update re-exports + INTEGRATION.md

**Files**: `packages/rust/src/susfactor/mod.rs`, `packages/rust/src/lib.rs`, `spec/INTEGRATION.md`

**Work**:
1. Export `SusFactorProvider`, `OnnxSusFactor`, `VertexSusFactor`, `ShadowSusFactor`, `ShadowDivergence` from `susfactor/mod.rs`
2. Update `lib.rs` to re-export `OnnxSusFactor` (and keep `SusFactorClassifier` alias) under `susfactor` feature; export `VertexSusFactor`, `ShadowSusFactor` under `susfactor-vertex`
3. Update `INTEGRATION.md §2` backend selection table to include `vertex` and `shadow` entries

**Done when**: `cargo doc --features susfactor,susfactor-vertex` builds without missing-doc warnings on public types.

---

## Task 7: Final validation

**Work**:
1. `cargo test --features susfactor` — all existing tests green
2. `cargo test --features susfactor,susfactor-vertex` — all new tests green
3. `cargo build --features susfactor-vertex --no-default-features` — no `ort`/`ndarray` in dep graph
4. `cargo clippy --features susfactor,susfactor-vertex -- -D warnings` — clean
5. Bump version to `0.8.0` in `Cargo.toml`; update `CHANGELOG.md`

**Done when**: All checks pass.
