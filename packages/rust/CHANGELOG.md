# Changelog

All notable changes to the `odin-prompt-toolkit` Rust crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-07-01

### Added
- `susfactor-vertex` Cargo feature: new `VertexSusFactor` backend that delegates
  ONNX graph execution to a Vertex AI Triton endpoint via `rawPredict`. Tokenization,
  chunking, softmax, and labeling remain client-side — results are byte-compatible
  with the ONNX backend.
- `SusFactorProvider` trait: abstraction over `OnnxSusFactor` and `VertexSusFactor`.
- `ShadowSusFactor`: runs both backends concurrently, returns the primary (ONNX)
  result, and emits a `ShadowDivergence` signal for divergence tracking during
  migration.
- `susfactor::common` module: shared tokenize/chunk/softmax/label/reduce functions
  used verbatim by both backends.

### Changed
- `SusFactorClassifier` renamed to `OnnxSusFactor`. `SusFactorClassifier` is
  retained as a deprecated type alias and will be removed in v0.9.0.

### Dependencies
- Added `gcp_auth = "0.12"` (optional, enabled by `susfactor-vertex` feature).

## [0.2.0] - Unreleased

Production-grade concurrency hardening for `OnnxProvider` (0DIN-1555). Adopts
Heimdall's proven ONNX Runtime provider into the SDK.

### Changed

- **BREAKING:** `OnnxProvider::new` now takes five arguments:
  `new(cache, model, name, intra_threads, pool_size)`. Existing 3-arg callers
  should pass `0, 0` for the new parameters to use auto intra-op threads and the
  default pool size (2).
- **BREAKING:** The ONNX inference runtime has been swapped from `tract` to
  ONNX Runtime (the `ort` crate), matching the sibling Python/TypeScript packages
  and Heimdall's production provider. Embedding floating-point output may differ
  slightly from the `tract` implementation; LSH signatures remain robust because
  SimHash depends only on hyperplane sign bits.
- Inference is now offloaded to `tokio::task::spawn_blocking`, so CPU-bound
  forward passes no longer block the async runtime's worker threads.
- The model file loaded is now `onnx/model.onnx`. The `tract`-specific
  `onnx/model_O4.onnx` artifact and its fallback have been removed; ONNX Runtime
  applies its own `GraphOptimizationLevel::Level3` fusions to the base graph.

### Added

- `OnnxSessionPool`: a lock-free round-robin pool of `Arc<Mutex<Session>>` ORT
  sessions, allowing `pool_size` inference requests to run concurrently.
- `pool_size` and `intra_threads` configuration on `OnnxProvider::new` for
  bounding ORT thread usage under Kubernetes cgroup CPU limits
  (keep `pool_size × intra_threads ≤ pod CPU limit`).
- Automatic `token_type_ids` input detection (BERT-style vs XLM-RoBERTa models).
- `examples/benchmark_onnx_load.rs`: a concurrent load-test harness reporting
  p50/p95 latency and throughput for a configurable concurrency / pool size.

### Dependencies

- Added `ort = "=2.0.0-rc.12"` (features: `std`, `ndarray`, `download-binaries`,
  `tls-rustls`) and `ndarray = "0.17"` under the `onnx` feature.
- Removed `tract-onnx`.
- Building the `onnx` feature now downloads a native ONNX Runtime binary at build
  time (or set `ORT_DYLIB_PATH` for offline/air-gapped builds).

## [0.1.1]

- Initial published baseline (tract-based `OnnxProvider`).
