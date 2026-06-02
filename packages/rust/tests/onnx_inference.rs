//! Gated integration tests for the ONNX Runtime provider.
//!
//! These tests run real inference and therefore require a downloaded/local
//! model. They are `#[ignore]`d by default so the standard `cargo test` run
//! stays fast and offline.
//!
//! To run them, point `ODIN_ONNX_TEST_MODEL` at a local model directory that
//! contains `onnx/model.onnx` plus the tokenizer, and pass `--ignored`:
//!
//! ```bash
//! ODIN_ONNX_TEST_MODEL=models/v1 \
//!   cargo test --features onnx --test onnx_inference -- --ignored
//! ```
//!
//! If `ODIN_ONNX_TEST_MODEL` is unset, the tests no-op (skip) rather than fail,
//! so they're safe to enable in environments where a model isn't provisioned.

#![cfg(feature = "onnx")]

use odin_prompt_toolkit::provider::EmbeddingProvider;
use odin_prompt_toolkit::providers::{ModelCache, OnnxProvider};

/// Returns the model path from `ODIN_ONNX_TEST_MODEL`, or `None` to skip.
fn test_model() -> Option<String> {
    std::env::var("ODIN_ONNX_TEST_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
}

#[tokio::test]
#[ignore = "requires a local ONNX model; set ODIN_ONNX_TEST_MODEL and run with --ignored"]
async fn test_real_inference_dimensions() {
    let Some(model) = test_model() else {
        eprintln!("ODIN_ONNX_TEST_MODEL not set; skipping real-inference test");
        return;
    };

    let cache = ModelCache::new().expect("model cache");
    // pool_size=1, intra_threads=1 keeps the test deterministic and light.
    let provider = OnnxProvider::new(&cache, Some(model), None, 1, 1)
        .await
        .expect("provider initializes");

    assert_eq!(provider.dimensions(), 1024);

    let result = provider
        .generate_embedding("hello world")
        .await
        .expect("embedding generated");

    assert_eq!(result.dimensions, 1024);
    assert_eq!(result.embedding.len(), 1024);
    assert_eq!(result.normalized_embedding.len(), 1024);
    assert!(result.token_count.is_some());

    // Normalized embedding should be unit length (L2 norm ~= 1.0).
    let norm: f32 = result
        .normalized_embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "normalized embedding L2 norm should be ~1.0, got {norm}"
    );
}

#[tokio::test]
#[ignore = "requires a local ONNX model; set ODIN_ONNX_TEST_MODEL and run with --ignored"]
async fn test_pool_concurrency_smoke() {
    let Some(model) = test_model() else {
        eprintln!("ODIN_ONNX_TEST_MODEL not set; skipping pool concurrency smoke test");
        return;
    };

    let cache = ModelCache::new().expect("model cache");
    // A 2-session pool exercised by more concurrent callers than sessions.
    let provider = std::sync::Arc::new(
        OnnxProvider::new(&cache, Some(model), None, 1, 2)
            .await
            .expect("provider initializes"),
    );

    let mut handles = Vec::new();
    for i in 0..8 {
        let p = std::sync::Arc::clone(&provider);
        handles.push(tokio::spawn(async move {
            p.generate_embedding(&format!("concurrent request {i}"))
                .await
        }));
    }

    for handle in handles {
        let result = handle.await.expect("task joins").expect("embedding ok");
        assert_eq!(result.dimensions, 1024);
    }
}
