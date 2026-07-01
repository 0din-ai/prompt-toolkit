//! Remote SusFactor classifier delegating ONNX graph execution to a Vertex AI
//! endpoint.
//!
//! Tokenization, chunking, softmax, and labeling remain client-side (shared
//! [`super::common`] logic) so results are identical to the in-pod ONNX backend
//! modulo floating-point rounding.
//!
//! ## Wire protocol
//!
//! The Vertex endpoint must speak the Triton Inference Server / Vertex Dedicated
//! Serving V2 HTTP JSON protocol:
//!
//! - **Request**: `POST <endpoint_url>` with body matching [`InferRequest`].
//! - **Response**: JSON body matching [`InferResponse`], where the `"logits"`
//!   output contains a flat `float32` array of length ≥ 2.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::error::{Result, SigError};
use crate::providers::ModelCache;
use crate::susfactor::common;
use crate::susfactor::provider::SusFactorProvider;
use crate::susfactor::types::{ChunkedSusFactorResult, SusFactorResult};

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct InferRequest {
    inputs: Vec<InferInput>,
}

#[derive(serde::Serialize)]
struct InferInput {
    name: &'static str,
    shape: [usize; 2],
    datatype: &'static str,
    data: Vec<i64>,
}

#[derive(serde::Deserialize)]
struct InferResponse {
    outputs: Vec<InferOutput>,
}

#[derive(serde::Deserialize)]
struct InferOutput {
    name: String,
    data: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Classifies prompts as safe vs. suspicious using the SusFactor model served
/// remotely on a Vertex AI endpoint.
///
/// Tokenization, chunking, softmax, and labeling run client-side using the
/// shared [`super::common`] helpers so results are identical to
/// [`super::onnx::OnnxSusFactor`] modulo network latency.
pub struct VertexSusFactor {
    client: reqwest::Client,
    endpoint_url: String,
    tokenizer: Arc<tokenizers::Tokenizer>,
    model_name: String,
    threshold: f32,
    auth: Arc<dyn gcp_auth::TokenProvider>,
    max_concurrent_chunks: usize,
}

impl VertexSusFactor {
    /// Canonical model identifier reported in results (shared across SDKs).
    pub const DEFAULT_MODEL: &'static str = "0dinai/susfactor-e5-large";

    /// HuggingFace repo holding the tokenizer downloaded at runtime.
    pub const DEFAULT_ONNX_REPO: &'static str = "0dinai/susfactor-e5-large-onnx";

    /// Default decision threshold.
    pub const DEFAULT_THRESHOLD: f32 = 0.5;

    /// Default number of in-flight chunk requests.
    pub const DEFAULT_MAX_CONCURRENT_CHUNKS: usize = 4;

    /// GCP scope required to call a Vertex AI endpoint.
    const VERTEX_SCOPE: &'static str = "https://www.googleapis.com/auth/cloud-platform";

    /// Create a new [`VertexSusFactor`] classifier.
    ///
    /// Downloads the tokenizer from HuggingFace (or uses the local cache) and
    /// initialises GCP authentication via Application Default Credentials.
    ///
    /// # Arguments
    ///
    /// * `cache` — Model cache for the tokenizer download.
    /// * `endpoint_url` — Full Vertex AI `rawPredict` URL.
    /// * `model_source` — HF repo ID for the tokenizer (default:
    ///   [`Self::DEFAULT_ONNX_REPO`]).
    /// * `model_name` — Canonical model identifier reported in results (default:
    ///   [`Self::DEFAULT_MODEL`]).
    /// * `threshold` — Decision threshold (default: [`Self::DEFAULT_THRESHOLD`]).
    /// * `max_concurrent_chunks` — In-flight chunk request limit (default:
    ///   [`Self::DEFAULT_MAX_CONCURRENT_CHUNKS`]).
    pub async fn new(
        cache: &ModelCache,
        endpoint_url: String,
        model_source: Option<String>,
        model_name: Option<String>,
        threshold: Option<f32>,
        max_concurrent_chunks: Option<usize>,
    ) -> Result<Self> {
        let source = model_source.unwrap_or_else(|| Self::DEFAULT_ONNX_REPO.to_string());
        let model_name = model_name.unwrap_or_else(|| Self::DEFAULT_MODEL.to_string());
        let threshold = threshold.unwrap_or(Self::DEFAULT_THRESHOLD);
        let max_concurrent_chunks =
            max_concurrent_chunks.unwrap_or(Self::DEFAULT_MAX_CONCURRENT_CHUNKS);

        // Load tokenizer only — no ONNX weights needed for the Vertex backend.
        let tokenizer_path = cache.get_tokenizer(&source).await?;
        let tokenizer = tokio::task::spawn_blocking(move || {
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| SigError::Model(format!("Failed to load SusFactor tokenizer: {e}")))
        })
        .await
        .map_err(|e| SigError::Model(format!("spawn_blocking panicked: {e}")))??;

        let auth = gcp_auth::provider()
            .await
            .map_err(|e| SigError::Provider(format!("GCP auth initialisation failed: {e}")))?;

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| SigError::Provider(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            client,
            endpoint_url,
            tokenizer: Arc::new(tokenizer),
            model_name,
            threshold,
            auth,
            max_concurrent_chunks,
        })
    }

    /// Classify a single chunk of token IDs against the remote endpoint.
    async fn classify_chunk(
        &self,
        chunk_ids: Vec<i64>,
        start_time: Instant,
    ) -> Result<SusFactorResult> {
        let seq_len = chunk_ids.len();
        let attention_mask: Vec<i64> = vec![1i64; seq_len];

        let body = InferRequest {
            inputs: vec![
                InferInput {
                    name: "input_ids",
                    shape: [1, seq_len],
                    datatype: "INT64",
                    data: chunk_ids,
                },
                InferInput {
                    name: "attention_mask",
                    shape: [1, seq_len],
                    datatype: "INT64",
                    data: attention_mask,
                },
            ],
        };

        let token = self
            .auth
            .token(&[Self::VERTEX_SCOPE])
            .await
            .map_err(|e| SigError::Provider(format!("Vertex AI token fetch failed: {e}")))?;

        let response = self
            .client
            .post(&self.endpoint_url)
            .bearer_auth(token.as_str())
            .json(&body)
            .send()
            .await
            .map_err(|e| SigError::Provider(format!("Vertex AI request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            return Err(SigError::Provider(format!(
                "Vertex AI returned HTTP {status}: {body_text}"
            )));
        }

        let infer: InferResponse = response
            .json()
            .await
            .map_err(|e| SigError::Provider(format!("Vertex AI response parse failed: {e}")))?;

        // Locate the logits output: prefer the output named "logits"; fall back
        // to the first output; error if there are no outputs at all.
        let output = infer
            .outputs
            .iter()
            .find(|o| o.name == "logits")
            .or_else(|| infer.outputs.first())
            .ok_or_else(|| {
                SigError::Model(
                    "Unexpected SusFactor output shape; got 0 elements, expected >= 2".to_string(),
                )
            })?;

        let logits = common::validate_logits(output.data.clone())?;

        Ok(common::result_from_logits(
            &logits,
            &self.model_name,
            self.threshold,
            common::elapsed_ms(start_time),
        ))
    }
}

#[async_trait]
impl SusFactorProvider for VertexSusFactor {
    fn model(&self) -> &str {
        &self.model_name
    }

    fn threshold(&self) -> f32 {
        self.threshold
    }

    async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult> {
        let wall_start = Instant::now();

        let (all_ids, _all_mask) = common::tokenize_full(&self.tokenizer, text)?;
        let id_chunks = common::chunk_token_ids(&all_ids);

        let concurrency = self.max_concurrent_chunks;

        // Fan out chunk requests concurrently, bounded by max_concurrent_chunks.
        let chunk_results: Vec<Result<SusFactorResult>> =
            futures_util::stream::iter(id_chunks.into_iter().map(|chunk| {
                let chunk_start = Instant::now();
                self.classify_chunk(chunk, chunk_start)
            }))
            .buffer_unordered(concurrency)
            .collect()
            .await;

        // Collect results, propagating the first error.
        let mut results = Vec::with_capacity(chunk_results.len());
        for r in chunk_results {
            results.push(r?);
        }

        Ok(common::reduce(results, common::elapsed_ms(wall_start)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SigError;
    use crate::susfactor::types::LABEL_SUSPICIOUS;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Test-only VertexSusFactor builder that bypasses the ModelCache and
    // gcp_auth initialisation.  All tests use a mock token provider and a
    // pre-loaded tokenizer downloaded from HuggingFace the first time the
    // test suite runs (cached under $HOME/.cache/signature-sdk/…).
    // -----------------------------------------------------------------------

    /// A no-op token provider that always returns a static bearer token.
    struct FakeTokenProvider;

    #[async_trait::async_trait]
    impl gcp_auth::TokenProvider for FakeTokenProvider {
        async fn token(&self, _scopes: &[&str]) -> std::result::Result<Arc<gcp_auth::Token>, gcp_auth::Error> {
            // gcp_auth::Token stores the raw JSON from a token endpoint; the
            // simplest way to construct a test token without private fields is
            // to parse a minimal JSON blob.
            let token: gcp_auth::Token = serde_json::from_str(
                r#"{"access_token":"test-token","expires_in":3600,"token_type":"Bearer"}"#,
            )
            .expect("static test token must parse");
            Ok(Arc::new(token))
        }

        async fn project_id(&self) -> std::result::Result<Arc<str>, gcp_auth::Error> {
            Ok(Arc::from("test-project"))
        }
    }

    /// Load a real tokenizer once per process (cached on disk after the first run).
    fn load_test_tokenizer() -> Arc<tokenizers::Tokenizer> {
        // Try the cached path first.
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
            .join("signature-sdk")
            .join("models")
            .join("0dinai")
            .join("susfactor-e5-large-onnx")
            .join("tokenizer.json");

        if cache_dir.exists() {
            let tok = tokenizers::Tokenizer::from_file(&cache_dir)
                .expect("cached tokenizer must load");
            return Arc::new(tok);
        }

        // Fall back: download synchronously via the blocking HuggingFace URL.
        let url = "https://huggingface.co/0dinai/susfactor-e5-large-onnx/resolve/main/tokenizer.json";
        let bytes = std::process::Command::new("curl")
            .args(["-fsSL", "--max-time", "30", url])
            .output()
            .expect("curl must be available to download test tokenizer");

        if !bytes.status.success() {
            panic!(
                "Failed to download test tokenizer; run tests with network access once to populate the cache.\ncurl stderr: {}",
                String::from_utf8_lossy(&bytes.stderr)
            );
        }

        // Write to cache so subsequent runs skip the download.
        if let Some(parent) = cache_dir.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&cache_dir, &bytes.stdout);

        let tok = tokenizers::Tokenizer::from_bytes(&bytes.stdout)
            .expect("downloaded tokenizer bytes must parse");
        Arc::new(tok)
    }

    fn build_vertex(server_url: &str, threshold: f32) -> VertexSusFactor {
        let tokenizer = load_test_tokenizer();
        VertexSusFactor {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap(),
            endpoint_url: format!("{server_url}/rawPredict"),
            tokenizer,
            model_name: VertexSusFactor::DEFAULT_MODEL.to_string(),
            threshold,
            auth: Arc::new(FakeTokenProvider),
            max_concurrent_chunks: 4,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1 — single chunk, correct softmax score
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn single_chunk_score_matches_softmax() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/rawPredict")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"outputs":[{"name":"logits","shape":[1,2],"datatype":"FP32","data":[-1.5,2.3]}]}"#,
            )
            .create_async()
            .await;

        let clf = build_vertex(&server.url(), 0.5);
        let result = clf.classify("hello world").await.expect("classify must succeed");

        assert_eq!(result.chunks.len(), 1);

        // softmax([-1.5, 2.3])[1] ≈ 0.9802
        let expected: f32 = {
            let logits = [-1.5_f32, 2.3_f32];
            common::suspicious_prob(&logits)
        };
        assert!(
            (result.chunks[0].score - expected).abs() < 1e-4,
            "score {} not within 1e-4 of expected {}",
            result.chunks[0].score,
            expected
        );
        assert_eq!(result.chunks[0].model, VertexSusFactor::DEFAULT_MODEL);
    }

    // -----------------------------------------------------------------------
    // Test 2 — multi-chunk prompt produces >= 2 chunks
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn multi_chunk_prompt_produces_multiple_chunks() {
        use crate::susfactor::types::MAX_CONTENT_TOKENS;

        let mut server = mockito::Server::new_async().await;
        // Always return the same body regardless of how many times it's called.
        let _mock = server
            .mock("POST", "/rawPredict")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"outputs":[{"name":"logits","shape":[1,2],"datatype":"FP32","data":[0.1,0.2]}]}"#,
            )
            .expect_at_least(2)
            .create_async()
            .await;

        let clf = build_vertex(&server.url(), 0.5);

        // Repeat a word enough times to exceed MAX_CONTENT_TOKENS after tokenization.
        // Each token is roughly one word; 1100 words guarantees > 1020 tokens.
        let long_text = "word ".repeat(MAX_CONTENT_TOKENS * 3);
        let result = clf
            .classify(&long_text)
            .await
            .expect("classify must succeed");

        assert!(
            result.chunks.len() >= 2,
            "expected >= 2 chunks, got {}",
            result.chunks.len()
        );

        // All chunks should have consistent (non-NaN, non-Inf) scores.
        for chunk in &result.chunks {
            assert!(chunk.score.is_finite(), "score must be finite");
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — threshold boundary: logits [0,0] → score == 0.5 → "suspicious"
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn threshold_boundary_score_at_threshold_is_suspicious() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/rawPredict")
            .with_status(200)
            .with_header("content-type", "application/json")
            // softmax([0.0, 0.0])[1] == 0.5 exactly
            .with_body(
                r#"{"outputs":[{"name":"logits","shape":[1,2],"datatype":"FP32","data":[0.0,0.0]}]}"#,
            )
            .create_async()
            .await;

        let clf = build_vertex(&server.url(), 0.5);
        let result = clf.classify("test").await.expect("classify must succeed");

        assert_eq!(result.chunks.len(), 1);
        let chunk = &result.chunks[0];
        assert!(
            (chunk.score - 0.5).abs() < 1e-6,
            "score {} should be exactly 0.5",
            chunk.score
        );
        assert_eq!(
            chunk.label, LABEL_SUSPICIOUS,
            "score == threshold must be labelled suspicious"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4 — HTTP 500 → SigError::Provider
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn http_500_returns_provider_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/rawPredict")
            .with_status(500)
            .with_header("content-type", "application/json")
            .with_body(r#"{"error":{"code":500,"message":"Internal","status":"INTERNAL"}}"#)
            .create_async()
            .await;

        let clf = build_vertex(&server.url(), 0.5);
        let err = clf.classify("test").await.expect_err("must return error");

        match &err {
            SigError::Provider(msg) => {
                assert!(
                    msg.contains("500") || msg.contains("Vertex AI"),
                    "error message should contain '500' or 'Vertex AI', got: {msg}"
                );
            }
            other => panic!("expected SigError::Provider, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — zero outputs → SigError::Model
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn zero_outputs_returns_model_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/rawPredict")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"outputs":[]}"#)
            .create_async()
            .await;

        let clf = build_vertex(&server.url(), 0.5);
        let err = clf.classify("test").await.expect_err("must return error");

        match &err {
            SigError::Model(msg) => {
                assert!(
                    msg.contains("output shape") || msg.contains("0 elements"),
                    "error message should reference output shape, got: {msg}"
                );
            }
            other => panic!("expected SigError::Model, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — missing "logits" output → fallback to first output
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn missing_logits_name_falls_back_to_first_output() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/rawPredict")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"outputs":[{"name":"scores","shape":[1,2],"datatype":"FP32","data":[-1.0,1.0]}]}"#,
            )
            .create_async()
            .await;

        let clf = build_vertex(&server.url(), 0.5);
        let result = clf
            .classify("test")
            .await
            .expect("should succeed with first-output fallback");

        let expected = common::suspicious_prob(&[-1.0_f32, 1.0_f32]);
        assert!(
            (result.chunks[0].score - expected).abs() < 1e-4,
            "score {} not close to expected {}",
            result.chunks[0].score,
            expected
        );
    }

    // -----------------------------------------------------------------------
    // Test 7 — refused connection → SigError::Provider
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn refused_connection_returns_provider_error() {
        // Port 1 is well-known to be refused on all platforms.
        let clf = VertexSusFactor {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .unwrap(),
            endpoint_url: "http://127.0.0.1:1/rawPredict".to_string(),
            tokenizer: load_test_tokenizer(),
            model_name: VertexSusFactor::DEFAULT_MODEL.to_string(),
            threshold: 0.5,
            auth: Arc::new(FakeTokenProvider),
            max_concurrent_chunks: 4,
        };

        let err = clf.classify("test").await.expect_err("must return error");

        match &err {
            SigError::Provider(_) => {}
            other => panic!("expected SigError::Provider, got {other:?}"),
        }
    }
}
