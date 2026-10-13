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
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use tracing::Instrument;

use crate::error::{Result, SigError};
use crate::providers::ModelCache;
use crate::susfactor::common;
use crate::susfactor::provider::SusFactorProvider;
use crate::susfactor::types::{ChunkedSusFactorResult, PhaseSpan, SusFactorResult};

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
    client: reqwest_middleware::ClientWithMiddleware,
    endpoint_url: String,
    tokenizer: Arc<tokenizers::Tokenizer>,
    model_name: String,
    threshold: f32,
    auth: Arc<dyn gcp_auth::TokenProvider>,
    max_concurrent_chunks: usize,
    /// `(bos_id, eos_id)`, resolved once at construction — the tokenizer's
    /// vocab never changes for the lifetime of this classifier.
    special_tokens: (i64, i64),
}

impl VertexSusFactor {
    /// Canonical model identifier reported in results (shared across SDKs).
    pub const DEFAULT_MODEL: &'static str = common::DEFAULT_MODEL;

    #[deprecated(since = "0.8.0", note = "use DEFAULT_TOKENIZER_REPO instead")]
    /// Alias kept for backward compatibility; prefer [`Self::DEFAULT_TOKENIZER_REPO`].
    pub const DEFAULT_ONNX_REPO: &'static str = common::DEFAULT_ONNX_REPO;

    /// HuggingFace repo holding the tokenizer downloaded at runtime.
    pub const DEFAULT_TOKENIZER_REPO: &'static str = common::DEFAULT_ONNX_REPO;

    /// Default decision threshold.
    pub const DEFAULT_THRESHOLD: f32 = common::DEFAULT_THRESHOLD;

    /// Default number of in-flight chunk requests.
    pub const DEFAULT_MAX_CONCURRENT_CHUNKS: usize = 4;

    /// GCP scope required to call a Vertex AI endpoint.
    const VERTEX_SCOPE: &'static str = "https://www.googleapis.com/auth/cloud-platform";

    /// Maximum bytes of an error response body embedded in a [`SigError::Provider`]
    /// message. Prevents accidentally logging megabyte-sized HTML error pages.
    const MAX_ERROR_BODY: usize = 512;

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
    ///   [`Self::DEFAULT_TOKENIZER_REPO`]).
    /// * `model_name` — Canonical model identifier reported in results (default:
    ///   [`Self::DEFAULT_MODEL`]).
    /// * `threshold` — Decision threshold (default: [`Self::DEFAULT_THRESHOLD`]).
    /// * `max_concurrent_chunks` — In-flight chunk request limit (default:
    ///   [`Self::DEFAULT_MAX_CONCURRENT_CHUNKS`]).
    /// * `connect_timeout` — TCP connect timeout (default: 5 s).
    /// * `request_timeout` — Full request timeout (default: 30 s).
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        cache: &ModelCache,
        endpoint_url: String,
        model_source: Option<String>,
        model_name: Option<String>,
        threshold: Option<f32>,
        max_concurrent_chunks: Option<usize>,
        connect_timeout: Option<Duration>,
        request_timeout: Option<Duration>,
    ) -> Result<Self> {
        let source = model_source.unwrap_or_else(|| Self::DEFAULT_TOKENIZER_REPO.to_string());
        let model_name = model_name.unwrap_or_else(|| Self::DEFAULT_MODEL.to_string());
        let threshold = threshold.unwrap_or(Self::DEFAULT_THRESHOLD);
        let max_concurrent_chunks =
            max_concurrent_chunks.unwrap_or(Self::DEFAULT_MAX_CONCURRENT_CHUNKS);
        let connect_timeout = connect_timeout.unwrap_or(Duration::from_secs(5));
        let request_timeout = request_timeout.unwrap_or(Duration::from_secs(30));

        // Load tokenizer only — no ONNX weights needed for the Vertex backend.
        let tokenizer_path = cache.get_tokenizer(&source).await?;
        let tokenizer =
            tokio::task::spawn_blocking(move || common::load_tokenizer(&tokenizer_path))
                .await
                .map_err(|e| SigError::Model(format!("spawn_blocking panicked: {e}")))??;

        let auth = gcp_auth::provider()
            .await
            .map_err(|e| SigError::Provider(format!("GCP auth initialisation failed: {e}")))?;

        let client = Self::build_traced_client(connect_timeout, request_timeout)?;
        let special_tokens = common::resolve_special_token_ids(&tokenizer)?;

        Ok(Self {
            client,
            endpoint_url,
            tokenizer: Arc::new(tokenizer),
            model_name,
            threshold,
            auth,
            max_concurrent_chunks,
            special_tokens,
        })
    }

    /// Build the `reqwest::Client` used for `rawPredict` calls, wrapped with a
    /// tracing middleware that emits a CLIENT-kind span per request (method,
    /// `url.full`, `server.address`, response status/error) so calls show up
    /// in downstream OpenTelemetry/Elastic APM pipelines.
    ///
    /// Trace-context propagation is disabled: Vertex is an external Google
    /// endpoint that won't honor W3C `traceparent` headers, and internal
    /// trace IDs should not be leaked to it.
    fn build_traced_client(
        connect_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<reqwest_middleware::ClientWithMiddleware> {
        let inner = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .timeout(request_timeout)
            .build()
            .map_err(|e| SigError::Provider(format!("Failed to create HTTP client: {e}")))?;

        Ok(reqwest_middleware::ClientBuilder::new(inner)
            .with_init(reqwest_middleware::Extension(
                reqwest_tracing::DisableOtelPropagation,
            ))
            .with(reqwest_tracing::TracingMiddleware::<
                reqwest_tracing::SpanBackendWithUrl,
            >::new())
            .build())
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

        // Time tokenization of the full text.
        let tokenize_start = Instant::now();
        let all_ids = common::tokenize_full(&self.tokenizer, text)?;
        let (bos_id, eos_id) = self.special_tokens;
        let tokenize_span = PhaseSpan {
            name: common::PHASE_TOKENIZE.to_string(),
            start_ms: common::offset_ms(tokenize_start, wall_start),
            duration_ms: common::elapsed_ms(tokenize_start),
            chunk_index: None,
            token_count: None,
        };

        // Time chunking of the token stream.
        let chunk_start_instant = Instant::now();
        let chunks = common::chunk_token_ids_with_special_tokens(&all_ids, bos_id, eos_id);
        let chunk_span = PhaseSpan {
            name: common::PHASE_CHUNK.to_string(),
            start_ms: common::offset_ms(chunk_start_instant, wall_start),
            duration_ms: common::elapsed_ms(chunk_start_instant),
            chunk_index: None,
            token_count: None,
        };

        let concurrency = self.max_concurrent_chunks;

        // Clone fields needed inside the async closures so `self` is not
        // borrowed across `.await` points inside the stream.
        let client = self.client.clone();
        let endpoint_url = self.endpoint_url.clone();
        let auth = Arc::clone(&self.auth);
        let model_name = self.model_name.clone();
        let threshold = self.threshold;

        // Fan out chunk requests concurrently, bounded by max_concurrent_chunks.
        // Each future captures its chunk index and the wall-clock baseline
        // (`Instant` is `Copy`) so its inference span records a real start
        // offset. `Instant::now()` for `chunk_start` is captured inside the
        // `async move` so overlap is visible on the timeline.
        let mut collected: Vec<(usize, SusFactorResult, PhaseSpan)> =
            futures_util::stream::iter(chunks.into_iter().enumerate().map(
                |(i, (chunk_ids, chunk_mask))| {
                    // Clone per-chunk so each future is self-contained.
                    let client = client.clone();
                    let endpoint_url = endpoint_url.clone();
                    let auth = Arc::clone(&auth);
                    let model_name = model_name.clone();
                    let seq_len = chunk_ids.len();
                    // `.instrument()` (not `.enter()`) so the span is correctly
                    // re-entered on every poll regardless of which executor
                    // thread resumes this future after an `.await` — holding
                    // an `Entered` guard across an await point corrupts
                    // whatever span happens to be ambient on that thread when
                    // this task is later polled. This span becomes the parent
                    // of the `HTTP request` span `reqwest_tracing` emits
                    // inside `send()`, nesting under whatever span was active
                    // when `classify()` was called (e.g. a caller's
                    // `rest_request` span), giving callers true parent-child
                    // trace linkage without the toolkit needing to touch
                    // OpenTelemetry context/baggage directly.
                    //
                    // `chunk_index`/`token_count` are cast to `i64`: `tracing`'s
                    // `Visit::record_u64` default implementation (what a bare
                    // `usize` field dispatches to) forwards to `record_debug`,
                    // and `tracing-opentelemetry`'s span visitor does not
                    // override `record_u64` — only `record_i64`. A `usize`
                    // field is therefore exported as a Debug-formatted STRING
                    // attribute (`"67"`) rather than a native OTEL integer,
                    // which can silently fail to appear in APM if the field is
                    // already mapped as numeric elsewhere (e.g. Elasticsearch
                    // rejects/drops a string write to a `long`-mapped field).
                    // Casting to `i64` routes through `record_i64`, which IS
                    // overridden, producing a proper integer attribute.
                    let span = tracing::info_span!(
                        "vertex_inference",
                        backend = common::BACKEND_VERTEX,
                        chunk_index = i as i64,
                        token_count = seq_len as i64,
                    );
                    async move {
                        let chunk_start = Instant::now();

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
                                    data: chunk_mask,
                                },
                            ],
                        };

                        let token =
                            auth.token(&[VertexSusFactor::VERTEX_SCOPE])
                                .await
                                .map_err(|e| {
                                    SigError::Provider(format!("Vertex AI token fetch failed: {e}"))
                                })?;

                        let response = client
                            .post(&endpoint_url)
                            .bearer_auth(token.as_str())
                            .json(&body)
                            .send()
                            .await
                            .map_err(|e| {
                                SigError::Provider(format!("Vertex AI request failed: {e}"))
                            })?;

                        if !response.status().is_success() {
                            let status = response.status();
                            let body_text = response
                                .text()
                                .await
                                .unwrap_or_else(|_| "<unreadable body>".to_string());
                            let truncated = if body_text.len() > VertexSusFactor::MAX_ERROR_BODY {
                                // Byte-index slicing panics if MAX_ERROR_BODY falls
                                // inside a multi-byte codepoint; walk char
                                // boundaries instead so non-ASCII error bodies
                                // (e.g. Vertex errors in other languages) can't
                                // crash the error-handling path itself.
                                let safe_prefix: String = body_text
                                    .chars()
                                    .take(VertexSusFactor::MAX_ERROR_BODY)
                                    .collect();
                                format!("{safe_prefix}… (truncated)")
                            } else {
                                body_text
                            };
                            return Err(SigError::Provider(format!(
                                "Vertex AI returned HTTP {status}: {truncated}"
                            )));
                        }

                        let infer: InferResponse = response.json().await.map_err(|e| {
                            SigError::Provider(format!("Vertex AI response parse failed: {e}"))
                        })?;

                        let output = infer
                            .outputs
                            .iter()
                            .find(|o| o.name == "logits")
                            .or_else(|| infer.outputs.first())
                            .ok_or_else(|| {
                                SigError::Model(
                                    "Unexpected SusFactor output shape; got 0 elements, expected >= 2"
                                        .to_string(),
                                )
                            })?;

                        common::validate_logits(&output.data)?;

                        let result = common::result_from_logits(
                            &output.data,
                            &model_name,
                            threshold,
                            common::elapsed_ms(chunk_start),
                        );
                        let span = PhaseSpan {
                            name: common::PHASE_INFERENCE.to_string(),
                            start_ms: common::offset_ms(chunk_start, wall_start),
                            duration_ms: result.timing_ms,
                            chunk_index: Some(i),
                            token_count: Some(seq_len),
                        };
                        Ok::<_, SigError>((i, result, span))
                    }
                    .instrument(span)
                },
            ))
            .buffer_unordered(concurrency)
            .try_collect()
            .await?;

        // `buffer_unordered` may complete futures out of order; restore chunk
        // order so both `chunks` and inference spans are deterministic.
        collected.sort_by_key(|(i, _, _)| *i);

        // Assemble the final result; time the reduction itself.
        let reduce_start = Instant::now();
        let mut chunk_results = Vec::with_capacity(collected.len());
        let mut spans = Vec::with_capacity(collected.len() + 3);
        spans.push(tokenize_span);
        spans.push(chunk_span);
        for (_, result, span) in collected {
            chunk_results.push(result);
            spans.push(span);
        }
        spans.push(PhaseSpan {
            name: common::PHASE_REDUCE.to_string(),
            start_ms: common::offset_ms(reduce_start, wall_start),
            duration_ms: common::elapsed_ms(reduce_start),
            chunk_index: None,
            token_count: None,
        });

        Ok(common::reduce(
            chunk_results,
            all_ids.len(),
            common::elapsed_ms(wall_start),
            spans,
        ))
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
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::instrument::WithSubscriber;
    use tracing::span::{Attributes, Id, Record};
    use tracing::Subscriber;
    use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
    use tracing_subscriber::registry::LookupSpan;

    // -----------------------------------------------------------------------
    // Minimal span-capture harness used to assert that the Vertex HTTP call
    // emits a tracing span with the expected CLIENT-kind attributes, without
    // pulling in an OpenTelemetry SDK or exporter for the test itself.
    // -----------------------------------------------------------------------

    #[derive(Default)]
    struct FieldMap(HashMap<String, String>);

    impl Visit for FieldMap {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_u64(&mut self, field: &Field, value: u64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }

        fn record_i64(&mut self, field: &Field, value: i64) {
            self.0.insert(field.name().to_string(), value.to_string());
        }
    }

    struct CapturedSpan {
        name: String,
        fields: HashMap<String, String>,
        parent_name: Option<String>,
    }

    #[derive(Clone, Default)]
    struct SpanCaptureLayer {
        captured: Arc<Mutex<Vec<CapturedSpan>>>,
    }

    impl<S> Layer<S> for SpanCaptureLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            let mut fields = FieldMap::default();
            attrs.record(&mut fields);
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(fields);
            }
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                let mut ext = span.extensions_mut();
                if let Some(fields) = ext.get_mut::<FieldMap>() {
                    values.record(fields);
                }
            }
        }

        fn on_close(&self, id: Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(&id) {
                let name = span.name().to_string();
                let fields = span
                    .extensions()
                    .get::<FieldMap>()
                    .map(|f| f.0.clone())
                    .unwrap_or_default();
                let parent_name = span.parent().map(|p| p.name().to_string());
                self.captured.lock().unwrap().push(CapturedSpan {
                    name,
                    fields,
                    parent_name,
                });
            }
        }
    }

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
        async fn token(
            &self,
            _scopes: &[&str],
        ) -> std::result::Result<Arc<gcp_auth::Token>, gcp_auth::Error> {
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
    async fn load_test_tokenizer() -> Arc<tokenizers::Tokenizer> {
        // Try the cached path first.
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
            .join("signature-sdk")
            .join("models")
            .join("0dinai")
            .join("susfactor-e5-large-onnx")
            .join("tokenizer.json");

        if cache_dir.exists() {
            let tok =
                tokenizers::Tokenizer::from_file(&cache_dir).expect("cached tokenizer must load");
            return Arc::new(tok);
        }

        // Fall back: download via the async reqwest client. Using the
        // blocking client here would panic ("Cannot drop a runtime in a
        // context where blocking is not allowed") because every caller of
        // this helper runs inside a `#[tokio::test]` async body already.
        let url =
            "https://huggingface.co/0dinai/susfactor-e5-large-onnx/resolve/main/tokenizer.json";
        let bytes = async {
            let response = reqwest::get(url).await?;
            response.bytes().await
        }
        .await
        .unwrap_or_else(|e| {
            panic!(
                "load_test_tokenizer: failed to download tokenizer fixture from {url} ({e}); \
                 this test requires network access on first run to populate the on-disk cache"
            );
        });

        // Write to cache so subsequent runs skip the download.
        if let Some(parent) = cache_dir.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&cache_dir, &bytes);

        let tok = tokenizers::Tokenizer::from_bytes(&bytes)
            .expect("downloaded tokenizer bytes must parse");
        Arc::new(tok)
    }

    async fn build_vertex(server_url: &str, threshold: f32) -> VertexSusFactor {
        let tokenizer = load_test_tokenizer().await;
        let special_tokens = common::resolve_special_token_ids(&tokenizer).unwrap();
        VertexSusFactor {
            client: VertexSusFactor::build_traced_client(
                Duration::from_secs(10),
                Duration::from_secs(10),
            )
            .unwrap(),
            endpoint_url: format!("{server_url}/rawPredict"),
            tokenizer,
            model_name: VertexSusFactor::DEFAULT_MODEL.to_string(),
            threshold,
            auth: Arc::new(FakeTokenProvider),
            max_concurrent_chunks: 4,
            special_tokens,
        }
    }

    /// `tracing` caches whether a given span callsite is "interesting"
    /// process-wide, based on whichever subscriber is active the first time
    /// that callsite fires. Other tests in this module call `classify()`
    /// under the ambient (no-op) default, which would otherwise cache the
    /// reqwest-tracing span callsite as permanently uninteresting before our
    /// test-local capturing subscriber ever gets a chance to see it. Installing
    /// a permissive global default once, before any test runs, guarantees the
    /// callsite is always marked interesting so per-test thread-local
    /// dispatches (see `with_subscriber` below) actually receive callbacks.
    fn ensure_global_subscriber_installed() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
        });
    }

    // -----------------------------------------------------------------------
    // Test 0 — rawPredict call emits a CLIENT-kind tracing span with standard
    // HTTP semantic attributes (method, url.full, server.address, response
    // status). This is the observable contract that lets downstream
    // consumers (heimdall) bridge these spans to OpenTelemetry/Elastic APM.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn rawpredict_call_emits_client_span_with_http_semantics() {
        ensure_global_subscriber_installed();

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/rawPredict")
            // DisableOtelPropagation must keep W3C trace-context headers from
            // leaking to Vertex, an external Google endpoint. Requiring the
            // header be absent means the mock only matches (and `classify`
            // only succeeds) when no `traceparent` header was sent.
            .match_header("traceparent", mockito::Matcher::Missing)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"outputs":[{"name":"logits","shape":[1,2],"datatype":"FP32","data":[-1.5,2.3]}]}"#,
            )
            .create_async()
            .await;

        let captured = Arc::new(Mutex::new(Vec::new()));
        let layer = SpanCaptureLayer {
            captured: Arc::clone(&captured),
        };
        let subscriber = tracing_subscriber::registry().with(layer);
        let dispatch = tracing::Dispatch::new(subscriber);

        let clf = build_vertex(&server.url(), 0.5).await;
        clf.classify("hello world")
            .with_subscriber(dispatch)
            .await
            .expect("classify must succeed");

        let captured = captured.lock().unwrap();
        let http_span = captured
            .iter()
            .find(|s| s.name == "HTTP request")
            .unwrap_or_else(|| {
                panic!(
                    "expected a span named \"HTTP request\", got spans: {:?}",
                    captured.iter().map(|s| &s.name).collect::<Vec<_>>()
                )
            });

        assert_eq!(
            http_span
                .fields
                .get("http.request.method")
                .map(String::as_str),
            Some("POST"),
            "span fields: {:?}",
            http_span.fields
        );
        assert_eq!(
            http_span.fields.get("otel.kind").map(String::as_str),
            Some("client"),
            "span fields: {:?}",
            http_span.fields
        );
        assert_eq!(
            http_span
                .fields
                .get("http.response.status_code")
                .map(String::as_str),
            Some("200"),
            "span fields: {:?}",
            http_span.fields
        );
        let url_full = http_span
            .fields
            .get("url.full")
            .unwrap_or_else(|| panic!("expected url.full field, got: {:?}", http_span.fields));
        assert!(
            url_full.ends_with("/rawPredict"),
            "url.full should target the rawPredict endpoint, got: {url_full}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 0b — a `vertex_inference` span is emitted per chunk, tagged with
    // `backend = "vertex"` so exported traces can distinguish Vertex calls
    // from in-pod ONNX inference, and it nests correctly as a child of
    // whatever span was active when `classify()` was called and as the
    // parent of `reqwest_tracing`'s `HTTP request` span — giving callers true
    // parent-child trace linkage (e.g. Heimdall's `rest_request` span) without
    // the toolkit touching OpenTelemetry context/baggage directly.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn vertex_inference_span_nests_under_caller_span_and_parents_http_span() {
        ensure_global_subscriber_installed();

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

        let captured = Arc::new(Mutex::new(Vec::new()));
        let layer = SpanCaptureLayer {
            captured: Arc::clone(&captured),
        };
        let subscriber = tracing_subscriber::registry().with(layer);
        let dispatch = tracing::Dispatch::new(subscriber);

        let clf = build_vertex(&server.url(), 0.5).await;
        // The outer span simulates a caller (e.g. Heimdall's `rest_request`
        // span) that is already active when `classify()` is invoked. It must
        // be created *inside* the instrumented/dispatched future (async
        // blocks defer their body to first poll) so `info_span!` picks up
        // the test-local dispatch rather than whatever is globally default.
        async {
            let caller_span = tracing::info_span!("rest_request");
            clf.classify("hello world").instrument(caller_span).await
        }
        .with_subscriber(dispatch)
        .await
        .expect("classify must succeed");

        let captured = captured.lock().unwrap();
        let inference_span = captured
            .iter()
            .find(|s| s.name == "vertex_inference")
            .unwrap_or_else(|| {
                panic!(
                    "expected a span named \"vertex_inference\", got spans: {:?}",
                    captured.iter().map(|s| &s.name).collect::<Vec<_>>()
                )
            });

        assert_eq!(
            inference_span.fields.get("backend").map(String::as_str),
            Some("vertex"),
            "span fields: {:?}",
            inference_span.fields
        );
        assert_eq!(
            inference_span.parent_name.as_deref(),
            Some("rest_request"),
            "vertex_inference span should nest under the caller's active span"
        );

        let http_span = captured
            .iter()
            .find(|s| s.name == "HTTP request")
            .unwrap_or_else(|| {
                panic!(
                    "expected a span named \"HTTP request\", got spans: {:?}",
                    captured.iter().map(|s| &s.name).collect::<Vec<_>>()
                )
            });
        assert_eq!(
            http_span.parent_name.as_deref(),
            Some("vertex_inference"),
            "HTTP request span should nest under vertex_inference"
        );
    }

    // -----------------------------------------------------------------------
    // Test 0c — `token_count`/`chunk_index` must export as native OTEL
    // integers, not Debug-formatted strings.
    //
    // `tracing-opentelemetry`'s span visitor overrides `record_i64` but NOT
    // `record_u64` — a bare `usize` field (as `chunk_index`/`token_count`
    // naturally are: `Vec::len()`, `Iterator::enumerate()`) dispatches via
    // `record_u64`, whose `tracing-core` default silently forwards to
    // `record_debug`, producing a STRING attribute (`"67"`) instead of an
    // integer. This is exactly the kind of regression the `SpanCaptureLayer`
    // harness above cannot catch: `FieldMap` implements `record_u64` directly,
    // so it observes the correct value regardless of which `Visit` method
    // tracing dispatched through. Only a real `tracing-opentelemetry` layer
    // exposes the gap, so this test uses the actual crate (dev-dependency
    // only — see Cargo.toml) with an in-process `SpanExporter` and asserts on
    // the exported `opentelemetry::Value` variant directly.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn vertex_inference_span_exports_token_count_as_integer_not_string() {
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry::Value as OtelValue;
        use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
        use tracing_opentelemetry::OpenTelemetryLayer;

        #[derive(Clone, Default, Debug)]
        struct CapturingExporter {
            spans: Arc<Mutex<Vec<SpanData>>>,
        }

        impl SpanExporter for CapturingExporter {
            fn export(
                &self,
                batch: Vec<SpanData>,
            ) -> impl std::future::Future<Output = opentelemetry_sdk::error::OTelSdkResult> + Send
            {
                self.spans.lock().unwrap().extend(batch);
                async { Ok(()) }
            }
        }

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

        let exporter = CapturingExporter::default();
        let spans = Arc::clone(&exporter.spans);
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .build();
        let tracer = provider.tracer("vertex-test");
        let otel_layer = OpenTelemetryLayer::new(tracer);
        let subscriber = tracing_subscriber::registry().with(otel_layer);
        let dispatch = tracing::Dispatch::new(subscriber);

        let clf = build_vertex(&server.url(), 0.5).await;
        clf.classify("hello world")
            .with_subscriber(dispatch)
            .await
            .expect("classify must succeed");
        provider.shutdown().expect("tracer provider shutdown");

        let spans = spans.lock().unwrap();
        let inference_span = spans
            .iter()
            .find(|s| s.name == "vertex_inference")
            .unwrap_or_else(|| {
                panic!(
                    "expected an exported span named \"vertex_inference\", got: {:?}",
                    spans.iter().map(|s| &s.name).collect::<Vec<_>>()
                )
            });

        let token_count = inference_span
            .attributes
            .iter()
            .find(|kv| kv.key.as_str() == "token_count")
            .unwrap_or_else(|| panic!("no token_count attribute on exported span"));
        assert!(
            matches!(token_count.value, OtelValue::I64(_)),
            "token_count must export as an OTEL integer, got {:?} \
             (a String here means the usize-record_u64-fallback regression is back)",
            token_count.value
        );

        let chunk_index = inference_span
            .attributes
            .iter()
            .find(|kv| kv.key.as_str() == "chunk_index")
            .unwrap_or_else(|| panic!("no chunk_index attribute on exported span"));
        assert!(
            matches!(chunk_index.value, OtelValue::I64(_)),
            "chunk_index must export as an OTEL integer, got {:?}",
            chunk_index.value
        );
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

        let clf = build_vertex(&server.url(), 0.5).await;
        let result = clf
            .classify("hello world")
            .await
            .expect("classify must succeed");

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

        let clf = build_vertex(&server.url(), 0.5).await;

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

        let clf = build_vertex(&server.url(), 0.5).await;
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

        let clf = build_vertex(&server.url(), 0.5).await;
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
    // Test 4b — HTTP error body with multi-byte UTF-8 straddling the
    // truncation boundary must not panic on a non-char-boundary byte slice.
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn http_error_body_with_multibyte_utf8_at_truncation_boundary_does_not_panic() {
        // 511 ASCII bytes followed by a 3-byte UTF-8 character ('中') means the
        // character spans bytes [511, 514), so byte offset 512 (MAX_ERROR_BODY)
        // falls inside it — `&body_text[..512]` panics on current code.
        let body = format!("{}{}", "a".repeat(511), "中".repeat(20));
        assert!(
            body.len() > VertexSusFactor::MAX_ERROR_BODY,
            "fixture must exceed MAX_ERROR_BODY to exercise the truncation path"
        );

        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/rawPredict")
            .with_status(500)
            .with_header("content-type", "application/json")
            .with_body(&body)
            .create_async()
            .await;

        let clf = build_vertex(&server.url(), 0.5).await;
        let err = clf.classify("test").await.expect_err("must return error");

        match &err {
            SigError::Provider(msg) => {
                assert!(
                    msg.contains("truncated"),
                    "expected truncated error message, got: {msg}"
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

        let clf = build_vertex(&server.url(), 0.5).await;
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

        let clf = build_vertex(&server.url(), 0.5).await;
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
        let tokenizer = load_test_tokenizer().await;
        let special_tokens = common::resolve_special_token_ids(&tokenizer).unwrap();
        let clf = VertexSusFactor {
            client: VertexSusFactor::build_traced_client(
                Duration::from_secs(5),
                Duration::from_secs(5),
            )
            .unwrap(),
            endpoint_url: "http://127.0.0.1:1/rawPredict".to_string(),
            tokenizer,
            model_name: VertexSusFactor::DEFAULT_MODEL.to_string(),
            threshold: 0.5,
            auth: Arc::new(FakeTokenProvider),
            max_concurrent_chunks: 4,
            special_tokens,
        };

        let err = clf.classify("test").await.expect_err("must return error");

        match &err {
            SigError::Provider(_) => {}
            other => panic!("expected SigError::Provider, got {other:?}"),
        }
    }
}
