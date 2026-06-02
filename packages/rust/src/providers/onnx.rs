//! ONNX embedding provider implementation using ONNX Runtime (`ort`).
//!
//! This module provides embedding generation using ONNX models via the `ort` crate,
//! which wraps Microsoft's ONNX Runtime for high-performance inference.
//!
//! The default model identifier is `"models/v1"`, which is treated as a **local
//! directory path** by [`ModelCache`]. In production, the model files
//! (`onnx/model.onnx`, `tokenizer.json`, etc.) should be pre-loaded into that
//! directory via a Kubernetes init container, Docker `COPY`, or volume mount.
//! To auto-download from HuggingFace, pass the HuggingFace model ID explicitly:
//! `Some("0dinai/0din-jailbreak-embeddings-small".to_string())`. The model
//! produces 1024-dimensional embeddings optimised for jailbreak/prompt-injection
//! detection.
//!
//! ## Features
//!
//! - Local ONNX model inference (no API calls)
//! - Automatic model download from HuggingFace
//! - Round-robin session pool for concurrent inference under load
//! - CPU-bound inference offloaded to `tokio::task::spawn_blocking`
//! - Configurable `pool_size` and `intra_threads` for K8s cgroup CPU limits
//! - Mean pooling with attention masking
//! - L2 normalization of embeddings
//! - Support for E5 models with "query: " prefix
//! - ORT Level 3 optimization (transformer-specific kernel fusions)
//!
//! ## Usage
//!
//! Enable the `onnx` feature in your `Cargo.toml`:
//!
//! ```toml
//! odin-prompt-toolkit = { version = "0.2", features = ["onnx"] }
//! ```
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use odin_prompt_toolkit::providers::{OnnxProvider, ModelCache};
//! use odin_prompt_toolkit::provider::EmbeddingProvider;
//!
//! let cache = ModelCache::new()?;
//! // intra_threads=0 (auto), pool_size=0 (use the default of 2).
//! let provider = OnnxProvider::new(&cache, None, None, 0, 0).await?;
//!
//! let result = provider.generate_embedding("Hello, world!").await?;
//! println!("Embedding dimensions: {}", result.embedding.len());
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use ndarray::{Array2, Axis};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use tracing::{debug, info};

use crate::error::{Result, SigError};
use crate::lsh::{compute_embedding_sha256, normalize_vector};
use crate::provider::EmbeddingProvider;
use crate::providers::ModelCache;
use crate::types::EmbeddingResult;

// ---------------------------------------------------------------------------
// OnnxSessionPool
// ---------------------------------------------------------------------------

/// A round-robin pool of ONNX Runtime sessions.
///
/// Each session in the pool is an independent `ort::Session` that can run
/// inference concurrently with the others. Requests are distributed across
/// sessions using an atomic round-robin counter, so N concurrent requests
/// can proceed in parallel without any single session becoming a bottleneck.
///
/// ## Design rationale
///
/// `ort::Session::run` requires `&mut self`, so each session is wrapped in
/// `Arc<Mutex<Session>>`. A pool of 2 sessions with `intra_threads=2` each
/// allows 2 requests to run fully in parallel (4 total ORT threads, same CPU
/// budget as 1 session with `intra_threads=4`) while halving the serialization
/// penalty under concurrent load.
///
/// ## Thread safety
///
/// `OnnxSessionPool` is `Send + Sync`. The atomic counter is lock-free.
/// Session acquisition is O(1) — no contention between callers selecting
/// different sessions.
#[derive(Debug)]
pub struct OnnxSessionPool {
    sessions: Vec<Arc<Mutex<Session>>>,
    next_index: AtomicUsize,
}

impl OnnxSessionPool {
    /// Default pool size: 2 sessions per process.
    ///
    /// 2 sessions allow 2 concurrent requests to run without serialization.
    /// Combined with `intra_threads=2` per session, the total ORT thread
    /// budget (4 threads) matches a typical small-pod CPU limit.
    pub const DEFAULT_POOL_SIZE: usize = 2;

    /// Build a pool of `pool_size` ORT sessions, all sharing the same
    /// `model_path` and `intra_threads` configuration.
    ///
    /// Returns the pool plus a flag indicating whether the model declares a
    /// `token_type_ids` input (BERT-style models do; XLM-RoBERTa models don't).
    ///
    /// # Errors
    ///
    /// Returns an error string if any session fails to initialize. The pool is
    /// all-or-nothing: a partial initialization is not left in place. The error
    /// is stringly-typed because `ort`'s `SessionBuilder` is `!Send`, so this
    /// function is intended to run inside `tokio::task::spawn_blocking` where a
    /// non-`Send` error must not cross the await boundary.
    pub fn build(
        pool_size: usize,
        model_path: &std::path::Path,
        intra_threads: usize,
    ) -> std::result::Result<(Self, bool), String> {
        if pool_size == 0 {
            return Err("pool_size must be at least 1".into());
        }

        let mut sessions = Vec::with_capacity(pool_size);
        let mut requires_token_type_ids = false;

        for i in 0..pool_size {
            let mut builder = Session::builder()
                .map_err(|e| format!("Failed to create session builder (session {i}): {e}"))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| format!("Failed to set optimization level (session {i}): {e}"))?;

            if intra_threads > 0 {
                builder = builder
                    .with_intra_threads(intra_threads)
                    .map_err(|e| format!("Failed to set intra-op threads (session {i}): {e}"))?;
            }

            let session = builder
                .commit_from_file(model_path)
                .map_err(|e| format!("Failed to load ONNX model (session {i}): {e}"))?;

            // Detect token_type_ids requirement from the first session only;
            // all sessions share the same model so the result is identical.
            if i == 0 {
                requires_token_type_ids = session
                    .inputs()
                    .iter()
                    .any(|inp| inp.name() == "token_type_ids");
            }

            sessions.push(Arc::new(Mutex::new(session)));
        }

        Ok((
            Self {
                sessions,
                next_index: AtomicUsize::new(0),
            },
            requires_token_type_ids,
        ))
    }

    /// Return the number of sessions in this pool.
    pub fn size(&self) -> usize {
        self.sessions.len()
    }

    /// Return the next session in round-robin order.
    ///
    /// This operation is lock-free. Concurrent callers will each receive a
    /// different session handle (modulo pool size), distributing inference
    /// load evenly across the pool.
    pub fn get_session(&self) -> Arc<Mutex<Session>> {
        let idx = self.next_index.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        Arc::clone(&self.sessions[idx])
    }
}

// ---------------------------------------------------------------------------
// mean_pool (free function)
// ---------------------------------------------------------------------------

/// Apply mean pooling to token embeddings, weighted by the attention mask.
///
/// This is a standard technique for transformer-based embeddings: token-level
/// embeddings are averaged based on which tokens were actual input
/// (attention_mask=1) vs padding (=0).
///
/// # Arguments
///
/// * `embeddings` - Token embeddings matrix (seq_len × hidden_dim)
/// * `attention_mask` - Binary mask indicating real tokens (1) vs padding (0)
///
/// # Returns
///
/// Pooled embedding vector (hidden_dim)
///
/// # Errors
///
/// Returns an error if the attention mask length doesn't match the sequence length.
fn mean_pool(embeddings: &Array2<f32>, attention_mask: &[i64]) -> Result<Vec<f32>> {
    let (seq_len, hidden_dim) = embeddings.dim();

    if seq_len != attention_mask.len() {
        return Err(SigError::Provider(format!(
            "Attention mask length ({}) does not match sequence length ({})",
            attention_mask.len(),
            seq_len
        )));
    }

    let mut pooled = vec![0.0f32; hidden_dim];
    let mut mask_sum = 0.0f32;

    // Sum embeddings weighted by attention mask
    for (i, &mask_val) in attention_mask.iter().enumerate() {
        if mask_val == 1 {
            let mask_f32 = mask_val as f32;
            mask_sum += mask_f32;

            for j in 0..hidden_dim {
                pooled[j] += embeddings[[i, j]] * mask_f32;
            }
        }
    }

    // Average
    if mask_sum > 0.0 {
        for val in &mut pooled {
            *val /= mask_sum;
        }
    }

    Ok(pooled)
}

// ---------------------------------------------------------------------------
// OnnxProvider
// ---------------------------------------------------------------------------

/// ONNX embedding provider using ONNX Runtime.
///
/// By default this provider looks for model files at the local path `models/v1`
/// (see [`OnnxProvider::DEFAULT_MODEL`]). Pass a HuggingFace model ID such as
/// `"0dinai/0din-jailbreak-embeddings-small"` to trigger auto-download instead.
/// The model produces 1024-dimensional embeddings optimised for
/// jailbreak/prompt-injection detection.
///
/// The model is loaded once at construction into a pool of ORT sessions.
///
/// ## Thread safety
///
/// `ort::Session::run` requires `&mut self`, so each session is wrapped in
/// `Arc<Mutex<Session>>`. Inference requests are distributed across the session
/// pool via round-robin, allowing `pool_size` requests to run concurrently
/// without serialization. CPU-bound inference is offloaded to
/// `tokio::task::spawn_blocking` so it never blocks the async runtime's worker
/// threads. ORT manages its own internal thread pool via `intra_threads` for
/// intra-op parallelism within a single call.
#[derive(Debug)]
pub struct OnnxProvider {
    /// Round-robin pool of ORT sessions for concurrent inference.
    pool: OnnxSessionPool,

    /// Tokenizer for text preprocessing (shared read-only across all callers).
    tokenizer: Arc<tokenizers::Tokenizer>,

    /// Model name (HuggingFace ID or local path)
    model_name: String,

    /// Embedding dimensions
    dimensions: usize,

    /// Provider name
    name: String,

    /// Input prefix (e.g., "query: " for E5 models, empty for others)
    input_prefix: String,

    /// Whether the model requires a `token_type_ids` input (BERT-style models).
    /// XLM-RoBERTa models don't use `token_type_ids`.
    requires_token_type_ids: bool,
}

impl OnnxProvider {
    /// Default model identifier — treated as a **local directory path** by
    /// [`ModelCache`].
    ///
    /// Production deployments should pre-populate `models/v1/onnx/model.onnx`
    /// and `models/v1/tokenizer.json` via init container, Docker `COPY`, or
    /// volume mount. To download the model from HuggingFace instead, pass
    /// `Some("0dinai/0din-jailbreak-embeddings-small".to_string())` as the
    /// `model` argument to [`OnnxProvider::new`].
    pub const DEFAULT_MODEL: &'static str = "models/v1";

    /// Default embedding dimensions.
    pub const DEFAULT_DIMENSIONS: usize = 1024;

    /// Maximum sequence length supported by the model.
    pub const MAX_SEQUENCE_LENGTH: usize = 512;

    /// Input prefix required by E5 models.
    pub const INPUT_PREFIX: &'static str = "query: ";

    /// ONNX model filename. With `ort` + `GraphOptimizationLevel::Level3`, ONNX
    /// Runtime applies its own transformer-aware fusions on the base graph, so
    /// we load the unoptimized export rather than a runtime-specific artifact.
    const ONNX_MODEL_FILE: &'static str = "onnx/model.onnx";

    /// Create a new ONNX provider.
    ///
    /// # Arguments
    ///
    /// * `cache`         - Model cache for downloading/locating local models
    /// * `model`         - Model name or HuggingFace ID (default: `models/v1`)
    /// * `name`          - Provider name identifier (default: `"onnx"`)
    /// * `intra_threads` - Number of threads ORT uses for a single inference call.
    ///   `0` = let ORT decide (uses available cores). For a pool of 2 sessions,
    ///   use 2 so the total ORT thread budget matches the CPU limit.
    /// * `pool_size`     - Number of independent ORT sessions in the pool.
    ///   `0` = use the default ([`OnnxSessionPool::DEFAULT_POOL_SIZE`]). Each
    ///   session can serve one concurrent request without blocking the others.
    ///
    /// Keep `pool_size × intra_threads ≤ pod CPU limit` to avoid CFS throttling.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model download fails
    /// - Any ONNX session fails to load
    /// - Tokenizer loading fails
    pub async fn new(
        cache: &ModelCache,
        model: Option<String>,
        name: Option<String>,
        intra_threads: usize,
        pool_size: usize,
    ) -> Result<Self> {
        let pool_size = if pool_size == 0 {
            OnnxSessionPool::DEFAULT_POOL_SIZE
        } else {
            pool_size
        };

        let model_name = model.unwrap_or_else(|| Self::DEFAULT_MODEL.to_string());
        let provider_name = name.unwrap_or_else(|| "onnx".to_string());

        info!(
            "Initializing ONNX provider with model: {}, pool_size: {}, intra_threads: {}",
            model_name,
            pool_size,
            if intra_threads == 0 {
                "auto".to_string()
            } else {
                intra_threads.to_string()
            }
        );

        // Download/locate model and tokenizer.
        let model_path = cache.get_model(&model_name, Self::ONNX_MODEL_FILE).await?;
        let tokenizer_path = cache.get_tokenizer(&model_name).await?;

        info!("Loading ONNX model from: {}", model_path.display());

        // Build the session pool on a blocking thread.
        //
        // GraphOptimizationLevel::Level3 enables all optimizations including
        // transformer-specific fusions (attention, skip-layer-norm, bias-gelu).
        //
        // SessionBuilder is !Send, so we must build sessions inside
        // spawn_blocking rather than holding builders across await points.
        let (pool, requires_token_type_ids) =
            tokio::task::spawn_blocking(move || -> std::result::Result<_, String> {
                OnnxSessionPool::build(pool_size, &model_path, intra_threads)
            })
            .await
            .map_err(|e| SigError::Provider(format!("Session pool build task panicked: {e}")))?
            .map_err(SigError::Provider)?;

        info!(
            "ONNX session pool created successfully ({} sessions, token_type_ids: {})",
            pool.size(),
            requires_token_type_ids
        );

        // Load tokenizer
        info!("Loading tokenizer from: {}", tokenizer_path.display());
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| SigError::Provider(format!("Failed to load tokenizer: {}", e)))?;

        info!("Tokenizer loaded successfully");

        // Detect input prefix from model configuration
        let input_prefix = Self::detect_input_prefix(&model_name, cache)
            .await
            .unwrap_or_else(|e| {
                debug!(
                    "Could not detect input prefix from config, using default: {}",
                    e
                );
                Self::INPUT_PREFIX.to_string()
            });

        if input_prefix.is_empty() {
            info!("Model does not require input prefix");
        } else {
            info!("Using input prefix: {:?}", input_prefix);
        }

        Ok(Self {
            pool,
            tokenizer: Arc::new(tokenizer),
            model_name,
            dimensions: Self::DEFAULT_DIMENSIONS,
            name: provider_name,
            input_prefix,
            requires_token_type_ids,
        })
    }

    /// Detect input prefix from model's sentence_transformers config.
    ///
    /// Returns empty string if no prefix is needed (custom models).
    /// Returns "query: " or similar if the model config specifies a prompt.
    /// Falls back to default E5 prefix on error.
    async fn detect_input_prefix(model_id: &str, cache: &ModelCache) -> Result<String> {
        // Try to load the sentence_transformers config
        let config_path = cache
            .get_model(model_id, "config_sentence_transformers.json")
            .await?;

        let config_str = tokio::fs::read_to_string(&config_path).await.map_err(|e| {
            SigError::Provider(format!(
                "Failed to read sentence_transformers config: {}",
                e
            ))
        })?;

        let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            SigError::Provider(format!(
                "Failed to parse sentence_transformers config: {}",
                e
            ))
        })?;

        // Check if prompts.query is non-empty
        if let Some(query_prompt) = config
            .get("prompts")
            .and_then(|p| p.get("query"))
            .and_then(|q| q.as_str())
        {
            if !query_prompt.is_empty() {
                // E5 models use "query: " with trailing space
                return Ok(format!("{} ", query_prompt));
            }
        }

        // No prefix needed (custom model case)
        Ok(String::new())
    }

    /// Tokenize text and prepare model inputs.
    ///
    /// 1. Prepends the input prefix if configured (e.g., "query: " for E5 models)
    /// 2. Tokenizes the text
    /// 3. Truncates to max sequence length
    /// 4. Returns input IDs and attention mask
    fn tokenize(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        // Prepend prefix if configured
        let prefixed_text = if self.input_prefix.is_empty() {
            text.to_string()
        } else {
            format!("{}{}", self.input_prefix, text)
        };

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(prefixed_text, true)
            .map_err(|e| SigError::Provider(format!("Tokenization failed: {}", e)))?;

        // Get token IDs and attention mask
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        // Truncate to max sequence length
        if input_ids.len() > Self::MAX_SEQUENCE_LENGTH {
            debug!(
                "Truncating sequence from {} to {} tokens",
                input_ids.len(),
                Self::MAX_SEQUENCE_LENGTH
            );
            input_ids.truncate(Self::MAX_SEQUENCE_LENGTH);
            attention_mask.truncate(Self::MAX_SEQUENCE_LENGTH);
        }

        Ok((input_ids, attention_mask))
    }

    /// Run a single inference forward pass.
    ///
    /// Called from `tokio::task::spawn_blocking` since ORT inference is CPU-bound.
    /// Acquires the session mutex for the duration of the forward pass. The
    /// returned [`EmbeddingResult`] has an empty `model` field; the caller fills it in.
    fn run_inference(
        session: &Mutex<Session>,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        requires_token_type_ids: bool,
        expected_dimensions: usize,
    ) -> Result<EmbeddingResult> {
        let seq_len = input_ids.len();
        let token_count = seq_len;

        // Build input tensors (shape: [1, seq_len])
        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| SigError::Provider(format!("Failed to create input_ids array: {e}")))?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask.clone())
            .map_err(|e| {
                SigError::Provider(format!("Failed to create attention_mask array: {e}"))
            })?;

        let input_ids_tensor = Tensor::<i64>::from_array(input_ids_array)
            .map_err(|e| SigError::Provider(format!("Failed to create input_ids tensor: {e}")))?;
        let attention_mask_tensor =
            Tensor::<i64>::from_array(attention_mask_array).map_err(|e| {
                SigError::Provider(format!("Failed to create attention_mask tensor: {e}"))
            })?;

        // Acquire the session lock only for the forward pass + tensor extraction.
        // We call `.to_owned()` on the extracted array to detach it from the
        // `SessionOutputs` borrow, allowing the lock to be released before the
        // purely CPU post-processing steps (pooling, normalization, sha256).
        let output_2d = {
            let mut session_guard = session
                .lock()
                .map_err(|_| SigError::Provider("Session mutex was poisoned".into()))?;

            // Run inference using named inputs. Only pass token_type_ids when the
            // model declares that input (BERT-style); XLM-RoBERTa models don't.
            let outputs = if requires_token_type_ids {
                let token_type_ids: Vec<i64> = vec![0i64; seq_len];
                let token_type_ids_array = Array2::from_shape_vec((1, seq_len), token_type_ids)
                    .map_err(|e| {
                        SigError::Provider(format!("Failed to create token_type_ids array: {e}"))
                    })?;
                let token_type_ids_tensor = Tensor::<i64>::from_array(token_type_ids_array)
                    .map_err(|e| {
                        SigError::Provider(format!("Failed to create token_type_ids tensor: {e}"))
                    })?;
                session_guard
                    .run(ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attention_mask_tensor,
                        "token_type_ids" => token_type_ids_tensor,
                    ])
                    .map_err(|e| SigError::Provider(format!("Model inference failed: {e}")))?
            } else {
                session_guard
                    .run(ort::inputs![
                        "input_ids" => input_ids_tensor,
                        "attention_mask" => attention_mask_tensor,
                    ])
                    .map_err(|e| SigError::Provider(format!("Model inference failed: {e}")))?
            };

            // Extract the first hidden-state output tensor.
            //
            // Standard ONNX exports from HuggingFace name this output
            // "last_hidden_state". Custom or re-exported models sometimes
            // omit the name or use "output_0", "output", etc.  Try the
            // canonical name first; fall back to the first output so the
            // provider works with a broader range of ONNX exports.
            let available_names: Vec<&str> = outputs.iter().map(|(k, _)| k).collect();
            let output_key: &str = if outputs.get("last_hidden_state").is_some() {
                "last_hidden_state"
            } else {
                let fallback = available_names.first().copied().ok_or_else(|| {
                    SigError::Provider(format!(
                        "Model produced no outputs. \
                         Expected 'last_hidden_state'. \
                         Available outputs: {:?}",
                        available_names
                    ))
                })?;
                debug!(
                    "Output 'last_hidden_state' not found; falling back to '{}'. \
                     Available outputs: {:?}",
                    fallback, available_names
                );
                fallback
            };
            let output_view = outputs
                .get(output_key)
                .expect("key was just confirmed present")
                .try_extract_array::<f32>()
                .map_err(|e| {
                    SigError::Provider(format!(
                        "Failed to extract output '{}' as f32 tensor \
                         (available outputs: {:?}): {e}",
                        output_key, available_names
                    ))
                })?;

            let shape = output_view.shape();
            if shape.len() != 3 || shape[0] != 1 {
                return Err(SigError::Provider(format!(
                    "Unexpected output shape: {:?}",
                    shape
                )));
            }

            let out_seq_len = shape[1];
            let hidden_dim = shape[2];

            // `.to_owned()` copies the data out of the ORT-owned buffer, letting
            // the `outputs` / `session_guard` borrow end here.
            let output_2d = output_view
                .index_axis(Axis(0), 0)
                .to_owned()
                .into_shape_with_order((out_seq_len, hidden_dim))
                .map_err(|e| SigError::Provider(format!("Failed to reshape output: {e}")))?;

            output_2d
            // session_guard (and outputs) are dropped here — lock released
            // before mean_pool, normalize_vector, and sha256.
        };

        // Mean pooling (weighted by attention mask)
        let embedding = mean_pool(&output_2d, &attention_mask)?;

        // Validate dimensions
        if embedding.len() != expected_dimensions {
            return Err(SigError::Provider(format!(
                "Model output dimensions ({}) don't match expected dimensions ({}).",
                embedding.len(),
                expected_dimensions
            )));
        }

        // L2 normalize
        let normalized = normalize_vector(&embedding);
        let sha256 = compute_embedding_sha256(&normalized);

        Ok(EmbeddingResult {
            embedding,
            normalized_embedding: normalized,
            normalized_embedding_sha256: sha256,
            model: String::new(), // Caller fills this in
            dimensions: expected_dimensions,
            token_count: Some(token_count),
            timing_ms: None,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OnnxProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model_name
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult> {
        debug!("Generating ONNX embedding (text length: {})", text.len());

        // Tokenization is CPU-bound but fast and Send, so we do it here on the
        // async executor rather than adding a spawn_blocking hop.
        let (input_ids, attention_mask) = self.tokenize(text)?;

        let requires_token_type_ids = self.requires_token_type_ids;
        let dimensions = self.dimensions;
        let model_name = self.model_name.clone();
        // Round-robin: pick the next available session from the pool.
        let session = self.pool.get_session();

        // ORT inference is CPU-bound; run it on the blocking thread pool so it
        // never starves the async runtime's worker threads.
        // Arc<Mutex<Session>> is Send + Sync so this is safe.
        let mut result = tokio::task::spawn_blocking(move || {
            Self::run_inference(
                &session,
                input_ids,
                attention_mask,
                requires_token_type_ids,
                dimensions,
            )
        })
        .await
        .map_err(|e| SigError::Provider(format!("Inference task panicked: {e}")))??;
        result.model = model_name;

        debug!("Generated {}-dimensional embedding", result.dimensions);
        Ok(result)
    }

    async fn close(&self) -> Result<()> {
        debug!("Closing ONNX provider");
        // Sessions are dropped automatically when their Arc refcount reaches zero.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(OnnxProvider::DEFAULT_MODEL, "models/v1");
        assert_eq!(OnnxProvider::DEFAULT_DIMENSIONS, 1024);
        assert_eq!(OnnxProvider::MAX_SEQUENCE_LENGTH, 512);
        assert_eq!(OnnxProvider::INPUT_PREFIX, "query: ");
    }

    // -----------------------------------------------------------------------
    // mean_pool (free function) tests — no model required
    // -----------------------------------------------------------------------

    #[test]
    fn test_mean_pool_basic() {
        // 3x4 embedding matrix:
        // [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        let embeddings = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let attention_mask = vec![1, 1, 1];

        let result = mean_pool(&embeddings, &attention_mask).unwrap();

        // Column means: [5, 6, 7, 8]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 6.0).abs() < 1e-6);
        assert!((result[2] - 7.0).abs() < 1e-6);
        assert!((result[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_with_padding() {
        let embeddings = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        // Last token is padding (mask=0): mean of first two rows.
        let attention_mask = vec![1, 1, 0];

        let result = mean_pool(&embeddings, &attention_mask).unwrap();

        // [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3, 4, 5, 6]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 5.0).abs() < 1e-6);
        assert!((result[3] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_single_token() {
        let embeddings = Array2::from_shape_vec((1, 4), vec![2.0, 4.0, 6.0, 8.0]).unwrap();
        let attention_mask = vec![1];

        let result = mean_pool(&embeddings, &attention_mask).unwrap();

        assert_eq!(result.len(), 4);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
        assert!((result[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_all_padding() {
        let embeddings =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let attention_mask = vec![0, 0];

        let result = mean_pool(&embeddings, &attention_mask).unwrap();

        // No real tokens → all zeros (mask_sum guard).
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_mismatched_lengths() {
        let embeddings = Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap();
        let attention_mask = vec![1, 1]; // wrong length (should be 3)

        let result = mean_pool(&embeddings, &attention_mask);

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("does not match sequence length"));
    }

    // -----------------------------------------------------------------------
    // Round-robin counter arithmetic tests (no model required)
    // These verify the modulo math used by OnnxSessionPool::get_session,
    // not the pool or session lifecycle end-to-end.
    // -----------------------------------------------------------------------

    #[test]
    fn test_session_pool_default_size() {
        assert_eq!(OnnxSessionPool::DEFAULT_POOL_SIZE, 2);
    }

    #[test]
    fn test_session_pool_build_rejects_zero_pool_size() {
        // pool_size=0 must return Err (not panic) so spawn_blocking propagates
        // a clean SigError::Provider rather than a JoinError.
        let result = OnnxSessionPool::build(0, std::path::Path::new("nonexistent"), 0);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("pool_size must be at least 1"),
            "unexpected: {msg}"
        );
    }

    #[test]
    fn test_session_pool_round_robin_wraps() {
        // Round-robin counter must wrap cleanly at pool_size.
        // 5 calls to a pool of 2: indices should be 0,1,0,1,0.
        let counter = std::sync::atomic::AtomicUsize::new(0);
        let pool_size = 2;
        let indices: Vec<usize> = (0..5)
            .map(|_| counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % pool_size)
            .collect();
        assert_eq!(indices, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_session_pool_size_one_always_returns_same() {
        let counter = std::sync::atomic::AtomicUsize::new(0);
        let pool_size = 1;
        let indices: Vec<usize> = (0..4)
            .map(|_| counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % pool_size)
            .collect();
        assert_eq!(indices, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_session_pool_round_robin_large_pool() {
        let counter = std::sync::atomic::AtomicUsize::new(0);
        let pool_size = 4;
        let indices: Vec<usize> = (0..8)
            .map(|_| counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % pool_size)
            .collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    // Note: real-inference tests require a downloaded model and live in the
    // gated integration tests (see tests/onnx_inference.rs, #[ignore]).
}
