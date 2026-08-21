//! In-pod SusFactor classifier using a local ONNX model via ONNX Runtime
//! (`ort`).
//!
//! The ONNX graph (exported via `scripts/export_susfactor_onnx.py`) bakes the
//! e5-large encoder, mean-pooling, and MLP head into a single model:
//!
//! - inputs:  `input_ids[1, seq]` int64, `attention_mask[1, seq]` int64
//! - output:  `logits[1, 2]` float32 (`softmax[1]` = P(suspicious))
//!
//! The model is downloaded from HuggingFace on first use and cached locally
//! (see [`ModelCache`]).
//!
//! Tokenization, chunking, softmax, and labeling are delegated to
//! [`crate::susfactor::common`] so this backend and the Vertex backend cannot
//! diverge.
//!
//! ## Notes on performance
//!
//! `ort::Session::run` requires `&mut self`, so the session is wrapped in
//! `Arc<Mutex<Session>>`. Inference is offloaded to
//! `tokio::task::spawn_blocking` so the async executor is never stalled.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ndarray::Array2;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::error::{Result, SigError};
use crate::providers::ModelCache;
use crate::susfactor::common;
use crate::susfactor::provider::SusFactorProvider;
use crate::susfactor::types::{ChunkedSusFactorResult, PhaseSpan, SusFactorResult};

/// Classifies prompts as safe vs. suspicious using SusFactor over a local ONNX
/// model.
#[derive(Debug, Clone)]
pub struct OnnxSusFactor {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    model_name: String,
    threshold: f32,
}

/// Backwards-compatible alias for [`OnnxSusFactor`].
///
/// Retained for one minor version so downstream imports of
/// `SusFactorClassifier` keep compiling; prefer [`OnnxSusFactor`].
#[deprecated(since = "0.8.0", note = "renamed to OnnxSusFactor")]
pub type SusFactorClassifier = OnnxSusFactor;

impl OnnxSusFactor {
    /// Canonical model identifier reported in results (shared across SDKs).
    pub const DEFAULT_MODEL: &'static str = common::DEFAULT_MODEL;

    /// HuggingFace repo holding the ONNX export downloaded at runtime.
    pub const DEFAULT_ONNX_REPO: &'static str = common::DEFAULT_ONNX_REPO;

    /// Default decision threshold.
    pub const DEFAULT_THRESHOLD: f32 = common::DEFAULT_THRESHOLD;

    /// Maximum sequence length supported by the model.
    pub const MAX_SEQUENCE_LENGTH: usize = 512;

    const ONNX_MODEL_FILE: &'static str = "onnx/model.onnx";

    /// Create a new SusFactor classifier, downloading the ONNX model if needed.
    ///
    /// # Arguments
    ///
    /// * `cache` - Model cache for downloading/caching the model.
    /// * `model` - Canonical model identifier reported in results (default:
    ///   [`Self::DEFAULT_MODEL`]).
    /// * `source` - HuggingFace repo ID or local directory to load the ONNX
    ///   weights + tokenizer from (default: [`Self::DEFAULT_ONNX_REPO`]).
    /// * `threshold` - Decision threshold (default: [`Self::DEFAULT_THRESHOLD`]).
    pub async fn new(
        cache: &ModelCache,
        model: Option<String>,
        source: Option<String>,
        threshold: Option<f32>,
    ) -> Result<Self> {
        let model_name = model.unwrap_or_else(|| Self::DEFAULT_MODEL.to_string());
        let source = source.unwrap_or_else(|| Self::DEFAULT_ONNX_REPO.to_string());
        let threshold = threshold.unwrap_or(Self::DEFAULT_THRESHOLD);

        let model_path = cache.get_model(&source, Self::ONNX_MODEL_FILE).await?;
        // The SusFactor ONNX export uses external weights stored alongside the
        // graph file. Download model.onnx_data before loading the session so
        // ORT can resolve the external data references.
        // Download the external weights file. Ignore 404-style errors (some
        // exports embed weights directly in the .onnx), but propagate real
        // failures (auth errors, network timeouts, I/O errors) so users see
        // the actual cause rather than a cryptic ORT load error later.
        let source_is_dir = std::path::Path::new(&source).is_dir();
        match cache.get_model(&source, "onnx/model.onnx_data").await {
            Ok(_) => {}
            Err(SigError::Provider(ref msg))
                if msg.contains("HTTP 404")
                    || (source_is_dir
                        && msg.contains("Local model directory")
                        && msg.contains("onnx/model.onnx_data")
                        && msg.contains("not found")) => {}
            Err(e) => return Err(e),
        }
        let tokenizer_path = cache.get_tokenizer(&source).await?;

        // Build the ORT session and load the tokenizer inside spawn_blocking
        // so we don't stall the async executor during the (potentially slow)
        // model-loading + optimization phase.
        let (session, tokenizer) =
            tokio::task::spawn_blocking(move || -> std::result::Result<_, String> {
                let session = Session::builder()
                    .map_err(|e| format!("Failed to create ORT session builder: {e}"))?
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .map_err(|e| format!("Failed to set optimization level: {e}"))?
                    .commit_from_file(&model_path)
                    .map_err(|e| format!("Failed to load SusFactor ONNX model: {e}"))?;

                let tokenizer =
                    common::load_tokenizer(&tokenizer_path).map_err(|e| e.to_string())?;

                Ok((session, tokenizer))
            })
            .await
            .map_err(|e| SigError::Model(format!("spawn_blocking panicked: {e}")))?
            .map_err(SigError::Model)?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            model_name,
            threshold,
        })
    }

    /// Model identifier.
    pub fn model(&self) -> &str {
        &self.model_name
    }

    /// Decision threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    fn run_inference_sync(
        session: &Mutex<Session>,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
    ) -> Result<Vec<f32>> {
        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| SigError::Model(format!("Failed to create input_ids array: {e}")))?;
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| SigError::Model(format!("Failed to create attention_mask array: {e}")))?;

        let input_ids_tensor = Tensor::<i64>::from_array(input_ids_array)
            .map_err(|e| SigError::Model(format!("Failed to create input_ids tensor: {e}")))?;
        let attention_mask_tensor = Tensor::<i64>::from_array(attention_mask_array)
            .map_err(|e| SigError::Model(format!("Failed to create attention_mask tensor: {e}")))?;

        let mut session_guard = session
            .lock()
            .map_err(|_| SigError::Model("Session mutex was poisoned".into()))?;

        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .map_err(|e| SigError::Model(format!("SusFactor inference failed: {e}")))?;

        // The ONNX export names the output "logits" (set in export script).
        // Fall back to first output key if absent.
        let available: Vec<&str> = outputs.iter().map(|(k, _)| k).collect();
        let key = if outputs.get("logits").is_some() {
            "logits"
        } else {
            available
                .first()
                .copied()
                .ok_or_else(|| SigError::Model("SusFactor model produced no outputs".into()))?
        };

        let logits = outputs[key]
            .try_extract_array::<f32>()
            .map_err(|e| SigError::Model(format!("Failed to extract logits: {e}")))?
            .to_owned();

        let flat: Vec<f32> = logits.iter().copied().collect();
        common::validate_logits(&flat)?;
        Ok(flat)
    }

    /// Classify a prompt of any length.
    ///
    /// Prompts that fit within `MAX_CONTENT_TOKENS` (510 tokens) are scored in a
    /// single inference call. Longer prompts are automatically split into
    /// overlapping chunks and each chunk is scored independently — callers do
    /// not need to check length or call a separate method.
    ///
    /// # Returns
    ///
    /// A [`ChunkedSusFactorResult`] containing one [`SusFactorResult`] per chunk.
    /// Short prompts produce exactly one chunk. No scores are aggregated.
    /// `is_suspicious` is `true` if **any** chunk is suspicious.
    ///
    /// # Scheduling
    ///
    /// Chunks are dispatched as concurrent `tokio::task::spawn_blocking` tasks.
    /// Actual concurrency depends on the ONNX Runtime session configuration —
    /// a single shared session serializes inference internally. Dispatching
    /// concurrently still allows the runtime to pipeline work efficiently;
    /// callers should not assume strict sequential execution.
    pub async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult> {
        let wall_start = Instant::now();

        // Time tokenization of the full text.
        let tokenize_start = Instant::now();
        let (all_ids, all_mask) = common::tokenize_full(&self.tokenizer, text)?;
        let tokenize_span = PhaseSpan {
            name: common::PHASE_TOKENIZE.to_string(),
            start_ms: common::offset_ms(tokenize_start, wall_start),
            duration_ms: common::elapsed_ms(tokenize_start),
            chunk_index: None,
        };

        // Time chunking of the token stream.
        let chunk_start_instant = Instant::now();
        let chunks = common::chunk_token_ids_with_mask(&all_ids, &all_mask);
        let chunk_span = PhaseSpan {
            name: common::PHASE_CHUNK.to_string(),
            start_ms: common::offset_ms(chunk_start_instant, wall_start),
            duration_ms: common::elapsed_ms(chunk_start_instant),
            chunk_index: None,
        };

        let mut handles = Vec::with_capacity(chunks.len());
        for (i, (chunk_ids, chunk_mask)) in chunks.into_iter().enumerate() {
            let session = Arc::clone(&self.session);
            let model_name = self.model_name.clone();
            let threshold = self.threshold;

            handles.push(tokio::task::spawn_blocking(
                move || -> Result<(SusFactorResult, PhaseSpan)> {
                    // Capture the inference start INSIDE the task so overlapping
                    // execution is visible on the timeline.
                    let chunk_start = Instant::now();
                    let logits = Self::run_inference_sync(&session, chunk_ids, chunk_mask)?;
                    let result = common::result_from_logits(
                        &logits,
                        &model_name,
                        threshold,
                        common::elapsed_ms(chunk_start),
                    );
                    let span = PhaseSpan {
                        name: common::PHASE_INFERENCE.to_string(),
                        start_ms: common::offset_ms(chunk_start, wall_start),
                        duration_ms: result.timing_ms,
                        chunk_index: Some(i),
                    };
                    Ok((result, span))
                },
            ));
        }

        let mut chunk_results = Vec::with_capacity(handles.len());
        let mut inference_spans = Vec::with_capacity(handles.len());
        for handle in handles {
            let (result, span) = handle
                .await
                .map_err(|e| SigError::Model(format!("spawn_blocking panicked: {e}")))??;
            chunk_results.push(result);
            inference_spans.push(span);
        }

        // Assemble the final result; time the reduction itself.
        let reduce_start = Instant::now();
        let mut spans = Vec::with_capacity(inference_spans.len() + 3);
        spans.push(tokenize_span);
        spans.push(chunk_span);
        spans.append(&mut inference_spans);
        spans.push(PhaseSpan {
            name: common::PHASE_REDUCE.to_string(),
            start_ms: common::offset_ms(reduce_start, wall_start),
            duration_ms: common::elapsed_ms(reduce_start),
            chunk_index: None,
        });

        Ok(common::reduce(
            chunk_results,
            common::elapsed_ms(wall_start),
            spans,
        ))
    }
}

#[async_trait]
impl SusFactorProvider for OnnxSusFactor {
    fn model(&self) -> &str {
        &self.model_name
    }

    fn threshold(&self) -> f32 {
        self.threshold
    }

    async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult> {
        OnnxSusFactor::classify(self, text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::susfactor::types::{LABEL_SAFE, LABEL_SUSPICIOUS};

    #[test]
    fn canonical_model_default_has_no_onnx_suffix() {
        assert_eq!(OnnxSusFactor::DEFAULT_MODEL, "0dinai/susfactor-e5-large");
        assert_eq!(
            OnnxSusFactor::DEFAULT_ONNX_REPO,
            "0dinai/susfactor-e5-large-onnx"
        );
    }

    #[test]
    fn max_content_tokens_fits_sequence_budget() {
        use crate::susfactor::types::MAX_CONTENT_TOKENS;
        assert!(MAX_CONTENT_TOKENS <= OnnxSusFactor::MAX_SEQUENCE_LENGTH - 2);
    }

    #[test]
    fn susfactor_result_is_suspicious_reflects_label() {
        let suspicious = SusFactorResult {
            score: 0.8,
            label: LABEL_SUSPICIOUS.to_string(),
            model: "m".to_string(),
            threshold: 0.5,
            timing_ms: 1.0,
        };
        let safe = SusFactorResult {
            score: 0.2,
            label: LABEL_SAFE.to_string(),
            model: "m".to_string(),
            threshold: 0.5,
            timing_ms: 1.0,
        };
        assert!(suspicious.is_suspicious());
        assert!(!safe.is_suspicious());
    }
}
