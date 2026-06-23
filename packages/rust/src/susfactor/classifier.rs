//! SusFactor classifier using a local ONNX model via ONNX Runtime (`ort`).
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
//! ## Notes on performance
//!
//! `ort::Session::run` requires `&mut self`, so the session is wrapped in
//! `Arc<Mutex<Session>>`. Inference is offloaded to
//! `tokio::task::spawn_blocking` so the async executor is never stalled.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use ndarray::Array2;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::error::{Result, SigError};
use crate::providers::ModelCache;
use crate::susfactor::types::{
    ChunkedSusFactorResult, SusFactorResult, CHUNK_STRIDE, LABEL_SAFE, LABEL_SUSPICIOUS,
    MAX_CONTENT_TOKENS,
};

/// Softmax over a 2-logit slice, returning P(class 1) = suspicious.
pub fn suspicious_prob(logits: &[f32]) -> f32 {
    debug_assert!(logits.len() >= 2);
    let m = logits[0].max(logits[1]);
    let e0 = (logits[0] - m).exp();
    let e1 = (logits[1] - m).exp();
    e1 / (e0 + e1)
}

/// Map a suspicious probability to a label using `threshold`.
pub fn label_for_score(score: f32, threshold: f32) -> &'static str {
    if score >= threshold {
        LABEL_SUSPICIOUS
    } else {
        LABEL_SAFE
    }
}

/// Classifies prompts as safe vs. suspicious using SusFactor.
#[derive(Debug, Clone)]
pub struct SusFactorClassifier {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    model_name: String,
    threshold: f32,
}

impl SusFactorClassifier {
    /// Canonical model identifier reported in results (shared across SDKs).
    pub const DEFAULT_MODEL: &'static str = "0dinai/susfactor-e5-large";

    /// HuggingFace repo holding the ONNX export downloaded at runtime.
    pub const DEFAULT_ONNX_REPO: &'static str = "0dinai/susfactor-e5-large-onnx";

    /// Default decision threshold.
    pub const DEFAULT_THRESHOLD: f32 = 0.5;

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

                let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                    .map_err(|e| format!("Failed to load SusFactor tokenizer: {e}"))?;

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

    fn tokenize_full(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| SigError::Model(format!("Tokenization failed: {e}")))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        Ok((input_ids, attention_mask))
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
        if flat.len() < 2 {
            return Err(SigError::Model(format!(
                "Unexpected SusFactor output shape; got {} elements, expected >= 2",
                flat.len()
            )));
        }
        Ok(flat)
    }

    /// Split a token-ID sequence into overlapping chunks of at most
    /// [`MAX_CONTENT_TOKENS`] tokens each.
    ///
    /// - Sequences at or below `MAX_CONTENT_TOKENS` produce exactly one chunk.
    /// - Adjacent chunks share [`CHUNK_OVERLAP`] tokens of context so that
    ///   sentence boundaries near a chunk edge are still scored in full context.
    /// - An empty input produces one empty chunk.
    pub fn chunk_token_ids(ids: &[i64]) -> Vec<Vec<i64>> {
        if ids.len() <= MAX_CONTENT_TOKENS {
            return vec![ids.to_vec()];
        }
        let mut chunks = Vec::new();
        let mut start = 0;
        loop {
            let end = (start + MAX_CONTENT_TOKENS).min(ids.len());
            chunks.push(ids[start..end].to_vec());
            if end == ids.len() {
                break;
            }
            start += CHUNK_STRIDE;
        }
        chunks
    }

    /// Classify a prompt of any length.
    ///
    /// Prompts that fit within [`MAX_CONTENT_TOKENS`] (510 tokens) are scored
    /// in a single inference call. Longer prompts are automatically split into
    /// overlapping chunks and each chunk is scored independently — callers do not
    /// need to check length or call a separate method.
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
    /// a single shared session serializes inference internally. True simultaneous
    /// execution would require multiple sessions (one per thread). Dispatching
    /// concurrently still allows the runtime to pipeline and schedule work
    /// efficiently; callers should not assume strict sequential execution.
    pub async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult> {
        let wall_start = Instant::now();

        let (all_ids, all_mask) = self.tokenize_full(text)?;
        let id_chunks = Self::chunk_token_ids(&all_ids);

        let mut handles = Vec::with_capacity(id_chunks.len());
        for chunk_ids in id_chunks {
            let chunk_len = chunk_ids.len();
            let chunk_mask: Vec<i64> = all_mask[..chunk_len.min(all_mask.len())].to_vec();

            let session = Arc::clone(&self.session);
            let model_name = self.model_name.clone();
            let threshold = self.threshold;

            handles.push(tokio::task::spawn_blocking(
                move || -> Result<SusFactorResult> {
                    let chunk_start = Instant::now();
                    let logits = Self::run_inference_sync(&session, chunk_ids, chunk_mask)?;
                    let score = suspicious_prob(&logits);
                    let label = label_for_score(score, threshold).to_string();
                    Ok(SusFactorResult {
                        score,
                        label,
                        model: model_name,
                        threshold,
                        timing_ms: chunk_start.elapsed().as_secs_f64() * 1000.0,
                    })
                },
            ));
        }

        let mut chunk_results = Vec::with_capacity(handles.len());
        for handle in handles {
            let result = handle
                .await
                .map_err(|e| SigError::Model(format!("spawn_blocking panicked: {e}")))?;
            chunk_results.push(result?);
        }

        let is_suspicious = chunk_results.iter().any(|r| r.is_suspicious());
        let total_timing_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

        Ok(ChunkedSusFactorResult {
            chunks: chunk_results,
            is_suspicious,
            total_timing_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::susfactor::types::{CHUNK_OVERLAP, CHUNK_STRIDE, MAX_CONTENT_TOKENS};

    #[test]
    fn suspicious_prob_favours_class_one() {
        assert!(suspicious_prob(&[-5.0, 5.0]) > 0.99);
        assert!(suspicious_prob(&[5.0, -5.0]) < 0.01);
    }

    #[test]
    fn label_uses_threshold_inclusive() {
        assert_eq!(label_for_score(0.9, 0.5), LABEL_SUSPICIOUS);
        assert_eq!(label_for_score(0.5, 0.5), LABEL_SUSPICIOUS);
        assert_eq!(label_for_score(0.49, 0.5), LABEL_SAFE);
    }

    #[test]
    fn canonical_model_default_has_no_onnx_suffix() {
        assert_eq!(
            SusFactorClassifier::DEFAULT_MODEL,
            "0dinai/susfactor-e5-large"
        );
        assert_eq!(
            SusFactorClassifier::DEFAULT_ONNX_REPO,
            "0dinai/susfactor-e5-large-onnx"
        );
    }

    #[test]
    fn susfactor_result_is_suspicious_reflects_label() {
        // SusFactorResult (individual chunk) still has is_suspicious().
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

    // -----------------------------------------------------------------------
    // Chunking logic tests — pure, no model required
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_constants_are_consistent() {
        assert_eq!(CHUNK_STRIDE, MAX_CONTENT_TOKENS - CHUNK_OVERLAP);
        assert!(MAX_CONTENT_TOKENS <= SusFactorClassifier::MAX_SEQUENCE_LENGTH - 2);
    }

    #[test]
    fn chunk_token_ids_short_prompt_produces_one_chunk() {
        // Any token sequence at or below MAX_CONTENT_TOKENS → exactly one chunk,
        // identical to the input.
        let ids: Vec<i64> = (0..100).map(|i| i as i64).collect();
        let chunks = SusFactorClassifier::chunk_token_ids(&ids);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], ids);
    }

    #[test]
    fn chunk_token_ids_exactly_at_limit_produces_one_chunk() {
        let ids: Vec<i64> = (0..MAX_CONTENT_TOKENS as i64).collect();
        let chunks = SusFactorClassifier::chunk_token_ids(&ids);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), MAX_CONTENT_TOKENS);
    }

    #[test]
    fn chunk_token_ids_one_over_limit_produces_two_chunks() {
        let ids: Vec<i64> = (0..(MAX_CONTENT_TOKENS + 1) as i64).collect();
        let chunks = SusFactorClassifier::chunk_token_ids(&ids);
        assert_eq!(chunks.len(), 2);
        // First chunk is full size.
        assert_eq!(chunks[0].len(), MAX_CONTENT_TOKENS);
        // Second chunk is the overflow — just the 1 token beyond stride, plus overlap.
        // It should start at CHUNK_STRIDE and contain everything to the end.
        assert_eq!(chunks[1], &ids[CHUNK_STRIDE..]);
    }

    #[test]
    fn chunk_token_ids_overlap_is_shared() {
        // For a 2-chunk case, the last CHUNK_OVERLAP tokens of chunk 0
        // must equal the first CHUNK_OVERLAP tokens of chunk 1.
        let ids: Vec<i64> = (0..(MAX_CONTENT_TOKENS + CHUNK_STRIDE) as i64).collect();
        let chunks = SusFactorClassifier::chunk_token_ids(&ids);
        assert!(chunks.len() >= 2);
        let tail_of_first = &chunks[0][MAX_CONTENT_TOKENS - CHUNK_OVERLAP..];
        let head_of_second = &chunks[1][..CHUNK_OVERLAP];
        assert_eq!(tail_of_first, head_of_second);
    }

    #[test]
    fn chunk_token_ids_all_tokens_covered() {
        // Every token in the original sequence must appear in at least one chunk.
        // Check by verifying the last chunk ends at the last token of `ids`.
        let n = MAX_CONTENT_TOKENS * 3; // force at least 3 chunks
        let ids: Vec<i64> = (0..n as i64).collect();
        let chunks = SusFactorClassifier::chunk_token_ids(&ids);
        assert!(chunks.len() >= 3);
        let last_chunk = chunks.last().unwrap();
        // The last token of the last chunk must be the last token of `ids`.
        assert_eq!(*last_chunk.last().unwrap(), ids.last().copied().unwrap());
    }

    #[test]
    fn chunk_token_ids_no_chunk_exceeds_max_content_tokens() {
        let n = MAX_CONTENT_TOKENS * 5;
        let ids: Vec<i64> = (0..n as i64).collect();
        let chunks = SusFactorClassifier::chunk_token_ids(&ids);
        for chunk in &chunks {
            assert!(
                chunk.len() <= MAX_CONTENT_TOKENS,
                "chunk length {} exceeds MAX_CONTENT_TOKENS {}",
                chunk.len(),
                MAX_CONTENT_TOKENS
            );
        }
    }

    #[test]
    fn chunk_token_ids_empty_input_produces_one_empty_chunk() {
        let chunks = SusFactorClassifier::chunk_token_ids(&[]);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_empty());
    }

    #[test]
    fn chunked_result_is_suspicious_if_any_chunk_is() {
        use crate::susfactor::types::{ChunkedSusFactorResult, LABEL_SAFE, LABEL_SUSPICIOUS};

        let make = |label: &str| SusFactorResult {
            score: if label == LABEL_SUSPICIOUS { 0.9 } else { 0.1 },
            label: label.to_string(),
            model: "m".to_string(),
            threshold: 0.5,
            timing_ms: 1.0,
        };

        let all_safe = ChunkedSusFactorResult {
            chunks: vec![make(LABEL_SAFE), make(LABEL_SAFE)],
            is_suspicious: false,
            total_timing_ms: 2.0,
        };
        assert!(!all_safe.is_suspicious);

        let one_suspicious = ChunkedSusFactorResult {
            chunks: vec![make(LABEL_SAFE), make(LABEL_SUSPICIOUS), make(LABEL_SAFE)],
            is_suspicious: true,
            total_timing_ms: 3.0,
        };
        assert!(one_suspicious.is_suspicious);
    }
}
