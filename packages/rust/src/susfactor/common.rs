//! Shared SusFactor logic used **verbatim** by both the ONNX and Vertex
//! backends, so the two cannot diverge.
//!
//! This module owns tokenization, chunking, softmax, labeling, per-chunk result
//! assembly, and the `ChunkedSusFactorResult` reduction. It depends only on
//! `tokenizers` (for [`tokenize_full`]) and is free of any inference runtime, so
//! it compiles under either the `onnx` or `susfactor-vertex` feature without
//! pulling in `ort`/`ndarray` or `gcp_auth`.

use std::time::Instant;

use crate::error::{Result, SigError};
use crate::susfactor::types::{
    ChunkedSusFactorResult, PhaseSpan, SusFactorResult, CHUNK_STRIDE, LABEL_SAFE, LABEL_SUSPICIOUS,
    MAX_CONTENT_TOKENS,
};

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Canonical model identifier reported in results (shared across SDKs).
pub const DEFAULT_MODEL: &str = "0dinai/susfactor-e5-large";

/// HuggingFace repo holding the ONNX export / tokenizer downloaded at runtime.
pub const DEFAULT_ONNX_REPO: &str = "0dinai/susfactor-e5-large-onnx";

/// Default decision threshold.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// Phase name for the tokenization span.
pub const PHASE_TOKENIZE: &str = "tokenize";
/// Phase name for the chunking span.
pub const PHASE_CHUNK: &str = "chunk";
/// Phase name for a per-chunk inference span.
pub const PHASE_INFERENCE: &str = "inference";
/// Phase name for the result-assembly span.
pub const PHASE_REDUCE: &str = "reduce";

/// Softmax over a 2-logit slice, returning P(class 1) = suspicious.
pub fn suspicious_prob(logits: &[f32]) -> f32 {
    debug_assert!(logits.len() >= 2);
    let m = logits[0].max(logits[1]);
    let e0 = (logits[0] - m).exp();
    let e1 = (logits[1] - m).exp();
    e1 / (e0 + e1)
}

/// Map a suspicious probability to a label using `threshold`.
///
/// `>=` is inclusive: a score exactly equal to the threshold is suspicious.
pub fn label_for_score(score: f32, threshold: f32) -> &'static str {
    if score >= threshold {
        LABEL_SUSPICIOUS
    } else {
        LABEL_SAFE
    }
}

/// Encode `text` into `(input_ids, attention_mask)` as `i64` vectors, adding the
/// special tokens (`[CLS]`/`[SEP]`) the model expects.
///
/// Extracted from the ONNX backend so the Vertex backend tokenizes identically.
pub fn tokenize_full(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let encoding = tokenizer
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

/// Load a SusFactor tokenizer with truncation disabled.
///
/// The bundled `tokenizer.json` embeds `truncation.max_length = 512`. Left in
/// place, [`tokenizers::Tokenizer::encode`] silently cuts every prompt to 512
/// tokens *before* [`chunk_token_ids`] runs — bypassing long-prompt chunking
/// and dropping any content past the limit. We tokenize the full input and
/// window it ourselves, so truncation must be cleared at load time.
pub fn load_tokenizer(path: impl AsRef<std::path::Path>) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
        .map_err(|e| SigError::Model(format!("Failed to load SusFactor tokenizer: {e}")))?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| SigError::Model(format!("Failed to clear tokenizer truncation: {e}")))?;
    Ok(tokenizer)
}

/// Split a token-ID sequence into overlapping chunks of at most
/// [`MAX_CONTENT_TOKENS`] tokens each.
///
/// - Sequences at or below `MAX_CONTENT_TOKENS` produce exactly one chunk.
/// - Adjacent chunks share [`crate::susfactor::types::CHUNK_OVERLAP`] tokens of
///   context so that sentence boundaries near a chunk edge are still scored in
///   full context.
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

/// Split token-ID and attention-mask sequences into overlapping chunks using
/// identical stride/overlap windows.
///
/// Returns `Vec<(id_chunk, mask_chunk)>` where each pair covers the same
/// token range. The chunking logic mirrors [`chunk_token_ids`] exactly; this
/// function exists so the mask is never reconstructed from scratch inside a
/// chunk handler (which would be wrong once padding is added).
pub fn chunk_token_ids_with_mask(ids: &[i64], mask: &[i64]) -> Vec<(Vec<i64>, Vec<i64>)> {
    if ids.len() <= MAX_CONTENT_TOKENS {
        return vec![(ids.to_vec(), mask.to_vec())];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + MAX_CONTENT_TOKENS).min(ids.len());
        chunks.push((ids[start..end].to_vec(), mask[start..end].to_vec()));
        if end == ids.len() {
            break;
        }
        start += CHUNK_STRIDE;
    }
    chunks
}

/// Validate a model's raw logits slice.
///
/// The classifier head emits `logits[1, 2]`; the slice must contain at least
/// two elements. Both backends route their raw output through this so the
/// "unexpected output shape" error is identical.
pub fn validate_logits(flat: &[f32]) -> Result<()> {
    if flat.len() < 2 {
        return Err(SigError::Model(format!(
            "Unexpected SusFactor output shape; got {} elements, expected >= 2",
            flat.len()
        )));
    }
    Ok(())
}

/// Assemble a single [`SusFactorResult`] from raw logits, applying the shared
/// softmax and labeling.
pub fn result_from_logits(
    logits: &[f32],
    model: &str,
    threshold: f32,
    timing_ms: f64,
) -> SusFactorResult {
    let score = suspicious_prob(logits);
    let label = label_for_score(score, threshold).to_string();
    SusFactorResult {
        score,
        label,
        model: model.to_string(),
        threshold,
        timing_ms,
    }
}

/// Reduce per-chunk results into a [`ChunkedSusFactorResult`].
///
/// `is_suspicious` is `true` if **any** chunk is suspicious. `total_tokens` is
/// the length of the full tokenized input sequence (before chunking). `spans`
/// carries the ordered per-phase timing waterfall assembled by the caller.
pub fn reduce(
    chunks: Vec<SusFactorResult>,
    total_tokens: usize,
    total_timing_ms: f64,
    spans: Vec<PhaseSpan>,
) -> ChunkedSusFactorResult {
    let is_suspicious = chunks.iter().any(|r| r.is_suspicious());
    ChunkedSusFactorResult {
        chunks,
        is_suspicious,
        total_tokens,
        total_timing_ms,
        spans,
    }
}

/// Convenience wrapper returning elapsed milliseconds since `start`.
pub fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Milliseconds between the wall-clock baseline `wall` and a later `start`.
///
/// Used to compute each [`PhaseSpan::start_ms`] offset against the single
/// baseline captured at the start of a classify call.
pub fn offset_ms(start: Instant, wall: Instant) -> f64 {
    start.duration_since(wall).as_secs_f64() * 1000.0
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
    fn validate_logits_rejects_short_output() {
        assert!(validate_logits(&[1.0]).is_err());
        assert!(validate_logits(&[]).is_err());
        assert!(validate_logits(&[1.0, 2.0]).is_ok());
    }

    /// Regression: the loaded tokenizer must NOT truncate long prompts, so that
    /// long-prompt chunking actually runs. Model-gated (needs the real
    /// tokenizer); skips when `SUSFACTOR_MODEL_DIR` is unset.
    #[test]
    fn load_tokenizer_disables_truncation_for_long_prompts() {
        let Ok(model_dir) = std::env::var("SUSFACTOR_MODEL_DIR") else {
            eprintln!("SUSFACTOR_MODEL_DIR unset; skipping truncation regression");
            return;
        };
        let path = std::path::Path::new(&model_dir).join("tokenizer.json");
        let tokenizer = load_tokenizer(&path).expect("load tokenizer");
        // Well over the model's 512-token window.
        let long = "The quarterly business review covered revenue and churn. ".repeat(150);
        let enc = tokenizer.encode(long.as_str(), true).expect("encode");
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        assert!(
            ids.len() > MAX_CONTENT_TOKENS,
            "truncation not disabled: got {} tokens (<= {MAX_CONTENT_TOKENS})",
            ids.len()
        );
        let chunks = chunk_token_ids(&ids);
        assert!(
            chunks.len() > 2,
            "expected >2 chunks for a long prompt, got {}",
            chunks.len()
        );
    }

    #[test]
    fn result_from_logits_applies_softmax_and_label() {
        let r = result_from_logits(&[-5.0, 5.0], "m", 0.5, 1.0);
        assert!(r.score > 0.99);
        assert_eq!(r.label, LABEL_SUSPICIOUS);
        assert_eq!(r.model, "m");
        assert_eq!(r.threshold, 0.5);

        let safe = result_from_logits(&[5.0, -5.0], "m", 0.5, 1.0);
        assert!(safe.score < 0.01);
        assert_eq!(safe.label, LABEL_SAFE);
    }

    #[test]
    fn reduce_is_suspicious_if_any_chunk_is() {
        let make = |label: &str| SusFactorResult {
            score: if label == LABEL_SUSPICIOUS { 0.9 } else { 0.1 },
            label: label.to_string(),
            model: "m".to_string(),
            threshold: 0.5,
            timing_ms: 1.0,
        };

        let all_safe = reduce(vec![make(LABEL_SAFE), make(LABEL_SAFE)], 0, 2.0, vec![]);
        assert!(!all_safe.is_suspicious);

        let one_suspicious = reduce(
            vec![make(LABEL_SAFE), make(LABEL_SUSPICIOUS), make(LABEL_SAFE)],
            0,
            3.0,
            vec![],
        );
        assert!(one_suspicious.is_suspicious);
    }

    #[test]
    fn spans_waterfall_has_expected_shape_and_ordering() {
        // Build a representative waterfall the way a backend does: one tokenize,
        // one chunk, one inference span per chunk (in order), then one reduce.
        let make = |i: usize| SusFactorResult {
            score: 0.1,
            label: LABEL_SAFE.to_string(),
            model: "m".to_string(),
            threshold: 0.5,
            timing_ms: 1.0 + i as f64,
        };
        let chunks = vec![make(0), make(1), make(2)];

        let mut spans = vec![
            PhaseSpan {
                name: PHASE_TOKENIZE.to_string(),
                start_ms: 0.0,
                duration_ms: 0.5,
                chunk_index: None,
                token_count: None,
            },
            PhaseSpan {
                name: PHASE_CHUNK.to_string(),
                start_ms: 0.5,
                duration_ms: 0.3,
                chunk_index: None,
                token_count: None,
            },
        ];
        for (i, c) in chunks.iter().enumerate() {
            spans.push(PhaseSpan {
                name: PHASE_INFERENCE.to_string(),
                start_ms: 1.0 + i as f64,
                duration_ms: c.timing_ms,
                chunk_index: Some(i),
                token_count: Some(510 - i),
            });
        }
        spans.push(PhaseSpan {
            name: PHASE_REDUCE.to_string(),
            start_ms: 10.0,
            duration_ms: 0.2,
            chunk_index: None,
            token_count: None,
        });

        let result = reduce(chunks, 1530, 12.0, spans);
        let spans = &result.spans;

        // Non-empty, correct bookends.
        assert!(!spans.is_empty());
        assert_eq!(spans.first().unwrap().name, PHASE_TOKENIZE);
        assert_eq!(spans[1].name, PHASE_CHUNK);
        assert_eq!(spans.last().unwrap().name, PHASE_REDUCE);

        // Exactly len(chunks) inference spans, indices 0..n-1 each once.
        let inference: Vec<&PhaseSpan> =
            spans.iter().filter(|s| s.name == PHASE_INFERENCE).collect();
        assert_eq!(inference.len(), result.chunks.len());
        for (pos, s) in inference.iter().enumerate() {
            assert_eq!(s.chunk_index, Some(pos));
        }
        let mut indices: Vec<usize> = inference.iter().map(|s| s.chunk_index.unwrap()).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices.len(), result.chunks.len());

        // Only inference spans carry a chunk_index.
        for s in spans.iter().filter(|s| s.name != PHASE_INFERENCE) {
            assert!(s.chunk_index.is_none());
        }

        // total_tokens is a non-negative integer surfaced on the result.
        let _: usize = result.total_tokens;

        // Every inference span carries a positive per-chunk token_count; all
        // non-inference spans leave it unset.
        for s in inference.iter() {
            assert!(s.token_count.is_some_and(|n| n > 0));
        }
        for s in spans.iter().filter(|s| s.name != PHASE_INFERENCE) {
            assert!(s.token_count.is_none());
        }

        // Durations finite/non-negative; start offsets non-negative.
        for s in spans {
            assert!(s.duration_ms.is_finite() && s.duration_ms >= 0.0);
            assert!(s.start_ms.is_finite() && s.start_ms >= 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Chunking logic tests — pure, no model required
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_constants_are_consistent() {
        assert_eq!(CHUNK_STRIDE, MAX_CONTENT_TOKENS - CHUNK_OVERLAP);
    }

    #[test]
    fn chunk_token_ids_short_prompt_produces_one_chunk() {
        let ids: Vec<i64> = (0..100).map(|i| i as i64).collect();
        let chunks = chunk_token_ids(&ids);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], ids);
    }

    #[test]
    fn chunk_token_ids_exactly_at_limit_produces_one_chunk() {
        let ids: Vec<i64> = (0..MAX_CONTENT_TOKENS as i64).collect();
        let chunks = chunk_token_ids(&ids);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), MAX_CONTENT_TOKENS);
    }

    #[test]
    fn chunk_token_ids_one_over_limit_produces_two_chunks() {
        let ids: Vec<i64> = (0..(MAX_CONTENT_TOKENS + 1) as i64).collect();
        let chunks = chunk_token_ids(&ids);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), MAX_CONTENT_TOKENS);
        assert_eq!(chunks[1], &ids[CHUNK_STRIDE..]);
    }

    #[test]
    fn chunk_token_ids_overlap_is_shared() {
        let ids: Vec<i64> = (0..(MAX_CONTENT_TOKENS + CHUNK_STRIDE) as i64).collect();
        let chunks = chunk_token_ids(&ids);
        assert!(chunks.len() >= 2);
        let tail_of_first = &chunks[0][MAX_CONTENT_TOKENS - CHUNK_OVERLAP..];
        let head_of_second = &chunks[1][..CHUNK_OVERLAP];
        assert_eq!(tail_of_first, head_of_second);
    }

    #[test]
    fn chunk_token_ids_all_tokens_covered() {
        let n = MAX_CONTENT_TOKENS * 3;
        let ids: Vec<i64> = (0..n as i64).collect();
        let chunks = chunk_token_ids(&ids);
        assert!(chunks.len() >= 3);
        let last_chunk = chunks.last().unwrap();
        assert_eq!(*last_chunk.last().unwrap(), ids.last().copied().unwrap());
    }

    #[test]
    fn chunk_token_ids_no_chunk_exceeds_max_content_tokens() {
        let n = MAX_CONTENT_TOKENS * 5;
        let ids: Vec<i64> = (0..n as i64).collect();
        let chunks = chunk_token_ids(&ids);
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
        let chunks = chunk_token_ids(&[]);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_empty());
    }

    #[test]
    fn chunk_token_ids_with_mask_short_produces_one_pair() {
        let ids: Vec<i64> = (0..100).map(|i| i as i64).collect();
        let mask: Vec<i64> = vec![1i64; 100];
        let chunks = chunk_token_ids_with_mask(&ids, &mask);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, ids);
        assert_eq!(chunks[0].1, mask);
    }

    #[test]
    fn chunk_token_ids_with_mask_produces_same_count_as_ids_only() {
        let n = MAX_CONTENT_TOKENS * 3;
        let ids: Vec<i64> = (0..n as i64).collect();
        let mask: Vec<i64> = vec![1i64; n];
        let paired = chunk_token_ids_with_mask(&ids, &mask);
        let ids_only = chunk_token_ids(&ids);
        assert_eq!(paired.len(), ids_only.len());
        for (i, ((pid, pmask), pid_only)) in paired.iter().zip(ids_only.iter()).enumerate() {
            assert_eq!(pid, pid_only, "chunk {i}: id mismatch");
            assert_eq!(
                pmask.len(),
                pid.len(),
                "chunk {i}: mask length != id length"
            );
        }
    }

    #[test]
    fn chunk_token_ids_with_mask_windows_match_ids() {
        let n = MAX_CONTENT_TOKENS + 1;
        let ids: Vec<i64> = (0..n as i64).collect();
        let mask: Vec<i64> = (0..n as i64)
            .map(|i| if i % 2 == 0 { 1 } else { 0 })
            .collect();
        let chunks = chunk_token_ids_with_mask(&ids, &mask);
        assert_eq!(chunks.len(), 2);
        // First chunk: ids[0..MAX] / mask[0..MAX]
        assert_eq!(chunks[0].0, &ids[..MAX_CONTENT_TOKENS]);
        assert_eq!(chunks[0].1, &mask[..MAX_CONTENT_TOKENS]);
        // Second chunk starts at CHUNK_STRIDE
        assert_eq!(chunks[1].0, &ids[CHUNK_STRIDE..]);
        assert_eq!(chunks[1].1, &mask[CHUNK_STRIDE..]);
    }
}
