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

/// Validate and flatten a model's raw logits output.
///
/// The classifier head emits `logits[1, 2]`; a flattened view must contain at
/// least two elements. Both backends route their raw output through this so the
/// "unexpected output shape" error is identical.
pub fn validate_logits(flat: Vec<f32>) -> Result<Vec<f32>> {
    if flat.len() < 2 {
        return Err(SigError::Model(format!(
            "Unexpected SusFactor output shape; got {} elements, expected >= 2",
            flat.len()
        )));
    }
    Ok(flat)
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
/// `is_suspicious` is `true` if **any** chunk is suspicious.
pub fn reduce(chunks: Vec<SusFactorResult>, total_timing_ms: f64) -> ChunkedSusFactorResult {
    let is_suspicious = chunks.iter().any(|r| r.is_suspicious());
    ChunkedSusFactorResult {
        chunks,
        is_suspicious,
        total_timing_ms,
    }
}

/// Convenience wrapper returning elapsed milliseconds since `start`.
pub fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
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
        assert!(validate_logits(vec![1.0]).is_err());
        assert!(validate_logits(vec![]).is_err());
        assert!(validate_logits(vec![1.0, 2.0]).is_ok());
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

        let all_safe = reduce(vec![make(LABEL_SAFE), make(LABEL_SAFE)], 2.0);
        assert!(!all_safe.is_suspicious);

        let one_suspicious = reduce(
            vec![make(LABEL_SAFE), make(LABEL_SUSPICIOUS), make(LABEL_SAFE)],
            3.0,
        );
        assert!(one_suspicious.is_suspicious);
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
}
