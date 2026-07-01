//! Shadow-mode SusFactor: runs both a primary and a shadow backend concurrently
//! and reports divergence between them.
//!
//! The primary result is always returned to the caller; the shadow result is
//! used only for observability. If the shadow call fails, the primary result is
//! still returned and divergence is `None`.
//!
//! This module compiles only when the `susfactor-vertex` feature is enabled.

use crate::error::Result;
use crate::susfactor::provider::SusFactorProvider;
use crate::susfactor::types::ChunkedSusFactorResult;

// ---------------------------------------------------------------------------
// Divergence types
// ---------------------------------------------------------------------------

/// Per-chunk divergence between two SusFactor backends.
///
/// Retained for callers that need the full per-chunk breakdown.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkDivergence {
    /// Score from the primary backend.
    pub primary_score: f32,
    /// Score from the shadow backend.
    pub shadow_score: f32,
    /// Absolute difference between the two scores.
    pub delta: f32,
    /// Whether the two backends disagree on the label for this chunk.
    pub label_mismatch: bool,
}

/// Divergence report comparing primary and shadow backends across all chunks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShadowDivergence {
    /// Absolute score delta per paired chunk: `|primary.score - shadow.score|`.
    ///
    /// If the backends produce different chunk counts, only `min(p, s)` pairs
    /// are included.
    pub chunk_score_deltas: Vec<f32>,
    /// `true` if any paired chunk has different labels between the two backends.
    pub label_mismatch: bool,
    /// `true` if the top-level `is_suspicious` flag differs between the two
    /// backends.
    pub is_suspicious_mismatch: bool,
}

// ---------------------------------------------------------------------------
// ShadowSusFactor
// ---------------------------------------------------------------------------

/// Shadow-mode classifier that runs two backends concurrently and returns the
/// primary result plus a divergence report.
///
/// Constructed with trait objects so it can be used in tests with mock
/// providers and in production with any pair of `SusFactorProvider`
/// implementations.
pub struct ShadowSusFactor {
    primary: Box<dyn SusFactorProvider>,
    shadow: Box<dyn SusFactorProvider>,
}

impl ShadowSusFactor {
    /// Create a new shadow classifier.
    ///
    /// * `primary` — the authoritative backend; its result is always returned.
    /// * `shadow` — the comparison backend; failures are silently ignored.
    pub fn new(primary: Box<dyn SusFactorProvider>, shadow: Box<dyn SusFactorProvider>) -> Self {
        Self { primary, shadow }
    }

    /// Classify `text` with both backends concurrently.
    ///
    /// Always returns the primary result. If the shadow call fails, divergence
    /// is `None`. If both succeed, divergence is `Some(ShadowDivergence)`.
    ///
    /// # Errors
    ///
    /// Returns an error only if the *primary* backend fails. Shadow failures
    /// are swallowed.
    pub async fn classify_with_divergence(
        &self,
        text: &str,
    ) -> Result<(ChunkedSusFactorResult, Option<ShadowDivergence>)> {
        let (primary_result, shadow_result) =
            tokio::join!(self.primary.classify(text), self.shadow.classify(text));

        // Primary failure is fatal.
        let primary = primary_result?;

        // Shadow failure → return primary with no divergence.
        let shadow = match shadow_result {
            Ok(r) => r,
            Err(_) => return Ok((primary, None)),
        };

        let divergence = compute_divergence(&primary, &shadow);
        Ok((primary, Some(divergence)))
    }
}

// ---------------------------------------------------------------------------
// Divergence computation
// ---------------------------------------------------------------------------

fn compute_divergence(
    primary: &ChunkedSusFactorResult,
    shadow: &ChunkedSusFactorResult,
) -> ShadowDivergence {
    let pair_count = primary.chunks.len().min(shadow.chunks.len());

    let mut chunk_score_deltas = Vec::with_capacity(pair_count);
    let mut label_mismatch = false;

    for i in 0..pair_count {
        let p = &primary.chunks[i];
        let s = &shadow.chunks[i];
        chunk_score_deltas.push((p.score - s.score).abs());
        if p.label != s.label {
            label_mismatch = true;
        }
    }

    ShadowDivergence {
        chunk_score_deltas,
        label_mismatch,
        is_suspicious_mismatch: primary.is_suspicious != shadow.is_suspicious,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::susfactor::types::{ChunkedSusFactorResult, SusFactorResult};
    use async_trait::async_trait;

    // -----------------------------------------------------------------------
    // FakeProvider: a test double for SusFactorProvider.
    //
    // `result` is `Some(ChunkedSusFactorResult)` for success or `None` for
    // an error. We avoid storing `SigError` directly because it does not
    // implement `Clone`.
    // -----------------------------------------------------------------------
    struct FakeProvider {
        result: Option<ChunkedSusFactorResult>,
        model_name: String,
    }

    impl FakeProvider {
        fn ok(score: f32, label: &str, model: &str) -> Self {
            FakeProvider {
                result: Some(ChunkedSusFactorResult {
                    chunks: vec![SusFactorResult {
                        score,
                        label: label.to_string(),
                        model: model.to_string(),
                        threshold: 0.5,
                        timing_ms: 1.0,
                    }],
                    is_suspicious: label == "suspicious",
                    total_timing_ms: 1.0,
                }),
                model_name: model.to_string(),
            }
        }

        fn err() -> Self {
            FakeProvider {
                result: None,
                model_name: "fake".to_string(),
            }
        }
    }

    #[async_trait]
    impl SusFactorProvider for FakeProvider {
        fn model(&self) -> &str {
            &self.model_name
        }

        fn threshold(&self) -> f32 {
            0.5
        }

        async fn classify(&self, _text: &str) -> crate::Result<ChunkedSusFactorResult> {
            match &self.result {
                Some(r) => Ok(r.clone()),
                None => Err(crate::error::SigError::Provider("fake error".to_string())),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: both succeed — primary result and correct divergence returned
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn both_succeed_returns_primary_result_with_divergence() {
        let shadow = ShadowSusFactor::new(
            Box::new(FakeProvider::ok(0.9, "suspicious", "primary-model")),
            Box::new(FakeProvider::ok(0.85, "suspicious", "shadow-model")),
        );

        let (result, divergence) = shadow
            .classify_with_divergence("test")
            .await
            .expect("must succeed");

        // Primary result returned.
        assert_eq!(result.chunks.len(), 1);
        assert!(
            (result.chunks[0].score - 0.9).abs() < 1e-6,
            "expected primary score 0.9, got {}",
            result.chunks[0].score
        );
        assert_eq!(result.chunks[0].model, "primary-model");

        // Divergence present.
        let div = divergence.expect("divergence must be Some when both succeed");
        assert_eq!(div.chunk_score_deltas.len(), 1);
        assert!(
            (div.chunk_score_deltas[0] - (0.9_f32 - 0.85_f32).abs()).abs() < 1e-5,
            "expected delta ≈ 0.05, got {}",
            div.chunk_score_deltas[0]
        );
        assert!(!div.label_mismatch, "both label 'suspicious' — no mismatch");
        assert!(
            !div.is_suspicious_mismatch,
            "both are suspicious — no mismatch"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: shadow fails — primary result returned, divergence is None
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn shadow_failure_returns_primary_with_no_divergence() {
        let shadow = ShadowSusFactor::new(
            Box::new(FakeProvider::ok(0.9, "suspicious", "primary-model")),
            Box::new(FakeProvider::err()),
        );

        let (result, divergence) = shadow
            .classify_with_divergence("test")
            .await
            .expect("primary must succeed");

        assert_eq!(result.chunks.len(), 1);
        assert!(
            (result.chunks[0].score - 0.9).abs() < 1e-6,
            "primary score must be 0.9, got {}",
            result.chunks[0].score
        );
        assert!(
            divergence.is_none(),
            "divergence must be None when shadow fails"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: label mismatch — divergence fields reflect disagreement
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn label_mismatch_reflected_in_divergence() {
        let shadow = ShadowSusFactor::new(
            Box::new(FakeProvider::ok(0.9, "suspicious", "primary-model")),
            Box::new(FakeProvider::ok(0.3, "safe", "shadow-model")),
        );

        let (result, divergence) = shadow
            .classify_with_divergence("test")
            .await
            .expect("must succeed");

        // Primary result unchanged.
        assert_eq!(result.chunks.len(), 1);
        assert!(
            (result.chunks[0].score - 0.9).abs() < 1e-6,
            "primary score must be 0.9"
        );

        let div = divergence.expect("divergence must be Some");
        assert!(div.label_mismatch, "labels differ — must be true");
        assert!(
            div.is_suspicious_mismatch,
            "primary is suspicious, shadow is safe — must be true"
        );
        assert_eq!(div.chunk_score_deltas.len(), 1);
        assert!(
            (div.chunk_score_deltas[0] - (0.9_f32 - 0.3_f32).abs()).abs() < 1e-5,
            "expected delta ≈ 0.6, got {}",
            div.chunk_score_deltas[0]
        );
    }
}
