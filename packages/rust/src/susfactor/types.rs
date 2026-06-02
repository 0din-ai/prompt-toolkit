//! Type definitions for SusFactor classification.

/// Label for a SusFactor classification.
pub const LABEL_SUSPICIOUS: &str = "suspicious";
/// Label for a benign prompt.
pub const LABEL_SAFE: &str = "safe";

/// Result of a SusFactor classification.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SusFactorResult {
    /// Probability that the prompt is suspicious/malicious, in `[0, 1]`.
    pub score: f32,
    /// `"suspicious"` if `score >= threshold`, else `"safe"`.
    pub label: String,
    /// Identifier of the model that produced the score.
    pub model: String,
    /// Decision threshold used to derive `label`.
    pub threshold: f32,
    /// Inference time in milliseconds.
    pub timing_ms: f64,
}

impl SusFactorResult {
    /// Whether the prompt was classified as suspicious.
    pub fn is_suspicious(&self) -> bool {
        self.label == LABEL_SUSPICIOUS
    }
}
