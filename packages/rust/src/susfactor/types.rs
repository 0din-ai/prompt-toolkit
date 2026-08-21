//! Type definitions for SusFactor classification.

/// Label for a SusFactor classification.
pub const LABEL_SUSPICIOUS: &str = "suspicious";
/// Label for a benign prompt.
pub const LABEL_SAFE: &str = "safe";

/// Maximum number of *content* tokens per inference chunk.
///
/// The model's hard limit is 512 tokens total, but the tokenizer adds a `[CLS]`
/// and a `[SEP]` token, leaving 510 usable positions for the prompt payload.
pub const MAX_CONTENT_TOKENS: usize = 510;

/// Overlap between adjacent chunks in tokens.
///
/// Each new chunk starts `CHUNK_STRIDE` tokens after the previous one begins,
/// keeping 50 tokens of shared context so sentence boundaries that fall near a
/// chunk edge are still scored in full context.
pub const CHUNK_OVERLAP: usize = 50;

/// Number of new tokens advanced per chunk (= MAX_CONTENT_TOKENS - CHUNK_OVERLAP).
pub const CHUNK_STRIDE: usize = MAX_CONTENT_TOKENS - CHUNK_OVERLAP;

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

/// A single timed phase of a SusFactor [`crate::susfactor::SusFactorProvider::classify`]
/// call, used to build a per-call timing waterfall.
///
/// All `start_ms` offsets are measured from one wall-clock baseline captured at
/// the very start of the call, so spans can be laid out on a shared timeline.
/// Because inference chunks may run concurrently, their spans can overlap.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PhaseSpan {
    /// Phase name: `"tokenize"` | `"chunk"` | `"inference"` | `"reduce"`.
    pub name: String,
    /// Offset from the call's wall-clock start, in milliseconds.
    pub start_ms: f64,
    /// Wall time of this phase, in milliseconds.
    pub duration_ms: f64,
    /// 0-based chunk index; `Some` only on `"inference"` spans.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<usize>,
}

/// Return type of [`crate::susfactor::SusFactorProvider::classify`] for prompts of any length.
///
/// Prompts within [`MAX_CONTENT_TOKENS`] (510 tokens) produce exactly one
/// chunk. Longer prompts are split automatically and produce one entry per
/// chunk — callers never need to check length or call a different method.
///
/// ## Displaying a single score
///
/// The previous API returned one `score` and one `label` directly. With
/// chunking, there is no single canonical score — each chunk is an
/// independent model inference. Callers that need one number for display
/// (dashboards, logs) should decide explicitly which value they want:
///
/// ```no_run
/// # use odin_prompt_toolkit::susfactor::{ChunkedSusFactorResult, SusFactorResult};
/// # let result: ChunkedSusFactorResult = unimplemented!();
/// // Highest suspicion score across all chunks (most conservative):
/// let max_score = result.chunks.iter().map(|c| c.score).fold(f32::MIN, f32::max);
///
/// // Score of the first chunk (equivalent to the old single-prompt score
/// // for short prompts; may miss a suspicious tail in long prompts):
/// let first_score = result.chunks[0].score;
/// ```
///
/// Using `is_suspicious` (any-chunk flag) is the recommended gate for
/// security decisions. A display score is a UX choice, not a security one.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkedSusFactorResult {
    /// Individual result for each chunk, in order.
    ///
    /// Short prompts (≤ [`MAX_CONTENT_TOKENS`] tokens) always produce exactly
    /// one entry. Access `chunks[0]` for the score and label in that case.
    pub chunks: Vec<SusFactorResult>,
    /// `true` if **any** chunk's label is `"suspicious"`.
    ///
    /// This is the recommended field for security gating. A prompt is
    /// considered suspicious if any portion of it is suspicious, regardless
    /// of how many chunks are safe.
    pub is_suspicious: bool,
    /// Total wall-clock time for all chunks, in milliseconds.
    pub total_timing_ms: f64,
    /// Ordered per-phase timing spans for this call, forming a waterfall:
    /// `tokenize`, `chunk`, one `inference` span per chunk (in chunk order,
    /// each carrying its `chunk_index`), then `reduce`.
    ///
    /// Spans share the single wall-clock baseline used for
    /// [`Self::total_timing_ms`]. Inference spans may overlap when chunks run
    /// concurrently, so the summed span durations intentionally do not equal
    /// `total_timing_ms` (the gap is scheduling/join overhead).
    pub spans: Vec<PhaseSpan>,
}
