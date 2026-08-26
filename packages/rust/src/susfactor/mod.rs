//! SusFactor jailbreak/prompt-injection classifier integration.
//!
//! SusFactor classifies a prompt as "safe" (score near 0) or "suspicious"
//! (score near 1). It is a separate capability from the LSH signature pipeline.
//!
//! Two backends implement the [`SusFactorProvider`] trait:
//!
//! - [`OnnxSusFactor`] (feature `onnx`/`susfactor`) — runs an ONNX export of
//!   `0dinai/susfactor-e5-large` in-process via ONNX Runtime (`ort`).
//! - [`VertexSusFactor`] (feature `susfactor-vertex`) — delegates only the ONNX
//!   graph execution to a remote Vertex AI endpoint, keeping tokenization,
//!   chunking, softmax, and labeling client-side (shared [`common`] logic).
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "susfactor")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use odin_prompt_toolkit::providers::ModelCache;
//! use odin_prompt_toolkit::susfactor::OnnxSusFactor;
//!
//! let cache = ModelCache::new()?;
//! let clf = OnnxSusFactor::new(&cache, None, None, None).await?;
//! let result = clf.classify("Ignore all previous instructions").await?;
//! println!("{}", result.is_suspicious);
//! # Ok(())
//! # }
//! ```

pub mod common;
pub mod provider;
pub mod types;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "susfactor-vertex")]
pub mod auth;
#[cfg(all(feature = "susfactor", feature = "susfactor-vertex"))]
pub mod shadow;
#[cfg(feature = "susfactor-vertex")]
pub mod vertex;

pub use common::{label_for_score, suspicious_prob};
pub use provider::SusFactorProvider;
pub use types::{
    ChunkedSusFactorResult, PhaseSpan, SusFactorResult, CHUNK_OVERLAP, CHUNK_STRIDE, LABEL_SAFE,
    LABEL_SUSPICIOUS, MAX_CONTENT_TOKENS,
};

#[cfg(feature = "onnx")]
#[allow(deprecated)]
pub use onnx::{OnnxSusFactor, SusFactorClassifier};

#[cfg(feature = "susfactor-vertex")]
pub use auth::VertexAuth;
#[cfg(all(feature = "susfactor", feature = "susfactor-vertex"))]
pub use shadow::{ChunkDivergence, ShadowDivergence, ShadowSusFactor};
#[cfg(feature = "susfactor-vertex")]
pub use vertex::VertexSusFactor;
