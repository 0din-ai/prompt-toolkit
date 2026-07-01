//! The [`SusFactorProvider`] trait: a backend-agnostic classification contract
//! analogous to [`crate::provider::EmbeddingProvider`].
//!
//! Both the in-pod ONNX backend ([`crate::susfactor::onnx::OnnxSusFactor`]) and
//! the remote Vertex backend ([`crate::susfactor::vertex::VertexSusFactor`])
//! implement this trait, so callers select a backend by configuration without
//! changing the caller-facing contract.

use async_trait::async_trait;

use crate::error::Result;
use crate::susfactor::types::ChunkedSusFactorResult;

/// Backend-agnostic SusFactor classification interface.
#[async_trait]
pub trait SusFactorProvider: Send + Sync {
    /// Canonical model identifier reported in results.
    fn model(&self) -> &str;

    /// Decision threshold used to derive labels.
    fn threshold(&self) -> f32;

    /// Classify a prompt of any length, returning one result per chunk.
    async fn classify(&self, text: &str) -> Result<ChunkedSusFactorResult>;
}
