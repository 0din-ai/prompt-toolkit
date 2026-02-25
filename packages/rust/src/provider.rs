use async_trait::async_trait;

use crate::error::Result;
use crate::types::EmbeddingResult;

/// Embedding provider interface
///
/// Defines the contract for embedding generation from text.
/// Implementations include OpenAI API client and ONNX local inference.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Provider identifier (e.g., "openai", "onnx")
    fn name(&self) -> &str;

    /// Model name being used
    fn model(&self) -> &str;

    /// Embedding dimensions
    fn dimensions(&self) -> usize;

    /// Generate embedding for text
    ///
    /// # Arguments
    /// * `text` - Input text to embed
    ///
    /// # Returns
    /// * `Result<EmbeddingResult>` - Embedding with metadata
    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult>;

    /// Cleanup resources
    ///
    /// Called when the provider is no longer needed.
    /// Implementations should close connections, free memory, etc.
    async fn close(&self) -> Result<()>;
}
