/// Embedding provider implementations.
///
/// This module contains implementations of the `EmbeddingProvider` trait
/// for different embedding services and models.
///
/// ## Available Providers
///
/// - `OpenAIProvider` - OpenAI/OpenRouter API embeddings (always available)
/// - `OnnxProvider` - Local ONNX embeddings with tract (requires `onnx` feature)
///
/// ## Feature Flags
///
/// - `openai` (default) - Enables OpenAI provider
/// - `onnx` (default) - Enables ONNX provider with tract runtime
#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "onnx")]
pub mod model_cache;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

#[cfg(feature = "onnx")]
pub use onnx::OnnxProvider;

#[cfg(feature = "onnx")]
pub use model_cache::ModelCache;
