//! ONNX embedding provider implementation using tract.
//!
//! This module provides embedding generation using ONNX models via the tract runtime.
//! The default model is `intfloat/multilingual-e5-small` which produces 384-dimensional embeddings.
//!
//! ## Features
//!
//! - Local ONNX model inference (no API calls)
//! - Automatic model download from HuggingFace
//! - Mean pooling with attention masking
//! - L2 normalization of embeddings
//! - Support for E5 models with "query: " prefix
//!
//! ## Usage
//!
//! Enable the `onnx` feature in your `Cargo.toml`:
//!
//! ```toml
//! odin-sig = { version = "0.1", features = ["onnx"] }
//! ```
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use odin_sig::providers::{OnnxProvider, ModelCache};
//! use odin_sig::provider::EmbeddingProvider;
//!
//! let cache = ModelCache::new()?;
//! let provider = OnnxProvider::new(&cache, None, None).await?;
//!
//! let result = provider.generate_embedding("Hello, world!").await?;
//! println!("Embedding dimensions: {}", result.embedding.len());
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};
use tract_ndarray::Array2;
use tract_onnx::prelude::*;

use crate::error::{SigError, Result};
use crate::lsh::{compute_embedding_sha256, normalize_vector};
use crate::provider::EmbeddingProvider;
use crate::providers::ModelCache;
use crate::types::EmbeddingResult;

/// Type alias for the tract ONNX inference plan to reduce complexity
type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// ONNX embedding provider using tract runtime.
///
/// This provider uses the `intfloat/multilingual-e5-small` model by default,
/// which produces 384-dimensional embeddings suitable for multilingual text similarity.
///
/// The model is automatically downloaded from HuggingFace on first use and cached locally.
#[derive(Debug)]
pub struct OnnxProvider {
    /// Tract inference plan
    model: Arc<TractModel>,

    /// Tokenizer for text preprocessing
    tokenizer: Arc<tokenizers::Tokenizer>,

    /// Model name (HuggingFace ID or local path)
    model_name: String,

    /// Embedding dimensions
    dimensions: usize,

    /// Provider name
    name: String,

    /// Input prefix (e.g., "query: " for E5 models, empty for others)
    input_prefix: String,
}

impl OnnxProvider {
    /// Default model for ONNX embeddings.
    pub const DEFAULT_MODEL: &'static str = "models/v1";

    /// Default embedding dimensions.
    pub const DEFAULT_DIMENSIONS: usize = 384;

    /// Maximum sequence length supported by the model.
    pub const MAX_SEQUENCE_LENGTH: usize = 512;

    /// Input prefix required by E5 models.
    pub const INPUT_PREFIX: &'static str = "query: ";

    /// ONNX model filename (optimized version).
    const ONNX_MODEL_FILE: &'static str = "onnx/model.onnx";

    /// Create a new ONNX provider.
    ///
    /// # Arguments
    ///
    /// * `cache` - Model cache for downloading/caching models
    /// * `model` - Model name or HuggingFace ID (default: intfloat/multilingual-e5-small)
    /// * `name` - Provider name identifier (default: "onnx")
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model download fails
    /// - ONNX model loading fails
    /// - Tokenizer loading fails
    pub async fn new(
        cache: &ModelCache,
        model: Option<String>,
        name: Option<String>,
    ) -> Result<Self> {
        let model_name = model.unwrap_or_else(|| Self::DEFAULT_MODEL.to_string());
        let provider_name = name.unwrap_or_else(|| "onnx".to_string());

        info!("Initializing ONNX provider with model: {}", model_name);

        // Download model and tokenizer
        let model_path = cache.get_model(&model_name, Self::ONNX_MODEL_FILE).await?;
        let tokenizer_path = cache.get_tokenizer(&model_name).await?;

        info!("Loading ONNX model from: {}", model_path.display());

        // Load ONNX model with tract
        let model = tract_onnx::onnx()
            .model_for_path(&model_path)
            .map_err(|e| SigError::Provider(format!("Failed to load ONNX model: {}", e)))?
            .into_optimized()
            .map_err(|e| SigError::Provider(format!("Failed to optimize ONNX model: {}", e)))?
            .into_runnable()
            .map_err(|e| {
                SigError::Provider(format!("Failed to create runnable model: {}", e))
            })?;

        info!("ONNX model loaded successfully");

        // Load tokenizer
        info!("Loading tokenizer from: {}", tokenizer_path.display());
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| SigError::Provider(format!("Failed to load tokenizer: {}", e)))?;

        info!("Tokenizer loaded successfully");

        // Detect input prefix from model configuration
        let input_prefix = Self::detect_input_prefix(&model_name, cache)
            .await
            .unwrap_or_else(|e| {
                debug!(
                    "Could not detect input prefix from config, using default: {}",
                    e
                );
                Self::INPUT_PREFIX.to_string()
            });

        if input_prefix.is_empty() {
            info!("Model does not require input prefix");
        } else {
            info!("Using input prefix: {:?}", input_prefix);
        }

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            model_name,
            dimensions: Self::DEFAULT_DIMENSIONS,
            name: provider_name,
            input_prefix,
        })
    }

    /// Detect input prefix from model's sentence_transformers config.
    ///
    /// Returns empty string if no prefix is needed (custom models).
    /// Returns "query: " or similar if the model config specifies a prompt.
    /// Falls back to default E5 prefix on error.
    async fn detect_input_prefix(model_id: &str, cache: &ModelCache) -> Result<String> {
        // Try to load the sentence_transformers config
        let config_path = cache
            .get_model(model_id, "config_sentence_transformers.json")
            .await?;

        let config_str = tokio::fs::read_to_string(&config_path).await.map_err(|e| {
            SigError::Provider(format!("Failed to read sentence_transformers config: {}", e))
        })?;

        let config: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            SigError::Provider(format!("Failed to parse sentence_transformers config: {}", e))
        })?;

        // Check if prompts.query is non-empty
        if let Some(query_prompt) = config
            .get("prompts")
            .and_then(|p| p.get("query"))
            .and_then(|q| q.as_str())
        {
            if !query_prompt.is_empty() {
                // E5 models use "query: " with trailing space
                return Ok(format!("{} ", query_prompt));
            }
        }

        // No prefix needed (custom model case)
        Ok(String::new())
    }

    /// Tokenize text and prepare inputs for the model.
    ///
    /// This method:
    /// 1. Prepends the input prefix if configured (e.g., "query: " for E5 models)
    /// 2. Tokenizes the text
    /// 3. Truncates to max sequence length
    /// 4. Returns input IDs and attention mask
    fn tokenize(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        // Prepend prefix if configured
        let prefixed_text = if self.input_prefix.is_empty() {
            text.to_string()
        } else {
            format!("{}{}", self.input_prefix, text)
        };

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(prefixed_text, true)
            .map_err(|e| SigError::Provider(format!("Tokenization failed: {}", e)))?;

        // Get token IDs and attention mask
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        // Truncate to max sequence length
        if input_ids.len() > Self::MAX_SEQUENCE_LENGTH {
            debug!(
                "Truncating sequence from {} to {} tokens",
                input_ids.len(),
                Self::MAX_SEQUENCE_LENGTH
            );
            input_ids.truncate(Self::MAX_SEQUENCE_LENGTH);
            attention_mask.truncate(Self::MAX_SEQUENCE_LENGTH);
        }

        Ok((input_ids, attention_mask))
    }

    /// Run inference on the ONNX model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs
    /// * `attention_mask` - Attention mask
    ///
    /// # Returns
    ///
    /// Raw token embeddings from the model (before pooling).
    fn run_inference(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Array2<f32>> {
        let seq_len = input_ids.len();

        // Create input tensors
        let input_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, seq_len), input_ids.to_vec()).map_err(
                |e| SigError::Provider(format!("Failed to create input_ids tensor: {}", e)),
            )?;

        let attention_mask_tensor = tract_ndarray::Array2::from_shape_vec(
            (1, seq_len),
            attention_mask.to_vec(),
        )
        .map_err(|e| {
            SigError::Provider(format!("Failed to create attention_mask tensor: {}", e))
        })?;

        // Create token_type_ids tensor (all zeros for single sentence input)
        // BERT-based models require this third input even for single sentences
        let token_type_ids: Vec<i64> = vec![0i64; seq_len];
        let token_type_ids_tensor =
            tract_ndarray::Array2::from_shape_vec((1, seq_len), token_type_ids).map_err(|e| {
                SigError::Provider(format!("Failed to create token_type_ids tensor: {}", e))
            })?;

        // Convert to Tensor and run inference
        let input_ids_value = Tensor::from(input_ids_tensor.into_dyn()).into();
        let attention_mask_value = Tensor::from(attention_mask_tensor.into_dyn()).into();
        let token_type_ids_value = Tensor::from(token_type_ids_tensor.into_dyn()).into();

        let outputs = self
            .model
            .run(tvec!(
                input_ids_value,
                attention_mask_value,
                token_type_ids_value
            ))
            .map_err(|e| SigError::Provider(format!("Model inference failed: {}", e)))?;

        // Extract output tensor (last_hidden_state)
        let output = outputs[0].to_array_view::<f32>().map_err(|e| {
            SigError::Provider(format!("Failed to extract output tensor: {}", e))
        })?;

        // Convert to Array2 (batch_size=1, seq_len, hidden_dim)
        let shape = output.shape();
        if shape.len() != 3 || shape[0] != 1 {
            return Err(SigError::Provider(format!(
                "Unexpected output shape: {:?}",
                shape
            )));
        }

        let seq_len = shape[1];
        let hidden_dim = shape[2];

        // Remove batch dimension and convert to Array2
        let output_2d = output
            .index_axis(tract_ndarray::Axis(0), 0)
            .to_owned()
            .into_shape_with_order((seq_len, hidden_dim))
            .map_err(|e| SigError::Provider(format!("Failed to reshape output: {}", e)))?;

        Ok(output_2d)
    }

    /// Apply mean pooling to token embeddings.
    ///
    /// This averages token embeddings, weighted by the attention mask.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Token embeddings (seq_len x hidden_dim)
    /// * `attention_mask` - Attention mask (seq_len)
    ///
    /// # Returns
    ///
    /// Pooled embedding vector (hidden_dim)
    fn mean_pool(&self, embeddings: &Array2<f32>, attention_mask: &[i64]) -> Result<Vec<f32>> {
        let (seq_len, hidden_dim) = embeddings.dim();

        if seq_len != attention_mask.len() {
            return Err(SigError::Provider(format!(
                "Attention mask length ({}) does not match sequence length ({})",
                attention_mask.len(),
                seq_len
            )));
        }

        let mut pooled = vec![0.0f32; hidden_dim];
        let mut mask_sum = 0.0f32;

        // Sum embeddings weighted by attention mask
        for (i, &mask_val) in attention_mask.iter().enumerate() {
            if mask_val == 1 {
                let mask_f32 = mask_val as f32;
                mask_sum += mask_f32;

                for j in 0..hidden_dim {
                    pooled[j] += embeddings[[i, j]] * mask_f32;
                }
            }
        }

        // Average
        if mask_sum > 0.0 {
            for val in &mut pooled {
                *val /= mask_sum;
            }
        }

        Ok(pooled)
    }
}

#[async_trait]
impl EmbeddingProvider for OnnxProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model_name
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult> {
        debug!("Generating embedding for text: {}", text);

        // Tokenize
        let (input_ids, attention_mask) = self.tokenize(text)?;
        let token_count = input_ids.len();

        // Run inference
        let token_embeddings = self.run_inference(&input_ids, &attention_mask)?;

        // Mean pooling
        let embedding = self.mean_pool(&token_embeddings, &attention_mask)?;

        // L2 normalization using the shared function
        let normalized = normalize_vector(&embedding);
        let sha256 = compute_embedding_sha256(&normalized);

        debug!("Generated {}-dimensional embedding", embedding.len());

        Ok(EmbeddingResult {
            embedding,
            normalized_embedding: normalized,
            normalized_embedding_sha256: sha256,
            model: self.model_name.clone(),
            dimensions: self.dimensions,
            token_count: Some(token_count),
            timing_ms: None,
        })
    }

    async fn close(&self) -> Result<()> {
        debug!("Closing ONNX provider");
        // tract doesn't require explicit cleanup
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(OnnxProvider::DEFAULT_MODEL, "models/v1");
        assert_eq!(OnnxProvider::DEFAULT_DIMENSIONS, 384);
        assert_eq!(OnnxProvider::MAX_SEQUENCE_LENGTH, 512);
        assert_eq!(OnnxProvider::INPUT_PREFIX, "query: ");
    }

    // Note: Full integration tests require model download and are in integration tests
}
