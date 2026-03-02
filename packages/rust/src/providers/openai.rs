//! OpenAI/OpenRouter embedding provider implementation.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{SigError, Result};
use crate::lsh::{compute_embedding_sha256, normalize_vector};
use crate::provider::EmbeddingProvider;
use crate::types::EmbeddingResult;

/// OpenAI API response structure for embeddings.
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    total_tokens: usize,
}

/// Request payload for OpenAI embeddings API.
#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: String,
    dimensions: usize,
}

/// Embedding provider using OpenAI or OpenRouter API.
///
/// This provider uses the OpenAI embeddings API to generate vector embeddings
/// for text. It can also be configured to use OpenRouter or other OpenAI-compatible
/// APIs by setting a custom base URL.
///
/// # Example
///
/// ```no_run
/// use odin_sig::providers::OpenAIProvider;
/// use odin_sig::provider::EmbeddingProvider;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = OpenAIProvider::new(
///     "sk-...".to_string(),
///     None,
///     None,
///     None,
///     None,
/// )?;
///
/// let result = provider.generate_embedding("Hello, world!").await?;
/// println!("Generated embedding with {} dimensions", result.dimensions);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
    dimensions: usize,
    base_url: String,
    name: String,
}

impl OpenAIProvider {
    /// Default model for OpenAI embeddings.
    pub const DEFAULT_MODEL: &'static str = "text-embedding-3-large";

    /// Default dimensions for text-embedding-3-large.
    pub const DEFAULT_DIMENSIONS: usize = 1536;

    /// Default base URL for OpenAI API.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";

    /// Create a new OpenAI provider.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI or OpenRouter API key
    /// * `model` - Model to use for embeddings (default: text-embedding-3-large)
    /// * `dimensions` - Number of embedding dimensions (default: 1536)
    /// * `base_url` - Base URL for OpenAI-compatible API (default: <https://api.openai.com/v1>)
    /// * `name` - Provider name identifier (default: "openai", use "openrouter" for OpenRouter)
    ///
    /// # Errors
    ///
    /// Returns an error if the API key is empty or if the HTTP client cannot be initialized.
    pub fn new(
        api_key: String,
        model: Option<String>,
        dimensions: Option<usize>,
        base_url: Option<String>,
        name: Option<String>,
    ) -> Result<Self> {
        if api_key.is_empty() {
            return Err(SigError::InvalidInput(
                "API key is required".to_string(),
            ));
        }

        let client = Client::builder()
            .build()
            .map_err(|e| SigError::Provider(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            api_key,
            model: model.unwrap_or_else(|| Self::DEFAULT_MODEL.to_string()),
            dimensions: dimensions.unwrap_or(Self::DEFAULT_DIMENSIONS),
            base_url: base_url.unwrap_or_else(|| Self::DEFAULT_BASE_URL.to_string()),
            name: name.unwrap_or_else(|| "openai".to_string()),
        })
    }

    /// Generate embeddings for the given text using the OpenAI API.
    ///
    /// This is an internal helper that makes the actual HTTP request.
    async fn call_api(&self, text: &str) -> Result<OpenAIEmbeddingResponse> {
        let url = format!("{}/embeddings", self.base_url);

        let request = OpenAIEmbeddingRequest {
            model: self.model.clone(),
            input: text.to_string(),
            dimensions: self.dimensions,
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| SigError::Provider(format!("Failed to call OpenAI API: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(SigError::Provider(format!(
                "OpenAI API error ({}): {}",
                status, error_text
            )));
        }

        response
            .json::<OpenAIEmbeddingResponse>()
            .await
            .map_err(|e| SigError::Provider(format!("Failed to parse OpenAI response: {}", e)))
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn generate_embedding(&self, text: &str) -> Result<EmbeddingResult> {
        let response = self.call_api(text).await?;

        let embedding = response
            .data
            .first()
            .ok_or_else(|| SigError::Provider("No embedding data returned".to_string()))?
            .embedding
            .clone();

        let normalized = normalize_vector(&embedding);
        let sha256 = compute_embedding_sha256(&normalized);
        let token_count = response.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0);

        Ok(EmbeddingResult {
            embedding,
            normalized_embedding: normalized,
            normalized_embedding_sha256: sha256,
            model: self.model.clone(),
            dimensions: self.dimensions,
            token_count: Some(token_count),
            timing_ms: None,
        })
    }

    async fn close(&self) -> Result<()> {
        // reqwest::Client doesn't need explicit closing
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_requires_api_key() {
        let result = OpenAIProvider::new(String::new(), None, None, None, None);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SigError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_new_with_defaults() {
        let provider = OpenAIProvider::new("sk-test".to_string(), None, None, None, None).unwrap();

        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model(), OpenAIProvider::DEFAULT_MODEL);
        assert_eq!(provider.dimensions(), OpenAIProvider::DEFAULT_DIMENSIONS);
    }

    #[test]
    fn test_new_with_custom_values() {
        let provider = OpenAIProvider::new(
            "sk-test".to_string(),
            Some("custom-model".to_string()),
            Some(512),
            Some("https://custom.api".to_string()),
            Some("custom".to_string()),
        )
        .unwrap();

        assert_eq!(provider.name(), "custom");
        assert_eq!(provider.model(), "custom-model");
        assert_eq!(provider.dimensions(), 512);
        assert_eq!(provider.base_url, "https://custom.api");
    }
}
