use std::time::Instant;

use crate::error::{Result, SigError};
use crate::lsh::simhash_lsh_multi;
use crate::provider::EmbeddingProvider;
use crate::types::{LshConfig, LshOutput, SignatureResult, SignatureVersion};

/// Generate a signature from text.
///
/// This is the high-level convenience function that orchestrates the full pipeline:
/// 1. Auto-construct provider (if not provided) based on version
/// 2. Generate embedding using the provider
/// 3. Normalize the embedding (already done by providers)
/// 4. Compute LSH signatures
/// 5. Build a `SignatureResult` with metadata
///
/// # Arguments
///
/// * `text` - The text prompt to sign
/// * `version` - Signature version (use `SignatureVersion::Latest` for the latest model).
///   If provider is given, version is inferred from provider dimensions
///   unless explicitly specified for validation.
/// * `provider` - Optional embedding provider. If `None`, auto-constructs the appropriate
///   provider based on version:
///   - V1: `OnnxProvider` (requires model cached, `onnx` feature enabled)
///   - V0: `OpenAIProvider` (requires `OPENAI_API_KEY` env var, `openai` feature enabled)
/// * `config` - Optional LSH configuration (defaults to 3 families, 256 bits, 16 bands)
///
/// # Examples
///
/// Simple usage (auto-constructs V1/ONNX provider):
/// ```no_run
/// # #[cfg(feature = "onnx")]
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use odin_prompt_toolkit::{sign_text, SignatureVersion};
///
/// let result = sign_text("How do I reset my password?", SignatureVersion::Latest, None, None).await?;
/// println!("Signature: {}", result.to_signature_string());
/// # Ok(())
/// # }
/// ```
///
/// Advanced - bring your own provider (version inferred):
/// ```no_run
/// # #[cfg(feature = "onnx")]
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use odin_prompt_toolkit::{sign_text, SignatureVersion};
/// use odin_prompt_toolkit::providers::{OnnxProvider, ModelCache};
/// use odin_prompt_toolkit::provider::EmbeddingProvider;
///
/// let cache = ModelCache::new()?;
/// let provider = OnnxProvider::new(&cache, None, None).await?;
/// let result = sign_text(
///     "How do I reset my password?",
///     SignatureVersion::Latest,
///     Some(&provider),
///     None,
/// ).await?;
/// provider.close().await?; // EmbeddingProvider trait method
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - Embedding generation fails
/// - LSH computation fails
/// - Invalid configuration provided
/// - Required feature flags are not enabled for auto-construction
/// - Version conflicts with provider dimensions
pub async fn sign_text(
    text: &str,
    version: SignatureVersion,
    provider: Option<&dyn EmbeddingProvider>,
    config: Option<LshConfig>,
) -> Result<SignatureResult> {
    let start = Instant::now();

    // Auto-construct provider if needed, or use provided one
    let owned_provider = if provider.is_none() {
        Some(create_provider_for_version(version).await?)
    } else {
        None
    };

    // Get the provider reference (either from owned or provided)
    let provider_ref: &dyn EmbeddingProvider = match (&owned_provider, provider) {
        (Some(owned), None) => owned.as_ref(),
        (None, Some(provided)) => provided,
        _ => unreachable!(),
    };

    // Execute the main logic, ensuring cleanup happens via Drop or explicit call
    let result = async {
        // Infer or validate version based on provider dimensions
        let resolved_version = resolve_version(version, provider_ref)?;

        // Generate embedding using provider
        let embedding_result = provider_ref.generate_embedding(text).await?;

        // Use provided config or default
        let lsh_config = config.unwrap_or_default();

        // Verify dimensions match expected for this version
        let expected_dims = resolved_version.embedding_dimensions();
        if embedding_result.dimensions != expected_dims {
            return Err(SigError::InvalidInput(format!(
                "Embedding dimensions mismatch: expected {} for {:?}, got {}",
                expected_dims, resolved_version, embedding_result.dimensions
            )));
        }

        // Compute LSH signatures (providers already normalize embeddings)
        let signatures = simhash_lsh_multi(&embedding_result.normalized_embedding, &lsh_config);

        // Build result
        let elapsed_ms = start.elapsed().as_millis() as f64;

        // Create prompt preview (first 50 chars)
        let prompt_preview = if text.len() <= 50 {
            text.to_string()
        } else {
            format!("{}...", &text[..47])
        };

        Ok(SignatureResult {
            signature: String::new(), // Will be computed by to_signature_string()
            version: resolved_version,
            prompt_preview,
            prompt_length: text.len(),
            provider: provider_ref.name().to_string(),
            model: embedding_result.model,
            dimensions: embedding_result.dimensions,
            embedding_sha256: embedding_result.normalized_embedding_sha256,
            lsh: LshOutput {
                config: lsh_config,
                signatures,
            },
            timing_ms: Some(elapsed_ms),
        })
    }
    .await;

    // Clean up owned provider if we created one
    if let Some(owned) = owned_provider {
        let _ = owned.close().await;
    }

    result
}

/// Auto-construct the appropriate provider for a given version.
///
/// # Errors
///
/// Returns an error if:
/// - Required feature flag is not enabled
/// - Required environment variables are missing
/// - Provider initialization fails
async fn create_provider_for_version(version: SignatureVersion) -> Result<Box<dyn EmbeddingProvider>> {
    let resolved = version.resolve();

    match resolved {
        SignatureVersion::V1 => {
            #[cfg(feature = "onnx")]
            {
                use crate::providers::{ModelCache, OnnxProvider};
                let cache = ModelCache::new()?;
                let provider = OnnxProvider::new(&cache, None, None).await?;
                Ok(Box::new(provider))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(SigError::InvalidInput(
                    "V1 signatures require the 'onnx' feature. \
                     Enable with: cargo add signature-sdk --features onnx"
                        .to_string(),
                ))
            }
        }
        SignatureVersion::V0 => {
            #[cfg(feature = "openai")]
            {
                use crate::providers::OpenAIProvider;
                let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
                    SigError::InvalidInput(
                        "OPENAI_API_KEY environment variable is required for V0 signatures. \
                         Set it with: export OPENAI_API_KEY='sk-...'"
                            .to_string(),
                    )
                })?;
                let provider = OpenAIProvider::new(api_key, None, None, None, None)?;
                Ok(Box::new(provider))
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(SigError::InvalidInput(
                    "V0 signatures require the 'openai' feature. \
                     Enable with: cargo add signature-sdk --features openai"
                        .to_string(),
                ))
            }
        }
        SignatureVersion::Latest => {
            // This should never happen since resolve() handles Latest -> V1
            unreachable!("Latest should have been resolved to V1")
        }
    }
}

/// Resolve version from provider dimensions or validate explicitly passed version.
///
/// # Errors
///
/// Returns an error if:
/// - Dimensions don't match any known version
/// - Version conflicts with provider dimensions
fn resolve_version(
    version: SignatureVersion,
    provider: &dyn EmbeddingProvider,
) -> Result<SignatureVersion> {
    let resolved_version = version.resolve();
    let provider_dims = provider.dimensions();

    // Infer version from provider dimensions
    let inferred_version = match provider_dims {
        1536 => SignatureVersion::V0,
        1024 => SignatureVersion::V1,
        _ => {
            return Err(SigError::InvalidInput(format!(
                "Cannot infer version from provider dimensions ({}). \
                 Expected 1536 (V0) or 1024 (V1). \
                 Please specify version explicitly.",
                provider_dims
            )));
        }
    };

    // If version was explicitly passed (not Latest), validate it matches
    if version != SignatureVersion::Latest && resolved_version != inferred_version {
        return Err(SigError::InvalidInput(format!(
            "Version mismatch: requested {:?} (expects {} dims) \
             but provider returns {} dims (matches {:?})",
            resolved_version,
            resolved_version.embedding_dimensions(),
            provider_dims,
            inferred_version
        )));
    }

    Ok(inferred_version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmbeddingResult, LshConfig};
    use async_trait::async_trait;

    // Mock provider for testing
    struct MockProvider {
        name: String,
        model: String,
        dimensions: usize,
        embedding: Vec<f32>,
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn model(&self) -> &str {
            &self.model
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        async fn generate_embedding(&self, _text: &str) -> Result<EmbeddingResult> {
            // Return pre-normalized test embedding
            Ok(EmbeddingResult {
                embedding: self.embedding.clone(),
                normalized_embedding: self.embedding.clone(),
                normalized_embedding_sha256: "test-sha256".to_string(),
                model: self.model.clone(),
                dimensions: self.dimensions,
                token_count: Some(10),
                timing_ms: Some(100.0),
            })
        }

        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_sign_text_v1_with_provider() {
        // Create mock provider with V1 dimensions (1024)
        let embedding = vec![0.5; 1024];
        let provider = MockProvider {
            name: "mock-onnx".to_string(),
            model: "test-model".to_string(),
            dimensions: 1024,
            embedding,
        };

        let result = sign_text("test prompt", SignatureVersion::V1, Some(&provider), None)
            .await
            .unwrap();

        assert_eq!(result.version, SignatureVersion::V1);
        assert_eq!(result.provider, "mock-onnx");
        assert_eq!(result.model, "test-model");
        assert_eq!(result.dimensions, 1024);
        assert_eq!(result.prompt_preview, "test prompt");
        assert_eq!(result.prompt_length, 11);
        assert!(result.timing_ms.is_some());

        // Verify signature format
        let sig_string = result.to_signature_string();
        assert!(sig_string.starts_with("0din-v1:"));
        assert_eq!(sig_string.len(), 72); // "0din-v1:" (8) + 64 hex chars
    }

    #[tokio::test]
    async fn test_sign_text_v0_with_provider() {
        // Create mock provider with V0 dimensions (1536)
        let embedding = vec![0.5; 1536];
        let provider = MockProvider {
            name: "mock-openai".to_string(),
            model: "text-embedding-3-large".to_string(),
            dimensions: 1536,
            embedding,
        };

        let result = sign_text("test prompt", SignatureVersion::V0, Some(&provider), None)
            .await
            .unwrap();

        assert_eq!(result.version, SignatureVersion::V0);
        assert_eq!(result.dimensions, 1536);

        let sig_string = result.to_signature_string();
        assert!(sig_string.starts_with("0din-v0:"));
    }

    #[tokio::test]
    async fn test_sign_text_latest_resolves_to_v1() {
        let embedding = vec![0.5; 1024]; // V1 dimensions
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };

        let result = sign_text("test", SignatureVersion::Latest, Some(&provider), None)
            .await
            .unwrap();

        // Latest should resolve to V1
        assert_eq!(result.version, SignatureVersion::V1);
    }

    #[tokio::test]
    async fn test_sign_text_infer_version_from_provider() {
        // V1 provider (1024 dims)
        let provider_v1 = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding: vec![0.5; 1024],
        };

        let result = sign_text("test", SignatureVersion::Latest, Some(&provider_v1), None)
            .await
            .unwrap();
        assert_eq!(result.version, SignatureVersion::V1);

        // V0 provider (1536 dims)
        let provider_v0 = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1536,
            embedding: vec![0.5; 1536],
        };

        let result = sign_text("test", SignatureVersion::Latest, Some(&provider_v0), None)
            .await
            .unwrap();
        assert_eq!(result.version, SignatureVersion::V0);
    }

    #[tokio::test]
    async fn test_sign_text_version_mismatch() {
        // Provider returns 1024 dimensions but we request V0 (expects 1536)
        let embedding = vec![0.5; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };

        let result = sign_text("test", SignatureVersion::V0, Some(&provider), None).await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Version mismatch"));
    }

    #[tokio::test]
    async fn test_sign_text_dimension_mismatch() {
        // Provider with unknown dimensions
        let embedding = vec![0.5; 512];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 512,
            embedding,
        };

        let result = sign_text("test", SignatureVersion::Latest, Some(&provider), None).await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Cannot infer version"));
    }

    #[tokio::test]
    async fn test_sign_text_custom_config() {
        let embedding = vec![0.5; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };

        let custom_config = LshConfig {
            families: 5,
            bits: 128,
            bands: 8,
        };

        let result = sign_text(
            "test",
            SignatureVersion::V1,
            Some(&provider),
            Some(custom_config.clone()),
        )
        .await
        .unwrap();

        assert_eq!(result.lsh.config.families, 5);
        assert_eq!(result.lsh.config.bits, 128);
        assert_eq!(result.lsh.config.bands, 8);
        assert_eq!(result.lsh.signatures.len(), 5); // 5 families
    }

    #[tokio::test]
    async fn test_sign_text_long_prompt_preview() {
        let embedding = vec![0.5; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };

        let long_text = "a".repeat(100);
        let result = sign_text(&long_text, SignatureVersion::V1, Some(&provider), None)
            .await
            .unwrap();

        // Preview should be truncated to 50 chars
        assert_eq!(result.prompt_preview.len(), 50);
        assert!(result.prompt_preview.ends_with("..."));
        assert_eq!(result.prompt_length, 100);
    }
}
