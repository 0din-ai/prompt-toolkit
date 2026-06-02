use std::time::Instant;

use crate::error::{Result, SigError};
use crate::lsh::{cosine_from_hamming, hamming_distance_hex, simhash_lsh_multi};
use crate::provider::EmbeddingProvider;
use crate::types::{signature_string, ComparisonResult, LshConfig, LshOutput, PromptInfo, SignatureResult, SignatureVersion};

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
/// let provider = OnnxProvider::new(&cache, None, None, 0, 0).await?;
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

/// Compare two prompts and return their similarity.
///
/// Generates embeddings for both prompts concurrently, computes LSH signatures,
/// then derives the Hamming distance and estimated cosine similarity. The
/// `version` field on the returned result reflects the resolved signature version
/// (always `V0` or `V1`, never `Latest`).
///
/// # Arguments
///
/// * `text_a` / `text_b` - The two prompts to compare
/// * `version` - Requested signature version. When a provider is supplied the
///   effective version is inferred from `provider.dimensions()` (1024 → V1,
///   1536 → V0), identical to `sign_text`. Passing `Latest` with a V0 provider
///   therefore yields `version = V0` in the result.
/// * `provider` - Optional provider; auto-constructed if `None` (same semantics as `sign_text`)
/// * `config` - Optional LSH config; defaults to 3 families, 256 bits, 16 bands
///
/// # Errors
///
/// Returns an error if embedding generation, LSH computation, or version
/// resolution fails.
pub async fn compare_text(
    text_a: &str,
    text_b: &str,
    version: SignatureVersion,
    provider: Option<&dyn EmbeddingProvider>,
    config: Option<LshConfig>,
) -> Result<ComparisonResult> {
    let start = Instant::now();

    let owned_provider = if provider.is_none() {
        Some(create_provider_for_version(version).await?)
    } else {
        None
    };

    let provider_ref: &dyn EmbeddingProvider = match (&owned_provider, provider) {
        (Some(owned), None) => owned.as_ref(),
        (None, Some(provided)) => provided,
        _ => unreachable!(),
    };

    let result = async {
        let resolved_version = resolve_version(version, provider_ref)?;
        let lsh_config = config.unwrap_or_default();

        // Generate both embeddings concurrently.
        let (emb_a, emb_b) = tokio::try_join!(
            provider_ref.generate_embedding(text_a),
            provider_ref.generate_embedding(text_b),
        )?;

        let expected_dims = resolved_version.embedding_dimensions();
        for (dims, label) in [(emb_a.dimensions, "text_a"), (emb_b.dimensions, "text_b")] {
            if dims != expected_dims {
                return Err(SigError::InvalidInput(format!(
                    "Embedding dimensions mismatch for {label}: expected {expected_dims} \
                     for {resolved_version:?}, got {dims}"
                )));
            }
        }

        let sigs_a = simhash_lsh_multi(&emb_a.normalized_embedding, &lsh_config);
        let sigs_b = simhash_lsh_multi(&emb_b.normalized_embedding, &lsh_config);

        // Use family 0 (primary) signature for distance computation.
        let hamming = hamming_distance_hex(&sigs_a[0].signature, &sigs_b[0].signature);
        // sigs_a[0].bits holds the value after simhash_lsh_multi's internal
        // clamping (min 64), so cosine_from_hamming is consistent with the
        // bits actually used.
        let cosine = cosine_from_hamming(hamming, sigs_a[0].bits);

        // Build the effective config from the clamped values so the returned
        // lsh_config matches what was actually used for the signatures.
        let effective_config = LshConfig {
            families: lsh_config.families.max(1),
            bits: lsh_config.bits.max(64),
            bands: lsh_config.bands.max(1),
        };

        let elapsed_ms = start.elapsed().as_millis() as f64;

        // Truncate at a char boundary to avoid panicking on non-ASCII UTF-8.
        // Use char count (not byte length) for the `length` field as well.
        let prompt_preview = |text: &str| -> String {
            if text.chars().count() <= 50 {
                text.to_string()
            } else {
                let cut = text
                    .char_indices()
                    .nth(47)
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());
                format!("{}...", &text[..cut])
            }
        };

        Ok(ComparisonResult {
            prompt_a: PromptInfo {
                preview: prompt_preview(text_a),
                length: text_a.len(), // byte length, consistent with sign_text's prompt_length
                signature: signature_string(resolved_version, &sigs_a[0].signature),
            },
            prompt_b: PromptInfo {
                preview: prompt_preview(text_b),
                length: text_b.len(), // byte length, consistent with sign_text's prompt_length
                signature: signature_string(resolved_version, &sigs_b[0].signature),
            },
            hamming_distance: hamming,
            cosine_similarity: cosine,
            lsh_config: effective_config,
            version: resolved_version,
            quality_stats: None,
            timing_ms: Some(elapsed_ms),
        })
    }
    .await;

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
                // intra_threads=0 (auto), pool_size=0 (default pool of 2).
                let provider = OnnxProvider::new(&cache, None, None, 0, 0).await?;
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

    // -----------------------------------------------------------------------
    // compare_text tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_compare_text_same_prompt_is_identical() {
        let embedding = vec![0.5f32; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };
        let result = compare_text("hello", "hello", SignatureVersion::V1, Some(&provider), None)
            .await
            .unwrap();

        assert_eq!(result.hamming_distance, 0);
        assert!((result.cosine_similarity - 1.0).abs() < 1e-6);
        assert_eq!(result.version, SignatureVersion::V1);
    }

    #[tokio::test]
    async fn test_compare_text_version_is_resolved() {
        let embedding = vec![0.5f32; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };
        let result = compare_text("a", "b", SignatureVersion::Latest, Some(&provider), None)
            .await
            .unwrap();
        // Latest should resolve to V1 (1024 dims)
        assert_eq!(result.version, SignatureVersion::V1);
    }

    #[tokio::test]
    async fn test_compare_text_version_never_latest_in_result() {
        let embedding = vec![0.5f32; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };
        let result = compare_text("a", "b", SignatureVersion::Latest, Some(&provider), None)
            .await
            .unwrap();
        assert_ne!(result.version, SignatureVersion::Latest);
    }

    #[tokio::test]
    async fn test_compare_text_unicode_preview_does_not_panic() {
        // "日" is 3 bytes; 51 of them = 153 bytes but only 51 chars.
        // Slicing at byte 47 would land mid-codepoint and panic without the fix.
        let long_unicode = "日".repeat(51);
        let embedding = vec![0.5f32; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };
        let result =
            compare_text(&long_unicode, "hello", SignatureVersion::V1, Some(&provider), None)
                .await
                .unwrap();
        // Should not panic; preview is at most 47 chars + "..." = 50 chars total.
        assert!(result.prompt_a.preview.chars().count() <= 50);
    }

    #[tokio::test]
    async fn test_compare_text_result_fields() {
        let embedding = vec![0.5f32; 1024];
        let provider = MockProvider {
            name: "mock".to_string(),
            model: "test".to_string(),
            dimensions: 1024,
            embedding,
        };
        let result = compare_text("prompt a", "prompt b", SignatureVersion::V1, Some(&provider), None)
            .await
            .unwrap();

        assert_eq!(result.prompt_a.preview, "prompt a");
        assert_eq!(result.prompt_a.length, 8); // 8 bytes (ASCII, matches sign_text byte semantics)
        assert!(result.prompt_a.signature.starts_with("0din-v1:"));
        assert_eq!(result.prompt_b.preview, "prompt b");
        assert_eq!(result.prompt_b.length, 8);
        assert!(result.prompt_b.signature.starts_with("0din-v1:"));
        assert!(result.timing_ms.is_some());
        assert_eq!(result.lsh_config, LshConfig::default()); // default is within clamped range
    }
}
