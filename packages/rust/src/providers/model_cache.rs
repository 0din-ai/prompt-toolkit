//! Model cache for downloading and managing ONNX models from HuggingFace.

use crate::error::{SigError, Result};
use std::env;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

/// Model cache for downloading and storing ONNX models.
///
/// Models are cached to `~/.cache/heimdall/models/` by default,
/// or to the path specified by the `HEIMDALL_MODEL_CACHE` environment variable.
#[derive(Debug, Clone)]
pub struct ModelCache {
    cache_dir: PathBuf,
}

impl ModelCache {
    /// Create a new model cache.
    ///
    /// # Returns
    ///
    /// A new `ModelCache` instance with the cache directory set to:
    /// - `$HEIMDALL_MODEL_CACHE` if set
    /// - `~/.cache/heimdall/models/` otherwise
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be determined or created.
    pub fn new() -> Result<Self> {
        let cache_dir = if let Ok(path) = env::var("HEIMDALL_MODEL_CACHE") {
            PathBuf::from(path)
        } else {
            dirs::cache_dir()
                .ok_or_else(|| SigError::Provider("Cannot determine cache directory".into()))?
                .join("heimdall")
                .join("models")
        };

        Ok(Self { cache_dir })
    }

    /// Get the path to a model file, downloading it if necessary.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace model ID (e.g., "intfloat/multilingual-e5-small")
    ///                or local directory path (e.g., "models/v1")
    /// * `filename` - Filename within the model repo (e.g., "onnx/model.onnx")
    ///
    /// # Returns
    ///
    /// Path to the cached or local model file.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be downloaded or cached.
    pub async fn get_model(&self, model_id: &str, filename: &str) -> Result<PathBuf> {
        // Check if model_id is a local directory path
        let local_path = PathBuf::from(model_id);
        if local_path.is_dir() {
            let model_file = local_path.join(filename);
            if model_file.exists() {
                info!("Using local model file: {}", model_file.display());
                return Ok(model_file);
            }
            return Err(SigError::Provider(format!(
                "Local model directory '{}' exists but file '{}' not found",
                model_id, filename
            )));
        }

        // Otherwise, use cache directory for HuggingFace models
        let model_path = self.cache_dir.join(model_id).join(filename);

        if model_path.exists() {
            debug!("Model already cached at: {}", model_path.display());
            return Ok(model_path);
        }

        info!("Downloading model {} from HuggingFace...", model_id);

        self.download_file(model_id, filename, &model_path).await?;

        Ok(model_path)
    }

    /// Get the path to a tokenizer file, downloading it if necessary.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace model ID (e.g., "intfloat/multilingual-e5-small")
    ///
    /// # Returns
    ///
    /// Path to the cached tokenizer.json file.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer cannot be downloaded or cached.
    pub async fn get_tokenizer(&self, model_id: &str) -> Result<PathBuf> {
        self.get_model(model_id, "tokenizer.json").await
    }

    /// Download a file from HuggingFace Hub.
    async fn download_file(
        &self,
        model_id: &str,
        filename: &str,
        dest_path: &PathBuf,
    ) -> Result<()> {
        // Create parent directories
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                SigError::Provider(format!("Failed to create cache directory: {}", e))
            })?;
        }

        // Construct HuggingFace URL
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_id, filename
        );

        debug!("Downloading from: {}", url);

        // Download the file
        let response = reqwest::get(&url)
            .await
            .map_err(|e| SigError::Provider(format!("Failed to download model: {}", e)))?;

        if !response.status().is_success() {
            return Err(SigError::Provider(format!(
                "Failed to download model: HTTP {}",
                response.status()
            )));
        }

        let total_size = response.content_length().unwrap_or(0);
        if total_size > 0 {
            info!(
                "Downloading {} ({:.1} MB)...",
                filename,
                total_size as f64 / 1_000_000.0
            );
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| SigError::Provider(format!("Failed to read model data: {}", e)))?;

        // Write to temporary file first
        let temp_path = dest_path.with_extension("tmp");
        let mut file = fs::File::create(&temp_path).await.map_err(|e| {
            SigError::Provider(format!("Failed to create temporary file: {}", e))
        })?;

        file.write_all(&bytes)
            .await
            .map_err(|e| SigError::Provider(format!("Failed to write model file: {}", e)))?;

        file.sync_all()
            .await
            .map_err(|e| SigError::Provider(format!("Failed to sync model file: {}", e)))?;

        drop(file);

        // Rename to final path (atomic on most systems)
        fs::rename(&temp_path, dest_path).await.map_err(|e| {
            SigError::Provider(format!("Failed to finalize model file: {}", e))
        })?;

        info!("Model cached at: {}", dest_path.display());

        Ok(())
    }

    /// Get the cache directory path.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Clear all cached models.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be removed.
    pub async fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)
                .await
                .map_err(|e| SigError::Provider(format!("Failed to clear cache: {}", e)))?;
            info!("Cache cleared: {}", self.cache_dir.display());
        } else {
            warn!(
                "Cache directory does not exist: {}",
                self.cache_dir.display()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let cache = ModelCache::new().unwrap();
        assert!(cache.cache_dir().to_string_lossy().contains("heimdall"));
    }

    #[test]
    fn test_cache_dir() {
        let cache = ModelCache::new().unwrap();
        let cache_dir = cache.cache_dir();
        assert!(cache_dir.ends_with("heimdall/models") || cache_dir.ends_with("heimdall\\models"));
    }

    #[tokio::test]
    async fn test_local_model_path() {
        use std::fs;
        use tempfile::TempDir;

        // Create a temporary directory structure
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path().join("test-model");
        let onnx_dir = model_dir.join("onnx");
        fs::create_dir_all(&onnx_dir).unwrap();

        // Create a dummy model file
        let model_file = onnx_dir.join("model.onnx");
        fs::write(&model_file, b"dummy onnx data").unwrap();

        // Test that local path is detected and used
        let cache = ModelCache::new().unwrap();
        let result = cache
            .get_model(model_dir.to_str().unwrap(), "onnx/model.onnx")
            .await
            .unwrap();

        assert_eq!(result, model_file);
    }

    #[tokio::test]
    async fn test_local_model_path_missing_file() {
        use std::fs;
        use tempfile::TempDir;

        // Create a temporary directory without the requested file
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path().join("test-model");
        fs::create_dir_all(&model_dir).unwrap();

        // Test that missing file is detected
        let cache = ModelCache::new().unwrap();
        let result = cache
            .get_model(model_dir.to_str().unwrap(), "onnx/model.onnx")
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("file 'onnx/model.onnx' not found"));
    }

    // Note: Download tests are skipped as they require network access
    // and would be slow. These should be tested manually or in integration tests.
}
