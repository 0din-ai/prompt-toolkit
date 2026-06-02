//! Model cache for downloading and managing ONNX models from HuggingFace.

use crate::error::{Result, SigError};
use futures_util::StreamExt;
use std::env;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

/// Process-global counter for temp file uniqueness.
///
/// `SystemTime` resolution can be coarser than nanoseconds on some platforms,
/// so two rapid calls may produce the same timestamp. The counter guarantees
/// each temp path is distinct within the process lifetime regardless of clock
/// resolution.
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Model cache for downloading and storing ONNX models.
///
/// Models are cached to `~/.cache/signature-sdk/models/` by default,
/// or to the path specified by the `SIGNATURE_SDK_MODEL_CACHE` environment variable.
#[derive(Debug, Clone)]
pub struct ModelCache {
    cache_dir: PathBuf,
}

impl ModelCache {
    /// HTTP connect timeout in seconds used when downloading models.
    pub const CONNECT_TIMEOUT_SECS: u64 = 30;
}

impl ModelCache {
    /// Create a new model cache.
    ///
    /// # Returns
    ///
    /// A new `ModelCache` instance with the cache directory set to:
    /// - `$SIGNATURE_SDK_MODEL_CACHE` if set
    /// - `~/.cache/signature-sdk/models/` otherwise
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be determined or created.
    pub fn new() -> Result<Self> {
        let cache_dir = if let Ok(path) = env::var("SIGNATURE_SDK_MODEL_CACHE") {
            PathBuf::from(path)
        } else {
            dirs::cache_dir()
                .ok_or_else(|| SigError::Provider("Cannot determine cache directory".into()))?
                .join("signature-sdk")
                .join("models")
        };

        Ok(Self { cache_dir })
    }

    /// Get the path to a model file, downloading it if necessary.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace model ID (e.g., "intfloat/multilingual-e5-small")
    ///   or local directory path (e.g., "models/v1")
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
        let base_url = "https://huggingface.co";
        self.download_file_from_url(base_url, model_id, filename, dest_path)
            .await
    }

    /// Download a file from a HuggingFace-compatible base URL.
    ///
    /// Separated from [`download_file`] so tests can point at a local mock server
    /// without touching the network.
    ///
    /// Robustness properties (matching Heimdall's production implementation):
    /// 1. **Streaming** — response body is written chunk-by-chunk; the entire
    ///    file is never buffered in memory.
    /// 2. **Connect timeout** — the HTTP client enforces a 30-second TCP
    ///    connect timeout (bounds connection establishment only; the overall
    ///    transfer duration is unbounded by design for large models).
    /// 3. **Race-safe rename** — a unique temp path (`<dest>.tmp.<pid>.<counter>.<nanos>.<tid>`)
    ///    is used so concurrent downloaders do not collide.  If the atomic rename
    ///    fails with `AlreadyExists` another caller already won the race; we clean
    ///    up the temp file and return `Ok(())`.
    pub(crate) async fn download_file_from_url(
        &self,
        base_url: &str,
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

        let url = format!("{}/{}/resolve/main/{}", base_url, model_id, filename);
        debug!("Downloading from: {}", url);

        // Build client with connect timeout (gap 2)
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(Self::CONNECT_TIMEOUT_SECS))
            .build()
            .map_err(|e| SigError::Provider(format!("Failed to create HTTP client: {}", e)))?;

        let response = client
            .get(&url)
            .send()
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

        // Unique temp suffix — appended to the full dest path (including its existing
        // extension) so `model.onnx` becomes `model.onnx.tmp.<pid>.<counter>.<nanos>.<tid>`.
        // Using OsString::push avoids with_extension(), which would silently replace
        // the existing extension (e.g. `.onnx` → `.tmp.…`).
        // The monotonic counter guarantees uniqueness even when SystemTime resolution
        // is coarser than nanoseconds (e.g. some Linux/VM configurations return
        // the same nanosecond timestamp across rapid successive calls).
        let unique_id = format!(
            "{}.{}.{}.{:?}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            std::thread::current().id()
        );
        let temp_path = {
            let mut s = dest_path.as_os_str().to_owned();
            s.push(format!(".tmp.{}", unique_id));
            PathBuf::from(s)
        };

        // Open temp file before starting the download so a mid-download failure
        // (disk full, network drop) can clean up the partial file via the guard below.
        let mut file = fs::File::create(&temp_path)
            .await
            .map_err(|e| SigError::Provider(format!("Failed to create temporary file: {}", e)))?;

        // Stream chunks to disk (gap 1) — never buffers the full body in memory.
        // On failure we clean up the partial temp file before returning.
        let mut stream = response.bytes_stream();
        let mut downloaded: u64 = 0;
        // Log progress each time we cross a 50 MB boundary.
        let progress_interval: u64 = 50 * 1024 * 1024;
        let mut next_log_threshold: u64 = progress_interval;

        let stream_result: Result<()> = async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk
                    .map_err(|e| SigError::Provider(format!("Download interrupted: {}", e)))?;

                file.write_all(&chunk).await.map_err(|e| {
                    SigError::Provider(format!("Failed to write model file: {}", e))
                })?;

                downloaded += chunk.len() as u64;

                // Log every crossed 50 MB boundary. Use a while loop so a single
                // large chunk that spans multiple thresholds logs each one.
                while total_size > 0 && downloaded >= next_log_threshold {
                    info!(
                        "Download progress: {:.0}%",
                        (next_log_threshold as f64 / total_size as f64) * 100.0
                    );
                    next_log_threshold += progress_interval;
                }
            }

            file.sync_all()
                .await
                .map_err(|e| SigError::Provider(format!("Failed to sync model file: {}", e)))?;

            Ok(())
        }
        .await;

        drop(file);

        if let Err(e) = stream_result {
            // Clean up the partial temp file before propagating the error.
            let _ = fs::remove_file(&temp_path).await;
            return Err(e);
        }

        // Atomic rename (gap 3).
        // On POSIX (Linux, macOS) rename(2) atomically replaces the destination, so
        // a concurrent winner's file is safely overwritten with identical bytes.
        // On Windows, rename fails with AlreadyExists if the destination exists; we
        // treat that as a successful race loss and clean up.
        if let Err(e) = fs::rename(&temp_path, dest_path).await {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                info!(
                    "Model already cached by another process at: {}",
                    dest_path.display()
                );
                let _ = fs::remove_file(&temp_path).await;
                return Ok(());
            }
            let _ = fs::remove_file(&temp_path).await;
            return Err(SigError::Provider(format!(
                "Failed to finalize model file: {}",
                e
            )));
        }

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
        assert!(cache
            .cache_dir()
            .to_string_lossy()
            .contains("signature-sdk"));
    }

    #[test]
    fn test_cache_dir() {
        let cache = ModelCache::new().unwrap();
        let cache_dir = cache.cache_dir();
        assert!(
            cache_dir.ends_with("signature-sdk/models")
                || cache_dir.ends_with("signature-sdk\\models")
        );
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

    // --- Gap 1: streaming download (chunks, not buffered) ---
    // Verifies that download_file_from_url correctly writes the response body
    // to disk and returns the expected bytes. Mockito serves a single response
    // body; the streaming code paths (bytes_stream + write_all loop) are
    // exercised regardless of how many TCP chunks the body arrives in.
    #[tokio::test]
    async fn test_download_streams_chunks_to_disk() {
        use mockito::Server;
        use tempfile::TempDir;

        let mut server = Server::new_async().await;
        let body = b"chunk-one-chunk-two";
        let mock = server
            .mock("GET", "/org/repo/resolve/main/model.onnx")
            .with_status(200)
            .with_header("content-type", "application/octet-stream")
            .with_body(body)
            .create_async()
            .await;

        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache {
            cache_dir: temp_dir.path().to_path_buf(),
        };
        let dest = temp_dir.path().join("model.onnx");

        // Call the internal helper directly via get_model by pointing at a fake HF URL.
        // We override via a method that accepts an explicit URL for testability.
        cache
            .download_file_from_url(&server.url(), "org/repo", "model.onnx", &dest)
            .await
            .unwrap();

        mock.assert_async().await;
        let written = tokio::fs::read(&dest).await.unwrap();
        assert_eq!(written, body);
    }

    // --- Gap 2: connect timeout ---
    // Behaviorally testing connect_timeout requires a listener that accepts the
    // TCP handshake but never sends bytes, plus waiting 30+ seconds — impractical
    // for a unit test. We verify the constant value matches the spec so any
    // future change to the timeout is visible at review time.
    #[test]
    fn test_connect_timeout_constant_is_thirty_seconds() {
        assert_eq!(ModelCache::CONNECT_TIMEOUT_SECS, 30);
    }

    // --- Gap 3: race-safe rename ---
    // Simulates two concurrent downloaders writing the same destination file.
    // On POSIX (Linux, macOS) rename(2) atomically replaces the destination, so
    // both calls succeed via the overwrite path. On Windows the second rename
    // would return AlreadyExists and be handled gracefully. Either way both
    // callers return Ok(()) and no temp files are left behind.
    #[tokio::test]
    async fn test_concurrent_download_both_succeed() {
        use mockito::Server;
        use tempfile::TempDir;

        let mut server = Server::new_async().await;
        let body = b"model-bytes";
        // Keep the Mock alive for the duration of the test — dropping it
        // de-registers it from mockito and would cause both requests to fail.
        let _mock = server
            .mock("GET", "/org/repo/resolve/main/model.onnx")
            .with_status(200)
            .with_body(body)
            .expect(2)
            .create_async()
            .await;

        let temp_dir = TempDir::new().unwrap();
        let cache = std::sync::Arc::new(ModelCache {
            cache_dir: temp_dir.path().to_path_buf(),
        });
        let dest = temp_dir.path().join("model.onnx");

        let base_url = server.url();
        let (r1, r2) = tokio::join!(
            {
                let c = cache.clone();
                let d = dest.clone();
                let u = base_url.clone();
                async move {
                    c.download_file_from_url(&u, "org/repo", "model.onnx", &d)
                        .await
                }
            },
            {
                let c = cache.clone();
                let d = dest.clone();
                let u = base_url.clone();
                async move {
                    c.download_file_from_url(&u, "org/repo", "model.onnx", &d)
                        .await
                }
            },
        );

        assert!(r1.is_ok(), "first downloader failed: {:?}", r1);
        assert!(r2.is_ok(), "second downloader failed: {:?}", r2);

        // Final file should contain valid content (one winner's write)
        let written = tokio::fs::read(&dest).await.unwrap();
        assert_eq!(written, body);

        // No leftover temp files
        let mut entries = tokio::fs::read_dir(temp_dir.path()).await.unwrap();
        let mut names = vec![];
        while let Some(e) = entries.next_entry().await.unwrap() {
            names.push(e.file_name().to_string_lossy().to_string());
        }
        let tmp_files: Vec<_> = names.iter().filter(|n| n.contains(".tmp.")).collect();
        assert!(tmp_files.is_empty(), "leftover temp files: {:?}", tmp_files);
    }
}
