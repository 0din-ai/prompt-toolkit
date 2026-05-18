//! Threat feed cache with band-indexed similarity lookup.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::error::{Result, SigError};
use crate::lsh::{cosine_from_hamming, hamming_distance_hex};
use crate::types::SignatureVersion;

use super::client::ThreatFeedClient;
use super::types::{CachedSignature, SyncResult, ThreatMatch};

/// Schema version for the cache file format.
const CACHE_SCHEMA_VERSION: u32 = 1;

/// Default number of bands for LSH indexing.
const DEFAULT_BANDS: usize = 16;

/// Default number of bits per signature.
const DEFAULT_BITS: usize = 256;

/// On-disk cache format.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CacheFile {
    schema_version: u32,
    signature_version: String,
    synced_at: String,
    source_url: String,
    entry_count: usize,
    lsh_config: CacheLshConfig,
    entries: Vec<CachedSignature>,
    band_index: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CacheLshConfig {
    bits: usize,
    bands: usize,
}

/// Threat feed cache with band-indexed similarity lookup.
///
/// Caches detection signatures from the 0din threat feed API and provides
/// fast similarity queries using LSH band indexing.
pub struct ThreatFeedCache {
    version: SignatureVersion,
    cache_dir: PathBuf,
    bits: usize,
    bands: usize,
    entries: Vec<CachedSignature>,
    band_index: HashMap<String, Vec<usize>>,
    synced_at: Option<String>,
    source_url: String,
}

impl ThreatFeedCache {
    /// Create a new threat feed cache.
    ///
    /// # Arguments
    ///
    /// * `version` - Signature version to cache (V0 or V1)
    /// * `cache_dir` - Override cache directory (default: `~/.odin-prompt-toolkit/threatfeed/`)
    /// * `bands` - Number of bands for LSH indexing (default: 16)
    pub fn new(
        version: SignatureVersion,
        cache_dir: Option<PathBuf>,
        bands: Option<usize>,
    ) -> Self {
        let cache_dir = cache_dir
            .or_else(|| {
                std::env::var("ODIN_PROMPT_TOOLKIT_THREATFEED_CACHE")
                    .ok()
                    .map(PathBuf::from)
            })
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".odin-prompt-toolkit")
                    .join("threatfeed")
            });

        Self {
            version,
            cache_dir,
            bits: DEFAULT_BITS,
            bands: bands.unwrap_or(DEFAULT_BANDS),
            entries: Vec::new(),
            band_index: HashMap::new(),
            synced_at: None,
            source_url: String::from("https://0din.ai"),
        }
    }

    /// Load cache from disk.
    ///
    /// Returns `true` if cache was loaded successfully, `false` if no cache exists.
    pub fn load(&mut self) -> Result<bool> {
        let path = self.cache_file_path();
        if !path.exists() {
            return Ok(false);
        }

        let content = std::fs::read_to_string(&path).map_err(|e| {
            SigError::ThreatFeedCache(format!("Failed to read cache file: {}", e))
        })?;

        let cache_file: CacheFile = serde_json::from_str(&content).map_err(|e| {
            tracing::warn!("Corrupt cache file, will be discarded: {}", e);
            SigError::ThreatFeedCache(format!("Corrupt cache file: {}", e))
        })?;

        if cache_file.schema_version != CACHE_SCHEMA_VERSION {
            tracing::warn!(
                "Cache schema version mismatch (got {}, expected {}), discarding",
                cache_file.schema_version,
                CACHE_SCHEMA_VERSION
            );
            return Ok(false);
        }

        self.entries = cache_file.entries;
        self.band_index = cache_file.band_index;
        self.synced_at = Some(cache_file.synced_at);
        self.source_url = cache_file.source_url;

        Ok(true)
    }

    /// Save cache to disk with atomic write (temp file + rename).
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir).map_err(|e| {
            SigError::ThreatFeedCache(format!("Failed to create cache directory: {}", e))
        })?;

        let cache_file = CacheFile {
            schema_version: CACHE_SCHEMA_VERSION,
            signature_version: self.version.resolve().to_string(),
            synced_at: self
                .synced_at
                .clone()
                .unwrap_or_else(|| chrono::Utc::now().to_rfc3339()),
            source_url: self.source_url.clone(),
            entry_count: self.entries.len(),
            lsh_config: CacheLshConfig {
                bits: self.bits,
                bands: self.bands,
            },
            entries: self.entries.clone(),
            band_index: self.band_index.clone(),
        };

        let json = serde_json::to_string_pretty(&cache_file).map_err(|e| {
            SigError::ThreatFeedCache(format!("Failed to serialize cache: {}", e))
        })?;

        // Atomic write: temp file + rename
        let path = self.cache_file_path();
        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, &json).map_err(|e| {
            SigError::ThreatFeedCache(format!("Failed to write temp cache file: {}", e))
        })?;
        std::fs::rename(&tmp_path, &path).map_err(|e| {
            SigError::ThreatFeedCache(format!("Failed to rename temp cache file: {}", e))
        })?;

        Ok(())
    }

    /// Sync signatures from the threat feed API.
    ///
    /// # Arguments
    ///
    /// * `client` - Threat feed API client
    /// * `full` - If true, fetch all entries; if false, fetch only entries updated since last sync
    pub async fn sync(&mut self, client: &ThreatFeedClient, full: bool) -> Result<SyncResult> {
        let since = if full {
            None
        } else {
            self.last_updated_at()
        };

        self.source_url = client.base_url().to_string();

        let entries = client.fetch_all(since.as_deref()).await?;
        let version_str = self.version.resolve().to_string();

        let mut new_cached: Vec<CachedSignature> = Vec::new();
        for entry in &entries {
            for sig in &entry.detection_signatures {
                if sig.version == version_str {
                    new_cached.push(CachedSignature {
                        uuid: entry.uuid.clone(),
                        title: entry.title.clone(),
                        severity: entry.severity.clone(),
                        security_boundary: entry.security_boundary.clone(),
                        signature: sig.signature.clone(),
                        bands: compute_bands(&sig.signature, self.bands),
                        updated_at: entry.updated_at.clone(),
                    });
                }
            }
        }

        let result = if full {
            let total = new_cached.len();
            self.entries = new_cached;
            SyncResult {
                added: total,
                updated: 0,
                total,
            }
        } else {
            self.merge_entries(new_cached)
        };

        self.rebuild_band_index();
        self.synced_at = Some(chrono::Utc::now().to_rfc3339());
        self.save()?;

        Ok(result)
    }

    /// Query the cache for signatures similar to the given query.
    ///
    /// Uses band-indexed candidate selection followed by Hamming distance verification.
    ///
    /// # Arguments
    ///
    /// * `signature` - 64 hex char signature to query (raw, no `0din-` prefix)
    /// * `threshold` - Minimum cosine similarity threshold (default: 0.85)
    /// * `max_results` - Maximum number of results to return (default: 10)
    pub fn query(
        &self,
        signature: &str,
        threshold: f64,
        max_results: usize,
    ) -> Vec<ThreatMatch> {
        let query_bands = compute_bands(signature, self.bands);

        // Collect candidate indices from band index
        let mut candidate_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        for (band_idx, band_val) in query_bands.iter().enumerate() {
            let key = format!("{}:{}", band_idx, band_val);
            if let Some(indices) = self.band_index.get(&key) {
                candidate_indices.extend(indices);
            }
        }

        // Verify candidates with Hamming distance
        let mut matches: Vec<ThreatMatch> = Vec::new();
        for &idx in &candidate_indices {
            if let Some(entry) = self.entries.get(idx) {
                let dist = hamming_distance_hex(signature, &entry.signature);
                let cosine = cosine_from_hamming(dist, self.bits);
                if cosine >= threshold {
                    matches.push(ThreatMatch {
                        uuid: entry.uuid.clone(),
                        title: entry.title.clone(),
                        severity: entry.severity.clone(),
                        security_boundary: entry.security_boundary.clone(),
                        signature: entry.signature.clone(),
                        hamming_distance: dist,
                        cosine_similarity: cosine,
                    });
                }
            }
        }

        // Sort by cosine similarity descending
        matches.sort_by(|a, b| {
            b.cosine_similarity
                .partial_cmp(&a.cosine_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(max_results);
        matches
    }

    /// Get the number of entries in the cache.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Get the timestamp of the last sync.
    pub fn last_synced(&self) -> Option<&str> {
        self.synced_at.as_deref()
    }

    /// Get all cached entries (for testing and inspection).
    pub fn entries(&self) -> &[CachedSignature] {
        &self.entries
    }

    /// Load entries directly (for testing without disk I/O).
    pub fn load_entries(&mut self, entries: Vec<CachedSignature>) {
        self.entries = entries;
        self.rebuild_band_index();
    }

    // --- Private methods ---

    fn cache_file_path(&self) -> PathBuf {
        let version = self.version.resolve().to_string();
        self.cache_dir.join(format!("cache-{}.json", version))
    }

    fn last_updated_at(&self) -> Option<String> {
        self.entries
            .iter()
            .filter_map(|e| e.updated_at.as_ref())
            .max()
            .cloned()
    }

    fn merge_entries(&mut self, new_entries: Vec<CachedSignature>) -> SyncResult {
        let mut existing: HashMap<String, usize> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.uuid.clone(), i))
            .collect();

        let mut added = 0;
        let mut updated = 0;

        for entry in new_entries {
            if let Some(&idx) = existing.get(&entry.uuid) {
                self.entries[idx] = entry;
                updated += 1;
            } else {
                existing.insert(entry.uuid.clone(), self.entries.len());
                self.entries.push(entry);
                added += 1;
            }
        }

        SyncResult {
            added,
            updated,
            total: self.entries.len(),
        }
    }

    fn rebuild_band_index(&mut self) {
        self.band_index.clear();
        for (idx, entry) in self.entries.iter().enumerate() {
            for (band_idx, band_val) in entry.bands.iter().enumerate() {
                let key = format!("{}:{}", band_idx, band_val);
                self.band_index.entry(key).or_default().push(idx);
            }
        }
    }
}

/// Compute bands from a hex signature string.
///
/// Splits a 64 hex char signature into `num_bands` equal-length bands.
pub fn compute_bands(signature: &str, num_bands: usize) -> Vec<String> {
    let band_len = signature.len() / num_bands;
    (0..num_bands)
        .map(|i| signature[i * band_len..(i + 1) * band_len].to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bands() {
        let sig = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
        let bands = compute_bands(sig, 16);
        assert_eq!(bands.len(), 16);
        assert_eq!(bands[0], "a1b2");
        assert_eq!(bands[1], "c3d4");
        assert_eq!(bands[15], "a1b2");
    }

    #[test]
    fn test_compute_bands_all_zeros() {
        let sig = "0000000000000000000000000000000000000000000000000000000000000000";
        let bands = compute_bands(sig, 16);
        for band in &bands {
            assert_eq!(band, "0000");
        }
    }

    #[test]
    fn test_cache_load_entries_and_query_exact() {
        let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
        let sig = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";

        cache.load_entries(vec![CachedSignature {
            uuid: "test-uuid".to_string(),
            title: "Test".to_string(),
            severity: "high".to_string(),
            security_boundary: "guardrail_jailbreak".to_string(),
            signature: sig.to_string(),
            bands: compute_bands(sig, 16),
            updated_at: None,
        }]);

        let matches = cache.query(sig, 0.85, 10);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].uuid, "test-uuid");
        assert_eq!(matches[0].hamming_distance, 0);
        assert!((matches[0].cosine_similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cache_query_no_match() {
        let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
        let sig = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";

        cache.load_entries(vec![CachedSignature {
            uuid: "test-uuid".to_string(),
            title: "Test".to_string(),
            severity: "high".to_string(),
            security_boundary: "guardrail_jailbreak".to_string(),
            signature: sig.to_string(),
            bands: compute_bands(sig, 16),
            updated_at: None,
        }]);

        // Query with completely different signature (no shared bands)
        let query = "5678901234567890567890123456789056789012345678905678901234567890";
        let matches = cache.query(query, 0.85, 10);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_cache_query_near_match() {
        let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
        let sig_a = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
        let sig_b = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b3";

        cache.load_entries(vec![
            CachedSignature {
                uuid: "entry-a".to_string(),
                title: "Entry A".to_string(),
                severity: "high".to_string(),
                security_boundary: "guardrail_jailbreak".to_string(),
                signature: sig_a.to_string(),
                bands: compute_bands(sig_a, 16),
                updated_at: None,
            },
            CachedSignature {
                uuid: "entry-b".to_string(),
                title: "Entry B".to_string(),
                severity: "medium".to_string(),
                security_boundary: "prompt_extraction".to_string(),
                signature: sig_b.to_string(),
                bands: compute_bands(sig_b, 16),
                updated_at: None,
            },
        ]);

        // Query with sig_a — should match both (sig_b differs by 1 bit)
        let matches = cache.query(sig_a, 0.85, 10);
        assert_eq!(matches.len(), 2);
        // Exact match should be first
        assert_eq!(matches[0].uuid, "entry-a");
        assert_eq!(matches[0].hamming_distance, 0);
        // Near match second
        assert_eq!(matches[1].uuid, "entry-b");
        assert!(matches[1].cosine_similarity > 0.99);
    }

    #[test]
    fn test_cache_save_and_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let cache_dir = tmp.path().to_path_buf();

        let sig = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";

        let mut cache = ThreatFeedCache::new(SignatureVersion::V1, Some(cache_dir.clone()), None);
        cache.load_entries(vec![CachedSignature {
            uuid: "test-uuid".to_string(),
            title: "Test Entry".to_string(),
            severity: "high".to_string(),
            security_boundary: "guardrail_jailbreak".to_string(),
            signature: sig.to_string(),
            bands: compute_bands(sig, 16),
            updated_at: Some("2025-03-01T10:00:00.000Z".to_string()),
        }]);
        cache.save().unwrap();

        // Load into a new cache
        let mut cache2 = ThreatFeedCache::new(SignatureVersion::V1, Some(cache_dir), None);
        let loaded = cache2.load().unwrap();
        assert!(loaded);
        assert_eq!(cache2.entry_count(), 1);

        // Query should still work after reload
        let matches = cache2.query(sig, 0.85, 10);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].uuid, "test-uuid");
    }

    #[test]
    fn test_cache_empty_query() {
        let cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
        let query = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
        let matches = cache.query(query, 0.85, 10);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_cache_load_nonexistent() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cache =
            ThreatFeedCache::new(SignatureVersion::V1, Some(tmp.path().to_path_buf()), None);
        let loaded = cache.load().unwrap();
        assert!(!loaded);
    }

    #[test]
    fn test_merge_entries() {
        let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
        let sig1 = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
        let sig2 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

        cache.load_entries(vec![CachedSignature {
            uuid: "existing".to_string(),
            title: "Existing".to_string(),
            severity: "low".to_string(),
            security_boundary: "guardrail_jailbreak".to_string(),
            signature: sig1.to_string(),
            bands: compute_bands(sig1, 16),
            updated_at: Some("2025-01-01T00:00:00Z".to_string()),
        }]);

        let result = cache.merge_entries(vec![
            CachedSignature {
                uuid: "existing".to_string(),
                title: "Updated Existing".to_string(),
                severity: "high".to_string(),
                security_boundary: "guardrail_jailbreak".to_string(),
                signature: sig1.to_string(),
                bands: compute_bands(sig1, 16),
                updated_at: Some("2025-02-01T00:00:00Z".to_string()),
            },
            CachedSignature {
                uuid: "new-entry".to_string(),
                title: "New Entry".to_string(),
                severity: "medium".to_string(),
                security_boundary: "prompt_extraction".to_string(),
                signature: sig2.to_string(),
                bands: compute_bands(sig2, 16),
                updated_at: Some("2025-02-01T00:00:00Z".to_string()),
            },
        ]);

        assert_eq!(result.added, 1);
        assert_eq!(result.updated, 1);
        assert_eq!(result.total, 2);
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.entries[0].title, "Updated Existing");
        assert_eq!(cache.entries[0].severity, "high");
    }

    #[test]
    fn test_cache_threshold_filtering() {
        let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
        let sig = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

        cache.load_entries(vec![CachedSignature {
            uuid: "test".to_string(),
            title: "Test".to_string(),
            severity: "high".to_string(),
            security_boundary: "guardrail_jailbreak".to_string(),
            signature: sig.to_string(),
            bands: compute_bands(sig, 16),
            updated_at: None,
        }]);

        // All zeros vs all ones = maximum Hamming distance (256 bits)
        // cosine_from_hamming(256, 256) = cos(pi) = -1.0
        let query = "0000000000000000000000000000000000000000000000000000000000000000";
        // But they DO share bands? No — "ffff" != "0000", so no candidates at all
        let matches = cache.query(query, 0.85, 10);
        assert!(matches.is_empty());
    }
}
