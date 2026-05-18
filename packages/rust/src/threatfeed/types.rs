//! Type definitions for threat feed operations.

use serde::{Deserialize, Serialize};

/// A detection signature from the threat feed API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionSignature {
    pub version: String,
    pub signature: String,
}

/// A single threat feed entry from the API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatFeedEntry {
    pub uuid: String,
    pub title: String,
    #[serde(default)]
    pub summary: Option<String>,
    pub severity: String,
    pub security_boundary: String,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub detection_signatures: Vec<DetectionSignature>,
}

/// Paginated API response from GET /api/v1/threatfeed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatFeedResponse {
    pub page: usize,
    pub total_pages: usize,
    pub total_count: usize,
    pub threat_feeds: Vec<ThreatFeedEntry>,
}

/// A cached signature entry with pre-computed bands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSignature {
    pub uuid: String,
    pub title: String,
    pub severity: String,
    pub security_boundary: String,
    pub signature: String,
    pub bands: Vec<String>,
    pub updated_at: Option<String>,
}

/// Result of a threat feed sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    pub added: usize,
    pub updated: usize,
    pub total: usize,
}

/// A match found when querying the threat feed cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatMatch {
    pub uuid: String,
    pub title: String,
    pub severity: String,
    pub security_boundary: String,
    pub signature: String,
    pub hamming_distance: usize,
    pub cosine_similarity: f64,
}

impl std::fmt::Display for ThreatMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}, {}) - cosine: {:.4}, hamming: {}",
            self.title, self.severity, self.security_boundary, self.cosine_similarity,
            self.hamming_distance
        )
    }
}
