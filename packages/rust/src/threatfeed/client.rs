//! Threat feed API client for fetching signatures from the 0din portal.

use crate::error::{Result, SigError};

use super::types::{ThreatFeedEntry, ThreatFeedResponse};

/// Client for the 0din threat feed API.
///
/// Fetches detection signatures from the paginated threat feed endpoint.
///
/// Token resolution order:
/// 1. Explicit `api_token` parameter
/// 2. `ODIN_THREATFEED_API_TOKEN` env var (dedicated)
/// 3. `ODIN_API_TOKEN` env var (shared with Thor / portal)
pub struct ThreatFeedClient {
    api_token: String,
    base_url: String,
    per_page: usize,
    client: reqwest::Client,
}

impl ThreatFeedClient {
    /// Create a new threat feed client.
    ///
    /// # Arguments
    ///
    /// * `api_token` - Optional raw API token (no Bearer prefix). Falls back to
    ///   `ODIN_THREATFEED_API_TOKEN`, then `ODIN_API_TOKEN` env vars.
    /// * `base_url` - Optional base URL override (default: `https://0din.ai`)
    /// * `per_page` - Optional page size override (default: 100)
    ///
    /// # Errors
    ///
    /// Returns `SigError::ThreatFeedApi` if no API token is found.
    pub fn new(
        api_token: Option<&str>,
        base_url: Option<&str>,
        per_page: Option<usize>,
    ) -> Result<Self> {
        let api_token = api_token
            .map(String::from)
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("ODIN_THREATFEED_API_TOKEN").ok().filter(|s| !s.is_empty()))
            .or_else(|| std::env::var("ODIN_API_TOKEN").ok().filter(|s| !s.is_empty()))
            .ok_or_else(|| {
                SigError::ThreatFeedApi(
                    "API token required: pass api_token or set \
                     ODIN_THREATFEED_API_TOKEN / ODIN_API_TOKEN"
                        .to_string(),
                )
            })?;

        let base_url = base_url
            .map(String::from)
            .or_else(|| std::env::var("ODIN_THREATFEED_BASE_URL").ok())
            .unwrap_or_else(|| String::from("https://0din.ai"));

        Ok(Self {
            api_token,
            base_url,
            per_page: per_page.unwrap_or(100),
            client: reqwest::Client::new(),
        })
    }

    /// Get the base URL of the API.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Fetch all threat feed entries, paginating through all pages.
    ///
    /// # Arguments
    ///
    /// * `since` - Optional ISO8601 timestamp to filter entries updated since this time
    pub async fn fetch_all(&self, since: Option<&str>) -> Result<Vec<ThreatFeedEntry>> {
        let mut all_entries = Vec::new();
        let mut page = 1;

        loop {
            let response = self.fetch_page(page, since).await?;
            all_entries.extend(response.threat_feeds);

            if page >= response.total_pages {
                break;
            }
            page += 1;

            // Rate limiting: 500ms delay between pages
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }

        Ok(all_entries)
    }

    /// Fetch a single threat feed entry by UUID.
    pub async fn fetch_one(&self, uuid: &str) -> Result<ThreatFeedEntry> {
        let url = format!("{}/api/v1/threatfeed/{}", self.base_url, uuid);

        let response = self
            .client
            .get(&url)
            .header("Authorization", &self.api_token)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| SigError::ThreatFeedApi(format!("Network error: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Err(SigError::ThreatFeedApi(format!(
                "API returned status {}: {}",
                status,
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown".to_string())
            )));
        }

        let entry: ThreatFeedEntry = response.json().await.map_err(|e| {
            SigError::ThreatFeedApi(format!("Failed to parse response: {}", e))
        })?;

        Ok(entry)
    }

    // --- Private methods ---

    async fn fetch_page(
        &self,
        page: usize,
        since: Option<&str>,
    ) -> Result<ThreatFeedResponse> {
        let mut url = format!(
            "{}/api/v1/threatfeed?page={}&per_page={}",
            self.base_url, page, self.per_page
        );

        if let Some(since) = since {
            url.push_str(&format!(
                "&q[updated_at_gteq]={}",
                urlencoding::encode(since)
            ));
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", &self.api_token)
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| SigError::ThreatFeedApi(format!("Network error: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            return Err(SigError::ThreatFeedApi(format!(
                "API returned status {}: {}",
                status,
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "unknown".to_string())
            )));
        }

        let data: ThreatFeedResponse = response.json().await.map_err(|e| {
            SigError::ThreatFeedApi(format!("Failed to parse response: {}", e))
        })?;

        Ok(data)
    }
}
