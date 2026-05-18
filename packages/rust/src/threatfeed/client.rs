//! Threat feed API client for fetching signatures from the 0din portal.

use crate::error::{Result, SigError};

use super::types::{ThreatFeedEntry, ThreatFeedResponse};

/// Client for the 0din threat feed API.
///
/// Fetches detection signatures from the paginated threat feed endpoint.
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
    /// * `api_token` - Raw API token (no Bearer prefix)
    /// * `base_url` - Optional base URL override (default: `https://0din.ai`)
    /// * `per_page` - Optional page size override (default: 100)
    pub fn new(api_token: &str, base_url: Option<&str>, per_page: Option<usize>) -> Self {
        let base_url = base_url
            .map(String::from)
            .or_else(|| std::env::var("ODIN_THREATFEED_BASE_URL").ok())
            .unwrap_or_else(|| String::from("https://0din.ai"));

        Self {
            api_token: api_token.to_string(),
            base_url,
            per_page: per_page.unwrap_or(100),
            client: reqwest::Client::new(),
        }
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
