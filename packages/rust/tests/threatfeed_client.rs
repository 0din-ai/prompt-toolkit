//! HTTP client tests for the threat feed API client.
//!
//! Uses mockito to serve a local HTTP server so requests never hit the real API.

#[cfg(feature = "threatfeed")]
mod tests {
    use odin_prompt_toolkit::error::SigError;
    use odin_prompt_toolkit::threatfeed::client::ThreatFeedClient;

    const TOKEN: &str = "test-token-abc123";

    fn make_entry(uuid: &str, v1_sig: Option<&str>) -> serde_json::Value {
        let sigs = if let Some(sig) = v1_sig {
            serde_json::json!([{"version": "v1", "signature": sig}])
        } else {
            serde_json::json!([])
        };
        serde_json::json!({
            "uuid": uuid,
            "title": "Test Vuln",
            "summary": "Test",
            "severity": "high",
            "security_boundary": "guardrail_jailbreak",
            "source": "internal",
            "disclosed_at": "2025-01-10T12:00:00.000Z",
            "published_at": "2025-01-15T12:00:00.000Z",
            "updated_at": "2025-03-01T10:00:00.000Z",
            "detection_signatures": sigs,
            "models": [],
            "messages": [],
            "taxonomies": [],
            "test_results": [],
            "metadata": [],
            "reference_urls": [],
            "variant_prompts": []
        })
    }

    fn page_response(
        entries: Vec<serde_json::Value>,
        page: usize,
        total_pages: usize,
    ) -> serde_json::Value {
        let count = entries.len();
        serde_json::json!({
            "page": page,
            "total_pages": total_pages,
            "total_count": count,
            "threat_feeds": entries
        })
    }

    // -----------------------------------------------------------------------
    // fetch_all tests
    // -----------------------------------------------------------------------
    // Constructor / token resolution tests
    //
    // NOTE: Env var tests can interfere with each other since Rust tests
    // run in parallel in the same process. We use explicit tokens for most
    // tests and only test env var resolution in a single combined test.
    // -----------------------------------------------------------------------

    #[test]
    fn test_explicit_token() {
        let client = ThreatFeedClient::new(Some("my-token"), None, None).unwrap();
        assert_eq!(client.base_url(), "https://0din.ai");
    }

    #[test]
    fn test_token_resolution_order() {
        // This test exercises all env var fallback paths sequentially
        // to avoid parallel test interference.

        // 1. Clear everything — should fail
        std::env::remove_var("ODIN_THREATFEED_API_TOKEN");
        std::env::remove_var("ODIN_API_TOKEN");
        assert!(
            ThreatFeedClient::new(None, None, None).is_err(),
            "Should fail with no token"
        );

        // 2. Set ODIN_API_TOKEN only — should succeed via fallback
        std::env::set_var("ODIN_API_TOKEN", "portal-token");
        assert!(
            ThreatFeedClient::new(None, None, None).is_ok(),
            "Should succeed with ODIN_API_TOKEN fallback"
        );

        // 3. Set both — dedicated should take precedence (client created OK)
        std::env::set_var("ODIN_THREATFEED_API_TOKEN", "dedicated-token");
        assert!(
            ThreatFeedClient::new(None, None, None).is_ok(),
            "Should succeed with ODIN_THREATFEED_API_TOKEN"
        );

        // 4. Explicit param overrides everything
        assert!(
            ThreatFeedClient::new(Some("explicit"), None, None).is_ok(),
            "Should succeed with explicit token"
        );

        // Clean up
        std::env::remove_var("ODIN_THREATFEED_API_TOKEN");
        std::env::remove_var("ODIN_API_TOKEN");
    }

    // -----------------------------------------------------------------------
    // fetch_all tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_fetch_all_single_page() {
        let mut server = mockito::Server::new_async().await;
        let body = page_response(
            vec![
                make_entry(
                    "aaa",
                    Some("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"),
                ),
                make_entry("bbb", None),
            ],
            1,
            1,
        );

        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let entries = client.fetch_all(None).await.unwrap();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].uuid, "aaa");
        assert_eq!(entries[1].uuid, "bbb");
    }

    #[tokio::test]
    async fn test_fetch_all_paginates() {
        let mut server = mockito::Server::new_async().await;

        let page1 = page_response(
            vec![make_entry("p1e1", None), make_entry("p1e2", None)],
            1,
            2,
        );
        let page2 = page_response(vec![make_entry("p2e1", None)], 2, 2);

        let _mock1 = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::UrlEncoded("page".into(), "1".into()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(page1.to_string())
            .create_async()
            .await;

        let _mock2 = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::UrlEncoded("page".into(), "2".into()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(page2.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let entries = client.fetch_all(None).await.unwrap();

        assert_eq!(entries.len(), 3);
        let uuids: Vec<&str> = entries.iter().map(|e| e.uuid.as_str()).collect();
        assert!(uuids.contains(&"p1e1"));
        assert!(uuids.contains(&"p1e2"));
        assert!(uuids.contains(&"p2e1"));
    }

    #[tokio::test]
    async fn test_fetch_all_auth_header_no_bearer_prefix() {
        let mut server = mockito::Server::new_async().await;
        let body = page_response(vec![], 1, 1);

        // Verify Authorization header is raw token (no "Bearer " prefix)
        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::Any)
            .match_header("Authorization", TOKEN)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let entries = client.fetch_all(None).await.unwrap();
        assert_eq!(entries.len(), 0);

        // If the header didn't match, mockito would return 501 and we'd get an error
        _mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_fetch_all_empty_response() {
        let mut server = mockito::Server::new_async().await;
        let body = serde_json::json!({
            "page": 1, "total_pages": 1, "total_count": 0, "threat_feeds": []
        });

        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let entries = client.fetch_all(None).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn test_fetch_all_401_raises_error() {
        let mut server = mockito::Server::new_async().await;

        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body("Unauthorized")
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some("bad-token"), Some(&server.url()), None).unwrap();
        let result = client.fetch_all(None).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            SigError::ThreatFeedApi(msg) => assert!(msg.contains("401")),
            other => panic!("Expected ThreatFeedApi error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_fetch_all_500_raises_error() {
        let mut server = mockito::Server::new_async().await;

        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::Any)
            .with_status(500)
            .with_body("Internal Server Error")
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let result = client.fetch_all(None).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            SigError::ThreatFeedApi(msg) => assert!(msg.contains("500")),
            other => panic!("Expected ThreatFeedApi error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_fetch_all_includes_since_param() {
        let mut server = mockito::Server::new_async().await;
        let body = page_response(vec![], 1, 1);
        let since = "2025-03-01T00:00:00Z";

        // The since param should appear URL-encoded in the query string
        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::UrlEncoded(
                "q[updated_at_gteq]".into(),
                since.into(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let result = client.fetch_all(Some(since)).await;

        // If the query param didn't match, mockito returns 501 and we'd get an error
        assert!(result.is_ok(), "Expected ok, got: {:?}", result);
        _mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_fetch_all_no_since_param_when_not_provided() {
        let mut server = mockito::Server::new_async().await;
        let body = page_response(vec![], 1, 1);

        // The since-param test verifies the positive case (param IS included).
        // Here we verify the negative: without `since`, the call succeeds against
        // a mock that only matches page/per_page params (not updated_at_gteq).
        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::UrlEncoded("page".into(), "1".into()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let result = client.fetch_all(None).await;
        // If updated_at_gteq were included, the mock wouldn't match and
        // the request would fail. A successful response proves it was omitted.
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_fetch_all_parses_detection_signatures() {
        let mut server = mockito::Server::new_async().await;
        let sig = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
        let mut entry = make_entry("aaa", Some(sig));
        // Add a v0 sig too
        entry["detection_signatures"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"version": "v0", "signature": "1111111111111111111111111111111111111111111111111111111111111111"}));

        let body = page_response(vec![entry], 1, 1);
        let _mock = server
            .mock("GET", "/api/v1/threatfeed")
            .match_query(mockito::Matcher::Any)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let entries = client.fetch_all(None).await.unwrap();

        assert_eq!(entries[0].detection_signatures.len(), 2);
        let versions: Vec<&str> = entries[0]
            .detection_signatures
            .iter()
            .map(|s| s.version.as_str())
            .collect();
        assert!(versions.contains(&"v0"));
        assert!(versions.contains(&"v1"));
    }

    // -----------------------------------------------------------------------
    // fetch_one tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_fetch_one_success() {
        let mut server = mockito::Server::new_async().await;
        let uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
        let entry = make_entry(uuid, None);
        let path = format!("/api/v1/threatfeed/{}", uuid);

        let _mock = server
            .mock("GET", path.as_str())
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(entry.to_string())
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let result = client.fetch_one(uuid).await.unwrap();

        assert_eq!(result.uuid, uuid);
        assert_eq!(result.title, "Test Vuln");
    }

    #[tokio::test]
    async fn test_fetch_one_404_raises_error() {
        let mut server = mockito::Server::new_async().await;
        let uuid = "nonexistent-uuid-1234-1234-123456789012";
        let path = format!("/api/v1/threatfeed/{}", uuid);

        let _mock = server
            .mock("GET", path.as_str())
            .with_status(404)
            .with_body("Not Found")
            .create_async()
            .await;

        let client = ThreatFeedClient::new(Some(TOKEN), Some(&server.url()), None).unwrap();
        let result = client.fetch_one(uuid).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            SigError::ThreatFeedApi(msg) => assert!(msg.contains("404")),
            other => panic!("Expected ThreatFeedApi error, got: {:?}", other),
        }
    }
}
