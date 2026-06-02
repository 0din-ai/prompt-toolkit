//! Cross-language validation tests using the shared fixture.

#[cfg(feature = "threatfeed")]
mod tests {
    use odin_prompt_toolkit::threatfeed::cache::{compute_bands, ThreatFeedCache};
    use odin_prompt_toolkit::threatfeed::types::CachedSignature;
    use odin_prompt_toolkit::types::SignatureVersion;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        expected_v1_cache: ExpectedCache,
        query_tests: QueryTests,
    }

    #[derive(Deserialize)]
    struct ExpectedCache {
        entry_count: usize,
        entries: Vec<FixtureEntry>,
    }

    #[derive(Deserialize)]
    struct FixtureEntry {
        uuid: String,
        title: String,
        severity: String,
        security_boundary: String,
        signature: String,
        bands: Vec<String>,
    }

    #[derive(Deserialize)]
    struct QueryTests {
        tests: Vec<QueryTest>,
    }

    #[derive(Deserialize)]
    struct QueryTest {
        name: String,
        query_signature: String,
        #[serde(default = "default_threshold")]
        threshold: f64,
        expected_match_uuids: Vec<String>,
        expected_top_match_uuid: Option<String>,
        expected_top_hamming_distance: Option<usize>,
        expected_top_cosine_similarity: Option<f64>,
    }

    fn default_threshold() -> f64 {
        0.85
    }

    fn load_fixture() -> Fixture {
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../spec/test-vectors/threatfeed-fixture.json");
        let content = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|e| panic!("Failed to read fixture at {:?}: {}", fixture_path, e));
        serde_json::from_str(&content).expect("Failed to parse fixture JSON")
    }

    fn build_v1_cache(fixture: &Fixture) -> ThreatFeedCache {
        let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
        let entries: Vec<CachedSignature> = fixture
            .expected_v1_cache
            .entries
            .iter()
            .map(|e| CachedSignature {
                uuid: e.uuid.clone(),
                title: e.title.clone(),
                severity: e.severity.clone(),
                security_boundary: e.security_boundary.clone(),
                signature: e.signature.clone(),
                bands: e.bands.clone(),
                updated_at: None,
            })
            .collect();
        cache.load_entries(entries);
        cache
    }

    #[test]
    fn test_bands_match_fixture() {
        let fixture = load_fixture();
        for entry in &fixture.expected_v1_cache.entries {
            let computed = compute_bands(&entry.signature, 16);
            assert_eq!(computed, entry.bands, "Band mismatch for {}", entry.uuid);
        }
    }

    #[test]
    fn test_v1_entry_count() {
        let fixture = load_fixture();
        assert_eq!(fixture.expected_v1_cache.entry_count, 6);
    }

    #[test]
    fn test_v1_excludes_no_signature_entries() {
        let fixture = load_fixture();
        let uuids: Vec<&str> = fixture
            .expected_v1_cache
            .entries
            .iter()
            .map(|e| e.uuid.as_str())
            .collect();
        assert!(!uuids.contains(&"dddddddd-dddd-dddd-dddd-dddddddddddd"));
    }

    #[test]
    fn test_v1_excludes_v0_only_entries() {
        let fixture = load_fixture();
        let uuids: Vec<&str> = fixture
            .expected_v1_cache
            .entries
            .iter()
            .map(|e| e.uuid.as_str())
            .collect();
        assert!(!uuids.contains(&"11111111-1111-1111-1111-111111111111"));
    }

    #[test]
    fn test_v1_includes_dual_version_entry() {
        let fixture = load_fixture();
        let uuids: Vec<&str> = fixture
            .expected_v1_cache
            .entries
            .iter()
            .map(|e| e.uuid.as_str())
            .collect();
        assert!(uuids.contains(&"22222222-2222-2222-2222-222222222222"));
    }

    #[test]
    fn test_dual_version_uses_v1_signature() {
        let fixture = load_fixture();
        let dual = fixture
            .expected_v1_cache
            .entries
            .iter()
            .find(|e| e.uuid == "22222222-2222-2222-2222-222222222222")
            .expect("Dual version entry not found");
        assert_eq!(
            dual.signature,
            "4444444444444444444444444444444444444444444444444444444444444444"
        );
    }

    #[test]
    fn test_query_exact_match() {
        let fixture = load_fixture();
        let cache = build_v1_cache(&fixture);
        let test = &fixture.query_tests.tests[0];
        assert_eq!(test.name, "exact_match");

        let matches = cache.query(&test.query_signature, test.threshold, 10);
        let match_uuids: Vec<&str> = matches.iter().map(|m| m.uuid.as_str()).collect();

        for expected_uuid in &test.expected_match_uuids {
            assert!(
                match_uuids.contains(&expected_uuid.as_str()),
                "Missing expected match: {}",
                expected_uuid
            );
        }

        if let Some(ref top_uuid) = test.expected_top_match_uuid {
            assert_eq!(matches[0].uuid, *top_uuid);
        }
        if let Some(top_dist) = test.expected_top_hamming_distance {
            assert_eq!(matches[0].hamming_distance, top_dist);
        }
        if let Some(top_cosine) = test.expected_top_cosine_similarity {
            assert!((matches[0].cosine_similarity - top_cosine).abs() < 1e-6);
        }
    }

    #[test]
    fn test_query_near_match() {
        let fixture = load_fixture();
        let cache = build_v1_cache(&fixture);
        let test = &fixture.query_tests.tests[1];
        assert_eq!(test.name, "near_match");

        let matches = cache.query(&test.query_signature, test.threshold, 10);
        let match_uuids: Vec<&str> = matches.iter().map(|m| m.uuid.as_str()).collect();

        for expected_uuid in &test.expected_match_uuids {
            assert!(
                match_uuids.contains(&expected_uuid.as_str()),
                "Missing expected match: {}",
                expected_uuid
            );
        }

        if let Some(ref top_uuid) = test.expected_top_match_uuid {
            assert_eq!(matches[0].uuid, *top_uuid);
        }
    }

    #[test]
    fn test_query_no_match() {
        let fixture = load_fixture();
        let cache = build_v1_cache(&fixture);
        let test = &fixture.query_tests.tests[2];
        assert_eq!(test.name, "no_match");

        let matches = cache.query(&test.query_signature, 0.85, 10);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_query_all_zeros_exact() {
        let fixture = load_fixture();
        let cache = build_v1_cache(&fixture);
        let test = &fixture.query_tests.tests[3];
        assert_eq!(test.name, "all_zeros_exact");

        let matches = cache.query(&test.query_signature, test.threshold, 10);
        let match_uuids: Vec<&str> = matches.iter().map(|m| m.uuid.as_str()).collect();

        for expected_uuid in &test.expected_match_uuids {
            assert!(
                match_uuids.contains(&expected_uuid.as_str()),
                "Missing expected match: {}",
                expected_uuid
            );
        }

        if let Some(ref top_uuid) = test.expected_top_match_uuid {
            assert_eq!(matches[0].uuid, *top_uuid);
        }
        if let Some(top_dist) = test.expected_top_hamming_distance {
            assert_eq!(matches[0].hamming_distance, top_dist);
        }
    }
}
