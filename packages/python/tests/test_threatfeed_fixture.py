"""Cross-language validation tests using the shared fixture."""

import json
from pathlib import Path

import pytest

from odin_prompt_toolkit.threatfeed.cache import ThreatFeedCache, compute_bands
from odin_prompt_toolkit.threatfeed.types import CachedSignature
from odin_prompt_toolkit.types import SignatureVersion

FIXTURE_PATH = Path(__file__).parent.parent.parent.parent / "spec" / "test-vectors" / "threatfeed-fixture.json"


@pytest.fixture
def fixture():
    """Load the shared threatfeed fixture."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture
def v1_cache(fixture):
    """Build a v1 cache from the fixture's expected_v1_cache."""
    cache = ThreatFeedCache(version=SignatureVersion.V1)
    expected = fixture["expected_v1_cache"]
    entries = [
        CachedSignature(
            uuid=e["uuid"],
            title=e["title"],
            severity=e["severity"],
            security_boundary=e["security_boundary"],
            signature=e["signature"],
            bands=e["bands"],
        )
        for e in expected["entries"]
    ]
    cache.load_entries(entries)
    return cache


class TestFixtureBands:
    """Verify band computation matches fixture expectations."""

    def test_bands_match_fixture(self, fixture):
        expected = fixture["expected_v1_cache"]
        for entry in expected["entries"]:
            computed = compute_bands(entry["signature"], 16)
            assert computed == entry["bands"], (
                f"Band mismatch for {entry['uuid']}: computed={computed}, expected={entry['bands']}"
            )


class TestFixtureVersionFiltering:
    """Verify correct version filtering from API response."""

    def test_v1_entry_count(self, fixture):
        expected = fixture["expected_v1_cache"]
        assert expected["entry_count"] == 6

    def test_v1_excludes_no_signature_entries(self, fixture):
        expected = fixture["expected_v1_cache"]
        uuids = {e["uuid"] for e in expected["entries"]}
        # dddddddd has no detection_signatures
        assert "dddddddd-dddd-dddd-dddd-dddddddddddd" not in uuids

    def test_v1_excludes_v0_only_entries(self, fixture):
        expected = fixture["expected_v1_cache"]
        uuids = {e["uuid"] for e in expected["entries"]}
        # 11111111 only has v0 signature
        assert "11111111-1111-1111-1111-111111111111" not in uuids

    def test_v1_includes_dual_version_entry(self, fixture):
        expected = fixture["expected_v1_cache"]
        uuids = {e["uuid"] for e in expected["entries"]}
        # 22222222 has both v0 and v1
        assert "22222222-2222-2222-2222-222222222222" in uuids

    def test_dual_version_uses_v1_signature(self, fixture):
        expected = fixture["expected_v1_cache"]
        dual = next(e for e in expected["entries"] if e["uuid"] == "22222222-2222-2222-2222-222222222222")
        # Should use the v1 signature "4444..." not the v0 "3333..."
        assert dual["signature"] == "4444444444444444444444444444444444444444444444444444444444444444"


class TestFixtureQueries:
    """Run the fixture's query test cases."""

    def test_exact_match(self, v1_cache, fixture):
        test = fixture["query_tests"]["tests"][0]
        assert test["name"] == "exact_match"

        matches = v1_cache.query(test["query_signature"], threshold=test["threshold"])
        match_uuids = [m.uuid for m in matches]

        # Should match the expected UUIDs
        for expected_uuid in test["expected_match_uuids"]:
            assert expected_uuid in match_uuids, f"Missing expected match: {expected_uuid}"

        # Top match should be exact
        assert matches[0].uuid == test["expected_top_match_uuid"]
        assert matches[0].hamming_distance == test["expected_top_hamming_distance"]
        assert abs(matches[0].cosine_similarity - test["expected_top_cosine_similarity"]) < 1e-6

    def test_near_match(self, v1_cache, fixture):
        test = fixture["query_tests"]["tests"][1]
        assert test["name"] == "near_match"

        matches = v1_cache.query(test["query_signature"], threshold=test["threshold"])
        match_uuids = [m.uuid for m in matches]

        for expected_uuid in test["expected_match_uuids"]:
            assert expected_uuid in match_uuids

        assert matches[0].uuid == test["expected_top_match_uuid"]
        assert matches[0].hamming_distance == test["expected_top_hamming_distance"]

    def test_no_match(self, v1_cache, fixture):
        test = fixture["query_tests"]["tests"][2]
        assert test["name"] == "no_match"

        matches = v1_cache.query(test["query_signature"], threshold=0.85)
        assert len(matches) == 0

    def test_all_zeros_exact(self, v1_cache, fixture):
        test = fixture["query_tests"]["tests"][3]
        assert test["name"] == "all_zeros_exact"

        matches = v1_cache.query(test["query_signature"], threshold=test["threshold"])
        match_uuids = [m.uuid for m in matches]

        for expected_uuid in test["expected_match_uuids"]:
            assert expected_uuid in match_uuids

        assert matches[0].uuid == test["expected_top_match_uuid"]
        assert matches[0].hamming_distance == test["expected_top_hamming_distance"]
