"""Tests for threat feed cache."""

import json
import tempfile
from pathlib import Path

from odin_prompt_toolkit.threatfeed.cache import ThreatFeedCache, compute_bands
from odin_prompt_toolkit.threatfeed.types import CachedSignature
from odin_prompt_toolkit.types import SignatureVersion

SIG_A = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
SIG_B = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b3"
SIG_ZEROS = "0000000000000000000000000000000000000000000000000000000000000000"
SIG_ONES = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
SIG_UNRELATED = "5678901234567890567890123456789056789012345678905678901234567890"


class TestComputeBands:
    def test_basic_split(self):
        bands = compute_bands(SIG_A, 16)
        assert len(bands) == 16
        assert bands[0] == "a1b2"
        assert bands[1] == "c3d4"
        assert bands[15] == "a1b2"

    def test_all_zeros(self):
        bands = compute_bands(SIG_ZEROS, 16)
        for band in bands:
            assert band == "0000"

    def test_all_ones(self):
        bands = compute_bands(SIG_ONES, 16)
        for band in bands:
            assert band == "ffff"


class TestThreatFeedCache:
    def _make_entry(
        self, uuid: str, sig: str, title: str = "Test", severity: str = "high"
    ) -> CachedSignature:
        return CachedSignature(
            uuid=uuid,
            title=title,
            severity=severity,
            security_boundary="guardrail_jailbreak",
            signature=sig,
            bands=compute_bands(sig, 16),
        )

    def test_empty_cache_query(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        matches = cache.query(SIG_A)
        assert matches == []

    def test_exact_match(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([self._make_entry("test-uuid", SIG_A)])

        matches = cache.query(SIG_A)
        assert len(matches) == 1
        assert matches[0].uuid == "test-uuid"
        assert matches[0].hamming_distance == 0
        assert abs(matches[0].cosine_similarity - 1.0) < 1e-10

    def test_near_match(self):
        """SIG_A and SIG_B differ by 1 bit in the last hex char."""
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([
            self._make_entry("entry-a", SIG_A),
            self._make_entry("entry-b", SIG_B, severity="medium"),
        ])

        matches = cache.query(SIG_A)
        assert len(matches) == 2
        # Exact match first
        assert matches[0].uuid == "entry-a"
        assert matches[0].hamming_distance == 0
        # Near match second
        assert matches[1].uuid == "entry-b"
        assert matches[1].cosine_similarity > 0.99

    def test_no_match_no_shared_bands(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([self._make_entry("test-uuid", SIG_A)])

        matches = cache.query(SIG_UNRELATED)
        assert matches == []

    def test_threshold_filtering(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([self._make_entry("test-uuid", SIG_ONES)])

        # SIG_ZEROS and SIG_ONES share no bands, so no candidates even checked
        matches = cache.query(SIG_ZEROS)
        assert matches == []

    def test_max_results(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        # Create many entries with the same signature
        entries = [self._make_entry(f"entry-{i}", SIG_A) for i in range(20)]
        cache.load_entries(entries)

        matches = cache.query(SIG_A, max_results=5)
        assert len(matches) == 5

    def test_entry_count(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        assert cache.entry_count == 0

        cache.load_entries([
            self._make_entry("a", SIG_A),
            self._make_entry("b", SIG_B),
        ])
        assert cache.entry_count == 2

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ThreatFeedCache(
                version=SignatureVersion.V1, cache_dir=tmp_dir
            )
            cache.load_entries([
                self._make_entry("test-uuid", SIG_A, title="Test Entry"),
            ])
            cache.save()

            # Load into a new cache
            cache2 = ThreatFeedCache(
                version=SignatureVersion.V1, cache_dir=tmp_dir
            )
            loaded = cache2.load()
            assert loaded is True
            assert cache2.entry_count == 1

            matches = cache2.query(SIG_A)
            assert len(matches) == 1
            assert matches[0].uuid == "test-uuid"

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ThreatFeedCache(
                version=SignatureVersion.V1, cache_dir=tmp_dir
            )
            loaded = cache.load()
            assert loaded is False

    def test_merge_entries(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([
            self._make_entry("existing", SIG_A, title="Original", severity="low"),
        ])

        result = cache._merge_entries([
            self._make_entry("existing", SIG_A, title="Updated", severity="high"),
            self._make_entry("new-entry", SIG_ONES, title="New"),
        ])

        assert result.added == 1
        assert result.updated == 1
        assert result.total == 2
        assert cache.entries[0].title == "Updated"
        assert cache.entries[0].severity == "high"

    def test_schema_version_mismatch(self):
        """Old schema version should cause load to return False."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cache-v1.json"
            path.write_text(json.dumps({
                "schema_version": 999,
                "entries": [],
                "band_index": {},
            }))

            cache = ThreatFeedCache(
                version=SignatureVersion.V1, cache_dir=tmp_dir
            )
            loaded = cache.load()
            assert loaded is False
