"""Tests for threat feed comparison API."""

from odin_prompt_toolkit.lsh import LSHFamily
from odin_prompt_toolkit.threatfeed.cache import ThreatFeedCache, compute_bands
from odin_prompt_toolkit.threatfeed.compare import compare_to_threatfeed
from odin_prompt_toolkit.threatfeed.types import CachedSignature
from odin_prompt_toolkit.types import (
    LshConfig,
    LshOutput,
    SignatureResult,
    SignatureVersion,
)


SIG_A = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"


def _make_signature_result(sig: str) -> SignatureResult:
    """Create a minimal SignatureResult for testing."""
    return SignatureResult(
        signature=f"0din-v1:{sig}",
        version=SignatureVersion.V1,
        prompt_preview="test prompt",
        prompt_length=11,
        provider="test",
        model="test-model",
        dimensions=1024,
        embedding_sha256="abc123",
        lsh=LshOutput(
            config=LshConfig(),
            signatures=[
                LSHFamily(
                    family=0,
                    bits=256,
                    signature=sig,
                    bands=compute_bands(sig, 16),
                )
            ],
        ),
    )


class TestCompareToThreatfeed:
    def test_exact_match(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([
            CachedSignature(
                uuid="threat-1",
                title="Known Threat",
                severity="high",
                security_boundary="guardrail_jailbreak",
                signature=SIG_A,
                bands=compute_bands(SIG_A, 16),
            ),
        ])

        result = _make_signature_result(SIG_A)
        matches = compare_to_threatfeed(result, cache)

        assert len(matches) == 1
        assert matches[0].uuid == "threat-1"
        assert matches[0].hamming_distance == 0
        assert abs(matches[0].cosine_similarity - 1.0) < 1e-10

    def test_no_match(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([
            CachedSignature(
                uuid="threat-1",
                title="Known Threat",
                severity="high",
                security_boundary="guardrail_jailbreak",
                signature=SIG_A,
                bands=compute_bands(SIG_A, 16),
            ),
        ])

        unrelated = "5678901234567890567890123456789056789012345678905678901234567890"
        result = _make_signature_result(unrelated)
        matches = compare_to_threatfeed(result, cache)

        assert len(matches) == 0

    def test_empty_cache(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        result = _make_signature_result(SIG_A)
        matches = compare_to_threatfeed(result, cache)
        assert len(matches) == 0

    def test_threshold_parameter(self):
        cache = ThreatFeedCache(version=SignatureVersion.V1)
        cache.load_entries([
            CachedSignature(
                uuid="threat-1",
                title="Known Threat",
                severity="high",
                security_boundary="guardrail_jailbreak",
                signature=SIG_A,
                bands=compute_bands(SIG_A, 16),
            ),
        ])

        result = _make_signature_result(SIG_A)

        # Very high threshold still matches exact
        matches = compare_to_threatfeed(result, cache, threshold=0.99)
        assert len(matches) == 1

        # Threshold of 1.0+ means nothing matches unless exact
        matches = compare_to_threatfeed(result, cache, threshold=1.0)
        assert len(matches) == 1  # Exact match has cosine = 1.0
