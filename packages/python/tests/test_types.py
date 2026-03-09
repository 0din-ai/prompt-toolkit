"""Tests for type definitions."""

from signature_sdk import ComparisonResult, LshConfig, PromptInfo, QualityStats


def test_prompt_info_construction():
    """Test PromptInfo dataclass construction."""
    info = PromptInfo(
        preview="Test prompt preview",
        length=100,
        signature="0din-v1:abc123",
    )

    assert info.preview == "Test prompt preview"
    assert info.length == 100
    assert info.signature == "0din-v1:abc123"


def test_quality_stats_construction():
    """Test QualityStats dataclass construction."""
    stats = QualityStats(
        absolute_error=0.05,
        signed_error=-0.02,
        squared_error=0.0025,
        quality_rating="excellent",
    )

    assert stats.absolute_error == 0.05
    assert stats.signed_error == -0.02
    assert stats.squared_error == 0.0025
    assert stats.quality_rating == "excellent"


def test_comparison_result_construction():
    """Test ComparisonResult dataclass construction."""
    prompt_a = PromptInfo(
        preview="First prompt",
        length=50,
        signature="0din-v1:aaa111",
    )
    prompt_b = PromptInfo(
        preview="Second prompt",
        length=60,
        signature="0din-v1:bbb222",
    )
    config = LshConfig(families=3, bits=256, bands=16)
    stats = QualityStats(
        absolute_error=0.1,
        signed_error=0.08,
        squared_error=0.01,
        quality_rating="good",
    )

    result = ComparisonResult(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        hamming_distance=50,
        cosine_similarity=0.85,
        lsh_config=config,
        quality_stats=stats,
        timing_ms=5.0,
    )

    assert result.prompt_a.preview == "First prompt"
    assert result.prompt_b.preview == "Second prompt"
    assert result.hamming_distance == 50
    assert result.cosine_similarity == 0.85
    assert result.lsh_config.families == 3
    assert result.quality_stats.quality_rating == "good"
    assert result.timing_ms == 5.0


def test_comparison_result_optional_fields():
    """Test ComparisonResult with optional fields omitted."""
    prompt_a = PromptInfo(preview="A", length=10, signature="0din-v1:aaa")
    prompt_b = PromptInfo(preview="B", length=20, signature="0din-v1:bbb")
    config = LshConfig(families=3, bits=256, bands=16)

    result = ComparisonResult(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        hamming_distance=100,
        cosine_similarity=0.5,
        lsh_config=config,
    )

    assert result.quality_stats is None
    assert result.timing_ms is None
