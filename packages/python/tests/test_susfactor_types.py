"""Tests for SusFactor result types."""

from odin_prompt_toolkit.susfactor import SusFactorResult


def test_susfactor_result_fields():
    """SusFactorResult holds score, label, model, threshold, and timing."""
    result = SusFactorResult(
        score=0.92,
        label="suspicious",
        model="0dinai/susfactor-e5-large",
        threshold=0.5,
        timing_ms=12.3,
    )
    assert result.score == 0.92
    assert result.label == "suspicious"
    assert result.model == "0dinai/susfactor-e5-large"
    assert result.threshold == 0.5
    assert result.timing_ms == 12.3


def test_susfactor_result_timing_optional():
    """timing_ms defaults to None."""
    result = SusFactorResult(
        score=0.1,
        label="safe",
        model="m",
        threshold=0.5,
    )
    assert result.timing_ms is None


def test_susfactor_result_is_suspicious():
    """is_suspicious reflects label == 'suspicious'."""
    suspicious = SusFactorResult(score=0.8, label="suspicious", model="m", threshold=0.5)
    safe = SusFactorResult(score=0.2, label="safe", model="m", threshold=0.5)
    assert suspicious.is_suspicious is True
    assert safe.is_suspicious is False
