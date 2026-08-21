"""Tests for SusFactor result types."""

from odin_prompt_toolkit.susfactor import (
    ChunkedSusFactorResult,
    PhaseSpan,
    SusFactorResult,
)


def assert_span_shape(result: ChunkedSusFactorResult) -> None:
    """Assert the PhaseSpan waterfall contract (shape/ordering only).

    Never asserts exact durations (nondeterministic). Reusable across the
    classifier span tests.
    """
    spans = result.spans
    assert spans, "spans must be non-empty"
    assert spans[0].name == "tokenize"
    assert spans[1].name == "chunk"
    assert spans[-1].name == "reduce"

    inference = [s for s in spans if s.name == "inference"]
    assert len(inference) == len(result.chunks)
    # chunk_index matches position 0..n-1 and is unique.
    indices = [s.chunk_index for s in inference]
    assert indices == list(range(len(result.chunks)))
    assert len(set(indices)) == len(indices)

    # Non-inference spans carry no chunk_index.
    for s in spans:
        if s.name != "inference":
            assert s.chunk_index is None

    # Durations finite/non-negative; start_ms non-negative.
    import math

    for s in spans:
        assert math.isfinite(s.start_ms)
        assert math.isfinite(s.duration_ms)
        assert s.duration_ms >= 0.0
        assert s.start_ms >= 0.0


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


def _result(label: str) -> SusFactorResult:
    return SusFactorResult(score=0.5, label=label, model="m", threshold=0.5, timing_ms=1.0)


def test_phase_span_fields():
    """PhaseSpan holds name, start_ms, duration_ms, and optional chunk_index."""
    span = PhaseSpan(name="inference", start_ms=1.5, duration_ms=2.5, chunk_index=3)
    assert span.name == "inference"
    assert span.start_ms == 1.5
    assert span.duration_ms == 2.5
    assert span.chunk_index == 3


def test_phase_span_chunk_index_optional():
    """chunk_index defaults to None (tokenize/chunk/reduce spans)."""
    span = PhaseSpan(name="tokenize", start_ms=0.0, duration_ms=1.0)
    assert span.chunk_index is None


def test_chunked_result_spans_default_empty():
    """spans defaults to an empty list when omitted."""
    result = ChunkedSusFactorResult(
        chunks=[_result("safe")],
        is_suspicious=False,
        total_timing_ms=1.0,
    )
    assert result.spans == []


def test_spans_waterfall_shape_and_ordering():
    """A representative two-chunk waterfall satisfies the span contract."""
    result = ChunkedSusFactorResult(
        chunks=[_result("safe"), _result("suspicious")],
        is_suspicious=True,
        total_timing_ms=10.0,
        spans=[
            PhaseSpan(name="tokenize", start_ms=0.0, duration_ms=1.0),
            PhaseSpan(name="chunk", start_ms=1.0, duration_ms=0.5),
            PhaseSpan(name="inference", start_ms=1.5, duration_ms=3.0, chunk_index=0),
            PhaseSpan(name="inference", start_ms=1.6, duration_ms=3.2, chunk_index=1),
            PhaseSpan(name="reduce", start_ms=8.0, duration_ms=0.2),
        ],
    )
    assert_span_shape(result)
