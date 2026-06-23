"""Type definitions and shared constants for SusFactor classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# Label strings used by the SusFactor classifier.
LABEL_SUSPICIOUS = "suspicious"
LABEL_SAFE = "safe"

# Shared inference constants — imported by both the torch and ONNX classifiers
# so neither depends on the other at import time.
MAX_SEQUENCE_LENGTH: int = 512
MODEL_VERSION: str = "susfactor-v1"
DEFAULT_THRESHOLD: float = 0.5

# Chunking constants.
# The model's hard limit is MAX_SEQUENCE_LENGTH tokens total, but the tokenizer
# adds [CLS] and [SEP], leaving 510 usable positions for the prompt payload.
MAX_CONTENT_TOKENS: int = MAX_SEQUENCE_LENGTH - 2  # 510
CHUNK_OVERLAP: int = 50
CHUNK_STRIDE: int = MAX_CONTENT_TOKENS - CHUNK_OVERLAP  # 460


@dataclass
class SusFactorResult:
    """Result of a SusFactor classification.

    Attributes:
        score: Probability that the prompt is suspicious/malicious, in [0, 1].
        label: ``"suspicious"`` if ``score >= threshold`` else ``"safe"``.
        model: Identifier of the model that produced the score.
        threshold: Decision threshold used to derive ``label`` from ``score``.
        timing_ms: Inference time in milliseconds, if measured.
    """

    score: float
    label: str
    model: str
    threshold: float
    timing_ms: float | None = None

    @property
    def is_suspicious(self) -> bool:
        """Whether the prompt was classified as suspicious."""
        return self.label == LABEL_SUSPICIOUS


@dataclass
class ChunkedSusFactorResult:
    """Return type of ``classify()`` for prompts of any length.

    Prompts within ``MAX_CONTENT_TOKENS`` (510 tokens) produce exactly one
    chunk. Longer prompts are split automatically — callers never need to
    check length or call a different method.

    Each chunk is an independent model inference; no scores are aggregated.

    Attributes:
        chunks: Individual result for each chunk, in order. Short prompts
            always produce exactly one entry; access ``chunks[0]`` for the
            score and label in that case.
        is_suspicious: ``True`` if **any** chunk's label is ``"suspicious"``.
            Use this field for security gating — a prompt is suspicious if
            any portion of it is suspicious.
        total_timing_ms: Wall-clock time for all chunks (parallel), in ms.

    Displaying a single score:
        The previous API returned one ``score`` and ``label`` directly. With
        chunking, there is no single canonical score. Callers that need one
        number for display should decide explicitly::

            # Highest suspicion across all chunks (most conservative):
            max_score = max(c.score for c in result.chunks)

            # First chunk only (equivalent to the old score for short prompts;
            # may miss a suspicious tail in long prompts):
            first_score = result.chunks[0].score

        Using ``is_suspicious`` is recommended for security decisions.
        A display score is a UX choice, not a security one.
    """

    chunks: List[SusFactorResult]
    is_suspicious: bool
    total_timing_ms: float


def suspicious_prob(logits: list[float]) -> float:
    """Softmax over a 2-logit list, returning P(class 1) = suspicious.

    This is a pure function with no torch dependency — importable without the
    ``susfactor`` extra.
    """
    import math

    m = max(logits[0], logits[1])
    e0 = math.exp(logits[0] - m)
    e1 = math.exp(logits[1] - m)
    return e1 / (e0 + e1)


def label_for_score(score: float, threshold: float) -> str:
    """Map a suspicious probability to a label using ``threshold``."""
    return LABEL_SUSPICIOUS if score >= threshold else LABEL_SAFE
