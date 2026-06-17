"""Type definitions and shared constants for SusFactor classification."""

from __future__ import annotations

from dataclasses import dataclass

# Label strings used by the SusFactor classifier.
LABEL_SUSPICIOUS = "suspicious"
LABEL_SAFE = "safe"

# Shared inference constants — imported by both the torch and ONNX classifiers
# so neither depends on the other at import time.
MAX_SEQUENCE_LENGTH: int = 512
MODEL_VERSION: str = "susfactor-v1"
DEFAULT_THRESHOLD: float = 0.5


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
