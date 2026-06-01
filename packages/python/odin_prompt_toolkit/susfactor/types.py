"""Type definitions for SusFactor classification."""

from __future__ import annotations

from dataclasses import dataclass

# Label strings used by the SusFactor classifier.
LABEL_SUSPICIOUS = "suspicious"
LABEL_SAFE = "safe"


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
