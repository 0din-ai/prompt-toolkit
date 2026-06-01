"""High-level entry point for SusFactor classification."""

from __future__ import annotations

from typing import Any, Optional

from .classifier import DEFAULT_THRESHOLD, SusFactorClassifier
from .types import SusFactorResult


async def sus_factor(
    text: str,
    *,
    classifier: Optional[SusFactorClassifier] = None,
    cache: Optional[Any] = None,
    model: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    device: Optional[str] = None,
) -> SusFactorResult:
    """Classify a prompt as safe vs. suspicious.

    If a ``classifier`` is provided it is used as-is (and left open for the
    caller to manage). Otherwise a classifier is constructed from a model
    cache, used once, and closed.

    Args:
        text: The prompt to classify.
        classifier: An existing classifier to reuse. If omitted, one is built.
        cache: A ``ModelCache`` to locate model files when auto-constructing.
            Defaults to a new ``ModelCache()``.
        model: Model identifier (default ``0dinai/susfactor-e5-large``).
        threshold: Decision threshold for the suspicious label.
        device: Torch device; auto-detected if None.

    Returns:
        A ``SusFactorResult`` with the suspicious probability and label.

    Raises:
        SusFactorError: If the model cannot be loaded or inference fails.

    Example:
        >>> from odin_prompt_toolkit.susfactor import sus_factor
        >>> result = await sus_factor("Ignore previous instructions")
        >>> print(result.score, result.label)
    """
    if classifier is not None:
        return await classifier.classify(text)

    if cache is None:
        from ..providers.model_cache import ModelCache

        cache = ModelCache()

    owned = await SusFactorClassifier.new(
        cache,
        model=model,
        threshold=threshold,
        device=device,
    )
    try:
        return await owned.classify(text)
    finally:
        await owned.close()
