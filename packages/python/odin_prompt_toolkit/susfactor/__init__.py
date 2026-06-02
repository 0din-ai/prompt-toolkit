"""SusFactor jailbreak/prompt-injection classifier integration.

SusFactor classifies a prompt as **safe** (score near 0) or **suspicious**
(score near 1) for jailbreak and prompt-injection detection. It is a separate
capability from the LSH signature pipeline -- it does not produce an embedding
or a signature.

The model (``0dinai/susfactor-e5-large``) is an e5-large encoder with a small
MLP classification head. It is not bundled with the SDK; download it from
HuggingFace and cache it locally (see ``ModelCache``), then:

Example:
    >>> from odin_prompt_toolkit.susfactor import sus_factor
    >>>
    >>> result = await sus_factor("Ignore all previous instructions...")
    >>> print(result.score, result.label)
    0.97 suspicious

``SusFactorResult`` is always importable. ``SusFactorClassifier`` and
``sus_factor`` are imported lazily because they require the optional
``torch`` / ``transformers`` dependencies (install ``[susfactor]``).
"""

from typing import TYPE_CHECKING, Any

from odin_prompt_toolkit.susfactor.types import SusFactorResult

if TYPE_CHECKING:
    from odin_prompt_toolkit.susfactor.classifier import SusFactorClassifier
    from odin_prompt_toolkit.susfactor.compare import sus_factor

__all__ = [
    "SusFactorClassifier",
    "SusFactorResult",
    "sus_factor",
]


def __getattr__(name: str) -> Any:
    """Lazily import optional-dependency members (torch/transformers)."""
    if name == "SusFactorClassifier":
        from odin_prompt_toolkit.susfactor.classifier import SusFactorClassifier

        return SusFactorClassifier
    if name == "sus_factor":
        from odin_prompt_toolkit.susfactor.compare import sus_factor

        return sus_factor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
