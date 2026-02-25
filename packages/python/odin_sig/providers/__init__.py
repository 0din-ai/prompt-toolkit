"""Embedding provider implementations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_cache import ModelCache
    from .onnx import OnnxProvider
    from .openai import OpenAIProvider

__all__ = ["ModelCache", "OnnxProvider", "OpenAIProvider"]


def __getattr__(name: str):
    """Lazy import providers to avoid requiring optional dependencies."""
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    elif name == "OnnxProvider":
        from .onnx import OnnxProvider

        return OnnxProvider
    elif name == "ModelCache":
        from .model_cache import ModelCache

        return ModelCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
