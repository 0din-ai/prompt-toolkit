"""Embedding provider protocol definition."""

from typing import Protocol, runtime_checkable

from .types import EmbeddingResult


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding generation providers.

    All embedding providers must implement this interface to work with
    the sign_text() function.

    Example:
        >>> class MyProvider:
        ...     def name(self) -> str:
        ...         return "my-provider"
        ...
        ...     def model(self) -> str:
        ...         return "my-model"
        ...
        ...     def dimensions(self) -> int:
        ...         return 1024
        ...
        ...     async def generate_embedding(self, text: str) -> EmbeddingResult:
        ...         # Generate and return embedding
        ...         pass
        ...
        ...     async def close(self) -> None:
        ...         # Cleanup resources
        ...         pass
    """

    def name(self) -> str:
        """Get the provider name.

        Returns:
            Provider name (e.g., "onnx", "openai")
        """
        ...

    def model(self) -> str:
        """Get the model identifier.

        Returns:
            Model name or path (e.g., "intfloat/multilingual-e5-small")
        """
        ...

    def dimensions(self) -> int:
        """Get the embedding dimensionality.

        Returns:
            Number of dimensions in the embedding vector
        """
        ...

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for the given text.

        Args:
            text: Input text to embed

        Returns:
            EmbeddingResult containing the raw embedding, normalized embedding,
            SHA256 hash, model info, and timing metrics

        Raises:
            Exception: If embedding generation fails
        """
        ...

    async def close(self) -> None:
        """Close the provider and clean up resources.

        This should be called when done using the provider to properly
        release any allocated resources (models, HTTP connections, etc.).
        """
        ...
