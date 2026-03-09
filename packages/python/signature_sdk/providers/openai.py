"""OpenAI embedding provider implementation."""

import time
from typing import Optional

from ..lsh import normalize_vector
from ..types import EmbeddingResult, compute_embedding_sha256

try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError(
        "OpenAI provider requires the 'openai' package. "
        "Install with: pip install '0din-sig[openai]'"
    ) from e


class OpenAIProvider:
    """Embedding provider using OpenAI API.

    This provider uses the OpenAI embeddings API to generate vector embeddings
    for text. It can also be configured to use OpenRouter or other OpenAI-compatible
    APIs by setting a custom base URL.

    Args:
        api_key: OpenAI API key
        model: Model name (default: "text-embedding-3-large")
        dimensions: Embedding dimensions (default: 1536)
        base_url: Custom API base URL (optional, for OpenRouter etc.)
        name: Provider name (default: "openai")

    Example:
        >>> provider = OpenAIProvider(api_key="sk-...")
        >>> result = await provider.generate_embedding("Hello, world!")
        >>> print(f"Generated {result.dimensions}-dimensional embedding")
    """

    DEFAULT_MODEL = "text-embedding-3-large"
    DEFAULT_DIMENSIONS = 1536
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-large)
            dimensions: Embedding dimensions (default: 1536)
            base_url: Custom API base URL (optional)
            name: Provider name (default: "openai")
        """
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL
        self._dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._name = name or "openai"

        # Initialize client
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def name(self) -> str:
        """Get provider name."""
        return self._name

    def model(self) -> str:
        """Get model identifier."""
        return self._model

    def dimensions(self) -> int:
        """Get embedding dimensionality."""
        return self._dimensions

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for the given text.

        Args:
            text: Input text to embed

        Returns:
            EmbeddingResult with embedding, normalized embedding, SHA256, etc.

        Raises:
            Exception: If API call fails
        """
        start_time = time.time()

        # Call OpenAI API
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Extract embedding
        embedding = response.data[0].embedding
        token_count = response.usage.total_tokens if response.usage else 0

        # Normalize and compute SHA256
        normalized = normalize_vector(embedding)
        sha256 = compute_embedding_sha256(normalized)

        return EmbeddingResult(
            embedding=embedding,
            normalized_embedding=normalized,
            normalized_embedding_sha256=sha256,
            model=self._model,
            dimensions=len(embedding),
            token_count=token_count,
            timing_ms=elapsed_ms,
        )

    async def close(self) -> None:
        """Close the provider and clean up resources."""
        await self._client.close()
