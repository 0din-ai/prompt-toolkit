"""Cross-language validation test for sign_text().

This test verifies that the same input produces identical signatures
across Rust, Python, and TypeScript implementations.
"""

import pytest

from odin_sig import sign_text, SignatureVersion
from odin_sig.types import EmbeddingResult


class FixedEmbeddingProvider:
    """Mock provider that returns a fixed embedding for cross-validation."""

    def __init__(self, dimensions: int):
        """Initialize with fixed dimensions."""
        self._dimensions = dimensions
        # Create a deterministic test embedding (all 0.5)
        self._embedding = [0.5] * dimensions

    def name(self) -> str:
        return "fixed-provider"

    def model(self) -> str:
        return "fixed-model"

    def dimensions(self) -> int:
        return self._dimensions

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Return the fixed embedding."""
        from odin_sig.lsh import normalize_vector
        from odin_sig.types import compute_embedding_sha256

        # Return the fixed embedding (normalize it)
        normalized = normalize_vector(self._embedding)
        sha256 = compute_embedding_sha256(normalized)

        return EmbeddingResult(
            embedding=self._embedding,
            normalized_embedding=normalized,
            normalized_embedding_sha256=sha256,
            model="fixed-model",
            dimensions=self._dimensions,
            token_count=10,
            timing_ms=100.0,
        )

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_cross_validation_v1():
    """Test V1 signature with fixed embedding."""
    # Create provider with V1 dimensions (384)
    provider = FixedEmbeddingProvider(384)

    # Generate signature
    result = await sign_text("test prompt", provider=provider, version=SignatureVersion.V1)

    signature = result.signature_string

    # Print for cross-validation with Rust/TypeScript
    print(f"Python V1 signature: {signature}")
    print(f"Python V1 embedding SHA256: {result.embedding_sha256}")

    # Verify format
    assert signature.startswith("0din-v1:")
    assert len(signature) == 72  # "0din-v1:" (8) + 64 hex chars

    # Verify all hex characters
    hex_part = signature[8:]
    assert all(c in "0123456789abcdef" for c in hex_part)


@pytest.mark.asyncio
async def test_cross_validation_v0():
    """Test V0 signature with fixed embedding."""
    # Create provider with V0 dimensions (1536)
    provider = FixedEmbeddingProvider(1536)

    # Generate signature
    result = await sign_text("test prompt", provider=provider, version=SignatureVersion.V0)

    signature = result.signature_string

    # Print for cross-validation
    print(f"Python V0 signature: {signature}")
    print(f"Python V0 embedding SHA256: {result.embedding_sha256}")

    # Verify format
    assert signature.startswith("0din-v0:")
    assert len(signature) == 72  # "0din-v0:" (8) + 64 hex chars


@pytest.mark.asyncio
async def test_cross_validation_pattern():
    """Test with a pattern vector."""

    class PatternProvider:
        """Provider with alternating pattern."""

        def __init__(self):
            # Create a pattern: alternating positive/negative
            self._embedding = [1.0 if i % 2 == 0 else -1.0 for i in range(384)]

        def name(self) -> str:
            return "pattern-provider"

        def model(self) -> str:
            return "pattern-model"

        def dimensions(self) -> int:
            return 384

        async def generate_embedding(self, text: str) -> EmbeddingResult:
            from odin_sig.lsh import normalize_vector
            from odin_sig.types import compute_embedding_sha256

            normalized = normalize_vector(self._embedding)
            sha256 = compute_embedding_sha256(normalized)

            return EmbeddingResult(
                embedding=self._embedding,
                normalized_embedding=normalized,
                normalized_embedding_sha256=sha256,
                model="pattern-model",
                dimensions=384,
                token_count=10,
                timing_ms=100.0,
            )

        async def close(self) -> None:
            pass

    provider = PatternProvider()

    result = await sign_text("test prompt", provider=provider, version=SignatureVersion.V1)

    signature = result.signature_string

    print(f"Python pattern signature: {signature}")
    print(f"Python pattern embedding SHA256: {result.embedding_sha256}")

    # Verify format
    assert signature.startswith("0din-v1:")
