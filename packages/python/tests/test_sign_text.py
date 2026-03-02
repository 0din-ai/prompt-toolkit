"""Tests for sign_text() high-level API."""

import os
import pytest

from odin_sig import sign_text, EmbeddingProvider, SignatureVersion
from odin_sig.types import EmbeddingResult, LshConfig


class MockProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dimensions: int = 384):
        self._dimensions = dimensions

    def name(self) -> str:
        return "mock-provider"

    def model(self) -> str:
        return "mock-model"

    def dimensions(self) -> int:
        return self._dimensions

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        # Return a pre-normalized test embedding
        embedding = [0.5] * self._dimensions
        return EmbeddingResult(
            embedding=embedding,
            normalized_embedding=embedding,
            normalized_embedding_sha256="test-sha256",
            model="mock-model",
            dimensions=self._dimensions,
            token_count=10,
            timing_ms=100.0,
        )

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_sign_text_v1_with_provider():
    """Test sign_text() with explicit V1 provider."""
    provider = MockProvider(dimensions=384)

    result = await sign_text("test prompt", provider=provider, version=SignatureVersion.V1)

    assert result.version == SignatureVersion.V1
    assert result.provider == "mock-provider"
    assert result.model == "mock-model"
    assert result.dimensions == 384
    assert result.prompt_preview == "test prompt"
    assert result.prompt_length == 11
    assert result.timing_ms is not None

    # Verify signature format
    sig_string = result.signature_string
    assert sig_string.startswith("0din-v1:")
    assert len(sig_string) == 72  # "0din-v1:" (8) + 64 hex chars


@pytest.mark.asyncio
async def test_sign_text_v0_with_provider():
    """Test sign_text() with explicit V0 provider."""
    provider = MockProvider(dimensions=1536)

    result = await sign_text("test prompt", provider=provider, version=SignatureVersion.V0)

    assert result.version == SignatureVersion.V0
    assert result.dimensions == 1536

    sig_string = result.signature_string
    assert sig_string.startswith("0din-v0:")


@pytest.mark.asyncio
async def test_sign_text_latest_resolves_to_v1():
    """Test that LATEST resolves to V1."""
    provider = MockProvider(dimensions=384)

    result = await sign_text("test", provider=provider, version=SignatureVersion.LATEST)

    assert result.version == SignatureVersion.V1


@pytest.mark.asyncio
async def test_sign_text_infer_version_from_provider():
    """Test version inference from provider dimensions."""
    # V1 provider (384 dims) - version inferred
    provider_v1 = MockProvider(dimensions=384)
    result = await sign_text("test", provider=provider_v1)
    assert result.version == SignatureVersion.V1

    # V0 provider (1536 dims) - version inferred
    provider_v0 = MockProvider(dimensions=1536)
    result = await sign_text("test", provider=provider_v0)
    assert result.version == SignatureVersion.V0


@pytest.mark.asyncio
async def test_sign_text_version_mismatch_with_provider():
    """Test that version mismatch with provider raises ValueError."""
    # Provider returns 384 dimensions but we request V0 (expects 1536)
    provider = MockProvider(dimensions=384)

    with pytest.raises(ValueError, match="Version mismatch"):
        await sign_text("test", provider=provider, version=SignatureVersion.V0)


@pytest.mark.asyncio
async def test_sign_text_dimension_mismatch():
    """Test that unknown dimensions raise ValueError."""
    # Provider with non-standard dimensions
    provider = MockProvider(dimensions=512)

    with pytest.raises(ValueError, match="Cannot infer version"):
        await sign_text("test", provider=provider)


@pytest.mark.asyncio
async def test_sign_text_custom_config():
    """Test sign_text() with custom LSH config."""
    provider = MockProvider(dimensions=384)

    custom_config = LshConfig(families=5, bits=128, bands=8)

    result = await sign_text("test", provider=provider, config=custom_config)

    assert result.lsh.config.families == 5
    assert result.lsh.config.bits == 128
    assert result.lsh.config.bands == 8
    assert len(result.lsh.signatures) == 5  # 5 families


@pytest.mark.asyncio
async def test_sign_text_long_prompt_preview():
    """Test that long prompts are truncated in preview."""
    provider = MockProvider(dimensions=384)

    long_text = "a" * 100
    result = await sign_text(long_text, provider=provider)

    # Preview should be truncated to 50 chars
    assert len(result.prompt_preview) == 50
    assert result.prompt_preview.endswith("...")
    assert result.prompt_length == 100


@pytest.mark.asyncio
@pytest.mark.skipif(
    not __import__("importlib.util").util.find_spec("openai"),
    reason="Requires openai package (install with pip install '0din-sig[openai]')",
)
async def test_sign_text_auto_construct_v0_missing_api_key(monkeypatch):
    """Test that auto-construction for V0 fails without API key."""
    # Remove OPENAI_API_KEY from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
        await sign_text("test", version=SignatureVersion.V0)


@pytest.mark.asyncio
async def test_sign_text_backward_compat_positional():
    """Test backward compatibility with old positional argument style."""
    provider = MockProvider(dimensions=384)

    # Old style: sign_text(text, provider, version, config)
    result = await sign_text("test", provider=provider, version=SignatureVersion.V1, config=None)

    assert result.version == SignatureVersion.V1
    assert result.provider == "mock-provider"
