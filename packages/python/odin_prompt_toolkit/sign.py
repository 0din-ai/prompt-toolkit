"""High-level sign_text() convenience API."""

import os
import time
from typing import Optional

from .lsh import simhash_lsh_multi
from .provider import EmbeddingProvider
from .types import LshConfig, LshOutput, SignatureResult, SignatureVersion


async def sign_text(
    text: str,
    *,
    version: SignatureVersion = SignatureVersion.LATEST,
    provider: Optional[EmbeddingProvider] = None,
    config: Optional[LshConfig] = None,
) -> SignatureResult:
    """Generate a signature from text.

    This is the high-level convenience function that orchestrates the full pipeline:
    1. Auto-construct provider (if not provided) based on version
    2. Generate embedding using the provider
    3. Normalize the embedding (already done by providers)
    4. Compute LSH signatures
    5. Build a SignatureResult with metadata

    Args:
        text: The text prompt to sign
        version: Signature version (default: LATEST, which resolves to V1).
                If provider is given, version is inferred from provider dimensions
                unless explicitly specified for validation.
        provider: Optional embedding provider. If None, auto-constructs the appropriate
                 provider based on version:
                 - V1: OnnxProvider (requires model cached, onnxruntime installed)
                 - V0: OpenAIProvider (requires OPENAI_API_KEY env var, openai installed)
        config: Optional LSH configuration (defaults to 3 families, 256 bits, 16 bands)

    Returns:
        SignatureResult containing the signature and metadata

    Raises:
        ValueError: If embedding dimensions don't match the version
        ImportError: If required provider dependencies are not installed
        Exception: If embedding generation fails

    Examples:
        Simple usage (auto-constructs V1/ONNX provider):
        >>> result = await sign_text("How do I reset my password?")
        >>> print(result.signature_string)
        0din-v1:8d000000ac854dae...

        Explicit V0 (auto-constructs OpenAI provider from env):
        >>> result = await sign_text(
        ...     "How do I reset my password?",
        ...     version=SignatureVersion.V0,
        ... )
        >>> print(result.signature_string)
        0din-v0:363b24ee2b817354...

        Advanced - bring your own provider (version inferred):
        >>> from odin_prompt_toolkit.providers import ModelCache, OnnxProvider
        >>> cache = ModelCache()
        >>> provider = await OnnxProvider.new(cache)
        >>> result = await sign_text(
        ...     "How do I reset my password?",
        ...     provider=provider,
        ... )
        >>> await provider.close()
    """
    start_time = time.time()

    # Determine if we auto-constructed the provider (for cleanup)
    auto_constructed = provider is None

    try:
        # Auto-construct provider if not provided
        if provider is None:
            provider = await _create_provider_for_version(version)

        # Infer or validate version based on provider dimensions
        resolved_version = _resolve_version(version, provider)

        # Generate embedding using provider
        embedding_result = await provider.generate_embedding(text)

        # Use provided config or default
        lsh_config = config or LshConfig()

        # Verify dimensions match expected for this version
        expected_dims = resolved_version.embedding_dimensions()
        if embedding_result.dimensions != expected_dims:
            raise ValueError(
                f"Embedding dimensions mismatch: expected {expected_dims} for {resolved_version.value}, "
                f"got {embedding_result.dimensions}"
            )

        # Compute LSH signatures (providers already normalize embeddings)
        signatures = simhash_lsh_multi(
            embedding_result.normalized_embedding,
            families=lsh_config.families,
            bits=lsh_config.bits,
            bands=lsh_config.bands,
        )

        # Build result
        elapsed_ms = (time.time() - start_time) * 1000

        # Create prompt preview (first 50 chars)
        if len(text) <= 50:
            prompt_preview = text
        else:
            prompt_preview = text[:47] + "..."

        result = SignatureResult(
            signature="",  # Will be computed by signature_string property
            version=resolved_version,
            prompt_preview=prompt_preview,
            prompt_length=len(text),
            provider=provider.name(),
            model=embedding_result.model,
            dimensions=embedding_result.dimensions,
            embedding_sha256=embedding_result.normalized_embedding_sha256,
            lsh=LshOutput(
                config=lsh_config,
                signatures=signatures,
            ),
            timing_ms=elapsed_ms,
        )

        return result

    finally:
        # Clean up auto-constructed provider
        if auto_constructed and provider is not None:
            await provider.close()


async def _create_provider_for_version(
    version: SignatureVersion,
) -> EmbeddingProvider:
    """Auto-construct the appropriate provider for a given version.

    Args:
        version: Signature version (may be LATEST)

    Returns:
        Initialized provider instance

    Raises:
        ImportError: If required dependencies are not installed
        ValueError: If required configuration is missing (e.g., API key)
    """
    resolved = version.resolve()

    if resolved == SignatureVersion.V1:
        # V1 uses ONNX provider (local inference)
        try:
            from .providers import ModelCache, OnnxProvider
        except ImportError as e:
            raise ImportError(
                "V1 signatures require the ONNX provider. "
                "Install with: pip install '0din-prompt-toolkit[onnx]'"
            ) from e

        cache = ModelCache()
        return await OnnxProvider.new(cache)

    elif resolved == SignatureVersion.V0:
        # V0 uses OpenAI provider (API-based)
        try:
            from .providers import OpenAIProvider
        except ImportError as e:
            raise ImportError(
                "V0 signatures require the OpenAI provider. "
                "Install with: pip install 'signature-sdk[openai]'"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for V0 signatures. "
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )

        return OpenAIProvider(api_key=api_key)

    else:
        raise ValueError(f"Unsupported signature version: {resolved}")


def _resolve_version(
    version: SignatureVersion,
    provider: EmbeddingProvider,
) -> SignatureVersion:
    """Resolve version from provider dimensions or validate explicitly passed version.

    Args:
        version: Explicitly passed version (may be LATEST)
        provider: Provider instance

    Returns:
        Resolved concrete version (V0 or V1)

    Raises:
        ValueError: If dimensions don't match any known version, or if version
                   conflicts with provider dimensions
    """
    resolved_version = version.resolve()
    provider_dims = provider.dimensions()

    # Infer version from provider dimensions
    if provider_dims == 1536:
        inferred_version = SignatureVersion.V0
    elif provider_dims == 1024:
        inferred_version = SignatureVersion.V1
    else:
        raise ValueError(
            f"Cannot infer version from provider dimensions ({provider_dims}). "
            f"Expected 1536 (V0) or 1024 (V1). "
            f"Please specify version explicitly."
        )

    # If version was explicitly passed (not LATEST), validate it matches
    if version != SignatureVersion.LATEST and resolved_version != inferred_version:
        raise ValueError(
            f"Version mismatch: requested {resolved_version.value} "
            f"(expects {resolved_version.embedding_dimensions()} dims) "
            f"but provider returns {provider_dims} dims "
            f"(matches {inferred_version.value})"
        )

    return inferred_version
