"""High-level comparison API for threat feed matching."""

from __future__ import annotations

from odin_prompt_toolkit.types import SignatureResult

from .cache import ThreatFeedCache
from .types import ThreatMatch


def compare_to_threatfeed(
    result: SignatureResult,
    cache: ThreatFeedCache,
    threshold: float = 0.85,
    max_results: int = 10,
) -> list[ThreatMatch]:
    """Compare a signature result against the threat feed cache.

    Extracts the primary signature (family 0) from the result and queries
    the cache for similar known threat signatures.

    Args:
        result: Signature result from sign_text().
        cache: Pre-loaded threat feed cache.
        threshold: Minimum cosine similarity threshold (default: 0.85).
        max_results: Maximum number of results to return (default: 10).

    Returns:
        List of ThreatMatch objects sorted by cosine similarity descending.

    Example:
        >>> from odin_prompt_toolkit import sign_text, SignatureVersion
        >>> from odin_prompt_toolkit.threatfeed import ThreatFeedCache, compare_to_threatfeed
        >>>
        >>> result = await sign_text("suspicious prompt", version=SignatureVersion.V1)
        >>> cache = ThreatFeedCache(version=SignatureVersion.V1)
        >>> cache.load()
        >>> matches = compare_to_threatfeed(result, cache, threshold=0.85)
        >>> for m in matches:
        ...     print(f"Match: {m.title} (similarity: {m.cosine_similarity:.3f})")
    """
    primary_sig = result.lsh.signatures[0].signature
    return cache.query(primary_sig, threshold=threshold, max_results=max_results)
