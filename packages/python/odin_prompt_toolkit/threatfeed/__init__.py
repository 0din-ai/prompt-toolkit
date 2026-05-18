"""Threat feed integration for fetching and caching known threat signatures.

This module provides the ability to fetch detection signatures from the 0din
portal's threat feed API, cache them locally with a band index, and perform
fast similarity lookup against the cache.

Example:
    >>> from odin_prompt_toolkit.threatfeed import ThreatFeedClient, ThreatFeedCache
    >>> from odin_prompt_toolkit import SignatureVersion
    >>>
    >>> # Sync signatures from the portal
    >>> client = ThreatFeedClient(api_token="your-api-token")
    >>> cache = ThreatFeedCache(version=SignatureVersion.V1)
    >>> await cache.sync(client, full=True)
    >>>
    >>> # Query for similar signatures
    >>> matches = cache.query("a1b2c3d4...", threshold=0.85)
"""

from odin_prompt_toolkit.threatfeed.cache import ThreatFeedCache
from odin_prompt_toolkit.threatfeed.client import ThreatFeedClient
from odin_prompt_toolkit.threatfeed.compare import compare_to_threatfeed
from odin_prompt_toolkit.threatfeed.types import (
    CachedSignature,
    SyncResult,
    ThreatFeedEntry,
    ThreatMatch,
)

__all__ = [
    "ThreatFeedClient",
    "ThreatFeedCache",
    "compare_to_threatfeed",
    "CachedSignature",
    "SyncResult",
    "ThreatFeedEntry",
    "ThreatMatch",
]
