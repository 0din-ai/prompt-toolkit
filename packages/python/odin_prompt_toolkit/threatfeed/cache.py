"""Threat feed cache with band-indexed similarity lookup."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from odin_prompt_toolkit.error import ThreatFeedCacheError
from odin_prompt_toolkit.lsh import cosine_from_hamming, hamming_distance_hex
from odin_prompt_toolkit.types import SignatureVersion

if TYPE_CHECKING:
    from odin_prompt_toolkit.threatfeed.client import ThreatFeedClient

from .types import CachedSignature, SyncResult, ThreatMatch

# Schema version for the cache file format.
CACHE_SCHEMA_VERSION = 1

# Default number of bands for LSH indexing.
DEFAULT_BANDS = 16

# Default number of bits per signature.
DEFAULT_BITS = 256


def compute_bands(signature: str, num_bands: int = DEFAULT_BANDS) -> list[str]:
    """Compute bands from a hex signature string.

    Splits a 64 hex char signature into `num_bands` equal-length bands.

    Args:
        signature: Hex string signature (e.g., 64 chars for 256 bits).
        num_bands: Number of bands to split into (default: 16).

    Returns:
        List of hex strings, one per band.

    Raises:
        ValueError: If the signature is too short to split into the requested bands.
    """
    if not signature or len(signature) < num_bands:
        raise ValueError(
            f"Signature too short to split into {num_bands} bands: "
            f"{len(signature)} chars (need at least {num_bands})"
        )
    band_len = len(signature) // num_bands
    return [signature[i * band_len : (i + 1) * band_len] for i in range(num_bands)]


class ThreatFeedCache:
    """Threat feed cache with band-indexed similarity lookup.

    Caches detection signatures from the 0din threat feed API and provides
    fast similarity queries using LSH band indexing.

    Args:
        version: Signature version to cache (V0 or V1).
        cache_dir: Override cache directory path.
        bands: Number of bands for LSH indexing (default: 16).
    """

    def __init__(
        self,
        version: SignatureVersion,
        cache_dir: str | Path | None = None,
        bands: int = DEFAULT_BANDS,
    ):
        self._version = version.resolve()
        self._bands = bands
        self._bits = DEFAULT_BITS
        self._entries: list[CachedSignature] = []
        self._band_index: dict[str, list[int]] = {}
        self._synced_at: str | None = None
        self._source_url: str = "https://0din.ai"

        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
        elif env_dir := os.environ.get("ODIN_PROMPT_TOOLKIT_THREATFEED_CACHE"):
            self._cache_dir = Path(env_dir)
        else:
            self._cache_dir = Path.home() / ".odin-prompt-toolkit" / "threatfeed"

    def load(self) -> bool:
        """Load cache from disk.

        Returns:
            True if cache was loaded successfully, False if no cache exists.

        Raises:
            ThreatFeedCacheError: If the cache file is corrupt.
        """
        path = self._cache_file_path()
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise ThreatFeedCacheError(f"Corrupt cache file: {e}") from e

        if data.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False

        self._entries = [
            CachedSignature(
                uuid=entry["uuid"],
                title=entry["title"],
                severity=entry["severity"],
                security_boundary=entry["security_boundary"],
                signature=entry["signature"],
                bands=entry["bands"],
                updated_at=entry.get("updated_at"),
            )
            for entry in data.get("entries", [])
        ]
        self._band_index = data.get("band_index", {})
        self._synced_at = data.get("synced_at")
        self._source_url = data.get("source_url", "https://0din.ai")

        return True

    def save(self) -> None:
        """Save cache to disk with atomic write (temp file + rename).

        Raises:
            ThreatFeedCacheError: If write fails.
        """
        from datetime import datetime, timezone

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "signature_version": self._version.value,
            "synced_at": self._synced_at or datetime.now(timezone.utc).isoformat(),
            "source_url": self._source_url,
            "entry_count": len(self._entries),
            "lsh_config": {"bits": self._bits, "bands": self._bands},
            "entries": [
                {
                    "uuid": e.uuid,
                    "title": e.title,
                    "severity": e.severity,
                    "security_boundary": e.security_boundary,
                    "signature": e.signature,
                    "bands": e.bands,
                    "updated_at": e.updated_at,
                }
                for e in self._entries
            ],
            "band_index": self._band_index,
        }

        path = self._cache_file_path()
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".tmp", prefix="cache-"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2)
                os.replace(tmp_path, str(path))
            except Exception:
                os.unlink(tmp_path)
                raise
        except OSError as e:
            raise ThreatFeedCacheError(f"Failed to write cache: {e}") from e

    async def sync(self, client: "ThreatFeedClient", full: bool = False) -> SyncResult:
        """Sync signatures from the threat feed API.

        Args:
            client: Threat feed API client.
            full: If True, fetch all entries. If False, incremental sync.

        Returns:
            SyncResult with counts of added/updated entries.
        """
        from datetime import datetime, timezone

        since = None if full else self._last_updated_at()
        self._source_url = client.base_url

        entries = await client.fetch_all(since=since)
        version_str = self._version.value

        new_cached: list[CachedSignature] = []
        for entry in entries:
            for sig in entry.detection_signatures:
                if sig.version == version_str:
                    new_cached.append(
                        CachedSignature(
                            uuid=entry.uuid,
                            title=entry.title,
                            severity=entry.severity,
                            security_boundary=entry.security_boundary,
                            signature=sig.signature,
                            bands=compute_bands(sig.signature, self._bands),
                            updated_at=entry.updated_at,
                        )
                    )

        if full:
            total = len(new_cached)
            self._entries = new_cached
            result = SyncResult(added=total, updated=0, total=total)
        else:
            result = self._merge_entries(new_cached)

        self._rebuild_band_index()
        self._synced_at = datetime.now(timezone.utc).isoformat()
        self.save()

        return result

    def query(
        self,
        signature: str,
        threshold: float = 0.85,
        max_results: int = 10,
    ) -> list[ThreatMatch]:
        """Query the cache for signatures similar to the given query.

        Uses band-indexed candidate selection followed by Hamming distance verification.

        Args:
            signature: 64 hex char signature to query (raw, no 0din- prefix).
            threshold: Minimum cosine similarity threshold (default: 0.85).
            max_results: Maximum number of results to return (default: 10).

        Returns:
            List of ThreatMatch objects sorted by cosine similarity descending.
        """
        query_bands = compute_bands(signature, self._bands)

        # Collect candidate indices from band index
        candidate_indices: set[int] = set()
        for band_idx, band_val in enumerate(query_bands):
            key = f"{band_idx}:{band_val}"
            if key in self._band_index:
                candidate_indices.update(self._band_index[key])

        # Verify candidates with Hamming distance
        matches: list[ThreatMatch] = []
        for idx in candidate_indices:
            if idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            dist = hamming_distance_hex(signature, entry.signature)
            cosine = cosine_from_hamming(dist, self._bits)
            if cosine >= threshold:
                matches.append(
                    ThreatMatch(
                        uuid=entry.uuid,
                        title=entry.title,
                        severity=entry.severity,
                        security_boundary=entry.security_boundary,
                        signature=entry.signature,
                        hamming_distance=dist,
                        cosine_similarity=cosine,
                    )
                )

        # Sort by cosine similarity descending
        matches.sort(key=lambda m: m.cosine_similarity, reverse=True)
        return matches[:max_results]

    @property
    def entry_count(self) -> int:
        """Get the number of entries in the cache."""
        return len(self._entries)

    @property
    def last_synced(self) -> str | None:
        """Get the timestamp of the last sync."""
        return self._synced_at

    @property
    def entries(self) -> list[CachedSignature]:
        """Get all cached entries."""
        return self._entries

    def load_entries(self, entries: list[CachedSignature]) -> None:
        """Load entries directly (for testing without disk I/O)."""
        self._entries = entries
        self._rebuild_band_index()

    # --- Private methods ---

    def _cache_file_path(self) -> Path:
        return self._cache_dir / f"cache-{self._version.value}.json"

    def _last_updated_at(self) -> str | None:
        updated_ats = [e.updated_at for e in self._entries if e.updated_at]
        return max(updated_ats) if updated_ats else None

    def _merge_entries(self, new_entries: list[CachedSignature]) -> SyncResult:
        existing: dict[str, int] = {e.uuid: i for i, e in enumerate(self._entries)}
        added = 0
        updated = 0

        for entry in new_entries:
            if entry.uuid in existing:
                self._entries[existing[entry.uuid]] = entry
                updated += 1
            else:
                existing[entry.uuid] = len(self._entries)
                self._entries.append(entry)
                added += 1

        return SyncResult(added=added, updated=updated, total=len(self._entries))

    def _rebuild_band_index(self) -> None:
        self._band_index = defaultdict(list)
        for idx, entry in enumerate(self._entries):
            for band_idx, band_val in enumerate(entry.bands):
                key = f"{band_idx}:{band_val}"
                self._band_index[key].append(idx)
        # Convert defaultdict to regular dict for JSON serialization
        self._band_index = dict(self._band_index)
