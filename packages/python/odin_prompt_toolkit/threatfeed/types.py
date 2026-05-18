"""Type definitions for threat feed operations."""

from dataclasses import dataclass, field


@dataclass
class DetectionSignature:
    """A detection signature from the threat feed API."""

    version: str
    signature: str


@dataclass
class ThreatFeedEntry:
    """A single threat feed entry from the API response."""

    uuid: str
    title: str
    severity: str
    security_boundary: str
    detection_signatures: list[DetectionSignature] = field(default_factory=list)
    summary: str | None = None
    updated_at: str | None = None


@dataclass
class ThreatFeedResponse:
    """Paginated API response from GET /api/v1/threatfeed."""

    page: int
    total_pages: int
    total_count: int
    threat_feeds: list[ThreatFeedEntry]

    @classmethod
    def from_dict(cls, data: dict) -> "ThreatFeedResponse":
        """Parse a raw API response dict into a ThreatFeedResponse."""
        entries = []
        for entry_data in data.get("threat_feeds", []):
            sigs = [
                DetectionSignature(version=s["version"], signature=s["signature"])
                for s in entry_data.get("detection_signatures", [])
            ]
            entries.append(
                ThreatFeedEntry(
                    uuid=entry_data["uuid"],
                    title=entry_data["title"],
                    severity=entry_data.get("severity", "low"),
                    security_boundary=entry_data.get("security_boundary", ""),
                    detection_signatures=sigs,
                    summary=entry_data.get("summary"),
                    updated_at=entry_data.get("updated_at"),
                )
            )
        return cls(
            page=data["page"],
            total_pages=data["total_pages"],
            total_count=data["total_count"],
            threat_feeds=entries,
        )


@dataclass
class CachedSignature:
    """A cached signature entry with pre-computed bands."""

    uuid: str
    title: str
    severity: str
    security_boundary: str
    signature: str
    bands: list[str]
    updated_at: str | None = None


@dataclass
class SyncResult:
    """Result of a threat feed sync operation."""

    added: int = 0
    updated: int = 0
    total: int = 0


@dataclass
class ThreatMatch:
    """A match found when querying the threat feed cache."""

    uuid: str
    title: str
    severity: str
    security_boundary: str
    signature: str
    hamming_distance: int
    cosine_similarity: float

    def __str__(self) -> str:
        return (
            f"{self.title} ({self.severity}, {self.security_boundary}) "
            f"- cosine: {self.cosine_similarity:.4f}, hamming: {self.hamming_distance}"
        )


__all__ = [
    "DetectionSignature",
    "ThreatFeedEntry",
    "ThreatFeedResponse",
    "CachedSignature",
    "SyncResult",
    "ThreatMatch",
]
