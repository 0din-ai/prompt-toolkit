"""Threat feed API client for fetching signatures from the 0din portal."""

from __future__ import annotations

import asyncio
import os

from odin_prompt_toolkit.error import ThreatFeedApiError

from .types import DetectionSignature, ThreatFeedEntry


class ThreatFeedClient:
    """Client for the 0din threat feed API.

    Fetches detection signatures from the paginated threat feed endpoint.

    Token resolution order:
        1. Explicit ``api_token`` parameter
        2. ``ODIN_THREATFEED_API_TOKEN`` env var (dedicated)
        3. ``ODIN_API_TOKEN`` env var (shared with Thor / portal)

    Args:
        api_token: Raw API token (no Bearer prefix). Falls back to
            ODIN_THREATFEED_API_TOKEN, then ODIN_API_TOKEN env vars.
        base_url: API base URL (default: https://0din.ai). Falls back to
            ODIN_THREATFEED_BASE_URL env var.
        per_page: Page size for paginated requests (default: 100).
    """

    def __init__(
        self,
        api_token: str | None = None,
        base_url: str | None = None,
        per_page: int = 100,
    ):
        self._api_token = (
            api_token
            or os.environ.get("ODIN_THREATFEED_API_TOKEN")
            or os.environ.get("ODIN_API_TOKEN")
            or ""
        )
        if not self._api_token:
            raise ThreatFeedApiError(
                "API token required: pass api_token or set "
                "ODIN_THREATFEED_API_TOKEN / ODIN_API_TOKEN"
            )

        self._base_url = (
            base_url
            or os.environ.get("ODIN_THREATFEED_BASE_URL")
            or "https://0din.ai"
        )
        self._per_page = per_page

    @property
    def base_url(self) -> str:
        """Get the base URL of the API."""
        return self._base_url

    async def fetch_all(self, since: str | None = None) -> list[ThreatFeedEntry]:
        """Fetch all threat feed entries, paginating through all pages.

        Args:
            since: Optional ISO8601 timestamp to filter entries updated since.

        Returns:
            List of all threat feed entries.

        Raises:
            ThreatFeedApiError: On network or API errors.
        """
        try:
            import aiohttp
        except ImportError:
            raise ThreatFeedApiError(
                "aiohttp is required for threat feed fetching. "
                "Install with: pip install 0din-prompt-toolkit[threatfeed]"
            )

        all_entries: list[ThreatFeedEntry] = []
        page = 1

        async with aiohttp.ClientSession() as session:
            while True:
                data = await self._fetch_page(session, page, since)
                entries = self._parse_entries(data.get("threat_feeds", []))
                all_entries.extend(entries)

                total_pages = data.get("total_pages", 1)
                if page >= total_pages:
                    break
                page += 1

                # Rate limiting: 500ms delay between pages
                await asyncio.sleep(0.5)

        return all_entries

    async def fetch_one(self, uuid: str) -> ThreatFeedEntry:
        """Fetch a single threat feed entry by UUID.

        Args:
            uuid: Threat feed entry UUID.

        Returns:
            ThreatFeedEntry for the specified UUID.

        Raises:
            ThreatFeedApiError: On network or API errors.
        """
        try:
            import aiohttp
        except ImportError:
            raise ThreatFeedApiError(
                "aiohttp is required for threat feed fetching. "
                "Install with: pip install 0din-prompt-toolkit[threatfeed]"
            )

        url = f"{self._base_url}/api/v1/threatfeed/{uuid}"
        headers = {
            "Authorization": self._api_token,
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ThreatFeedApiError(
                        f"API returned status {response.status}: {text}",
                        status_code=response.status,
                    )
                data = await response.json()

        entries = self._parse_entries([data])
        if not entries:
            raise ThreatFeedApiError(f"No entry found for UUID: {uuid}")
        return entries[0]

    # --- Private methods ---

    async def _fetch_page(
        self,
        session: "aiohttp.ClientSession",
        page: int,
        since: str | None = None,
    ) -> dict:
        """Fetch a single page of threat feed entries."""
        import aiohttp

        params: dict[str, str | int] = {
            "page": page,
            "per_page": self._per_page,
        }
        if since:
            params["q[updated_at_gteq]"] = since

        url = f"{self._base_url}/api/v1/threatfeed"
        headers = {
            "Authorization": self._api_token,
            "Content-Type": "application/json",
        }

        try:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise ThreatFeedApiError(
                        f"API returned status {response.status}: {text}",
                        status_code=response.status,
                    )
                return await response.json()
        except aiohttp.ClientError as e:
            raise ThreatFeedApiError(f"Network error: {e}") from e

    @staticmethod
    def _parse_entries(entries_data: list[dict]) -> list[ThreatFeedEntry]:
        """Parse raw API response entries into ThreatFeedEntry objects."""
        result = []
        for data in entries_data:
            sigs = [
                DetectionSignature(version=s["version"], signature=s["signature"])
                for s in data.get("detection_signatures", [])
            ]
            result.append(
                ThreatFeedEntry(
                    uuid=data["uuid"],
                    title=data["title"],
                    severity=data.get("severity", "low"),
                    security_boundary=data.get("security_boundary", ""),
                    detection_signatures=sigs,
                    summary=data.get("summary"),
                    updated_at=data.get("updated_at"),
                )
            )
        return result
