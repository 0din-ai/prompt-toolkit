"""Tests for the threat feed API client."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from aioresponses import aioresponses as mock_aiohttp

from odin_prompt_toolkit.error import ThreatFeedApiError
from odin_prompt_toolkit.threatfeed.client import ThreatFeedClient


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://test.0din.ai"
TOKEN = "test-token-abc123"


def make_entry(
    uuid: str,
    title: str = "Test Vuln",
    severity: str = "high",
    security_boundary: str = "guardrail_jailbreak",
    v1_sig: str | None = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
) -> dict:
    sigs = []
    if v1_sig:
        sigs.append({"version": "v1", "signature": v1_sig})
    return {
        "uuid": uuid,
        "title": title,
        "summary": "Test summary",
        "severity": severity,
        "security_boundary": security_boundary,
        "source": "internal",
        "disclosed_at": "2025-01-10T12:00:00.000Z",
        "published_at": "2025-01-15T12:00:00.000Z",
        "updated_at": "2025-03-01T10:00:00.000Z",
        "detection_signatures": sigs,
        "models": [],
        "messages": [],
        "taxonomies": [],
        "test_results": [],
        "metadata": [],
        "reference_urls": [],
        "variant_prompts": [],
    }


def page_response(entries: list, page: int = 1, total_pages: int = 1) -> dict:
    return {
        "page": page,
        "total_pages": total_pages,
        "total_count": len(entries),
        "threat_feeds": entries,
    }


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestThreatFeedClientConstructor:
    def test_requires_token(self, monkeypatch):
        monkeypatch.delenv("ODIN_THREATFEED_API_TOKEN", raising=False)
        monkeypatch.delenv("ODIN_API_TOKEN", raising=False)
        with pytest.raises(ThreatFeedApiError, match="API token required"):
            ThreatFeedClient()

    def test_token_from_dedicated_env(self, monkeypatch):
        monkeypatch.delenv("ODIN_API_TOKEN", raising=False)
        monkeypatch.setenv("ODIN_THREATFEED_API_TOKEN", "dedicated-token")
        client = ThreatFeedClient()
        assert client._api_token == "dedicated-token"

    def test_token_falls_back_to_odin_api_token(self, monkeypatch):
        monkeypatch.delenv("ODIN_THREATFEED_API_TOKEN", raising=False)
        monkeypatch.setenv("ODIN_API_TOKEN", "portal-token")
        client = ThreatFeedClient()
        assert client._api_token == "portal-token"

    def test_dedicated_env_takes_precedence_over_shared(self, monkeypatch):
        monkeypatch.setenv("ODIN_THREATFEED_API_TOKEN", "dedicated-token")
        monkeypatch.setenv("ODIN_API_TOKEN", "portal-token")
        client = ThreatFeedClient()
        assert client._api_token == "dedicated-token"

    def test_explicit_token_overrides_all_env(self, monkeypatch):
        monkeypatch.setenv("ODIN_THREATFEED_API_TOKEN", "dedicated-token")
        monkeypatch.setenv("ODIN_API_TOKEN", "portal-token")
        client = ThreatFeedClient(api_token="explicit-token")
        assert client._api_token == "explicit-token"

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("ODIN_THREATFEED_BASE_URL", raising=False)
        client = ThreatFeedClient(api_token=TOKEN)
        assert client.base_url == "https://0din.ai"

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("ODIN_THREATFEED_BASE_URL", "https://staging.0din.ai")
        client = ThreatFeedClient(api_token=TOKEN)
        assert client.base_url == "https://staging.0din.ai"

    def test_explicit_base_url(self, monkeypatch):
        monkeypatch.delenv("ODIN_THREATFEED_BASE_URL", raising=False)
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)
        assert client.base_url == BASE_URL


# ---------------------------------------------------------------------------
# fetch_all tests
# ---------------------------------------------------------------------------


def _url_with_params(page: int = 1, per_page: int = 100, since: str | None = None) -> str:
    """Build the exact URL that the client will request (params merged in)."""
    url = f"{BASE_URL}/api/v1/threatfeed?page={page}&per_page={per_page}"
    if since:
        from urllib.parse import quote
        url += f"&q%5Bupdated_at_gteq%5D={quote(since, safe='')}"
    return url


class TestFetchAll:
    @pytest.mark.asyncio
    async def test_single_page(self):
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)
        entries = [make_entry("aaa"), make_entry("bbb")]

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                payload=page_response(entries),
            )
            result = await client.fetch_all()

        assert len(result) == 2
        assert result[0].uuid == "aaa"
        assert result[1].uuid == "bbb"

    @pytest.mark.asyncio
    async def test_pagination_fetches_all_pages(self):
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL, per_page=2)
        page1 = [make_entry("p1e1"), make_entry("p1e2")]
        page2 = [make_entry("p2e1")]

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(page=1, per_page=2),
                payload=page_response(page1, page=1, total_pages=2),
            )
            m.get(
                _url_with_params(page=2, per_page=2),
                payload=page_response(page2, page=2, total_pages=2),
            )
            result = await client.fetch_all()

        assert len(result) == 3
        uuids = [e.uuid for e in result]
        assert "p1e1" in uuids
        assert "p1e2" in uuids
        assert "p2e1" in uuids

    @pytest.mark.asyncio
    async def test_auth_header_no_bearer_prefix(self):
        """API token must be sent as-is, without 'Bearer ' prefix."""
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                payload=page_response([]),
            )
            await client.fetch_all()

        # Verify the token stored on the client is raw (no Bearer prefix)
        assert not client._api_token.startswith("Bearer ")
        assert client._api_token == TOKEN

    @pytest.mark.asyncio
    async def test_empty_response(self):
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                payload={"page": 1, "total_pages": 1, "total_count": 0, "threat_feeds": []},
            )
            result = await client.fetch_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_401_raises_api_error(self):
        client = ThreatFeedClient(api_token="bad-token", base_url=BASE_URL)

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                status=401,
                body="Unauthorized",
            )
            with pytest.raises(ThreatFeedApiError) as exc_info:
                await client.fetch_all()

        assert exc_info.value.status_code == 401
        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_500_raises_api_error(self):
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                status=500,
                body="Internal Server Error",
            )
            with pytest.raises(ThreatFeedApiError) as exc_info:
                await client.fetch_all()

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_incremental_since_param_included(self):
        """q[updated_at_gteq] must be included when since is provided."""
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)
        since = "2025-03-01T00:00:00Z"

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(since=since),
                payload=page_response([]),
            )
            # If the URL didn't match (since param missing), aioresponses raises
            # ClientConnectionError, so a successful call proves the param was sent.
            result = await client.fetch_all(since=since)

        assert result == []

    @pytest.mark.asyncio
    async def test_no_since_param_when_not_provided(self):
        """q[updated_at_gteq] must NOT appear when since is None."""
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                payload=page_response([]),
            )
            # Exact URL match (no since param) — any extra params would cause a
            # ClientConnectionError, proving the client omitted updated_at_gteq.
            result = await client.fetch_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_parses_detection_signatures(self):
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)
        entry = make_entry(
            "aaa",
            v1_sig="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
        )
        entry["detection_signatures"].append(
            {"version": "v0", "signature": "1111111111111111111111111111111111111111111111111111111111111111"}
        )

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                payload=page_response([entry]),
            )
            result = await client.fetch_all()

        assert len(result) == 1
        sigs = result[0].detection_signatures
        assert len(sigs) == 2
        versions = {s.version for s in sigs}
        assert "v0" in versions
        assert "v1" in versions

    @pytest.mark.asyncio
    async def test_entry_with_no_signatures(self):
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)
        entry = make_entry("aaa", v1_sig=None)

        with mock_aiohttp() as m:
            m.get(
                _url_with_params(),
                payload=page_response([entry]),
            )
            result = await client.fetch_all()

        assert len(result) == 1
        assert result[0].detection_signatures == []


# ---------------------------------------------------------------------------
# fetch_one tests
# ---------------------------------------------------------------------------


class TestFetchOne:
    @pytest.mark.asyncio
    async def test_fetch_one_success(self):
        uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)
        entry = make_entry(uuid)

        with mock_aiohttp() as m:
            m.get(
                f"{BASE_URL}/api/v1/threatfeed/{uuid}",
                payload=entry,
            )
            result = await client.fetch_one(uuid)

        assert result.uuid == uuid
        assert result.title == "Test Vuln"

    @pytest.mark.asyncio
    async def test_fetch_one_404_raises(self):
        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)

        with mock_aiohttp() as m:
            m.get(
                f"{BASE_URL}/api/v1/threatfeed/nonexistent",
                status=404,
                body="Not Found",
            )
            with pytest.raises(ThreatFeedApiError) as exc_info:
                await client.fetch_one("nonexistent")

        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# aiohttp import guard
# ---------------------------------------------------------------------------


class TestAiohttpImportGuard:
    @pytest.mark.asyncio
    async def test_missing_aiohttp_raises_helpful_error(self, monkeypatch):
        import sys
        # Simulate aiohttp not being installed
        monkeypatch.setitem(sys.modules, "aiohttp", None)

        client = ThreatFeedClient(api_token=TOKEN, base_url=BASE_URL)
        with pytest.raises(ThreatFeedApiError, match="aiohttp is required"):
            await client.fetch_all()
