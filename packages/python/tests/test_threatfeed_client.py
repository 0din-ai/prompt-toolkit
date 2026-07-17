"""Tests for the threat feed API client."""


from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from odin_prompt_toolkit.error import ThreatFeedApiError
from odin_prompt_toolkit.threatfeed.client import ThreatFeedClient

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://test.0din.ai"
TOKEN = "test-token-abc123"

THREATFEED_PATH = "/api/v1/threatfeed"
FETCH_ONE_PATH = "/api/v1/threatfeed/{uuid}"


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
# Real aiohttp server helpers (replaces aioresponses URL mocking)
#
# aioresponses is incompatible with aiohttp>=3.14 (it never adapted to the
# `stream_writer` kwarg aiohttp added to ClientResponse.__init__). Instead of
# mocking at the transport layer, these tests spin up a genuine aiohttp
# server on loopback and point the client under test at it, using aiohttp's
# own first-party test utilities.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def threatfeed_server(*routes: tuple):
    """Start a real aiohttp server exposing the given (method, path, handler) routes.

    Yields the base URL the ThreatFeedClient should be pointed at.
    """
    app = web.Application()
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)

    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


def _threatfeed_handler(
    *,
    per_page: int = 100,
    since: str | None = None,
    pages: dict[int, dict] | None = None,
    status: int | None = None,
    body: str | None = None,
):
    """Build a GET handler for /api/v1/threatfeed.

    Validates the query string matches exactly what the client is expected
    to send (page/per_page/optional since) — the same strict URL matching
    aioresponses performed. A mismatch responds 400, which the client
    surfaces as a ThreatFeedApiError, so a successful call still proves the
    expected query params were sent (or omitted).
    """

    async def handler(request: web.Request) -> web.Response:
        page = int(request.query.get("page", "0"))
        expected = {"page": str(page), "per_page": str(per_page)}
        if since:
            expected["q[updated_at_gteq]"] = since
        if dict(request.query) != expected:
            return web.Response(status=400, text="unexpected query params")

        if status is not None:
            return web.Response(status=status, text=body or "")

        assert pages is not None
        return web.json_response(pages[page])

    return handler


def _fetch_one_handler(
    *,
    entries_by_uuid: dict[str, dict] | None = None,
    status: int | None = None,
    body: str | None = None,
):
    """Build a GET handler for /api/v1/threatfeed/{uuid}."""

    async def handler(request: web.Request) -> web.Response:
        if status is not None:
            return web.Response(status=status, text=body or "")

        assert entries_by_uuid is not None
        uuid = request.match_info["uuid"]
        return web.json_response(entries_by_uuid[uuid])

    return handler


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


class TestFetchAll:
    @pytest.mark.asyncio
    async def test_single_page(self):
        entries = [make_entry("aaa"), make_entry("bbb")]
        handler = _threatfeed_handler(pages={1: page_response(entries)})

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            result = await client.fetch_all()

        assert len(result) == 2
        assert result[0].uuid == "aaa"
        assert result[1].uuid == "bbb"

    @pytest.mark.asyncio
    async def test_pagination_fetches_all_pages(self):
        page1 = [make_entry("p1e1"), make_entry("p1e2")]
        page2 = [make_entry("p2e1")]
        handler = _threatfeed_handler(
            per_page=2,
            pages={
                1: page_response(page1, page=1, total_pages=2),
                2: page_response(page2, page=2, total_pages=2),
            },
        )

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url, per_page=2)
            result = await client.fetch_all()

        assert len(result) == 3
        uuids = [e.uuid for e in result]
        assert "p1e1" in uuids
        assert "p1e2" in uuids
        assert "p2e1" in uuids

    @pytest.mark.asyncio
    async def test_auth_header_no_bearer_prefix(self):
        """API token must be sent as-is, without 'Bearer ' prefix."""
        handler = _threatfeed_handler(pages={1: page_response([])})

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            await client.fetch_all()

        # Verify the token stored on the client is raw (no Bearer prefix)
        assert not client._api_token.startswith("Bearer ")
        assert client._api_token == TOKEN

    @pytest.mark.asyncio
    async def test_empty_response(self):
        handler = _threatfeed_handler(
            pages={1: {"page": 1, "total_pages": 1, "total_count": 0, "threat_feeds": []}}
        )

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            result = await client.fetch_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_401_raises_api_error(self):
        handler = _threatfeed_handler(status=401, body="Unauthorized")

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token="bad-token", base_url=base_url)
            with pytest.raises(ThreatFeedApiError) as exc_info:
                await client.fetch_all()

        assert exc_info.value.status_code == 401
        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_500_raises_api_error(self):
        handler = _threatfeed_handler(status=500, body="Internal Server Error")

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            with pytest.raises(ThreatFeedApiError) as exc_info:
                await client.fetch_all()

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_incremental_since_param_included(self):
        """q[updated_at_gteq] must be included when since is provided."""
        since = "2025-03-01T00:00:00Z"
        handler = _threatfeed_handler(since=since, pages={1: page_response([])})

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            # If the request didn't include the since param, the handler's
            # exact query match fails and the client raises ThreatFeedApiError,
            # so a successful call proves the param was sent.
            result = await client.fetch_all(since=since)

        assert result == []

    @pytest.mark.asyncio
    async def test_no_since_param_when_not_provided(self):
        """q[updated_at_gteq] must NOT appear when since is None."""
        handler = _threatfeed_handler(pages={1: page_response([])})

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            # Exact query match (no since param) — any extra params would
            # cause a ThreatFeedApiError, proving the client omitted
            # updated_at_gteq.
            result = await client.fetch_all()

        assert result == []

    @pytest.mark.asyncio
    async def test_parses_detection_signatures(self):
        entry = make_entry(
            "aaa",
            v1_sig="a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
        )
        entry["detection_signatures"].append(
            {"version": "v0", "signature": "1111111111111111111111111111111111111111111111111111111111111111"}
        )
        handler = _threatfeed_handler(pages={1: page_response([entry])})

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            result = await client.fetch_all()

        assert len(result) == 1
        sigs = result[0].detection_signatures
        assert len(sigs) == 2
        versions = {s.version for s in sigs}
        assert "v0" in versions
        assert "v1" in versions

    @pytest.mark.asyncio
    async def test_entry_with_no_signatures(self):
        entry = make_entry("aaa", v1_sig=None)
        handler = _threatfeed_handler(pages={1: page_response([entry])})

        async with threatfeed_server(("GET", THREATFEED_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
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
        entry = make_entry(uuid)
        handler = _fetch_one_handler(entries_by_uuid={uuid: entry})

        async with threatfeed_server(("GET", FETCH_ONE_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
            result = await client.fetch_one(uuid)

        assert result.uuid == uuid
        assert result.title == "Test Vuln"

    @pytest.mark.asyncio
    async def test_fetch_one_404_raises(self):
        handler = _fetch_one_handler(status=404, body="Not Found")

        async with threatfeed_server(("GET", FETCH_ONE_PATH, handler)) as base_url:
            client = ThreatFeedClient(api_token=TOKEN, base_url=base_url)
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
