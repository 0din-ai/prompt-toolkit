/**
 * Tests for the threat feed API client.
 *
 * Uses Jest's global fetch mock (no extra deps required — the project
 * already targets Node 18+ where fetch is built-in).
 */

import { ThreatFeedApiError } from '../src/error';
import { ThreatFeedClient } from '../src/threatfeed/client';

const BASE_URL = 'https://test.0din.ai';
const TOKEN = 'test-token-abc123';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeEntry(
  uuid: string,
  {
    title = 'Test Vuln',
    severity = 'high',
    securityBoundary = 'guardrail_jailbreak',
    v1Sig = 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2',
  }: {
    title?: string;
    severity?: string;
    securityBoundary?: string;
    v1Sig?: string | null;
  } = {},
) {
  return {
    uuid,
    title,
    summary: 'Test summary',
    severity,
    security_boundary: securityBoundary,
    source: 'internal',
    disclosed_at: '2025-01-10T12:00:00.000Z',
    published_at: '2025-01-15T12:00:00.000Z',
    updated_at: '2025-03-01T10:00:00.000Z',
    detection_signatures: v1Sig ? [{ version: 'v1', signature: v1Sig }] : [],
    models: [],
    messages: [],
    taxonomies: [],
    test_results: [],
    metadata: [],
    reference_urls: [],
    variant_prompts: [],
  };
}

function pageResponse(
  entries: ReturnType<typeof makeEntry>[],
  page = 1,
  totalPages = 1,
) {
  return {
    page,
    total_pages: totalPages,
    total_count: entries.length,
    threat_feeds: entries,
  };
}

function mockFetch(responses: Array<{ body: unknown; status?: number }>): jest.SpyInstance {
  let callCount = 0;
  return jest.spyOn(global, 'fetch').mockImplementation(async (_url, _init) => {
    const resp = responses[callCount] ?? responses[responses.length - 1];
    callCount++;
    const status = resp.status ?? 200;
    return {
      ok: status >= 200 && status < 300,
      status,
      text: async () => (typeof resp.body === 'string' ? resp.body : JSON.stringify(resp.body)),
      json: async () => (typeof resp.body === 'string' ? JSON.parse(resp.body) : resp.body),
    } as Response;
  });
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

describe('ThreatFeedClient constructor', () => {
  const OLD_ENV = process.env;

  beforeEach(() => {
    process.env = { ...OLD_ENV };
    delete process.env.ODIN_THREATFEED_API_TOKEN;
    delete process.env.ODIN_API_TOKEN;
    delete process.env.ODIN_THREATFEED_BASE_URL;
  });

  afterEach(() => {
    process.env = OLD_ENV;
  });

  test('throws when no token provided', () => {
    expect(() => new ThreatFeedClient()).toThrow(ThreatFeedApiError);
    expect(() => new ThreatFeedClient()).toThrow('API token required');
  });

  test('reads token from dedicated env', () => {
    process.env.ODIN_THREATFEED_API_TOKEN = 'dedicated-token';
    const client = new ThreatFeedClient();
    expect(client['apiToken']).toBe('dedicated-token');
  });

  test('falls back to ODIN_API_TOKEN', () => {
    process.env.ODIN_API_TOKEN = 'portal-token';
    const client = new ThreatFeedClient();
    expect(client['apiToken']).toBe('portal-token');
  });

  test('dedicated env takes precedence over shared', () => {
    process.env.ODIN_THREATFEED_API_TOKEN = 'dedicated-token';
    process.env.ODIN_API_TOKEN = 'portal-token';
    const client = new ThreatFeedClient();
    expect(client['apiToken']).toBe('dedicated-token');
  });

  test('explicit token overrides all env', () => {
    process.env.ODIN_THREATFEED_API_TOKEN = 'dedicated-token';
    process.env.ODIN_API_TOKEN = 'portal-token';
    const client = new ThreatFeedClient({ apiToken: 'explicit-token' });
    expect(client['apiToken']).toBe('explicit-token');
  });

  test('default base URL is 0din.ai', () => {
    const client = new ThreatFeedClient({ apiToken: TOKEN });
    expect(client.baseUrl).toBe('https://0din.ai');
  });

  test('base URL from env', () => {
    process.env.ODIN_THREATFEED_BASE_URL = 'https://staging.0din.ai';
    const client = new ThreatFeedClient({ apiToken: TOKEN });
    expect(client.baseUrl).toBe('https://staging.0din.ai');
  });

  test('explicit base URL overrides env', () => {
    process.env.ODIN_THREATFEED_BASE_URL = 'https://staging.0din.ai';
    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    expect(client.baseUrl).toBe(BASE_URL);
  });
});

// ---------------------------------------------------------------------------
// fetchAll
// ---------------------------------------------------------------------------

describe('ThreatFeedClient.fetchAll', () => {
  afterEach(() => jest.restoreAllMocks());

  test('fetches single page', async () => {
    const entries = [makeEntry('aaa'), makeEntry('bbb')];
    mockFetch([{ body: pageResponse(entries) }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    const result = await client.fetchAll();

    expect(result).toHaveLength(2);
    expect(result[0].uuid).toBe('aaa');
    expect(result[1].uuid).toBe('bbb');
  });

  test('paginates through all pages', async () => {
    const page1 = [makeEntry('p1e1'), makeEntry('p1e2')];
    const page2 = [makeEntry('p2e1')];

    mockFetch([
      { body: pageResponse(page1, 1, 2) },
      { body: pageResponse(page2, 2, 2) },
    ]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    const result = await client.fetchAll();

    expect(result).toHaveLength(3);
    const uuids = result.map((e) => e.uuid);
    expect(uuids).toContain('p1e1');
    expect(uuids).toContain('p1e2');
    expect(uuids).toContain('p2e1');
  });

  test('auth header has no Bearer prefix', async () => {
    const fetchSpy = mockFetch([{ body: pageResponse([]) }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    await client.fetchAll();

    const [, init] = fetchSpy.mock.calls[0];
    const headers = init?.headers as Record<string, string>;
    expect(headers['Authorization']).toBe(TOKEN);
    expect(headers['Authorization']).not.toMatch(/^Bearer /);
  });

  test('empty response returns empty array', async () => {
    mockFetch([
      {
        body: { page: 1, total_pages: 1, total_count: 0, threat_feeds: [] },
      },
    ]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    const result = await client.fetchAll();
    expect(result).toHaveLength(0);
  });

  test('401 throws ThreatFeedApiError with status code', async () => {
    mockFetch([{ body: 'Unauthorized', status: 401 }]);

    const client = new ThreatFeedClient({ apiToken: 'bad-token', baseUrl: BASE_URL });
    await expect(client.fetchAll()).rejects.toThrow(ThreatFeedApiError);
    await expect(client.fetchAll()).rejects.toMatchObject({ statusCode: 401 });
  });

  test('500 throws ThreatFeedApiError with status code', async () => {
    mockFetch([{ body: 'Internal Server Error', status: 500 }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    await expect(client.fetchAll()).rejects.toMatchObject({ statusCode: 500 });
  });

  test('includes q[updated_at_gteq] when since is provided', async () => {
    const fetchSpy = mockFetch([{ body: pageResponse([]) }]);
    const since = '2025-03-01T00:00:00Z';

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    await client.fetchAll({ since });

    const [url] = fetchSpy.mock.calls[0];
    expect(String(url)).toContain('updated_at_gteq');
    expect(String(url)).toContain(encodeURIComponent(since));
  });

  test('omits q[updated_at_gteq] when since is not provided', async () => {
    const fetchSpy = mockFetch([{ body: pageResponse([]) }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    await client.fetchAll();

    const [url] = fetchSpy.mock.calls[0];
    expect(String(url)).not.toContain('updated_at_gteq');
  });

  test('parses detection signatures', async () => {
    const entry = makeEntry('aaa');
    entry.detection_signatures.push({
      version: 'v0',
      signature: '1111111111111111111111111111111111111111111111111111111111111111',
    });
    mockFetch([{ body: pageResponse([entry]) }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    const result = await client.fetchAll();

    expect(result[0].detectionSignatures).toHaveLength(2);
    const versions = result[0].detectionSignatures.map((s) => s.version);
    expect(versions).toContain('v0');
    expect(versions).toContain('v1');
  });

  test('entry with no detection signatures', async () => {
    const entry = makeEntry('aaa', { v1Sig: null });
    mockFetch([{ body: pageResponse([entry]) }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    const result = await client.fetchAll();

    expect(result[0].detectionSignatures).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// fetchOne
// ---------------------------------------------------------------------------

describe('ThreatFeedClient.fetchOne', () => {
  afterEach(() => jest.restoreAllMocks());

  test('fetches single entry by UUID', async () => {
    const uuid = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';
    mockFetch([{ body: makeEntry(uuid) }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    const result = await client.fetchOne(uuid);

    expect(result.uuid).toBe(uuid);
    expect(result.title).toBe('Test Vuln');
  });

  test('404 throws ThreatFeedApiError', async () => {
    mockFetch([{ body: 'Not Found', status: 404 }]);

    const client = new ThreatFeedClient({ apiToken: TOKEN, baseUrl: BASE_URL });
    await expect(client.fetchOne('nonexistent')).rejects.toMatchObject({
      statusCode: 404,
    });
  });
});
