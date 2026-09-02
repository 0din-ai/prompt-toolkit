import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { ThreatFeedCache, computeBands } from '../src/threatfeed/cache';
import { ThreatFeedClient } from '../src/threatfeed/client';
import { SignatureVersion } from '../src/types';
import type { CachedSignature, ThreatFeedEntry } from '../src/threatfeed/types';

const SIG_A = 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2';
const SIG_B = 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b3';
const SIG_ZEROS = '0000000000000000000000000000000000000000000000000000000000000000';
const SIG_ONES = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff';
const SIG_UNRELATED = '5678901234567890567890123456789056789012345678905678901234567890';

function makeEntry(
  uuid: string,
  sig: string,
  title: string = 'Test',
  severity: string = 'high',
): CachedSignature {
  return {
    uuid,
    title,
    severity,
    securityBoundary: 'guardrail_jailbreak',
    signature: sig,
    bands: computeBands(sig, 16),
  };
}

describe('computeBands', () => {
  test('basic split', () => {
    const bands = computeBands(SIG_A, 16);
    expect(bands).toHaveLength(16);
    expect(bands[0]).toBe('a1b2');
    expect(bands[1]).toBe('c3d4');
    expect(bands[15]).toBe('a1b2');
  });

  test('all zeros', () => {
    const bands = computeBands(SIG_ZEROS, 16);
    for (const band of bands) {
      expect(band).toBe('0000');
    }
  });

  test('all ones', () => {
    const bands = computeBands(SIG_ONES, 16);
    for (const band of bands) {
      expect(band).toBe('ffff');
    }
  });
});

describe('ThreatFeedCache', () => {
  test('empty cache query', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    const matches = cache.query(SIG_A);
    expect(matches).toHaveLength(0);
  });

  test('exact match', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    cache.loadEntries([makeEntry('test-uuid', SIG_A)]);

    const matches = cache.query(SIG_A);
    expect(matches).toHaveLength(1);
    expect(matches[0].uuid).toBe('test-uuid');
    expect(matches[0].hammingDistance).toBe(0);
    expect(Math.abs(matches[0].cosineSimilarity - 1.0)).toBeLessThan(1e-10);
  });

  test('near match', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    cache.loadEntries([
      makeEntry('entry-a', SIG_A),
      makeEntry('entry-b', SIG_B, 'Test B', 'medium'),
    ]);

    const matches = cache.query(SIG_A);
    expect(matches).toHaveLength(2);
    // Exact match first
    expect(matches[0].uuid).toBe('entry-a');
    expect(matches[0].hammingDistance).toBe(0);
    // Near match second
    expect(matches[1].uuid).toBe('entry-b');
    expect(matches[1].cosineSimilarity).toBeGreaterThan(0.99);
  });

  test('no match - no shared bands', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    cache.loadEntries([makeEntry('test-uuid', SIG_A)]);

    const matches = cache.query(SIG_UNRELATED);
    expect(matches).toHaveLength(0);
  });

  test('threshold filtering', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    cache.loadEntries([makeEntry('test-uuid', SIG_ONES)]);

    const matches = cache.query(SIG_ZEROS);
    expect(matches).toHaveLength(0);
  });

  test('max results', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    const entries = Array.from({ length: 20 }, (_, i) => makeEntry(`entry-${i}`, SIG_A));
    cache.loadEntries(entries);

    const matches = cache.query(SIG_A, { maxResults: 5 });
    expect(matches).toHaveLength(5);
  });

  test('entry count', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    expect(cache.entryCount).toBe(0);

    cache.loadEntries([makeEntry('a', SIG_A), makeEntry('b', SIG_B)]);
    expect(cache.entryCount).toBe(2);
  });

  test('save and load roundtrip', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'threatfeed-test-'));

    try {
      const cache = new ThreatFeedCache({
        version: SignatureVersion.V1,
        cacheDir: tmpDir,
      });
      cache.loadEntries([makeEntry('test-uuid', SIG_A, 'Test Entry')]);
      cache.save();

      // Load into a new cache
      const cache2 = new ThreatFeedCache({
        version: SignatureVersion.V1,
        cacheDir: tmpDir,
      });
      const loaded = cache2.load();
      expect(loaded).toBe(true);
      expect(cache2.entryCount).toBe(1);

      const matches = cache2.query(SIG_A);
      expect(matches).toHaveLength(1);
      expect(matches[0].uuid).toBe('test-uuid');
    } finally {
      fs.rmSync(tmpDir, { recursive: true });
    }
  });

  test('load nonexistent', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'threatfeed-test-'));

    try {
      const cache = new ThreatFeedCache({
        version: SignatureVersion.V1,
        cacheDir: tmpDir,
      });
      const loaded = cache.load();
      expect(loaded).toBe(false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true });
    }
  });

  test('schema version mismatch', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'threatfeed-test-'));

    try {
      const filePath = path.join(tmpDir, 'cache-v1.json');
      fs.writeFileSync(
        filePath,
        JSON.stringify({
          schema_version: 999,
          entries: [],
          band_index: {},
        }),
      );

      const cache = new ThreatFeedCache({
        version: SignatureVersion.V1,
        cacheDir: tmpDir,
      });
      const loaded = cache.load();
      expect(loaded).toBe(false);
    } finally {
      fs.rmSync(tmpDir, { recursive: true });
    }
  });
});

describe('ThreatFeedCache.sync', () => {
  afterEach(() => jest.restoreAllMocks());

  function fakeEntry(uuid: string, sig: string, taxonomies?: unknown[]): ThreatFeedEntry {
    return {
      uuid,
      title: 'Test',
      severity: 'high',
      securityBoundary: 'guardrail_jailbreak',
      detectionSignatures: [{ version: 'v1', signature: sig }],
      extra: taxonomies ? { taxonomies: taxonomies as never } : undefined,
    };
  }

  test('sync with fields carries extra.taxonomies onto cached entries', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'threatfeed-test-'));
    try {
      const client = new ThreatFeedClient({ apiToken: 'test-token' });
      const taxonomies = [{ category: 'c', strategy: 's', technique: 't' }];
      jest.spyOn(client, 'fetchAll').mockResolvedValue([fakeEntry('entry-a', SIG_A, taxonomies)]);

      const cache = new ThreatFeedCache({
        version: SignatureVersion.V1,
        cacheDir: tmpDir,
      });

      await cache.sync(client, { full: true, fields: ['taxonomies'] });

      expect(cache.entries).toHaveLength(1);
      expect(cache.entries[0].extra?.taxonomies).toEqual(taxonomies);
    } finally {
      fs.rmSync(tmpDir, { recursive: true });
    }
  });

  test('sync without fields leaves extra undefined', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'threatfeed-test-'));
    try {
      const client = new ThreatFeedClient({ apiToken: 'test-token' });
      jest.spyOn(client, 'fetchAll').mockResolvedValue([fakeEntry('entry-a', SIG_A)]);

      const cache = new ThreatFeedCache({
        version: SignatureVersion.V1,
        cacheDir: tmpDir,
      });

      await cache.sync(client, { full: true });

      expect(cache.entries).toHaveLength(1);
      expect(cache.entries[0].extra).toBeUndefined();
    } finally {
      fs.rmSync(tmpDir, { recursive: true });
    }
  });
});
