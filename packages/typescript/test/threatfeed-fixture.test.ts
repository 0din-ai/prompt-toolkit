/**
 * Cross-language validation tests using the shared fixture.
 */
import * as fs from 'fs';
import * as path from 'path';
import { ThreatFeedCache, computeBands } from '../src/threatfeed/cache';
import { SignatureVersion } from '../src/types';
import type { CachedSignature } from '../src/threatfeed/types';

const FIXTURE_PATH = path.join(
  __dirname,
  '..',
  '..',
  '..',
  'spec',
  'test-vectors',
  'threatfeed-fixture.json',
);

interface FixtureEntry {
  uuid: string;
  title: string;
  severity: string;
  security_boundary: string;
  signature: string;
  bands: string[];
}

interface QueryTest {
  name: string;
  query_signature: string;
  threshold: number;
  expected_match_uuids: string[];
  expected_top_match_uuid?: string;
  expected_top_hamming_distance?: number;
  expected_top_cosine_similarity?: number;
}

const fixture = JSON.parse(fs.readFileSync(FIXTURE_PATH, 'utf-8'));

function buildV1Cache(): ThreatFeedCache {
  const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
  const entries: CachedSignature[] = fixture.expected_v1_cache.entries.map(
    (e: FixtureEntry) => ({
      uuid: e.uuid,
      title: e.title,
      severity: e.severity,
      securityBoundary: e.security_boundary,
      signature: e.signature,
      bands: e.bands,
    }),
  );
  cache.loadEntries(entries);
  return cache;
}

describe('fixture bands', () => {
  test('bands match fixture expectations', () => {
    for (const entry of fixture.expected_v1_cache.entries as FixtureEntry[]) {
      const computed = computeBands(entry.signature, 16);
      expect(computed).toEqual(entry.bands);
    }
  });
});

describe('fixture version filtering', () => {
  test('v1 entry count', () => {
    expect(fixture.expected_v1_cache.entry_count).toBe(6);
  });

  test('v1 excludes no-signature entries', () => {
    const uuids = new Set(
      (fixture.expected_v1_cache.entries as FixtureEntry[]).map((e) => e.uuid),
    );
    expect(uuids.has('dddddddd-dddd-dddd-dddd-dddddddddddd')).toBe(false);
  });

  test('v1 excludes v0-only entries', () => {
    const uuids = new Set(
      (fixture.expected_v1_cache.entries as FixtureEntry[]).map((e) => e.uuid),
    );
    expect(uuids.has('11111111-1111-1111-1111-111111111111')).toBe(false);
  });

  test('v1 includes dual version entry', () => {
    const uuids = new Set(
      (fixture.expected_v1_cache.entries as FixtureEntry[]).map((e) => e.uuid),
    );
    expect(uuids.has('22222222-2222-2222-2222-222222222222')).toBe(true);
  });

  test('dual version uses v1 signature', () => {
    const dual = (fixture.expected_v1_cache.entries as FixtureEntry[]).find(
      (e) => e.uuid === '22222222-2222-2222-2222-222222222222',
    )!;
    expect(dual.signature).toBe(
      '4444444444444444444444444444444444444444444444444444444444444444',
    );
  });
});

describe('fixture queries', () => {
  const cache = buildV1Cache();

  for (const test of fixture.query_tests.tests as QueryTest[]) {
    it(test.name, () => {
      const matches = cache.query(test.query_signature, {
        threshold: test.threshold ?? 0.85,
      });
      const matchUuids = matches.map((m) => m.uuid);

      for (const expectedUuid of test.expected_match_uuids) {
        expect(matchUuids).toContain(expectedUuid);
      }

      if (test.expected_match_uuids.length === 0) {
        expect(matches).toHaveLength(0);
      }

      if (test.expected_top_match_uuid) {
        expect(matches[0].uuid).toBe(test.expected_top_match_uuid);
      }

      if (test.expected_top_hamming_distance !== undefined) {
        expect(matches[0].hammingDistance).toBe(test.expected_top_hamming_distance);
      }

      if (test.expected_top_cosine_similarity !== undefined) {
        expect(
          Math.abs(matches[0].cosineSimilarity - test.expected_top_cosine_similarity),
        ).toBeLessThan(1e-6);
      }
    });
  }
});
