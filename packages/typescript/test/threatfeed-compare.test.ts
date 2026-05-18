import { ThreatFeedCache, computeBands } from '../src/threatfeed/cache';
import { compareToThreatfeed } from '../src/threatfeed/compare';
import { SignatureVersion, type SignatureResult } from '../src/types';
import type { CachedSignature } from '../src/threatfeed/types';

const SIG_A = 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2';
const SIG_UNRELATED = '5678901234567890567890123456789056789012345678905678901234567890';

function makeSignatureResult(sig: string): SignatureResult {
  return {
    signature: `0din-v1:${sig}`,
    version: SignatureVersion.V1,
    promptPreview: 'test prompt',
    promptLength: 11,
    provider: 'test',
    model: 'test-model',
    dimensions: 1024,
    embeddingSha256: 'abc123',
    lsh: {
      config: { families: 3, bits: 256, bands: 16 },
      signatures: [
        {
          family: 0,
          bits: 256,
          signature: sig,
          bands: computeBands(sig, 16),
        },
      ],
    },
  };
}

function makeEntry(uuid: string, sig: string): CachedSignature {
  return {
    uuid,
    title: 'Known Threat',
    severity: 'high',
    securityBoundary: 'guardrail_jailbreak',
    signature: sig,
    bands: computeBands(sig, 16),
  };
}

describe('compareToThreatfeed', () => {
  test('exact match', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    cache.loadEntries([makeEntry('threat-1', SIG_A)]);

    const result = makeSignatureResult(SIG_A);
    const matches = compareToThreatfeed(result, cache);

    expect(matches).toHaveLength(1);
    expect(matches[0].uuid).toBe('threat-1');
    expect(matches[0].hammingDistance).toBe(0);
    expect(Math.abs(matches[0].cosineSimilarity - 1.0)).toBeLessThan(1e-10);
  });

  test('no match', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    cache.loadEntries([makeEntry('threat-1', SIG_A)]);

    const result = makeSignatureResult(SIG_UNRELATED);
    const matches = compareToThreatfeed(result, cache);

    expect(matches).toHaveLength(0);
  });

  test('empty cache', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    const result = makeSignatureResult(SIG_A);
    const matches = compareToThreatfeed(result, cache);
    expect(matches).toHaveLength(0);
  });

  test('threshold parameter', () => {
    const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
    cache.loadEntries([makeEntry('threat-1', SIG_A)]);

    const result = makeSignatureResult(SIG_A);

    // Very high threshold still matches exact
    const matches = compareToThreatfeed(result, cache, { threshold: 0.99 });
    expect(matches).toHaveLength(1);

    // Threshold of 1.0 still matches exact (cosine = 1.0)
    const matches2 = compareToThreatfeed(result, cache, { threshold: 1.0 });
    expect(matches2).toHaveLength(1);
  });
});
