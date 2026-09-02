/**
 * Tests for threat feed type helpers.
 */

import { extractThreatFeedExtraFields } from '../src/threatfeed/types';

describe('extractThreatFeedExtraFields', () => {
  const raw = {
    taxonomies: [{ category: 'c', strategy: 's', technique: 't' }],
    models: ['model-a'],
    test_results: [{ passed: true }],
    metadata: { foo: 'bar' },
    reference_urls: ['https://example.com'],
    variant_prompts: ['variant'],
    disclosed_at: '2025-01-10T12:00:00.000Z',
    published_at: '2025-01-15T12:00:00.000Z',
    source: 'internal',
  };

  test('no fields argument returns undefined', () => {
    expect(extractThreatFeedExtraFields(raw)).toBeUndefined();
  });

  test('empty fields array returns undefined', () => {
    expect(extractThreatFeedExtraFields(raw, [])).toBeUndefined();
  });

  test('requesting a field not present in raw data omits the key', () => {
    const result = extractThreatFeedExtraFields({}, ['taxonomies']);
    expect(result).toBeDefined();
    expect(result).not.toHaveProperty('taxonomies');
  });

  test('requesting multiple fields maps raw snake_case keys to camelCase', () => {
    const result = extractThreatFeedExtraFields(raw, ['testResults', 'referenceUrls', 'source']);
    expect(result?.testResults).toEqual(raw.test_results);
    expect(result?.referenceUrls).toEqual(raw.reference_urls);
    expect(result?.source).toBe('internal');
    expect(result).not.toHaveProperty('taxonomies');
    expect(result).not.toHaveProperty('models');
  });
});
