/**
 * High-level comparison API for threat feed matching.
 */

import type { SignatureResult } from '../types';
import type { ThreatFeedCache } from './cache';
import type { ThreatMatch } from './types';

/**
 * Compare a signature result against the threat feed cache.
 *
 * Extracts the primary signature (family 0) from the result and queries
 * the cache for similar known threat signatures.
 *
 * @param result Signature result from signText()
 * @param cache Pre-loaded threat feed cache
 * @param options Comparison options
 * @param options.threshold Minimum cosine similarity threshold (default: 0.85)
 * @param options.maxResults Maximum number of results to return (default: 10)
 * @returns Array of ThreatMatch objects sorted by cosine similarity descending
 *
 * @example
 * ```typescript
 * import { signText, SignatureVersion } from '@0din/prompt-toolkit';
 * import { ThreatFeedCache, compareToThreatfeed } from '@0din/prompt-toolkit/threatfeed';
 *
 * const result = await signText('suspicious prompt', provider, SignatureVersion.V1);
 * const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
 * cache.load();
 *
 * const matches = compareToThreatfeed(result, cache, { threshold: 0.85 });
 * for (const m of matches) {
 *   console.log(`Match: ${m.title} (similarity: ${m.cosineSimilarity.toFixed(3)})`);
 * }
 * ```
 */
export function compareToThreatfeed(
  result: SignatureResult,
  cache: ThreatFeedCache,
  options?: { threshold?: number; maxResults?: number },
): ThreatMatch[] {
  const primarySig = result.lsh.signatures[0].signature;
  return cache.query(primarySig, options);
}
