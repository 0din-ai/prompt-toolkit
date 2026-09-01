/**
 * Threat feed integration for fetching and caching known threat signatures.
 *
 * This module provides the ability to fetch detection signatures from the 0din
 * portal's threat feed API, cache them locally with a band index, and perform
 * fast similarity lookup against the cache.
 *
 * @example
 * ```typescript
 * import { ThreatFeedClient, ThreatFeedCache, compareToThreatfeed } from '@0din/prompt-toolkit/threatfeed';
 * import { SignatureVersion } from '@0din/prompt-toolkit';
 *
 * // Sync signatures from the portal
 * const client = new ThreatFeedClient({ apiToken: 'your-api-token' });
 * const cache = new ThreatFeedCache({ version: SignatureVersion.V1 });
 * await cache.sync(client, { full: true });
 *
 * // Query for similar signatures
 * const matches = cache.query('a1b2c3d4...', { threshold: 0.85 });
 * ```
 *
 * @module threatfeed
 */

export { ThreatFeedCache } from './cache';
export { ThreatFeedClient } from './client';
export { compareToThreatfeed } from './compare';
export type {
  CachedSignature,
  DetectionSignature,
  SyncResult,
  ThreatFeedEntry,
  ThreatFeedExtraField,
  ThreatFeedExtraFields,
  ThreatFeedResponse,
  ThreatFeedTaxonomy,
  ThreatMatch,
} from './types';
export { extractThreatFeedExtraFields, parseThreatFeedResponse } from './types';
