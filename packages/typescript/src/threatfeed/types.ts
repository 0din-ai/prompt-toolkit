/**
 * Type definitions for threat feed operations.
 */

/**
 * A detection signature from the threat feed API.
 */
export interface DetectionSignature {
  version: string;
  signature: string;
}

/**
 * A taxonomy classification attached to a threat feed entry.
 */
export interface ThreatFeedTaxonomy {
  category: string;
  strategy: string;
  technique: string;
}

/**
 * Names of additional threat feed fields present in the raw API response
 * but omitted from the default `ThreatFeedEntry` shape for backward
 * compatibility. Pass a subset via the `fields` option on
 * `fetchOne`/`fetchAll`/`fetchPage`/`ThreatFeedCache.sync()` to opt in.
 */
export type ThreatFeedExtraField =
  | 'taxonomies'
  | 'models'
  | 'testResults'
  | 'metadata'
  | 'referenceUrls'
  | 'variantPrompts'
  | 'disclosedAt'
  | 'publishedAt'
  | 'source';

/** Extra, opt-in fields attached to a ThreatFeedEntry when requested. */
export interface ThreatFeedExtraFields {
  taxonomies?: ThreatFeedTaxonomy[];
  models?: unknown[];
  testResults?: unknown[];
  metadata?: unknown;
  referenceUrls?: unknown[];
  variantPrompts?: unknown[];
  disclosedAt?: string;
  publishedAt?: string;
  source?: string;
}

/**
 * A single threat feed entry from the API response.
 */
export interface ThreatFeedEntry {
  uuid: string;
  title: string;
  summary?: string;
  severity: string;
  securityBoundary: string;
  updatedAt?: string;
  detectionSignatures: DetectionSignature[];
  /** Extra fields, populated only when explicitly requested via the `fields` option. */
  extra?: ThreatFeedExtraFields;
}

/**
 * Paginated API response from GET /api/v1/threatfeed.
 */
export interface ThreatFeedResponse {
  page: number;
  totalPages: number;
  totalCount: number;
  threatFeeds: ThreatFeedEntry[];
}

/**
 * A cached signature entry with pre-computed bands.
 */
export interface CachedSignature {
  uuid: string;
  title: string;
  severity: string;
  securityBoundary: string;
  signature: string;
  bands: string[];
  updatedAt?: string;
  extra?: ThreatFeedExtraFields;
}

/**
 * Result of a threat feed sync operation.
 */
export interface SyncResult {
  added: number;
  updated: number;
  total: number;
}

/**
 * A match found when querying the threat feed cache.
 */
export interface ThreatMatch {
  uuid: string;
  title: string;
  severity: string;
  securityBoundary: string;
  signature: string;
  hammingDistance: number;
  cosineSimilarity: number;
}

const EXTRA_FIELD_RAW_KEYS: Record<ThreatFeedExtraField, string> = {
  taxonomies: 'taxonomies',
  models: 'models',
  testResults: 'test_results',
  metadata: 'metadata',
  referenceUrls: 'reference_urls',
  variantPrompts: 'variant_prompts',
  disclosedAt: 'disclosed_at',
  publishedAt: 'published_at',
  source: 'source',
};

/**
 * Extract the requested opt-in extra fields from a raw API entry object.
 * Returns undefined when no fields were requested (default, backward-compatible
 * behavior — `ThreatFeedEntry.extra` stays undefined and is omitted).
 */
export function extractThreatFeedExtraFields(
  raw: Record<string, unknown>,
  fields?: ThreatFeedExtraField[],
): ThreatFeedExtraFields | undefined {
  if (!fields || fields.length === 0) {
    return undefined;
  }
  const extra: Record<string, unknown> = {};
  for (const field of fields) {
    const rawKey = EXTRA_FIELD_RAW_KEYS[field];
    if (rawKey in raw) {
      extra[field] = raw[rawKey];
    }
  }
  return extra as ThreatFeedExtraFields;
}

/**
 * Parse a raw API response object into a ThreatFeedResponse.
 *
 * The API uses snake_case keys; this converts them to camelCase.
 */
export function parseThreatFeedResponse(
  data: Record<string, unknown>,
  fields?: ThreatFeedExtraField[],
): ThreatFeedResponse {
  const feeds = (data.threat_feeds as Record<string, unknown>[]) || [];
  return {
    page: data.page as number,
    totalPages: data.total_pages as number,
    totalCount: data.total_count as number,
    threatFeeds: feeds.map((entry) => ({
      uuid: entry.uuid as string,
      title: entry.title as string,
      summary: entry.summary as string | undefined,
      severity: (entry.severity as string) || 'low',
      securityBoundary: (entry.security_boundary as string) || '',
      updatedAt: entry.updated_at as string | undefined,
      detectionSignatures: ((entry.detection_signatures as Record<string, string>[]) || []).map(
        (sig) => ({
          version: sig.version,
          signature: sig.signature,
        }),
      ),
      extra: extractThreatFeedExtraFields(entry, fields),
    })),
  };
}
