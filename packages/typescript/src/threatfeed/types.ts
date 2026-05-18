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

/**
 * Parse a raw API response object into a ThreatFeedResponse.
 *
 * The API uses snake_case keys; this converts them to camelCase.
 */
export function parseThreatFeedResponse(data: Record<string, unknown>): ThreatFeedResponse {
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
    })),
  };
}
