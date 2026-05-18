/**
 * Threat feed API client for fetching signatures from the 0din portal.
 */

import { ThreatFeedApiError } from '../error';
import type { ThreatFeedEntry } from './types';
import { parseThreatFeedResponse } from './types';

/**
 * Options for creating a ThreatFeedClient.
 */
export interface ThreatFeedClientOptions {
  /** Raw API token (no Bearer prefix). Falls back to ODIN_THREATFEED_API_TOKEN env var. */
  apiToken?: string;
  /** API base URL (default: https://0din.ai). Falls back to ODIN_THREATFEED_BASE_URL env var. */
  baseUrl?: string;
  /** Page size for paginated requests (default: 100). */
  perPage?: number;
}

/**
 * Client for the 0din threat feed API.
 *
 * Fetches detection signatures from the paginated threat feed endpoint.
 */
export class ThreatFeedClient {
  private readonly apiToken: string;
  private readonly _baseUrl: string;
  private readonly perPage: number;

  constructor(options: ThreatFeedClientOptions = {}) {
    this.apiToken =
      options.apiToken ?? process.env.ODIN_THREATFEED_API_TOKEN ?? '';

    if (!this.apiToken) {
      throw new ThreatFeedApiError(
        'API token required: pass apiToken or set ODIN_THREATFEED_API_TOKEN',
      );
    }

    this._baseUrl =
      options.baseUrl ??
      process.env.ODIN_THREATFEED_BASE_URL ??
      'https://0din.ai';

    this.perPage = options.perPage ?? 100;
  }

  /** Get the base URL of the API. */
  get baseUrl(): string {
    return this._baseUrl;
  }

  /**
   * Fetch all threat feed entries, paginating through all pages.
   *
   * @param options Fetch options
   * @param options.since Optional ISO8601 timestamp to filter entries updated since
   * @returns Array of all threat feed entries
   * @throws ThreatFeedApiError on network or API errors
   */
  async fetchAll(options?: { since?: string }): Promise<ThreatFeedEntry[]> {
    const since = options?.since;
    const allEntries: ThreatFeedEntry[] = [];
    let page = 1;

    while (true) {
      const data = await this.fetchPage(page, since);
      const response = parseThreatFeedResponse(data);
      allEntries.push(...response.threatFeeds);

      if (page >= response.totalPages) {
        break;
      }
      page++;

      // Rate limiting: 500ms delay between pages
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    return allEntries;
  }

  /**
   * Fetch a single threat feed entry by UUID.
   *
   * @param uuid Threat feed entry UUID
   * @returns ThreatFeedEntry for the specified UUID
   * @throws ThreatFeedApiError on network or API errors
   */
  async fetchOne(uuid: string): Promise<ThreatFeedEntry> {
    const url = `${this._baseUrl}/api/v1/threatfeed/${uuid}`;
    const headers = {
      Authorization: this.apiToken,
      'Content-Type': 'application/json',
    };

    let response: Response;
    try {
      response = await fetch(url, { headers });
    } catch (e) {
      throw new ThreatFeedApiError(`Network error: ${e}`);
    }

    if (!response.ok) {
      const text = await response.text();
      throw new ThreatFeedApiError(
        `API returned status ${response.status}: ${text}`,
        response.status,
      );
    }

    const data = (await response.json()) as Record<string, unknown>;
    const sigs = ((data.detection_signatures as Array<Record<string, string>>) || []).map(
      (sig) => ({
        version: sig.version,
        signature: sig.signature,
      }),
    );

    return {
      uuid: data.uuid as string,
      title: data.title as string,
      summary: data.summary as string | undefined,
      severity: (data.severity as string) || 'low',
      securityBoundary: (data.security_boundary as string) || '',
      updatedAt: data.updated_at as string | undefined,
      detectionSignatures: sigs,
    };
  }

  // --- Private methods ---

  private async fetchPage(
    page: number,
    since?: string,
  ): Promise<Record<string, unknown>> {
    let url = `${this._baseUrl}/api/v1/threatfeed?page=${page}&per_page=${this.perPage}`;
    if (since) {
      url += `&q[updated_at_gteq]=${encodeURIComponent(since)}`;
    }

    const headers = {
      Authorization: this.apiToken,
      'Content-Type': 'application/json',
    };

    let response: Response;
    try {
      response = await fetch(url, { headers });
    } catch (e) {
      throw new ThreatFeedApiError(`Network error: ${e}`);
    }

    if (!response.ok) {
      const text = await response.text();
      throw new ThreatFeedApiError(
        `API returned status ${response.status}: ${text}`,
        response.status,
      );
    }

    return (await response.json()) as Record<string, unknown>;
  }
}
