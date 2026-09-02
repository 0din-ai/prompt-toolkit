/**
 * Threat feed cache with band-indexed similarity lookup.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { ThreatFeedCacheError } from '../error';
import { hammingDistanceHex, cosineFromHamming } from '../lsh';
import { SignatureVersion, resolveVersion } from '../types';
import { ThreatFeedClient } from './client';
import type {
  CachedSignature,
  SyncResult,
  ThreatFeedExtraField,
  ThreatFeedExtraFields,
  ThreatMatch,
} from './types';

/** Schema version for the cache file format. */
const CACHE_SCHEMA_VERSION = 1;

/** Default number of bands for LSH indexing. */
const DEFAULT_BANDS = 16;

/** Default number of bits per signature. */
const DEFAULT_BITS = 256;

/** On-disk cache format. */
interface CacheFile {
  schema_version: number;
  signature_version: string;
  synced_at: string;
  source_url: string;
  entry_count: number;
  lsh_config: { bits: number; bands: number };
  entries: Array<{
    uuid: string;
    title: string;
    severity: string;
    security_boundary: string;
    signature: string;
    bands: string[];
    updated_at?: string;
    extra?: ThreatFeedExtraFields;
  }>;
  band_index: Record<string, number[]>;
}

/**
 * Compute bands from a hex signature string.
 *
 * Splits a 64 hex char signature into `numBands` equal-length bands.
 *
 * @throws ThreatFeedCacheError if the signature is too short to split into the requested bands.
 */
export function computeBands(signature: string, numBands: number = DEFAULT_BANDS): string[] {
  if (!signature || signature.length < numBands) {
    throw new ThreatFeedCacheError(
      `Signature too short to split into ${numBands} bands: ${signature.length} chars (need at least ${numBands})`,
    );
  }
  const bandLen = Math.floor(signature.length / numBands);
  const bands: string[] = [];
  for (let i = 0; i < numBands; i++) {
    bands.push(signature.slice(i * bandLen, (i + 1) * bandLen));
  }
  return bands;
}

/**
 * Threat feed cache with band-indexed similarity lookup.
 *
 * Caches detection signatures from the 0din threat feed API and provides
 * fast similarity queries using LSH band indexing.
 */
export class ThreatFeedCache {
  private readonly version: SignatureVersion;
  private readonly cacheDir: string;
  private readonly bits: number;
  private readonly bands: number;
  private _entries: CachedSignature[] = [];
  private bandIndex: Record<string, number[]> = {};
  private syncedAt?: string;
  private sourceUrl: string = 'https://0din.ai';

  /**
   * Create a new threat feed cache.
   *
   * @param options Configuration options
   * @param options.version Signature version to cache (V0 or V1)
   * @param options.cacheDir Override cache directory path
   * @param options.bands Number of bands for LSH indexing (default: 16)
   */
  constructor(options: { version: SignatureVersion; cacheDir?: string; bands?: number }) {
    this.version = resolveVersion(options.version);
    this.bits = DEFAULT_BITS;
    this.bands = options.bands ?? DEFAULT_BANDS;

    this.cacheDir =
      options.cacheDir ??
      process.env.ODIN_PROMPT_TOOLKIT_THREATFEED_CACHE ??
      path.join(os.homedir(), '.odin-prompt-toolkit', 'threatfeed');
  }

  /**
   * Load cache from disk.
   *
   * @returns true if cache was loaded successfully, false if no cache exists.
   * @throws ThreatFeedCacheError if the cache file is corrupt.
   */
  load(): boolean {
    const filePath = this.cacheFilePath();
    if (!fs.existsSync(filePath)) {
      return false;
    }

    let data: CacheFile;
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      data = JSON.parse(content);
    } catch (e) {
      throw new ThreatFeedCacheError(`Corrupt cache file: ${e}`);
    }

    if (data.schema_version !== CACHE_SCHEMA_VERSION) {
      return false;
    }

    this._entries = (data.entries || []).map((entry) => ({
      uuid: entry.uuid,
      title: entry.title,
      severity: entry.severity,
      securityBoundary: entry.security_boundary,
      signature: entry.signature,
      bands: entry.bands,
      updatedAt: entry.updated_at,
      extra: entry.extra,
    }));
    this.bandIndex = data.band_index || {};
    this.syncedAt = data.synced_at;
    this.sourceUrl = data.source_url || 'https://0din.ai';

    return true;
  }

  /**
   * Save cache to disk with atomic write (temp file + rename).
   *
   * @throws ThreatFeedCacheError if write fails.
   */
  save(): void {
    try {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    } catch (e) {
      throw new ThreatFeedCacheError(`Failed to create cache directory: ${e}`);
    }

    const cacheData: CacheFile = {
      schema_version: CACHE_SCHEMA_VERSION,
      signature_version: this.version,
      synced_at: this.syncedAt ?? new Date().toISOString(),
      source_url: this.sourceUrl,
      entry_count: this._entries.length,
      lsh_config: { bits: this.bits, bands: this.bands },
      entries: this._entries.map((e) => ({
        uuid: e.uuid,
        title: e.title,
        severity: e.severity,
        security_boundary: e.securityBoundary,
        signature: e.signature,
        bands: e.bands,
        updated_at: e.updatedAt,
        extra: e.extra,
      })),
      band_index: this.bandIndex,
    };

    const filePath = this.cacheFilePath();
    const tmpPath = filePath + '.tmp';

    try {
      fs.writeFileSync(tmpPath, JSON.stringify(cacheData, null, 2), 'utf-8');
      fs.renameSync(tmpPath, filePath);
    } catch (e) {
      try {
        fs.unlinkSync(tmpPath);
      } catch {
        // Ignore cleanup failure
      }
      throw new ThreatFeedCacheError(`Failed to write cache: ${e}`);
    }
  }

  /**
   * Sync signatures from the threat feed API.
   *
   * @param client Threat feed API client
   * @param options Sync options
   * @param options.full If true, fetch all entries. If false, incremental sync.
   * @param options.fields Optional list of opt-in extra fields to populate on each entry
   * @returns SyncResult with counts of added/updated entries.
   */
  async sync(
    client: ThreatFeedClient,
    options?: { full?: boolean; fields?: ThreatFeedExtraField[] },
  ): Promise<SyncResult> {
    const full = options?.full ?? false;
    const since = full ? undefined : this.lastUpdatedAt();
    this.sourceUrl = client.baseUrl;

    const entries = await client.fetchAll({ since, fields: options?.fields });
    const versionStr = this.version;

    const newCached: CachedSignature[] = [];
    for (const entry of entries) {
      for (const sig of entry.detectionSignatures) {
        if (sig.version === versionStr) {
          newCached.push({
            uuid: entry.uuid,
            title: entry.title,
            severity: entry.severity,
            securityBoundary: entry.securityBoundary,
            signature: sig.signature,
            bands: computeBands(sig.signature, this.bands),
            updatedAt: entry.updatedAt,
            extra: entry.extra,
          });
        }
      }
    }

    let result: SyncResult;
    if (full) {
      const total = newCached.length;
      this._entries = newCached;
      result = { added: total, updated: 0, total };
    } else {
      result = this.mergeEntries(newCached);
    }

    this.rebuildBandIndex();
    this.syncedAt = new Date().toISOString();
    this.save();

    return result;
  }

  /**
   * Query the cache for signatures similar to the given query.
   *
   * Uses band-indexed candidate selection followed by Hamming distance verification.
   *
   * @param signature 64 hex char signature to query (raw, no 0din- prefix)
   * @param options Query options
   * @param options.threshold Minimum cosine similarity threshold (default: 0.85)
   * @param options.maxResults Maximum number of results to return (default: 10)
   * @returns Array of ThreatMatch objects sorted by cosine similarity descending.
   */
  query(signature: string, options?: { threshold?: number; maxResults?: number }): ThreatMatch[] {
    const threshold = options?.threshold ?? 0.85;
    const maxResults = options?.maxResults ?? 10;

    const queryBands = computeBands(signature, this.bands);

    // Collect candidate indices from band index
    const candidateIndices = new Set<number>();
    for (let bandIdx = 0; bandIdx < queryBands.length; bandIdx++) {
      const key = `${bandIdx}:${queryBands[bandIdx]}`;
      const indices = this.bandIndex[key];
      if (indices) {
        for (const idx of indices) {
          candidateIndices.add(idx);
        }
      }
    }

    // Verify candidates with Hamming distance
    const matches: ThreatMatch[] = [];
    for (const idx of candidateIndices) {
      if (idx >= this._entries.length) continue;
      const entry = this._entries[idx];
      const dist = hammingDistanceHex(signature, entry.signature);
      const cosine = cosineFromHamming(dist, this.bits);
      if (cosine >= threshold) {
        matches.push({
          uuid: entry.uuid,
          title: entry.title,
          severity: entry.severity,
          securityBoundary: entry.securityBoundary,
          signature: entry.signature,
          hammingDistance: dist,
          cosineSimilarity: cosine,
        });
      }
    }

    // Sort by cosine similarity descending
    matches.sort((a, b) => b.cosineSimilarity - a.cosineSimilarity);
    return matches.slice(0, maxResults);
  }

  /** Get the number of entries in the cache. */
  get entryCount(): number {
    return this._entries.length;
  }

  /** Get the timestamp of the last sync. */
  get lastSynced(): string | undefined {
    return this.syncedAt;
  }

  /** Get all cached entries. */
  get entries(): CachedSignature[] {
    return this._entries;
  }

  /** Load entries directly (for testing without disk I/O). */
  loadEntries(entries: CachedSignature[]): void {
    this._entries = entries;
    this.rebuildBandIndex();
  }

  // --- Private methods ---

  private cacheFilePath(): string {
    return path.join(this.cacheDir, `cache-${this.version}.json`);
  }

  private lastUpdatedAt(): string | undefined {
    const updatedAts = this._entries
      .map((e) => e.updatedAt)
      .filter((v): v is string => v !== undefined);
    return updatedAts.length > 0 ? updatedAts.sort().pop() : undefined;
  }

  private mergeEntries(newEntries: CachedSignature[]): SyncResult {
    const existing = new Map<string, number>();
    this._entries.forEach((e, i) => existing.set(e.uuid, i));

    let added = 0;
    let updated = 0;

    for (const entry of newEntries) {
      const idx = existing.get(entry.uuid);
      if (idx !== undefined) {
        this._entries[idx] = entry;
        updated++;
      } else {
        existing.set(entry.uuid, this._entries.length);
        this._entries.push(entry);
        added++;
      }
    }

    return { added, updated, total: this._entries.length };
  }

  private rebuildBandIndex(): void {
    this.bandIndex = {};
    for (let idx = 0; idx < this._entries.length; idx++) {
      const entry = this._entries[idx];
      for (let bandIdx = 0; bandIdx < entry.bands.length; bandIdx++) {
        const key = `${bandIdx}:${entry.bands[bandIdx]}`;
        if (!this.bandIndex[key]) {
          this.bandIndex[key] = [];
        }
        this.bandIndex[key].push(idx);
      }
    }
  }
}
