/**
 * Abstract hasher interface for hash algorithm implementations.
 * 
 * Each hasher takes a normalized embedding vector and LSH configuration,
 * and produces LSH signatures suitable for similarity matching.
 * 
 * @example
 * ```typescript
 * import { getHasher, HashAlgorithm } from '@0din/sig';
 * import type { LshConfig } from '@0din/sig/types';
 * import { normalizeVector } from '@0din/sig';
 * 
 * const hasher = getHasher(HashAlgorithm.LSH);
 * const vector = normalizeVector([1.0, 2.0, 3.0]);
 * const config: LshConfig = { families: 3, bits: 256, bands: 16 };
 * const families = hasher.compute(vector, config);
 * ```
 */

import type { LSHFamily } from './lsh';
import type { LshConfig } from './types';

/**
 * Abstract interface for hash algorithm implementations.
 */
export interface Hasher {
  /**
   * Algorithm name (e.g., 'lsh', 'cm-lsh')
   * @returns Algorithm identifier string
   */
  name(): string;

  /**
   * Compute LSH signatures from a normalized embedding vector.
   * 
   * @param embedding - L2-normalized embedding vector
   * @param config - LSH configuration parameters
   * @returns List of LSH families, one per family index
   */
  compute(embedding: number[], config: LshConfig): LSHFamily[];
}
