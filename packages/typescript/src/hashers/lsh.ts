/**
 * SimHash LSH implementation (current default algorithm).
 * 
 * Uses deterministic random hyperplane LSH with SplitMix64-based
 * hyperplane generation. This is the canonical LSH implementation
 * used for V0 and V1 signatures.
 */

import type { Hasher } from '../hasher';
import { simhashLshMulti, type LSHFamily } from '../lsh';
import type { LshConfig } from '../types';

/**
 * SimHash LSH implementation.
 * 
 * @example
 * ```typescript
 * import { SimHashLsh } from '@0din/prompt-toolkit/hashers';
 * import { normalizeVector } from '@0din/prompt-toolkit';
 * 
 * const hasher = new SimHashLsh();
 * const vector = normalizeVector([1.0, 2.0, 3.0]);
 * const config = { families: 3, bits: 256, bands: 16 };
 * const families = hasher.compute(vector, config);
 * console.log(hasher.name()); // 'lsh'
 * ```
 */
export class SimHashLsh implements Hasher {
  /**
   * Return algorithm name.
   * @returns 'lsh'
   */
  name(): string {
    return 'lsh';
  }

  /**
   * Compute LSH signatures from normalized embedding.
   * 
   * Delegates to the simhashLshMulti function with the provided
   * configuration parameters.
   * 
   * @param embedding - L2-normalized embedding vector
   * @param config - LSH configuration parameters
   * @returns List of LSH families, one per family index
   */
  compute(embedding: number[], config: LshConfig): LSHFamily[] {
    return simhashLshMulti(embedding, config);
  }
}
