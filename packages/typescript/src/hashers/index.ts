/**
 * Hash algorithm implementations.
 */

import type { Hasher } from '../hasher';
import { HashAlgorithm } from '../types';
import { SimHashLsh } from './lsh';

/**
 * Get a hasher instance by algorithm.
 * 
 * @param algorithm - Hash algorithm to use
 * @returns Hasher instance for the specified algorithm
 * @throws {Error} If the algorithm is not supported
 * 
 * @example
 * ```typescript
 * import { getHasher, HashAlgorithm } from '@0din/prompt-toolkit';
 * 
 * const hasher = getHasher(HashAlgorithm.LSH);
 * console.log(hasher.name()); // 'lsh'
 * ```
 */
export function getHasher(algorithm: HashAlgorithm): Hasher {
  if (algorithm === HashAlgorithm.LSH) {
    return new SimHashLsh();
  }
  throw new Error(`Unknown algorithm: ${algorithm}`);
}

export { SimHashLsh };
