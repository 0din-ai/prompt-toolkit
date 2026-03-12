/**
 * Robust, deterministic SimHash (random-hyperplane LSH) utilities.
 * 
 * This module provides locality-sensitive hashing for similarity search on
 * high-dimensional vectors (e.g., embeddings).
 * 
 * - Uses normalized vectors (direction only)
 * - Deterministic per-family, per-bit, per-dimension sign via SplitMix64-based hash
 * - Supports multiple independent hash families for robustness
 * 
 * Ported from thor/src/utils/lsh.ts (canonical TypeScript implementation).
 */

/**
 * Result of LSH hashing for one family.
 */
export interface LSHFamily {
  family: number;
  bits: number;
  signature: string; // hex string (bits/4 hex chars)
  bands: string[]; // contiguous slices of hex
}

/**
 * LSH configuration parameters.
 */
export interface LshConfig {
  families?: number; // Number of independent hash families (default: 3)
  bits?: number; // Number of bits per signature (default: 256)
  bands?: number; // Number of bands for LSH indexing (default: 16)
}

const MASK64 = (1n << 64n) - 1n;

/**
 * SplitMix64 hash function for deterministic random generation.
 * @internal
 */
function splitmix64(x: bigint): bigint {
  let z = (x + 0x9E3779B97F4A7C15n) & MASK64;
  z = (z ^ (z >> 30n));
  z = (z * 0xBF58476D1CE4E5B9n) & MASK64;
  z = (z ^ (z >> 27n));
  z = (z * 0x94D049BB133111EBn) & MASK64;
  z = z ^ (z >> 31n);
  return z & MASK64;
}

/**
 * Get deterministic +1/-1 sign from (family, bit, dim).
 * @internal
 */
function signFor(family: number, bit: number, dim: number): number {
  // Combine into 64-bit seed and mix
  const seed = (BigInt(family) << 48n) ^ (BigInt(bit) << 24n) ^ BigInt(dim);
  const h = splitmix64(seed);
  // Use lowest bit for sign
  return (h & 1n) === 1n ? 1 : -1;
}

/**
 * Compute SimHash LSH signatures for a normalized vector.
 * 
 * Uses random hyperplane LSH with deterministic hyperplanes generated
 * via SplitMix64. Multiple independent hash families can be computed
 * for improved recall in similarity search.
 * 
 * @param normalizedVector - Input vector (should be L2-normalized for best results)
 * @param config - LSH configuration options
 * @returns Array of LSHFamily results, one per family
 * 
 * @example
 * ```typescript
 * import { simhashLshMulti, normalizeVector } from '@0din/prompt-toolkit';
 * 
 * const vector = [0.5, 0.5, 0.5, 0.5];
 * const normalized = normalizeVector(vector);
 * const families = simhashLshMulti(normalized);
 * 
 * console.log(families[0].signature); // hex string
 * ```
 */
export function simhashLshMulti(
  normalizedVector: number[],
  config: LshConfig = {},
): LSHFamily[] {
  const families = Math.max(1, config.families ?? 3);
  const bits = Math.max(64, config.bits ?? 256);
  const bands = Math.max(1, config.bands ?? 16);
  const out: LSHFamily[] = [];

  for (let f = 0; f < families; f++) {
    const boolBits: boolean[] = [];
    for (let b = 0; b < bits; b++) {
      let dot = 0;
      for (let j = 0; j < normalizedVector.length; j++) {
        const s = signFor(f, b, j);
        dot += normalizedVector[j] * s;
      }
      boolBits.push(dot > 0);
    }

    // Pack into hex string
    let hex = '';
    for (let i = 0; i < boolBits.length; i += 4) {
      const n =
        (boolBits[i] ? 8 : 0) +
        (boolBits[i + 1] ? 4 : 0) +
        (boolBits[i + 2] ? 2 : 0) +
        (boolBits[i + 3] ? 1 : 0);
      hex += n.toString(16);
    }

    // Split into bands (contiguous slices)
    const bandLen = Math.floor(hex.length / bands) || hex.length;
    const bandArr: string[] = [];
    for (let i = 0; i < hex.length; i += bandLen) {
      bandArr.push(hex.slice(i, i + bandLen));
      if (bandArr.length === bands) {
        break;
      }
    }

    out.push({ family: f, bits, signature: hex, bands: bandArr });
  }

  return out;
}

/**
 * Compute Hamming distance between two hex signatures.
 * 
 * Each hex character represents 4 bits, so the distance is computed
 * by XORing corresponding nibbles and counting set bits.
 * 
 * @param a - First hex string
 * @param b - Second hex string
 * @returns Hamming distance in bits
 * 
 * @example
 * ```typescript
 * import { hammingDistanceHex } from '@0din/prompt-toolkit';
 * 
 * const distance = hammingDistanceHex('abcd', 'abce');
 * console.log(distance); // 1
 * ```
 */
export function hammingDistanceHex(a: string, b: string): number {
  const clean = (s: string) => s.toLowerCase().replace(/[^0-9a-f]/g, '');
  const x = clean(a);
  const y = clean(b);
  const L = Math.min(x.length, y.length);
  let dist = 0;

  for (let i = 0; i < L; i++) {
    const n1 = Number.parseInt(x[i], 16);
    const n2 = Number.parseInt(y[i], 16);
    const xor = n1 ^ n2;
    dist += ((xor >> 3) & 1) + ((xor >> 2) & 1) + ((xor >> 1) & 1) + (xor & 1);
  }

  // Extra nibbles count as differing bits (4 bits per nibble)
  dist += Math.abs(x.length - y.length) * 4;

  return dist;
}

/**
 * Estimate cosine similarity from Hamming distance.
 * 
 * For random hyperplane LSH, the probability that two vectors have
 * different signs for a random hyperplane is proportional to the
 * angle between them. This function converts Hamming distance back
 * to an estimated cosine similarity.
 * 
 * @param distanceBits - Hamming distance in bits
 * @param totalBits - Total number of bits in the signature
 * @returns Estimated cosine similarity in range [-1, 1]
 * 
 * @example
 * ```typescript
 * import { cosineFromHamming } from '@0din/prompt-toolkit';
 * 
 * const similarity = cosineFromHamming(64, 256);
 * console.log(similarity); // ~0.707 (quarter bits different)
 * ```
 */
export function cosineFromHamming(distanceBits: number, totalBits: number): number {
  if (totalBits <= 0) {
    return 0;
  }
  const pDiff = distanceBits / totalBits;
  return Math.cos(Math.PI * pDiff);
}

/**
 * L2-normalize a vector.
 * 
 * @param vector - Input vector
 * @returns L2-normalized vector (unit length)
 * 
 * @example
 * ```typescript
 * import { normalizeVector } from '@0din/prompt-toolkit';
 * 
 * const normalized = normalizeVector([3, 4]);
 * console.log(normalized); // [0.6, 0.8]
 * ```
 */
export function normalizeVector(vector: number[]): number[] {
  const magnitude = Math.sqrt(vector.reduce((sum, x) => sum + x * x, 0));
  if (magnitude === 0) {
    return vector;
  }
  return vector.map((x) => x / magnitude);
}

/**
 * Export internal functions for testing.
 * @internal
 */
export const _internal = {
  splitmix64,
  signFor,
};
