/**
 * Confidence Matrix LSH (CM-LSH) utilities.
 * 
 * This module provides an enhanced LSH implementation that combines:
 * - Standard LSH-TS (256-bit random hyperplane hash)
 * - ITQ (Iterative Quantization) for improved quantization
 * - Confidence matrix to weight reliable bits higher
 * - Isotonic calibration for accurate similarity estimates
 * 
 * The result is a DualHash with:
 * - hashA: 512-bit signature (first 256 bits = LSH-TS compatible)
 * - hashB: 512-bit confidence matrix
 * - Calibrated similarity function
 * 
 * Ported from python/odin_sig/cm_lsh.py to match signature_cli code style.
 */

import { _internal } from './lsh';

/**
 * Result of CM-LSH hashing with confidence matrix.
 */
export interface DualHash {
  /** 512-bit signature (128 hex chars, first 64 = LSH-TS compat) */
  hashA: string;
  /** 512-bit confidence matrix (128 hex chars) */
  hashB: string;
  /** Band slices for LSH indexing */
  bands: string[];
  /** Number of bits (always 512) */
  bits: number;
}

/**
 * Get LSH-TS compatible 256-bit signature (first 64 hex chars).
 */
export function lshTsCompat(hash: DualHash): string {
  return hash.hashA.substring(0, 64);
}

/**
 * Parameters for Iterative Quantization transformation.
 */
export interface ITQParams {
  /** Mean vector for centering */
  mean: Float32Array;
  /** PCA projection matrix (256 x dims) */
  pca: Float32Array[];
  /** ITQ rotation matrix (256 x 256) */
  rotation: Float32Array[];
}

/**
 * Combined parameters for hybrid LSH+ITQ.
 */
export interface HybridParams {
  /** LSH-TS hyperplanes (256 x dims) */
  lshTsHyperplanes: Float32Array[];
  /** ITQ parameters */
  itq: ITQParams;
}

/**
 * Isotonic regression calibrator configuration.
 */
export interface CalibratorConfig {
  /** X thresholds for piecewise linear function */
  xThresh: number[];
  /** Y thresholds for piecewise linear function */
  yThresh: number[];
  /** Minimum x value for clipping */
  xMin: number;
  /** Maximum x value for clipping */
  xMax: number;
}

// ============================================================================
// Helper functions (private)
// ============================================================================

/**
 * Pack boolean array into hex string (MSB-first, 4 bits per char).
 */
function packBits(bits: boolean[]): string {
  const hexChars: string[] = [];
  for (let i = 0; i < bits.length; i += 4) {
    const n =
      (bits[i] ? 8 : 0) +
      (bits[i + 1] ? 4 : 0) +
      (bits[i + 2] ? 2 : 0) +
      (bits[i + 3] ? 1 : 0);
    hexChars.push(n.toString(16));
  }
  return hexChars.join('');
}

/**
 * Unpack hex string into boolean array (MSB-first).
 */
function unpackBits(hex: string): boolean[] {
  const bits: boolean[] = [];
  for (const c of hex) {
    const n = Number.parseInt(c, 16);
    bits.push((n & 8) !== 0, (n & 4) !== 0, (n & 2) !== 0, (n & 1) !== 0);
  }
  return bits;
}

/**
 * Matrix-vector multiply (matrix rows × vector).
 */
function matmulVec(matrix: Float32Array[], vec: Float32Array): Float32Array {
  const result = new Float32Array(matrix.length);
  for (let i = 0; i < matrix.length; i++) {
    let sum = 0;
    for (let j = 0; j < vec.length; j++) {
      sum += matrix[i][j] * vec[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * Compute percentile of array (linear interpolation, matching numpy).
 */
function percentile(arr: Float32Array, p: number): number {
  if (arr.length === 0) return 0;
  
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const index = (sorted.length - 1) * (p / 100);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  
  if (lower === upper) {
    return sorted[lower];
  }
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/**
 * Piecewise linear interpolation (matching numpy.interp).
 */
function interp(x: number, xp: number[], fp: number[]): number {
  // Clamp to range
  if (x <= xp[0]) return fp[0];
  if (x >= xp[xp.length - 1]) return fp[fp.length - 1];
  
  // Find interval
  for (let i = 0; i < xp.length - 1; i++) {
    if (x >= xp[i] && x <= xp[i + 1]) {
      const t = (x - xp[i]) / (xp[i + 1] - xp[i]);
      return fp[i] + t * (fp[i + 1] - fp[i]);
    }
  }
  
  return fp[fp.length - 1];
}

/**
 * Clamp value to range.
 */
function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x));
}

// ============================================================================
// Calibrator class
// ============================================================================

/**
 * Isotonic regression calibrator for similarity scores.
 * 
 * Maps raw similarity scores to calibrated cosine similarity estimates
 * using piecewise linear interpolation.
 */
class Calibrator {
  constructor(
    private xThresh: number[],
    private yThresh: number[],
    private xMin: number,
    private xMax: number,
  ) {}

  /**
   * Predict calibrated similarity from raw score.
   */
  predict(x: number): number {
    const clipped = clamp(x, this.xMin, this.xMax);
    return interp(clipped, this.xThresh, this.yThresh);
  }
}

// ============================================================================
// HybridCMLSH class
// ============================================================================

/**
 * Hybrid Confidence Matrix LSH implementation.
 * 
 * Combines LSH-TS and ITQ for improved similarity search:
 * - LSH-TS: 256-bit random hyperplane hash (deterministic, backward compatible)
 * - ITQ: 256-bit quantized projection (learned, rotation optimized)
 * - Confidence: weights reliable bits higher in similarity computation
 * - Calibration: maps raw scores to accurate cosine similarities
 */
export class HybridCMLSH {
  private calibrator: Calibrator;
  
  constructor(
    private params: HybridParams,
    calibratorConfig: CalibratorConfig,
    private alpha: number = 0.65,
    private family: number = 0,
  ) {
    this.calibrator = new Calibrator(
      calibratorConfig.xThresh,
      calibratorConfig.yThresh,
      calibratorConfig.xMin,
      calibratorConfig.xMax,
    );
  }

  /**
   * Generate CM-LSH hash from embedding.
   * 
   * @param embedding - Input embedding vector (will be L2-normalized)
   * @returns DualHash with hashA (signature), hashB (confidence), and bands
   */
  hash(embedding: number[]): DualHash {
    // Normalize embedding to f32
    const emb32 = new Float32Array(embedding);
    let norm = 0;
    for (let i = 0; i < emb32.length; i++) {
      norm += emb32[i] * emb32[i];
    }
    norm = Math.sqrt(norm);
    
    if (norm > 1e-8) {
      for (let i = 0; i < emb32.length; i++) {
        emb32[i] /= norm;
      }
    }
    
    return this._genHash(emb32);
  }

  /**
   * Generate hash from normalized embedding.
   */
  private _genHash(emb: Float32Array): DualHash {
    // 1. LSH-TS projection (256 bits)
    const p1 = matmulVec(this.params.lshTsHyperplanes, emb);
    
    // 2. ITQ projection (256 bits)
    // Center
    const centered = new Float32Array(emb.length);
    for (let i = 0; i < emb.length; i++) {
      centered[i] = emb[i] - this.params.itq.mean[i];
    }
    
    // PCA projection
    const pcaProj = matmulVec(this.params.itq.pca, centered);
    
    // ITQ rotation
    const p2 = matmulVec(this.params.itq.rotation, pcaProj);
    
    // 3. Combine projections (512 bits total)
    const proj = new Float32Array(512);
    proj.set(p1, 0);
    proj.set(p2, 256);
    
    // 4. Sign bits (hashA)
    const signs: boolean[] = [];
    for (let i = 0; i < proj.length; i++) {
      signs.push(proj[i] > 0);
    }
    
    // 5. Confidence bits (hashB)
    // Use 45th percentile as threshold
    const absProj = new Float32Array(proj.length);
    for (let i = 0; i < proj.length; i++) {
      absProj[i] = Math.abs(proj[i]);
    }
    const confThresh = percentile(absProj, 45);
    
    const confBits: boolean[] = [];
    for (let i = 0; i < absProj.length; i++) {
      confBits.push(absProj[i] > confThresh);
    }
    
    // 6. Pack into hex strings
    const hashA = packBits(signs);
    const hashB = packBits(confBits);
    
    // 7. Split into bands (64 bands for LSH indexing)
    const bandLen = Math.floor(hashA.length / 64);
    const bands: string[] = [];
    for (let i = 0; i < hashA.length; i += bandLen) {
      bands.push(hashA.substring(i, i + bandLen));
      if (bands.length === 64) break;
    }
    
    return { hashA, hashB, bands, bits: 512 };
  }

  /**
   * Compute calibrated similarity between two hashes.
   * 
   * @param h1 - First hash
   * @param h2 - Second hash
   * @returns Calibrated cosine similarity estimate in [0, 1]
   */
  sim(h1: DualHash, h2: DualHash): number {
    const rawSim = this._rawSim(h1, h2);
    return this.calibrator.predict(rawSim);
  }

  /**
   * Compute raw similarity before calibration.
   */
  private _rawSim(h1: DualHash, h2: DualHash): number {
    const a1 = unpackBits(h1.hashA);
    const a2 = unpackBits(h2.hashA);
    const b1 = unpackBits(h1.hashB);
    const b2 = unpackBits(h2.hashB);
    
    // Compute agreement and confidence overlap
    let agreeCount = 0;
    let totalCount = 0;
    let confAgreeCount = 0;
    let confTotalCount = 0;
    
    for (let i = 0; i < a1.length; i++) {
      const agree = a1[i] === a2[i];
      const bothConf = b1[i] && b2[i];
      
      if (agree) agreeCount++;
      totalCount++;
      
      if (bothConf) {
        if (agree) confAgreeCount++;
        confTotalCount++;
      }
    }
    
    const overallAgreeRate = agreeCount / totalCount;
    
    // Weighted similarity: alpha * (confident agreement) + (1-alpha) * (overall agreement)
    if (confTotalCount > 0) {
      const confAgreeRate = confAgreeCount / confTotalCount;
      return this.alpha * confAgreeRate + (1 - this.alpha) * overallAgreeRate;
    } else {
      return overallAgreeRate;
    }
  }

  /**
   * Compare two embeddings via CM-LSH.
   * 
   * @param e1 - First embedding
   * @param e2 - Second embedding
   * @returns Calibrated cosine similarity estimate
   */
  cmp(e1: number[], e2: number[]): number {
    return this.sim(this.hash(e1), this.hash(e2));
  }

  /**
   * Check if two hashes represent duplicates.
   * 
   * @param h1 - First hash
   * @param h2 - Second hash
   * @param threshold - Similarity threshold (default: 0.85)
   * @returns True if similarity >= threshold
   */
  isDup(h1: DualHash, h2: DualHash, threshold: number = 0.85): boolean {
    return this.sim(h1, h2) >= threshold;
  }

  /**
   * Verify that LSH-TS portion matches standalone LSH-TS hash.
   * 
   * @param embedding - Input embedding vector
   * @returns True if first 256 bits match standalone LSH-TS hash
   */
  verifyLshTs(embedding: number[]): boolean {
    const h = this.hash(embedding);
    
    // Normalize embedding to f32
    const emb32 = new Float32Array(embedding);
    let norm = 0;
    for (let i = 0; i < emb32.length; i++) {
      norm += emb32[i] * emb32[i];
    }
    norm = Math.sqrt(norm);
    
    if (norm > 1e-8) {
      for (let i = 0; i < emb32.length; i++) {
        emb32[i] /= norm;
      }
    }
    
    // Generate standalone LSH-TS hash
    const lshTs = lshTsHash(emb32, this.family, 256);
    
    return lshTsCompat(h) === lshTs;
  }
}

// ============================================================================
// Factory functions
// ============================================================================

/**
 * Generate deterministic random hyperplanes for LSH.
 * 
 * @param family - Hash family index
 * @param bits - Number of bits (hyperplanes)
 * @param dims - Dimensionality of input vectors
 * @returns Matrix of shape (bits, dims) with +1/-1 entries
 */
export function genHyperplanes(family: number, bits: number, dims: number): Float32Array[] {
  const hp: Float32Array[] = [];
  for (let b = 0; b < bits; b++) {
    const row = new Float32Array(dims);
    for (let d = 0; d < dims; d++) {
      row[d] = _internal.signFor(family, b, d);
    }
    hp.push(row);
  }
  return hp;
}

/**
 * Generate standalone LSH-TS hash (for verification).
 */
function lshTsHash(emb: Float32Array, family: number, bits: number): string {
  const boolBits: boolean[] = [];
  for (let b = 0; b < bits; b++) {
    let dot = 0;
    for (let d = 0; d < emb.length; d++) {
      dot += emb[d] * _internal.signFor(family, b, d);
    }
    boolBits.push(dot > 0);
  }
  return packBits(boolBits);
}

/**
 * Create a default CM-LSH instance without training.
 * 
 * This creates a minimal CM-LSH configuration using:
 * - Random hyperplanes for LSH-TS (deterministic based on family)
 * - Identity transformations for ITQ (no learned rotation)
 * - Linear calibrator (no adjustment)
 * 
 * Note: Always produces 512-bit output (256 from LSH-TS + 256 from ITQ).
 * For dimensions < 256, ITQ output is padded with zeros.
 * 
 * @param dimensions - Embedding dimensionality
 * @param family - LSH family index (default: 0)
 * @returns HybridCMLSH instance with default parameters
 */
export function createDefaultCmLsh(dimensions: number, family: number = 0): HybridCMLSH {
  // Generate LSH-TS hyperplanes (always 256 bits)
  const lshHp = genHyperplanes(family, 256, dimensions);
  
  // Create identity ITQ parameters (always 256 bits output)
  // If dimensions < 256, we'll pad the output
  const itqDims = Math.min(256, dimensions);
  const mean = new Float32Array(dimensions); // zeros
  
  // PCA: identity for first itqDims dimensions
  // Output is always 256 dimensions (padded if needed)
  const pca: Float32Array[] = [];
  for (let i = 0; i < 256; i++) {
    const row = new Float32Array(dimensions);
    if (i < itqDims && i < dimensions) {
      row[i] = 1.0;
    }
    pca.push(row);
  }
  
  // Rotation: identity 256x256
  const rotation: Float32Array[] = [];
  for (let i = 0; i < 256; i++) {
    const row = new Float32Array(256);
    row[i] = 1.0;
    rotation.push(row);
  }
  
  const itq: ITQParams = { mean, pca, rotation };
  const params: HybridParams = { lshTsHyperplanes: lshHp, itq };
  
  // Create linear calibrator (identity function)
  const calibratorConfig: CalibratorConfig = {
    xThresh: [0.0, 1.0],
    yThresh: [0.0, 1.0],
    xMin: 0.0,
    xMax: 1.0,
  };
  
  return new HybridCMLSH(params, calibratorConfig, 0.65, family);
}
