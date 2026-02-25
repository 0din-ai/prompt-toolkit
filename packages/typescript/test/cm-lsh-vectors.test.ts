/**
 * Test CM-LSH implementation against canonical test vectors.
 */

import * as fs from 'fs';
import * as path from 'path';
import { createDefaultCmLsh, lshTsCompat } from '../src/cm-lsh';
import { hammingDistanceHex } from '../src/lsh';

const VECTORS_DIR = path.join(__dirname, '../../../spec/test-vectors');

function loadVectors(filename: string): any {
  const filepath = path.join(VECTORS_DIR, filename);
  const content = fs.readFileSync(filepath, 'utf-8');
  return JSON.parse(content);
}

describe('CM-LSH', () => {
  describe('Hash Vectors', () => {
    it('should match canonical test vectors', () => {
      const vectors = loadVectors('cm_lsh.json');

      for (const testCase of vectors.hash_vectors) {
        const { name, input, expected } = testCase;

        // Create CM-LSH hasher
        const cmLsh = createDefaultCmLsh(input.length, 0);

        // Generate hash
        const hashResult = cmLsh.hash(input);

        // Check structure
        expect(hashResult.bits).toBe(expected.bits);
        expect(hashResult.bands.length).toBe(expected.bands.length);

        // Check LSH-TS portion has reasonable similarity (allow some bit differences)
        const lshTsActual = lshTsCompat(hashResult);
        const lshTsExpected = expected.lsh_ts_compat;
        const hammingDist = hammingDistanceHex(lshTsActual, lshTsExpected);

        // Allow up to 7% bit difference due to floating-point precision
        // (TypeScript f32 vs Rust f32 can cause small differences in dot products)
        const maxAllowedDiff = Math.floor(256 * 0.07);
        
        if (hammingDist > maxAllowedDiff) {
          console.error(`${name}: LSH-TS compatibility has too many bit differences`);
          console.error(`Hamming distance: ${hammingDist} (max allowed: ${maxAllowedDiff})`);
          console.error(`Expected: ${lshTsExpected}`);
          console.error(`Actual:   ${lshTsActual}`);
        }
        
        expect(hammingDist).toBeLessThanOrEqual(maxAllowedDiff);
      }
    });
  });

  describe('Similarity Vectors', () => {
    it('should match canonical similarity computations', () => {
      const vectors = loadVectors('cm_lsh.json');

      // Create hasher
      const cmLsh = createDefaultCmLsh(384, 0);

      for (const testCase of vectors.similarity_vectors) {
        const { name, embedding1, embedding2, similarity: expectedSim } = testCase;

        // Compute similarity
        const h1 = cmLsh.hash(embedding1);
        const h2 = cmLsh.hash(embedding2);
        const actualSim = cmLsh.sim(h1, h2);

        // Allow 1% relative difference or 0.01 absolute difference (whichever is larger)
        const relativeDiff = Math.abs(actualSim - expectedSim) / Math.max(Math.abs(expectedSim), 0.01);
        const absoluteDiff = Math.abs(actualSim - expectedSim);

        if (relativeDiff >= 0.01 && absoluteDiff >= 0.01) {
          console.error(`${name}: similarity mismatch`);
          console.error(`Expected: ${expectedSim.toFixed(6)}`);
          console.error(`Actual:   ${actualSim.toFixed(6)}`);
          console.error(`Relative diff: ${(relativeDiff * 100).toFixed(2)}%`);
          console.error(`Absolute diff: ${absoluteDiff.toFixed(6)}`);
        }

        expect(relativeDiff < 0.01 || absoluteDiff < 0.01).toBe(true);
      }
    });
  });

  describe('Self Similarity', () => {
    it('should have similarity ~1.0 for identical vectors', () => {
      const cmLsh = createDefaultCmLsh(384, 0);

      // Create random vector
      const vector = Array.from({ length: 384 }, () => Math.random() - 0.5);
      const h1 = cmLsh.hash(vector);
      const h2 = cmLsh.hash(vector);

      const similarity = cmLsh.sim(h1, h2);
      expect(similarity).toBeGreaterThan(0.99);
    });
  });

  describe('LSH-TS Compatibility', () => {
    it('should verify first 256 bits match standalone LSH-TS', () => {
      const cmLsh = createDefaultCmLsh(384, 0);

      // Create random vector
      const vector = Array.from({ length: 384 }, () => Math.random() - 0.5);

      // Verify LSH-TS compatibility
      expect(cmLsh.verifyLshTs(vector)).toBe(true);
    });
  });
});
