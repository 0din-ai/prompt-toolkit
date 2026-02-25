/**
 * Confidence Matrix LSH (CM-LSH) example.
 * 
 * This example demonstrates:
 * - Enhanced LSH with confidence matrix
 * - Dual hash structure (512-bit signature + 512-bit confidence)
 * - Backward compatibility with standard LSH (first 256 bits)
 * - Calibrated similarity estimation
 * 
 * Run with: npx ts-node typescript/examples/cm_lsh.ts
 */

import { createDefaultCmLsh, lshTsCompat } from '../src/cm-lsh';
import { normalizeVector } from '../src/lsh';

function main() {
  console.log('=== Confidence Matrix LSH (CM-LSH) ===\n');

  // Example vectors
  const vectorA = [1.0, 1.0, 1.0, 1.0];
  const vectorB = [1.0, 0.9, 1.1, 1.0]; // Similar to A
  const vectorC = [-1.0, -1.0, -1.0, -1.0]; // Opposite to A

  console.log('Input vectors:');
  console.log(`  A: ${JSON.stringify(vectorA)}`);
  console.log(`  B: ${JSON.stringify(vectorB)} (similar to A)`);
  console.log(`  C: ${JSON.stringify(vectorC)} (opposite to A)\n`);

  // Normalize vectors
  const normA = normalizeVector(vectorA);
  const normB = normalizeVector(vectorB);
  const normC = normalizeVector(vectorC);

  // Create CM-LSH hasher with default configuration
  // This uses identity ITQ (no learned rotation) for simplicity
  // Family 0 for deterministic results
  const hasher = createDefaultCmLsh(normA.length, 0);

  console.log('CM-LSH Configuration:');
  console.log('  Total bits: 512 (256 LSH-TS + 256 ITQ)');
  console.log('  First 256 bits: LSH-TS compatible');
  console.log('  Confidence matrix: Alpha-weighted agreement\n');

  // Generate dual hashes
  const hashA = hasher.hash(normA);
  const hashB = hasher.hash(normB);
  const hashC = hasher.hash(normC);

  console.log('Dual hashes (showing first 32 hex chars of 128):');
  console.log(`  A: hash=${hashA.hashA.substring(0, 32)} conf=${hashA.hashB.substring(0, 32)}`);
  console.log(`  B: hash=${hashB.hashA.substring(0, 32)} conf=${hashB.hashB.substring(0, 32)}`);
  console.log(`  C: hash=${hashC.hashA.substring(0, 32)} conf=${hashC.hashB.substring(0, 32)}`);
  console.log();

  // Demonstrate LSH-TS backward compatibility
  console.log('LSH-TS compatibility (first 256 bits):');
  console.log(`  A: ${lshTsCompat(hashA).substring(0, 16)}`);
  console.log(`  B: ${lshTsCompat(hashB).substring(0, 16)}`);
  console.log(`  C: ${lshTsCompat(hashC).substring(0, 16)}`);
  console.log('     (showing first 16 hex chars of 64)\n');

  // Compute calibrated similarities
  console.log('Calibrated similarities:\n');

  const simAB = hasher.sim(hashA, hashB);
  const simAC = hasher.sim(hashA, hashC);
  const simBC = hasher.sim(hashB, hashC);

  console.log(`  A vs B: ${simAB.toFixed(4)}`);
  console.log(`  A vs C: ${simAC.toFixed(4)}`);
  console.log(`  B vs C: ${simBC.toFixed(4)}`);

  console.log('\n✓ CM-LSH example complete!');
  console.log('\nKey advantages of CM-LSH:');
  console.log('  - Confidence matrix weights reliable bits higher');
  console.log('  - Isotonic calibration improves similarity estimates');
  console.log('  - Dual hash (LSH + ITQ) for better quantization');
  console.log('  - Backward compatible with standard LSH');
}

main();
