#!/usr/bin/env ts-node
/**
 * Similarity comparison example.
 *
 * This example demonstrates:
 * - Comparing multiple vectors using LSH signatures
 * - Computing Hamming distance between signatures
 * - Estimating cosine similarity from Hamming distance
 *
 * Run with: npx ts-node typescript/examples/similarity_comparison.ts
 */

import {
  cosineFromHamming,
  hammingDistanceHex,
  normalizeVector,
  simhashLshMulti,
} from '../src/lsh';

function main() {
  console.log('=== LSH Similarity Comparison ===\n');

  // Three example vectors (unnormalized)
  const vectorA = [1.0, 1.0, 1.0, 1.0]; // Original
  const vectorB = [1.0, 0.9, 1.1, 1.0]; // Similar (small perturbation)
  const vectorC = [-1.0, -1.0, -1.0, -1.0]; // Opposite direction

  console.log('Input vectors:');
  console.log(`  A: ${JSON.stringify(vectorA)}`);
  console.log(`  B: ${JSON.stringify(vectorB)} (similar to A)`);
  console.log(`  C: ${JSON.stringify(vectorC)} (opposite to A)\n`);

  // Normalize all vectors
  const normA = normalizeVector(vectorA);
  const normB = normalizeVector(vectorB);
  const normC = normalizeVector(vectorC);

  // Generate signatures
  const sigA = simhashLshMulti(normA);
  const sigB = simhashLshMulti(normB);
  const sigC = simhashLshMulti(normC);

  console.log('Signatures (first family only):');
  console.log(`  A: ${sigA[0].signature.substring(0, 16)}`);
  console.log(`  B: ${sigB[0].signature.substring(0, 16)}`);
  console.log(`  C: ${sigC[0].signature.substring(0, 16)}`);
  console.log('     (showing first 16 hex chars of 64)\n');

  // Compare all pairs
  console.log('Pairwise comparisons:\n');

  compareSignatures('A vs B', sigA[0].signature, sigB[0].signature);
  compareSignatures('A vs C', sigA[0].signature, sigC[0].signature);
  compareSignatures('B vs C', sigB[0].signature, sigC[0].signature);

  console.log('\n✓ Comparison complete!');
  console.log('\nInterpretation:');
  console.log('  - Similarity > 0.9: Very similar');
  console.log('  - Similarity 0.7-0.9: Moderately similar');
  console.log('  - Similarity < 0.5: Dissimilar');
}

function compareSignatures(label: string, sig1: string, sig2: string): void {
  const hamming = hammingDistanceHex(sig1, sig2);
  const similarity = cosineFromHamming(hamming, 256);

  console.log(`${label}:`);
  console.log(`  Hamming distance: ${hamming}/256 bits differ`);
  console.log(`  Estimated cosine similarity: ${similarity.toFixed(4)}`);
  console.log();
}

main();
