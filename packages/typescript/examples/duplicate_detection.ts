#!/usr/bin/env ts-node
/**
 * Duplicate detection using LSH bands.
 *
 * This example demonstrates:
 * - Batch signature generation for multiple vectors
 * - Using LSH bands for efficient candidate generation
 * - Finding near-duplicates in a collection
 *
 * Run with: npx ts-node typescript/examples/duplicate_detection.ts
 */

import {
  cosineFromHamming,
  hammingDistanceHex,
  normalizeVector,
  simhashLshMulti,
  type LSHFamily,
} from '../src/lsh';

function main() {
  console.log('=== Duplicate Detection with LSH ===\n');

  // Example vectors representing different documents
  // Vectors 0, 1, 2 are similar (duplicates)
  // Vectors 3, 4 are different
  const vectors = [
    [1.0, 1.0, 1.0, 1.0],      // Doc 0
    [1.0, 0.95, 1.05, 1.0],    // Doc 1 (near-duplicate of 0)
    [0.98, 1.02, 1.0, 1.01],   // Doc 2 (near-duplicate of 0, 1)
    [0.0, 1.0, 0.0, 1.0],      // Doc 3 (different)
    [-1.0, -1.0, -1.0, -1.0],  // Doc 4 (opposite, different)
  ];

  console.log(`Processing ${vectors.length} documents...\n`);

  // Normalize and generate signatures
  const signatures: LSHFamily[][] = vectors.map(v => 
    simhashLshMulti(normalizeVector(v))
  );

  // Build band index for candidate generation
  // Map: (band_index, band_value) -> [doc_ids]
  const bandIndex = new Map<string, number[]>();

  for (let docId = 0; docId < signatures.length; docId++) {
    // Use first family only for this example
    const family = signatures[docId][0];

    for (let bandIdx = 0; bandIdx < family.bands.length; bandIdx++) {
      const key = `${bandIdx}:${family.bands[bandIdx]}`;
      const docs = bandIndex.get(key) || [];
      docs.push(docId);
      bandIndex.set(key, docs);
    }
  }

  // Find candidate pairs (documents that share at least one band)
  const candidates = new Set<string>();

  for (const docs of bandIndex.values()) {
    if (docs.length > 1) {
      // Multiple documents match this band
      for (let i = 0; i < docs.length; i++) {
        for (let j = i + 1; j < docs.length; j++) {
          const pair = [docs[i], docs[j]].sort((a, b) => a - b).join(',');
          candidates.add(pair);
        }
      }
    }
  }

  console.log(`Found ${candidates.size} candidate pairs from band matching\n`);

  // Verify candidates with full Hamming distance
  const threshold = 0.85; // Cosine similarity threshold for duplicates
  const duplicates: [number, number, number][] = [];

  for (const pair of candidates) {
    const [id1, id2] = pair.split(',').map(Number);
    const sig1 = signatures[id1][0].signature;
    const sig2 = signatures[id2][0].signature;

    const hamming = hammingDistanceHex(sig1, sig2);
    const similarity = cosineFromHamming(hamming, 256);

    if (similarity >= threshold) {
      duplicates.push([id1, id2, similarity]);
    }
  }

  // Sort by similarity (descending)
  duplicates.sort((a, b) => b[2] - a[2]);

  console.log(`Detected duplicates (similarity >= ${threshold}):\n`);
  for (const [id1, id2, sim] of duplicates) {
    console.log(`  Doc ${id1} <-> Doc ${id2}: ${sim.toFixed(4)}`);
  }

  console.log('\n✓ Duplicate detection complete!');
  console.log('\nKey insight:');
  console.log('  - Band matching reduces comparisons from O(n²) to O(n)');
  console.log('  - Only candidate pairs need full Hamming distance computation');
  console.log('  - Tune bands/bits ratio for precision/recall tradeoff');
}

main();
