#!/usr/bin/env ts-node
/**
 * Basic LSH signature generation example.
 *
 * This example demonstrates:
 * - Generating an LSH signature from a normalized vector
 * - Default configuration (3 families, 256 bits, 16 bands)
 * - Formatting and parsing signature strings
 *
 * Run with: npx ts-node typescript/examples/basic_signature.ts
 */

import { simhashLshMulti } from '../src/lsh';

function main() {
  console.log('=== Basic LSH Signature Generation ===\n');

  // Example normalized vector (4 dimensions for clarity)
  // In practice, this would come from an embedding model (384 or 1536 dims)
  const normalizedVector = [0.5, 0.5, 0.5, 0.5];

  console.log(`Input vector: ${JSON.stringify(normalizedVector)}`);
  console.log(`Vector dimensions: ${normalizedVector.length}\n`);

  // Generate LSH signatures with default configuration
  const config = {
    families: 3,
    bits: 256,
    bands: 16,
  };

  console.log('Configuration:');
  console.log(`  Families: ${config.families}`);
  console.log(`  Bits per signature: ${config.bits}`);
  console.log(`  Bands: ${config.bands}\n`);

  const families = simhashLshMulti(normalizedVector, config);

  // Display results for each family
  for (const family of families) {
    console.log(`Family ${family.family}:`);
    console.log(`  Signature (hex): ${family.signature}`);
    console.log(`  Signature length: ${family.signature.length} hex chars = ${family.bits} bits`);
    console.log(`  Number of bands: ${family.bands.length}`);
    console.log(`  Band 0: ${family.bands[0]} (first ${family.bands[0].length} hex chars)`);
    console.log();
  }

  // Format as 0din signature string (V1 format)
  const primarySig = families[0].signature;
  const signatureString = `0din-v1:${primarySig}`;
  
  console.log('Formatted signature string:');
  console.log(`  ${signatureString}`);
  console.log();

  // In a real application, you would:
  // 1. Store this signature in a database with the original text
  // 2. Use bands for efficient similarity search (LSH indexing)
  // 3. Compare signatures using hamming distance
  
  console.log('✓ Signature generation complete!');
}

main();
