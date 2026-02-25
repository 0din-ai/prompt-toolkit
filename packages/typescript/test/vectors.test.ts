/**
 * Test TypeScript implementation against canonical test vectors.
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  simhashLshMulti,
  hammingDistanceHex,
  cosineFromHamming,
  normalizeVector,
  _internal,
} from '../src/lsh';
import {
  parseSignatureString,
  computeEmbeddingSha256,
  SignatureVersion,
} from '../src/types';

const VECTORS_DIR = path.join(__dirname, '../../../spec/test-vectors');

function loadVectors(filename: string): any {
  const filepath = path.join(VECTORS_DIR, filename);
  let content = fs.readFileSync(filepath, 'utf-8');
  
  // For SplitMix64 vectors, wrap large integers in quotes to preserve precision
  // This regex finds large integers in JSON (>2^53-1) and wraps them in quotes
  if (filename === 'splitmix64.json') {
    content = content.replace(/"input":\s*(\d{16,})/g, '"input": "$1"');
  }
  
  return JSON.parse(content);
}

describe('SplitMix64', () => {
  it('should match canonical test vectors', () => {
    const vectors = loadVectors('splitmix64.json');

    for (const testCase of vectors.vectors) {
      // Convert to BigInt - JSON loads large integers as numbers which lose precision
      const input = typeof testCase.input === 'string' 
        ? BigInt(testCase.input)
        : BigInt(Math.floor(testCase.input));
      const expected = testCase.output;
      const actual = _internal.splitmix64(input).toString(16).toUpperCase().padStart(16, '0');

      expect(actual).toBe(expected);
    }
  });
});

describe('SignFor', () => {
  it('should match canonical test vectors', () => {
    const vectors = loadVectors('sign_for.json');

    for (const testCase of vectors.vectors) {
      const { family, bit, dim, sign: expected } = testCase;
      const actual = _internal.signFor(family, bit, dim);

      expect(actual).toBe(expected);
    }
  });
});

describe('SimHash', () => {
  it('should match canonical test vectors', () => {
    const vectors = loadVectors('simhash.json');

    for (const testCase of vectors.vectors) {
      const { name, input, config, expected: expectedFamilies } = testCase;

      const families = simhashLshMulti(input, config);

      for (let i = 0; i < families.length; i++) {
        const actual = families[i];
        const expected = expectedFamilies[i];

        expect(actual.family).toBe(expected.family);
        expect(actual.bits).toBe(expected.bits);
        expect(actual.signature).toBe(expected.signature);
        expect(actual.bands).toEqual(expected.bands);
      }
    }
  });
});

describe('HammingDistance', () => {
  it('should match canonical test vectors', () => {
    const vectors = loadVectors('hamming.json');

    for (const testCase of vectors.vectors) {
      const { a, b, distance: expected, description } = testCase;
      const actual = hammingDistanceHex(a, b);

      expect(actual).toBe(expected);
    }
  });
});

describe('CosineFromHamming', () => {
  it('should match canonical test vectors', () => {
    const vectors = loadVectors('cosine.json');

    for (const testCase of vectors.vectors) {
      const { distance, total_bits, cosine_similarity: expected } = testCase;
      const actual = cosineFromHamming(distance, total_bits);

      expect(Math.abs(actual - expected)).toBeLessThan(1e-10);
    }
  });
});

describe('SHA256', () => {
  it('should match canonical test vectors', () => {
    const vectors = loadVectors('sha256.json');

    for (const testCase of vectors.vectors) {
      const { input, expected_json, expected_sha256, description } = testCase;
      const actual = computeEmbeddingSha256(input);

      expect(actual).toBe(expected_sha256);
    }
  });
});

describe('SignatureFormat', () => {
  it('should match canonical test vectors', () => {
    const vectors = loadVectors('signature_format.json');

    for (const testCase of vectors.vectors) {
      const { input, valid, description } = testCase;

      if (valid) {
        const { expected_version, expected_signature } = testCase;
        const result = parseSignatureString(input);

        expect(result.version).toBe(expected_version);
        expect(result.signature).toBe(expected_signature);
      } else {
        expect(() => parseSignatureString(input)).toThrow();
      }
    }
  });
});
