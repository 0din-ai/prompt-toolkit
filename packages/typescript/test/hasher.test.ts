/**
 * Tests for hasher abstraction
 */

import { getHasher, HashAlgorithm, SimHashLsh } from '../src';
import type { Hasher } from '../src/hasher';
import type { LshConfig } from '../src/types';
import { normalizeVector } from '../src/lsh';

describe('Hasher abstraction', () => {
  test('getHasher returns SimHashLsh for LSH algorithm', () => {
    const hasher = getHasher(HashAlgorithm.LSH);
    expect(hasher).toBeInstanceOf(SimHashLsh);
    expect(hasher.name()).toBe('lsh');
  });

  test('getHasher throws for unknown algorithm', () => {
    expect(() => {
      getHasher(HashAlgorithm.OPENAI);
    }).toThrow('Unknown algorithm');
  });

  test('SimHashLsh.name() returns "lsh"', () => {
    const hasher = new SimHashLsh();
    expect(hasher.name()).toBe('lsh');
  });

  test('SimHashLsh.compute() produces correct signatures', () => {
    const hasher = new SimHashLsh();
    const vector = normalizeVector([1.0, 2.0, 3.0, 4.0]);
    const config: LshConfig = {
      families: 2,
      bits: 64,
      bands: 4,
    };

    const families = hasher.compute(vector, config);

    expect(families).toHaveLength(2);
    expect(families[0].family).toBe(0);
    expect(families[1].family).toBe(1);
    expect(families[0].bits).toBe(64);
    expect(families[1].bits).toBe(64);
    expect(families[0].signature).toHaveLength(16); // 64 bits = 16 hex chars
    expect(families[1].signature).toHaveLength(16);
    expect(families[0].bands).toHaveLength(4);
    expect(families[1].bands).toHaveLength(4);
  });

  test('SimHashLsh conforms to Hasher interface', () => {
    const hasher: Hasher = new SimHashLsh();
    const vector = normalizeVector([0.5, 0.5, 0.5, 0.5]);
    const config: LshConfig = {
      families: 1,
      bits: 256,
      bands: 16,
    };

    // Should satisfy the Hasher interface
    expect(typeof hasher.name).toBe('function');
    expect(typeof hasher.compute).toBe('function');

    const name = hasher.name();
    expect(typeof name).toBe('string');

    const families = hasher.compute(vector, config);
    expect(Array.isArray(families)).toBe(true);
    expect(families).toHaveLength(1);
  });
});
