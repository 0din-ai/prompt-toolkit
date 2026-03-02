/**
 * Tests for type definitions
 */

import { ComparisonResult, PromptInfo, QualityStats } from '../src/types';
import type { LshConfig } from '../src/types';

describe('Type definitions', () => {
  test('PromptInfo construction', () => {
    const info: PromptInfo = {
      preview: 'Test prompt preview',
      length: 100,
      signature: '0din-v1:abc123',
    };

    expect(info.preview).toBe('Test prompt preview');
    expect(info.length).toBe(100);
    expect(info.signature).toBe('0din-v1:abc123');
  });

  test('QualityStats construction', () => {
    const stats: QualityStats = {
      absoluteError: 0.05,
      signedError: -0.02,
      squaredError: 0.0025,
      qualityRating: 'excellent',
    };

    expect(stats.absoluteError).toBe(0.05);
    expect(stats.signedError).toBe(-0.02);
    expect(stats.squaredError).toBe(0.0025);
    expect(stats.qualityRating).toBe('excellent');
  });

  test('ComparisonResult construction', () => {
    const promptA: PromptInfo = {
      preview: 'First prompt',
      length: 50,
      signature: '0din-v1:aaa111',
    };

    const promptB: PromptInfo = {
      preview: 'Second prompt',
      length: 60,
      signature: '0din-v1:bbb222',
    };

    const config: LshConfig = {
      families: 3,
      bits: 256,
      bands: 16,
    };

    const stats: QualityStats = {
      absoluteError: 0.1,
      signedError: 0.08,
      squaredError: 0.01,
      qualityRating: 'good',
    };

    const result: ComparisonResult = {
      promptA,
      promptB,
      hammingDistance: 50,
      cosineSimilarity: 0.85,
      lshConfig: config,
      qualityStats: stats,
      timingMs: 5.0,
    };

    expect(result.promptA.preview).toBe('First prompt');
    expect(result.promptB.preview).toBe('Second prompt');
    expect(result.hammingDistance).toBe(50);
    expect(result.cosineSimilarity).toBe(0.85);
    expect(result.lshConfig.families).toBe(3);
    expect(result.qualityStats?.qualityRating).toBe('good');
    expect(result.timingMs).toBe(5.0);
  });

  test('ComparisonResult with optional fields omitted', () => {
    const promptA: PromptInfo = {
      preview: 'A',
      length: 10,
      signature: '0din-v1:aaa',
    };

    const promptB: PromptInfo = {
      preview: 'B',
      length: 20,
      signature: '0din-v1:bbb',
    };

    const config: LshConfig = {
      families: 3,
      bits: 256,
      bands: 16,
    };

    const result: ComparisonResult = {
      promptA,
      promptB,
      hammingDistance: 100,
      cosineSimilarity: 0.5,
      lshConfig: config,
    };

    expect(result.qualityStats).toBeUndefined();
    expect(result.timingMs).toBeUndefined();
  });
});
