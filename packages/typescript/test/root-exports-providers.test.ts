/**
 * Verifies that provider symbols are re-exported from the package root.
 *
 * Consumers should be able to import ModelCache, OnnxProvider, and
 * OpenAIProvider from '@0din/prompt-toolkit' rather than the deep subpath
 * '@0din/prompt-toolkit/providers'.
 */

import {
  ModelCache,
  OnnxProvider,
  OpenAIProvider,
} from '../src/index';

describe('Provider root re-exports', () => {
  it('exports ModelCache class', () => {
    expect(typeof ModelCache).toBe('function');
  });

  it('exports OnnxProvider class', () => {
    expect(typeof OnnxProvider).toBe('function');
    expect(typeof OnnxProvider.create).toBe('function');
  });

  it('exports OpenAIProvider class', () => {
    expect(typeof OpenAIProvider).toBe('function');
  });

  it('ModelCache can be instantiated', () => {
    const cache = new ModelCache();
    expect(typeof cache.modelDirectory).toBe('function');
    expect(typeof cache.hasSusfactorModel).toBe('function');
  });
});
