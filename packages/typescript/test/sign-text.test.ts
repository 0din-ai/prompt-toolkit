/**
 * Tests for signText() high-level API.
 */

import { signText } from '../src/sign';
import { EmbeddingProvider } from '../src/provider';
import { SignatureVersion, EmbeddingResult, LshConfig, getSignatureString } from '../src/types';

/**
 * Mock embedding provider for testing.
 */
class MockProvider implements EmbeddingProvider {
  private dims: number;

  constructor(dimensions: number = 768) {
    this.dims = dimensions;
  }

  name(): string {
    return 'mock-provider';
  }

  model(): string {
    return 'mock-model';
  }

  dimensions(): number {
    return this.dims;
  }

  async generateEmbedding(_text: string): Promise<EmbeddingResult> {
    // Return a pre-normalized test embedding
    const embedding = new Array(this.dims).fill(0.5);
    return {
      embedding,
      normalizedEmbedding: embedding,
      normalizedEmbeddingSha256: 'test-sha256',
      model: 'mock-model',
      dimensions: this.dims,
      tokenCount: 10,
      timingMs: 100.0,
    };
  }

  async close(): Promise<void> {
    // No-op
  }
}

describe('signText', () => {
  test('sign_text with explicit V1 provider', async () => {
    const provider = new MockProvider(768);

    const result = await signText('test prompt', {
      provider,
      version: SignatureVersion.V1,
    });

    expect(result.version).toBe(SignatureVersion.V1);
    expect(result.provider).toBe('mock-provider');
    expect(result.model).toBe('mock-model');
    expect(result.dimensions).toBe(768);
    expect(result.promptPreview).toBe('test prompt');
    expect(result.promptLength).toBe(11);
    expect(result.timingMs).toBeGreaterThan(0);

    // Verify signature format
    const sigString = getSignatureString(result);
    expect(sigString).toMatch(/^0din-v1:[0-9a-f]{64}$/);
  });

  test('sign_text with explicit V0 provider', async () => {
    const provider = new MockProvider(1536);

    const result = await signText('test prompt', {
      provider,
      version: SignatureVersion.V0,
    });

    expect(result.version).toBe(SignatureVersion.V0);
    expect(result.dimensions).toBe(1536);

    const sigString = getSignatureString(result);
    expect(sigString).toMatch(/^0din-v0:/);
  });

  test('LATEST resolves to V1', async () => {
    const provider = new MockProvider(768);

    const result = await signText('test', {
      provider,
      version: SignatureVersion.LATEST,
    });

    expect(result.version).toBe(SignatureVersion.V1);
  });

  test('infer version from provider dimensions', async () => {
    // V1 provider (768 dims) - version inferred
    const providerV1 = new MockProvider(768);
    const resultV1 = await signText('test', { provider: providerV1 });
    expect(resultV1.version).toBe(SignatureVersion.V1);

    // V0 provider (1536 dims) - version inferred
    const providerV0 = new MockProvider(1536);
    const resultV0 = await signText('test', { provider: providerV0 });
    expect(resultV0.version).toBe(SignatureVersion.V0);
  });

  test('version mismatch with provider throws error', async () => {
    // Provider returns 768 dimensions but we request V0 (expects 1536)
    const provider = new MockProvider(768);

    await expect(
      signText('test', { provider, version: SignatureVersion.V0 })
    ).rejects.toThrow('Version mismatch');
  });

  test('unknown dimensions throw error', async () => {
    // Provider with non-standard dimensions
    const provider = new MockProvider(512);

    await expect(
      signText('test', { provider })
    ).rejects.toThrow('Cannot infer version');
  });

  test('sign_text with custom config', async () => {
    const provider = new MockProvider(768);

    const customConfig: LshConfig = {
      families: 5,
      bits: 128,
      bands: 8,
    };

    const result = await signText('test', {
      provider,
      config: customConfig,
    });

    expect(result.lsh.config.families).toBe(5);
    expect(result.lsh.config.bits).toBe(128);
    expect(result.lsh.config.bands).toBe(8);
    expect(result.lsh.signatures.length).toBe(5); // 5 families
  });

  test('long prompt preview truncation', async () => {
    const provider = new MockProvider(768);

    const longText = 'a'.repeat(100);
    const result = await signText(longText, { provider });

    // Preview should be truncated to 50 chars
    expect(result.promptPreview.length).toBe(50);
    expect(result.promptPreview).toMatch(/\.\.\.$/);
    expect(result.promptLength).toBe(100);
  });

  test('auto-construct fails without OPENAI_API_KEY', async () => {
    // Remove OPENAI_API_KEY from environment
    const oldKey = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;

    await expect(
      signText('test', { version: SignatureVersion.V0 })
    ).rejects.toThrow('OPENAI_API_KEY environment variable is required');

    // Restore
    if (oldKey) process.env.OPENAI_API_KEY = oldKey;
  });

  test('backward compat - provider-only call', async () => {
    const provider = new MockProvider(768);

    // Should still work with just provider (version inferred)
    const result = await signText('test', { provider });

    expect(result.version).toBe(SignatureVersion.V1);
    expect(result.provider).toBe('mock-provider');
  });
});
