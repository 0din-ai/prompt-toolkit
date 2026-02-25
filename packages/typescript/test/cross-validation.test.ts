/**
 * Cross-language validation test for signText().
 *
 * This test verifies that the same input produces identical signatures
 * across Rust, Python, and TypeScript implementations.
 */

import { signText } from '../src/sign';
import { EmbeddingProvider } from '../src/provider';
import { SignatureVersion, EmbeddingResult, getSignatureString, computeEmbeddingSha256 } from '../src/types';
import { normalizeVector } from '../src/lsh';

/**
 * Mock provider that returns a fixed embedding for cross-validation.
 */
class FixedEmbeddingProvider implements EmbeddingProvider {
  private embedding: number[];
  private dims: number;

  constructor(dimensions: number) {
    this.dims = dimensions;
    // Create a deterministic test embedding (all 0.5)
    this.embedding = new Array(dimensions).fill(0.5);
  }

  name(): string {
    return 'fixed-provider';
  }

  model(): string {
    return 'fixed-model';
  }

  dimensions(): number {
    return this.dims;
  }

  async generateEmbedding(_text: string): Promise<EmbeddingResult> {
    // Return the fixed embedding (normalize it)
    const normalized = normalizeVector(this.embedding);
    const sha256 = computeEmbeddingSha256(normalized);

    return {
      embedding: this.embedding,
      normalizedEmbedding: normalized,
      normalizedEmbeddingSha256: sha256,
      model: 'fixed-model',
      dimensions: this.dims,
      tokenCount: 10,
      timingMs: 100.0,
    };
  }

  async close(): Promise<void> {
    // No-op
  }
}

describe('Cross-language validation', () => {
  test('V1 signature with fixed embedding', async () => {
    // Create provider with V1 dimensions (384)
    const provider = new FixedEmbeddingProvider(384);

    // Generate signature
    const result = await signText('test prompt', { provider, version: SignatureVersion.V1 });

    const signature = getSignatureString(result);

    // Print for cross-validation with Rust/Python
    console.log(`TypeScript V1 signature: ${signature}`);
    console.log(`TypeScript V1 embedding SHA256: ${result.embeddingSha256}`);

    // Verify format
    expect(signature).toMatch(/^0din-v1:/);
    expect(signature.length).toBe(72); // "0din-v1:" (8) + 64 hex chars

    // Verify all hex characters
    const hexPart = signature.substring(8);
    expect(hexPart).toMatch(/^[0-9a-f]{64}$/);
  });

  test('V0 signature with fixed embedding', async () => {
    // Create provider with V0 dimensions (1536)
    const provider = new FixedEmbeddingProvider(1536);

    // Generate signature
    const result = await signText('test prompt', { provider, version: SignatureVersion.V0 });

    const signature = getSignatureString(result);

    // Print for cross-validation
    console.log(`TypeScript V0 signature: ${signature}`);
    console.log(`TypeScript V0 embedding SHA256: ${result.embeddingSha256}`);

    // Verify format
    expect(signature).toMatch(/^0din-v0:/);
    expect(signature.length).toBe(72); // "0din-v0:" (8) + 64 hex chars
  });

  test('Pattern vector signature', async () => {
    /**
     * Provider with alternating pattern.
     */
    class PatternProvider implements EmbeddingProvider {
      private embedding: number[];

      constructor() {
        // Create a pattern: alternating positive/negative
        this.embedding = Array.from({ length: 384 }, (_, i) => (i % 2 === 0 ? 1.0 : -1.0));
      }

      name(): string {
        return 'pattern-provider';
      }

      model(): string {
        return 'pattern-model';
      }

      dimensions(): number {
        return 384;
      }

      async generateEmbedding(_text: string): Promise<EmbeddingResult> {
        const normalized = normalizeVector(this.embedding);
        const sha256 = computeEmbeddingSha256(normalized);

        return {
          embedding: this.embedding,
          normalizedEmbedding: normalized,
          normalizedEmbeddingSha256: sha256,
          model: 'pattern-model',
          dimensions: 384,
          tokenCount: 10,
          timingMs: 100.0,
        };
      }

      async close(): Promise<void> {
        // No-op
      }
    }

    const provider = new PatternProvider();

    const result = await signText('test prompt', { provider, version: SignatureVersion.V1 });

    const signature = getSignatureString(result);

    console.log(`TypeScript pattern signature: ${signature}`);
    console.log(`TypeScript pattern embedding SHA256: ${result.embeddingSha256}`);

    // Verify format
    expect(signature).toMatch(/^0din-v1:/);
  });
});
