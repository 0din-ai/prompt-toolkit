/**
 * OpenAI embedding provider implementation.
 */

import { normalizeVector } from '../lsh';
import { EmbeddingResult, computeEmbeddingSha256 } from '../types';
import { EmbeddingProvider } from '../provider';

/**
 * Embedding provider using OpenAI API.
 *
 * This provider uses the OpenAI embeddings API to generate vector embeddings
 * for text. It can also be configured to use OpenRouter or other OpenAI-compatible
 * APIs by setting a custom base URL.
 *
 * @example
 * ```typescript
 * const provider = new OpenAIProvider({ apiKey: 'sk-...' });
 * const result = await provider.generateEmbedding("Hello, world!");
 * console.log(`Generated ${result.dimensions}-dimensional embedding`);
 * await provider.close();
 * ```
 */
export class OpenAIProvider implements EmbeddingProvider {
  private static readonly DEFAULT_MODEL = 'text-embedding-3-large';
  private static readonly DEFAULT_DIMENSIONS = 1536;
  private static readonly DEFAULT_BASE_URL = 'https://api.openai.com/v1';

  private client: any; // OpenAI client type
  private providerName: string;
  private modelName: string;
  private dims: number;

  /**
   * Initialize OpenAI provider.
   *
   * Note: Requires the 'openai' package to be installed:
   * ```bash
   * npm install openai
   * ```
   *
   * @param options - Configuration options
   */
  constructor(options: {
    apiKey: string;
    model?: string;
    dimensions?: number;
    baseURL?: string;
    name?: string;
  }) {
    this.modelName = options.model || OpenAIProvider.DEFAULT_MODEL;
    this.dims = options.dimensions || OpenAIProvider.DEFAULT_DIMENSIONS;
    this.providerName = options.name || 'openai';

    // Dynamically import OpenAI
    try {
      const { OpenAI } = require('openai');
      this.client = new OpenAI({
        apiKey: options.apiKey,
        baseURL: options.baseURL || OpenAIProvider.DEFAULT_BASE_URL,
      });
    } catch (error) {
      throw new Error(
        "OpenAI provider requires the 'openai' package. Install with: npm install openai",
        { cause: error }
      );
    }
  }

  name(): string {
    return this.providerName;
  }

  model(): string {
    return this.modelName;
  }

  dimensions(): number {
    return this.dims;
  }

  async generateEmbedding(text: string): Promise<EmbeddingResult> {
    const startTime = Date.now();

    // Call OpenAI API
    const response = await this.client.embeddings.create({
      model: this.modelName,
      input: text,
      dimensions: this.dims,
    });

    const elapsedMs = Date.now() - startTime;

    // Extract embedding
    const embedding = response.data[0].embedding;
    const tokenCount = response.usage?.total_tokens || 0;

    // Normalize and compute SHA256
    const normalized = normalizeVector(embedding);
    const sha256 = computeEmbeddingSha256(normalized);

    return {
      embedding,
      normalizedEmbedding: normalized,
      normalizedEmbeddingSha256: sha256,
      model: this.modelName,
      dimensions: embedding.length,
      tokenCount,
      timingMs: elapsedMs,
    };
  }

  async close(): Promise<void> {
    // OpenAI client doesn't require explicit cleanup
  }
}
