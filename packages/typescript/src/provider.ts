/**
 * Embedding provider interface definition.
 */

import { EmbeddingResult } from './types';

/**
 * Interface for embedding generation providers.
 *
 * All embedding providers must implement this interface to work with
 * the signText() function.
 *
 * @example
 * ```typescript
 * class MyProvider implements EmbeddingProvider {
 *   name(): string {
 *     return 'my-provider';
 *   }
 *
 *   model(): string {
 *     return 'my-model';
 *   }
 *
 *   dimensions(): number {
 *     return 384;
 *   }
 *
 *   async generateEmbedding(text: string): Promise<EmbeddingResult> {
 *     // Generate and return embedding
 *   }
 *
 *   async close(): Promise<void> {
 *     // Cleanup resources
 *   }
 * }
 * ```
 */
export interface EmbeddingProvider {
  /**
   * Get the provider name.
   *
   * @returns Provider name (e.g., "onnx", "openai")
   */
  name(): string;

  /**
   * Get the model identifier.
   *
   * @returns Model name or path (e.g., "intfloat/multilingual-e5-small")
   */
  model(): string;

  /**
   * Get the embedding dimensionality.
   *
   * @returns Number of dimensions in the embedding vector
   */
  dimensions(): number;

  /**
   * Generate embedding for the given text.
   *
   * @param text - Input text to embed
   * @returns EmbeddingResult containing the raw embedding, normalized embedding,
   *          SHA256 hash, model info, and timing metrics
   * @throws Error if embedding generation fails
   */
  generateEmbedding(text: string): Promise<EmbeddingResult>;

  /**
   * Close the provider and clean up resources.
   *
   * This should be called when done using the provider to properly
   * release any allocated resources (models, HTTP connections, etc.).
   */
  close(): Promise<void>;
}
