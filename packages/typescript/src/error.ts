/**
 * Error types for signature-sdk operations.
 *
 * All errors inherit from SigError base class for easy catching.
 * @module error
 */

/**
 * Base error class for all signature-sdk operations.
 *
 * All exceptions thrown by the signature-sdk library inherit from this base class,
 * making it easy to catch all library-specific errors.
 *
 * @example
 * ```typescript
 * try {
 *   // some signature-sdk operation
 * } catch (error) {
 *   if (error instanceof SigError) {
 *     console.error(`Signature operation failed: ${error.message}`);
 *   }
 * }
 * ```
 */
export class SigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SigError';
    // Restore prototype chain for instanceof checks
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Configuration error.
 *
 * Thrown when LSH configuration parameters are invalid or incompatible.
 *
 * @example
 * ```typescript
 * import { LshConfig } from 'signature-sdk';
 * // Invalid configuration would throw ConfigError
 * ```
 */
export class ConfigError extends SigError {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigError';
  }
}

/**
 * Embedding provider error.
 *
 * Thrown when an embedding provider fails to generate embeddings,
 * such as API failures, authentication errors, or network issues.
 *
 * @example
 * ```typescript
 * // API call failure would throw ProviderError
 * ```
 */
export class ProviderError extends SigError {
  constructor(message: string) {
    super(message);
    this.name = 'ProviderError';
  }
}

/**
 * Model loading or inference error.
 *
 * Thrown when ONNX model loading fails or inference encounters an error.
 *
 * @example
 * ```typescript
 * // Model file not found would throw ModelError
 * ```
 */
export class ModelError extends SigError {
  constructor(message: string) {
    super(message);
    this.name = 'ModelError';
  }
}

/**
 * Invalid input data.
 *
 * Thrown when input data doesn't meet requirements, such as empty text,
 * invalid embedding dimensions, or malformed signature strings.
 *
 * @example
 * ```typescript
 * import { parseSignatureString } from 'signature-sdk';
 * try {
 *   parseSignatureString('invalid');
 * } catch (error) {
 *   if (error instanceof InvalidInputError) {
 *     console.error(`Invalid signature: ${error.message}`);
 *   }
 * }
 * ```
 */
export class InvalidInputError extends SigError {
  constructor(message: string) {
    super(message);
    this.name = 'InvalidInputError';
  }
}

/**
 * Threat feed API error.
 *
 * Thrown when HTTP requests to the 0din threat feed API fail,
 * such as authentication errors, network issues, or bad responses.
 */
export class ThreatFeedApiError extends SigError {
  public readonly statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'ThreatFeedApiError';
    this.statusCode = statusCode;
  }
}

/**
 * Threat feed cache error.
 *
 * Thrown when cache I/O operations fail, such as corrupt cache files,
 * write failures, or schema version mismatches.
 */
export class ThreatFeedCacheError extends SigError {
  constructor(message: string) {
    super(message);
    this.name = 'ThreatFeedCacheError';
  }
}
