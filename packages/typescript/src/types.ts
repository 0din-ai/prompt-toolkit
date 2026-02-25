/**
 * Type definitions and utilities for signature generation.
 */

import * as crypto from 'crypto';

/**
 * Signature version enumeration.
 * 
 * Each version corresponds to a specific embedding model and dimensionality.
 * V0 and V1 signatures are NOT comparable due to different embedding spaces.
 */
export enum SignatureVersion {
  V0 = 'v0', // OpenAI text-embedding-3-large (1536 dims)
  V1 = 'v1', // multilingual-e5-small ONNX (384 dims)
  LATEST = 'latest', // Resolves to V1
}

/**
 * Hash algorithm enumeration.
 */
export enum HashAlgorithm {
  LSH = 'lsh', // Generic LSH (used with any embedding)
  OPENAI = 'openai', // OpenAI embeddings (V0, 1536 dims)
  ONNX = 'onnx', // ONNX local embeddings (V1, 384 dims)
}

/**
 * Parsed signature string.
 */
export interface ParsedSignature {
  version: SignatureVersion;
  signature: string; // Hex string (family 0 only for V0/V1)
}

/**
 * Embedding generation result.
 */
export interface EmbeddingResult {
  embedding: number[];
  normalizedEmbedding: number[];
  normalizedEmbeddingSha256: string;
  model: string;
  dimensions: number;
  tokenCount?: number;
  timingMs?: number;
}

/**
 * LSH configuration parameters.
 */
export interface LshConfig {
  families: number;
  bits: number;
  bands: number;
}

/**
 * LSH computation output.
 */
export interface LshOutput {
  config: LshConfig;
  signatures: Array<{
    family: number;
    bits: number;
    signature: string;
    bands: string[];
  }>;
}

/**
 * Complete signature generation result.
 *
 * This contains all metadata from the signature generation process,
 * including the primary signature string, provider info, embedding hash,
 * and LSH families.
 */
export interface SignatureResult {
  signature: string; // Formatted signature string (e.g., "0din-v1:...")
  version: SignatureVersion;
  promptPreview: string;
  promptLength: number;
  provider: string;
  model: string;
  dimensions: number;
  embeddingSha256: string;
  lsh: LshOutput;
  timingMs?: number;
}

/**
 * Get the formatted signature string from a SignatureResult.
 */
export function getSignatureString(result: SignatureResult): string {
  const resolvedVersion = resolveVersion(result.version);
  const primarySig = result.lsh.signatures[0].signature;
  return `0din-${resolvedVersion}:${primarySig}`;
}

/**
 * Resolve 'latest' to the current version.
 */
export function resolveVersion(version: SignatureVersion): SignatureVersion {
  if (version === SignatureVersion.LATEST) {
    return SignatureVersion.V1;
  }
  return version;
}

/**
 * Get expected embedding dimensions for a signature version.
 */
export function embeddingDimensions(version: SignatureVersion): number {
  const resolved = resolveVersion(version);
  if (resolved === SignatureVersion.V0) {
    return 1536;
  } else if (resolved === SignatureVersion.V1) {
    return 384;
  }
  throw new Error(`Unknown version: ${resolved}`);
}

/**
 * Get hash algorithm for a signature version.
 */
export function versionToAlgorithm(version: SignatureVersion): HashAlgorithm {
  const resolved = resolveVersion(version);
  if (resolved === SignatureVersion.V0) {
    return HashAlgorithm.OPENAI;
  } else if (resolved === SignatureVersion.V1) {
    return HashAlgorithm.ONNX;
  }
  throw new Error(`Unknown version: ${resolved}`);
}

/**
 * Get version from hash algorithm.
 */
export function algorithmToVersion(algorithm: HashAlgorithm): SignatureVersion {
  if (algorithm === HashAlgorithm.OPENAI) {
    return SignatureVersion.V0;
  } else if (algorithm === HashAlgorithm.ONNX) {
    return SignatureVersion.V1;
  }
  throw new Error(`Unknown algorithm: ${algorithm}`);
}

/**
 * Format signature as versioned string.
 * 
 * @param version - Signature version (v0, v1, etc.)
 * @param signature - Hex-encoded signature string
 * @returns Formatted string like "0din-v0:deadbeef..."
 * 
 * @example
 * ```typescript
 * import { signatureString, SignatureVersion } from '@0din/sig';
 * 
 * const sig = signatureString(SignatureVersion.V1, 'abcd1234');
 * console.log(sig); // "0din-v1:abcd1234"
 * ```
 */
export function signatureString(version: SignatureVersion, signature: string): string {
  const resolved = resolveVersion(version);
  return `0din-${resolved}:${signature}`;
}

/**
 * Parse versioned signature string.
 * 
 * @param s - Signature string like "0din-v0:deadbeef..."
 * @returns Parsed signature with version and hex string
 * @throws Error if format is invalid or version unsupported
 * 
 * @example
 * ```typescript
 * import { parseSignatureString } from '@0din/sig';
 * 
 * const parsed = parseSignatureString('0din-v1:abcd1234');
 * console.log(parsed.version); // 'v1'
 * console.log(parsed.signature); // 'abcd1234'
 * ```
 */
export function parseSignatureString(s: string): ParsedSignature {
  if (!s.startsWith('0din-')) {
    throw new Error(`Invalid signature prefix: ${s}`);
  }

  const parts = s.split(':', 2);
  if (parts.length !== 2) {
    throw new Error(`Invalid signature format: ${s}`);
  }

  const versionStr = parts[0].slice(5); // Remove "0din-" prefix
  const signature = parts[1];

  // Validate version
  if (!Object.values(SignatureVersion).includes(versionStr as SignatureVersion)) {
    throw new Error(`Unsupported signature version: ${versionStr}`);
  }

  // Validate hex signature
  if (!/^[0-9a-f]+$/.test(signature)) {
    throw new Error(`Invalid hex signature: ${signature}`);
  }

  return {
    version: versionStr as SignatureVersion,
    signature,
  };
}

/**
 * Compute SHA256 hash of normalized embedding.
 * 
 * This implementation matches the canonical specification:
 * 1. Quantize each value to 6 decimal places: round(x * 1e6) / 1e6
 * 2. Serialize as JSON array: [0.001234, 0.005678, ...]
 *    - Space after comma
 *    - Whole numbers must include .0 (e.g., 1.0 not 1)
 *    - Preserve sign for negative zero (-0.0)
 * 3. Hash the JSON string representation
 * 
 * The 6-decimal quantization eliminates floating-point jitter from:
 * - OpenAI API non-determinism (different servers/GPUs)
 * - Cross-platform float representation differences
 * - Numerical precision variations
 * 
 * @param normalizedEmbedding - L2-normalized embedding vector
 * @returns Hex string of SHA256 hash
 * 
 * @example
 * ```typescript
 * import { computeEmbeddingSha256 } from '@0din/sig';
 * 
 * const hash = computeEmbeddingSha256([0.1, 0.2, 0.3]);
 * console.log(hash); // SHA256 hash as hex string
 * ```
 */
export function computeEmbeddingSha256(normalizedEmbedding: number[]): string {
  // Quantize to 6 decimals
  const quantized = normalizedEmbedding.map((x) => Math.round(x * 1_000_000) / 1_000_000);

  // Format as JSON with specific rules
  const jsonParts = quantized.map((x) => {
    // Check for negative zero (preserve sign)
    if (Object.is(x, -0)) {
      return '-0.0';
    }

    let s = String(x);
    // Ensure whole numbers have .0
    if (x === Math.floor(x) && !s.includes('.')) {
      s = `${s}.0`;
    }
    return s;
  });

  const jsonStr = `[${jsonParts.join(', ')}]`;

  return crypto.createHash('sha256').update(jsonStr).digest('hex');
}
