/**
 * @0din/prompt-toolkit - Multi-language SDK for LSH signature generation
 * 
 * This package provides locality-sensitive hashing (LSH) for AI prompt similarity
 * detection, with support for both standard LSH and Confidence Matrix LSH (CM-LSH).
 * 
 * @packageDocumentation
 * 
 * @example Quick Start with signText()
 * ```typescript
 * import { signText, SignatureVersion, getSignatureString } from '@0din/prompt-toolkit';
 * import { ModelCache, OnnxProvider } from '@0din/prompt-toolkit/providers';
 * 
 * const cache = new ModelCache();
 * const provider = await OnnxProvider.create(cache);
 * 
 * const result = await signText(
 *   "How do I reset my password?",
 *   provider,
 *   SignatureVersion.V1
 * );
 * 
 * console.log(getSignatureString(result));
 * // => "0din-v1:8d000000ac854dae..."
 * 
 * await provider.close();
 * ```
 * 
 * @example Low-level LSH API
 * ```typescript
 * import { simhashLshMulti, normalizeVector } from '@0din/prompt-toolkit';
 * 
 * const vector = [0.5, 0.5, 0.5, 0.5];
 * const normalized = normalizeVector(vector);
 * const families = simhashLshMulti(normalized);
 * 
 * console.log(families[0].signature);
 * ```
 * 
 * @example Similarity Comparison
 * ```typescript
 * import { 
 *   simhashLshMulti, 
 *   hammingDistanceHex, 
 *   cosineFromHamming 
 * } from '@0din/prompt-toolkit';
 * 
 * const families1 = simhashLshMulti(vector1);
 * const families2 = simhashLshMulti(vector2);
 * 
 * const distance = hammingDistanceHex(
 *   families1[0].signature, 
 *   families2[0].signature
 * );
 * const similarity = cosineFromHamming(distance, 256);
 * 
 * console.log(`Estimated cosine similarity: ${similarity.toFixed(3)}`);
 * ```
 */

// High-level convenience API (recommended)
export { signText } from './sign';

// Provider interface (for custom implementations)
export { type EmbeddingProvider } from './provider';

// Hasher abstraction
export { type Hasher } from './hasher';
export { getHasher, SimHashLsh } from './hashers';

// Error types
export {
  SigError,
  ConfigError,
  ProviderError,
  ModelError,
  InvalidInputError,
  ThreatFeedApiError,
  ThreatFeedCacheError,
  SusFactorError,
} from './error';

// Core LSH functions
export {
  simhashLshMulti,
  hammingDistanceHex,
  cosineFromHamming,
  normalizeVector,
  type LSHFamily,
  type LshConfig,
} from './lsh';

// Types and utilities
export {
  SignatureVersion,
  HashAlgorithm,
  type ParsedSignature,
  type EmbeddingResult,
  type LshOutput,
  type SignatureResult,
  type ComparisonResult,
  type PromptInfo,
  type QualityStats,
  resolveVersion,
  embeddingDimensions,
  versionToAlgorithm,
  algorithmToVersion,
  signatureString,
  parseSignatureString,
  computeEmbeddingSha256,
  getSignatureString,
} from './types';

// CM-LSH (Confidence Matrix LSH)
export {
  HybridCMLSH,
  createDefaultCmLsh,
  genHyperplanes,
  lshTsCompat,
  type DualHash,
  type ITQParams,
  type HybridParams,
  type CalibratorConfig,
} from './cm-lsh';

// Threat feed integration
export {
  ThreatFeedCache,
  ThreatFeedClient,
  compareToThreatfeed,
  parseThreatFeedResponse,
  type CachedSignature,
  type DetectionSignature,
  type SyncResult,
  type ThreatFeedEntry,
  type ThreatFeedResponse,
  type ThreatMatch,
} from './threatfeed';

// Embedding providers
export {
  ModelCache,
  OnnxProvider,
  OpenAIProvider,
} from './providers';

// SusFactor jailbreak/prompt-injection classifier
// Constants are aliased with SUSFACTOR_ prefix to avoid collisions with the
// generic model/version names already present at this scope.
export {
  susFactor,
  SusFactorClassifier,
  softmaxSuspicious,
  labelForScore,
  LABEL_SAFE,
  LABEL_SUSPICIOUS,
  DEFAULT_MODEL as SUSFACTOR_DEFAULT_MODEL,
  DEFAULT_ONNX_REPO as SUSFACTOR_DEFAULT_ONNX_REPO,
  DEFAULT_THRESHOLD as SUSFACTOR_DEFAULT_THRESHOLD,
  MODEL_VERSION as SUSFACTOR_MODEL_VERSION,
  type SusFactorOptions,
  type SusFactorLabel,
  type SusFactorResult,
} from './susfactor';
