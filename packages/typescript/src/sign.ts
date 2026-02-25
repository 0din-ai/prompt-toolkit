/**
 * High-level sign_text() convenience API.
 */

import { simhashLshMulti } from './lsh';
import { EmbeddingProvider } from './provider';
import {
  LshConfig,
  LshOutput,
  SignatureResult,
  SignatureVersion,
  getSignatureString,
} from './types';

/**
 * Generate a signature from text.
 *
 * This is the high-level convenience function that orchestrates the full pipeline:
 * 1. Auto-construct provider (if not provided) based on version
 * 2. Generate embedding using the provider
 * 3. Normalize the embedding (already done by providers)
 * 4. Compute LSH signatures
 * 5. Build a SignatureResult with metadata
 *
 * @param text - The text prompt to sign
 * @param options - Configuration options
 * @param options.version - Signature version (default: LATEST, which resolves to V1).
 *                         If provider is given, version is inferred from provider dimensions
 *                         unless explicitly specified for validation.
 * @param options.provider - Optional embedding provider. If undefined, auto-constructs the
 *                          appropriate provider based on version:
 *                          - V1: OnnxProvider (requires model cached, onnxruntime-node installed)
 *                          - V0: OpenAIProvider (requires OPENAI_API_KEY env var, openai installed)
 * @param options.config - Optional LSH configuration (defaults to 3 families, 256 bits, 16 bands)
 * @returns SignatureResult containing the signature and metadata
 * @throws Error if embedding dimensions don't match the version or if embedding generation fails
 *
 * @example Simple usage (auto-constructs V1/ONNX provider)
 * ```typescript
 * const result = await signText("How do I reset my password?");
 * console.log(getSignatureString(result));
 * // => "0din-v1:8d000000ac854dae..."
 * ```
 *
 * @example Explicit V0 (auto-constructs OpenAI provider from env)
 * ```typescript
 * const result = await signText("How do I reset my password?", {
 *   version: SignatureVersion.V0,
 * });
 * console.log(getSignatureString(result));
 * // => "0din-v0:363b24ee2b817354..."
 * ```
 *
 * @example Advanced - bring your own provider (version inferred)
 * ```typescript
 * import { ModelCache, OnnxProvider } from '@0din/sig/providers';
 *
 * const cache = new ModelCache();
 * const provider = await OnnxProvider.create(cache);
 * const result = await signText("How do I reset my password?", { provider });
 * await provider.close();
 * ```
 */
export async function signText(
  text: string,
  options?: {
    version?: SignatureVersion;
    provider?: EmbeddingProvider;
    config?: LshConfig;
  }
): Promise<SignatureResult> {
  const startTime = Date.now();

  const version = options?.version ?? SignatureVersion.LATEST;
  const config = options?.config;
  let provider = options?.provider;

  // Track if we auto-constructed the provider for cleanup
  const autoConstructed = provider === undefined;

  try {
    // Auto-construct provider if not provided
    if (provider === undefined) {
      provider = await createProviderForVersion(version);
    }

    // Infer or validate version based on provider dimensions
    const resolvedVersion = resolveVersion(version, provider);

    // Generate embedding using provider
    const embeddingResult = await provider.generateEmbedding(text);

    // Use provided config or default
    const lshConfig = config || {
      families: 3,
      bits: 256,
      bands: 16,
    };

    // Verify dimensions match expected for this version
    const expectedDims = getExpectedDimensions(resolvedVersion);
    if (embeddingResult.dimensions !== expectedDims) {
      throw new Error(
        `Embedding dimensions mismatch: expected ${expectedDims} for ${resolvedVersion}, ` +
          `got ${embeddingResult.dimensions}`
      );
    }

    // Compute LSH signatures (providers already normalize embeddings)
    const signatures = simhashLshMulti(
      embeddingResult.normalizedEmbedding,
      lshConfig
    );

    // Build result
    const elapsedMs = Date.now() - startTime;

    // Create prompt preview (first 50 chars)
    const promptPreview =
      text.length <= 50 ? text : text.substring(0, 47) + '...';

    const lshOutput: LshOutput = {
      config: lshConfig,
      signatures,
    };

    const result: SignatureResult = {
      signature: '', // Computed by getSignatureString()
      version: resolvedVersion,
      promptPreview,
      promptLength: text.length,
      provider: provider.name(),
      model: embeddingResult.model,
      dimensions: embeddingResult.dimensions,
      embeddingSha256: embeddingResult.normalizedEmbeddingSha256,
      lsh: lshOutput,
      timingMs: elapsedMs,
    };

    return result;
  } finally {
    // Clean up auto-constructed provider
    if (autoConstructed && provider) {
      await provider.close();
    }
  }
}

/**
 * Auto-construct the appropriate provider for a given version.
 *
 * @param version - Signature version (may be LATEST)
 * @returns Initialized provider instance
 * @throws Error if required dependencies are not installed or configuration is missing
 */
async function createProviderForVersion(
  version: SignatureVersion
): Promise<EmbeddingProvider> {
  const resolved = resolveVersionEnum(version);

  if (resolved === SignatureVersion.V1) {
    // V1 uses ONNX provider (local inference)
    try {
      const { ModelCache } = await import('./providers/model-cache');
      const { OnnxProvider } = await import('./providers/onnx');
      const cache = new ModelCache();
      return await OnnxProvider.create(cache);
    } catch (error: any) {
      if (error.code === 'MODULE_NOT_FOUND') {
        throw new Error(
          "V1 signatures require the ONNX provider. " +
          "Install with: npm install onnxruntime-node"
        );
      }
      throw error;
    }
  } else if (resolved === SignatureVersion.V0) {
    // V0 uses OpenAI provider (API-based)
    try {
      const { OpenAIProvider } = await import('./providers/openai');
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error(
          "OPENAI_API_KEY environment variable is required for V0 signatures. " +
          "Set it with: export OPENAI_API_KEY='sk-...'"
        );
      }
      return new OpenAIProvider({ apiKey });
    } catch (error: any) {
      if (error.code === 'MODULE_NOT_FOUND') {
        throw new Error(
          "V0 signatures require the OpenAI provider. " +
          "Install with: npm install openai"
        );
      }
      throw error;
    }
  } else {
    throw new Error(`Unsupported signature version: ${resolved}`);
  }
}

/**
 * Resolve version from provider dimensions or validate explicitly passed version.
 *
 * @param version - Explicitly passed version (may be LATEST)
 * @param provider - Provider instance
 * @returns Resolved concrete version (V0 or V1)
 * @throws Error if dimensions don't match any known version or if version conflicts with provider
 */
function resolveVersion(
  version: SignatureVersion,
  provider: EmbeddingProvider
): SignatureVersion {
  const resolvedVersion = resolveVersionEnum(version);
  const providerDims = provider.dimensions();

  // Infer version from provider dimensions
  let inferredVersion: SignatureVersion;
  if (providerDims === 1536) {
    inferredVersion = SignatureVersion.V0;
  } else if (providerDims === 384) {
    inferredVersion = SignatureVersion.V1;
  } else {
    throw new Error(
      `Cannot infer version from provider dimensions (${providerDims}). ` +
        `Expected 1536 (V0) or 384 (V1). ` +
        `Please specify version explicitly.`
    );
  }

  // If version was explicitly passed (not LATEST), validate it matches
  if (version !== SignatureVersion.LATEST && resolvedVersion !== inferredVersion) {
    throw new Error(
      `Version mismatch: requested ${resolvedVersion} ` +
        `(expects ${getExpectedDimensions(resolvedVersion)} dims) ` +
        `but provider returns ${providerDims} dims ` +
        `(matches ${inferredVersion})`
    );
  }

  return inferredVersion;
}

/**
 * Resolve LATEST to concrete version.
 */
function resolveVersionEnum(version: SignatureVersion): SignatureVersion {
  return version === SignatureVersion.LATEST ? SignatureVersion.V1 : version;
}

/**
 * Get expected embedding dimensions for a version.
 */
function getExpectedDimensions(version: SignatureVersion): number {
  switch (version) {
    case SignatureVersion.V0:
      return 1536;
    case SignatureVersion.V1:
      return 384;
    case SignatureVersion.LATEST:
      return 384;
    default:
      throw new Error(`Unknown version: ${version}`);
  }
}
