/**
 * ONNX embedding provider implementation using onnxruntime-node.
 */

import { normalizeVector } from '../lsh';
import { EmbeddingResult, computeEmbeddingSha256 } from '../types';
import { EmbeddingProvider } from '../provider';
import { ModelCache } from './model-cache';

/**
 * Embedding provider using local ONNX model inference.
 *
 * This provider uses the intfloat/multilingual-e5-small model by default,
 * which produces 384-dimensional embeddings suitable for multilingual text similarity.
 *
 * The model is automatically loaded from the local cache directory.
 *
 * @example
 * ```typescript
 * const cache = new ModelCache();
 * const provider = await OnnxProvider.create(cache);
 * const result = await provider.generateEmbedding("Hello, world!");
 * console.log(`Generated ${result.dimensions}-dimensional embedding`);
 * await provider.close();
 * ```
 */
export class OnnxProvider implements EmbeddingProvider {
  private static readonly DEFAULT_MODEL = 'intfloat/multilingual-e5-small';
  private static readonly DEFAULT_DIMENSIONS = 384;
  private static readonly MAX_SEQUENCE_LENGTH = 512;

  private session: any; // InferenceSession type
  private tokenizer: any; // Tokenizer type
  private providerName: string;
  private modelName: string;
  private dims: number;
  private inputPrefix: string;

  /**
   * Private constructor - use OnnxProvider.create() instead.
   */
  private constructor(
    session: any,
    tokenizer: any,
    modelName: string,
    dimensions: number,
    name: string,
    inputPrefix: string
  ) {
    this.session = session;
    this.tokenizer = tokenizer;
    this.modelName = modelName;
    this.dims = dimensions;
    this.providerName = name;
    this.inputPrefix = inputPrefix;
  }

  /**
   * Create a new ONNX provider instance.
   *
   * Note: Requires 'onnxruntime-node' package to be installed:
   * ```bash
   * npm install onnxruntime-node
   * ```
   *
   * @param cache - ModelCache instance for managing model files
   * @param model - Model name or path (default: intfloat/multilingual-e5-small)
   * @param name - Provider name (default: "onnx")
   * @returns Initialized OnnxProvider
   * @throws Error if model files are not found or if onnxruntime-node is not installed
   */
  static async create(
    cache: ModelCache,
    model?: string,
    name?: string
  ): Promise<OnnxProvider> {
    const modelName = model || OnnxProvider.DEFAULT_MODEL;
    const providerName = name || 'onnx';

    // Check if model is cached
    if (!cache.hasModel('v1')) {
      throw new Error(
        `Model not found in cache at ${cache.modelDirectory('v1')}. ` +
          'Please ensure the model files are present.'
      );
    }

    // Dynamically import onnxruntime-node
    let ort: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      ort = require('onnxruntime-node');
    } catch (error) {
      throw new Error(
        "ONNX provider requires the 'onnxruntime-node' package. " +
          'Install with: npm install onnxruntime-node'
      );
    }

    // Load ONNX model
    const modelPath = cache.getModelPath('v1');
    const session = await ort.InferenceSession.create(modelPath);

    // Load tokenizer - use the built-in tokenizer loading
    const tokenizerPath = cache.getTokenizerPath('v1');
    const fs = require('fs');
    const tokenizerData = JSON.parse(fs.readFileSync(tokenizerPath, 'utf-8'));

    // Create a simple tokenizer wrapper
    const tokenizer = new Tokenizer(tokenizerData);

    // Use default dimensions (model output shape not needed here)
    const dimensions = OnnxProvider.DEFAULT_DIMENSIONS;

    // Load input prefix from config
    const config = cache.loadConfig('v1');
    const inputPrefix = config.inference?.input_prefix || '';

    return new OnnxProvider(
      session,
      tokenizer,
      modelName,
      dimensions,
      providerName,
      inputPrefix
    );
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

    // Prepend input prefix if configured
    const prefixedText = this.inputPrefix ? `${this.inputPrefix}${text}` : text;

    // Tokenize input
    const tokens = this.tokenizer.encode(prefixedText, OnnxProvider.MAX_SEQUENCE_LENGTH);

    // Prepare ONNX inputs
    const inputIds = new BigInt64Array(tokens.inputIds);
    const attentionMask = new BigInt64Array(tokens.attentionMask);
    const tokenTypeIds = new BigInt64Array(tokens.inputIds.length).fill(0n);

    // Create tensors
    const ort = require('onnxruntime-node');
    const inputIdsTensor = new ort.Tensor('int64', inputIds, [1, tokens.inputIds.length]);
    const attentionMaskTensor = new ort.Tensor('int64', attentionMask, [1, tokens.attentionMask.length]);
    const tokenTypeIdsTensor = new ort.Tensor('int64', tokenTypeIds, [1, tokens.inputIds.length]);

    // Run inference
    const feeds = {
      input_ids: inputIdsTensor,
      attention_mask: attentionMaskTensor,
      token_type_ids: tokenTypeIdsTensor,
    };

    const results = await this.session.run(feeds);
    const lastHiddenState = results[Object.keys(results)[0]].data;

    // Mean pooling with attention mask
    const seqLen = tokens.inputIds.length;
    const hiddenSize = this.dims;
    const embedding = this.meanPool(lastHiddenState, tokens.attentionMask, seqLen, hiddenSize);

    const elapsedMs = Date.now() - startTime;

    // Normalize and compute SHA256
    const normalized = normalizeVector(embedding);
    const sha256 = computeEmbeddingSha256(normalized);

    // Count tokens
    const tokenCount = tokens.attentionMask.reduce((sum: number, val: number) => sum + val, 0);

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

  /**
   * Apply mean pooling to token embeddings.
   */
  private meanPool(
    hiddenStates: Float32Array | number[],
    attentionMask: number[],
    seqLen: number,
    hiddenSize: number
  ): number[] {
    const pooled = new Array(hiddenSize).fill(0);
    let maskSum = 0;

    // Sum embeddings weighted by attention mask
    for (let i = 0; i < seqLen; i++) {
      const maskVal = attentionMask[i];
      if (maskVal === 1) {
        maskSum += maskVal;
        for (let j = 0; j < hiddenSize; j++) {
          pooled[j] += hiddenStates[i * hiddenSize + j] * maskVal;
        }
      }
    }

    // Average
    if (maskSum > 0) {
      for (let j = 0; j < hiddenSize; j++) {
        pooled[j] /= maskSum;
      }
    }

    return pooled;
  }

  async close(): Promise<void> {
    // ONNX session doesn't require explicit cleanup in Node.js
  }
}

/**
 * Simple tokenizer implementation for XLM-RoBERTa.
 */
class Tokenizer {
  private vocab: Map<string, number>;
  private merges: Map<string, number>;
  private addedTokens: Map<string, number>;
  private bosToken: number;
  private eosToken: number;
  private padToken: number;
  private unkToken: number;

  constructor(config: any) {
    // Load vocabulary
    this.vocab = new Map();
    if (config.model && config.model.vocab) {
      for (const [token, id] of Object.entries(config.model.vocab)) {
        this.vocab.set(token, id as number);
      }
    }

    // Load merges
    this.merges = new Map();
    if (config.model && config.model.merges) {
      config.model.merges.forEach((merge: string, idx: number) => {
        this.merges.set(merge, idx);
      });
    }

    // Load added tokens
    this.addedTokens = new Map();
    if (config.added_tokens) {
      config.added_tokens.forEach((token: any) => {
        this.addedTokens.set(token.content, token.id);
      });
    }

    // Special tokens
    this.bosToken = this.vocab.get('<s>') || 0;
    this.eosToken = this.vocab.get('</s>') || 2;
    this.padToken = this.vocab.get('<pad>') || 1;
    this.unkToken = this.vocab.get('<unk>') || 3;
  }

  /**
   * Encode text to token IDs with attention mask.
   */
  encode(text: string, maxLength: number): { inputIds: number[]; attentionMask: number[] } {
    // Simple whitespace tokenization (fallback)
    // In production, this should use proper BPE tokenization
    const tokens = text.toLowerCase().split(/\s+/).filter(t => t.length > 0);
    
    // Map to IDs
    const ids = [this.bosToken];
    for (const token of tokens) {
      const id = this.vocab.get(token) || this.unkToken;
      ids.push(id);
      if (ids.length >= maxLength - 1) break;
    }
    ids.push(this.eosToken);

    // Pad to max length
    const attentionMask = new Array(ids.length).fill(1);
    while (ids.length < maxLength) {
      ids.push(this.padToken);
      attentionMask.push(0);
    }

    return {
      inputIds: ids.slice(0, maxLength),
      attentionMask: attentionMask.slice(0, maxLength),
    };
  }
}
