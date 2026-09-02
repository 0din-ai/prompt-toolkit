/**
 * ONNX embedding provider implementation using onnxruntime-node.
 */

import * as path from 'path';

import { normalizeVector } from '../lsh';
import { EmbeddingResult, computeEmbeddingSha256 } from '../types';
import { EmbeddingProvider } from '../provider';
import { ModelCache } from './model-cache';

/**
 * Embedding provider using local ONNX model inference.
 *
 * This provider uses the 0dinai/0din-jailbreak-embeddings-large model by default,
 * which produces 1024-dimensional embeddings suitable for multilingual text similarity.
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
  private static readonly DEFAULT_MODEL = 'intfloat/multilingual-e5-large';
  private static readonly DEFAULT_DIMENSIONS = 1024;
  private static readonly MAX_SEQUENCE_LENGTH = 512;

  private session: any; // InferenceSession type
  private tokenizer: any; // AutoTokenizer instance
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
   * Requires both 'onnxruntime-node' and '@huggingface/transformers' packages:
   * ```bash
   * npm install onnxruntime-node @huggingface/transformers
   * ```
   *
   * @param cache - ModelCache instance for managing model files
   * @param model - Model name (default: intfloat/multilingual-e5-large)
   * @param name - Provider name (default: "onnx")
   * @returns Initialized OnnxProvider
   * @throws Error if model files are not found or required packages are not installed
   */
  static async create(
    cache: ModelCache,
    model?: string,
    name?: string
  ): Promise<OnnxProvider> {
    const modelName = model || OnnxProvider.DEFAULT_MODEL;
    const providerName = name || 'onnx';

    // Auto-download model files if not already cached.
    await cache.downloadModel('v1');

    // Dynamically import onnxruntime-node
    let ort: any;
    try {
      ort = require('onnxruntime-node');
    } catch (error) {
      throw new Error(
        "ONNX provider requires the 'onnxruntime-node' package. " +
          'Install with: npm install onnxruntime-node',
        { cause: error }
      );
    }

    // Load ONNX model
    const modelPath = cache.getModelPath('v1');
    const session = await ort.InferenceSession.create(modelPath);

    // Load tokenizer using @huggingface/transformers AutoTokenizer.
    // This provides proper SentencePiece/Unigram tokenization via tokenizer.json,
    // matching the Python (transformers.AutoTokenizer) and Rust (tokenizers crate)
    // implementations exactly.
    let AutoTokenizer: any;
    let hfEnv: any;
    try {
      const hf = require('@huggingface/transformers');
      AutoTokenizer = hf.AutoTokenizer;
      hfEnv = hf.env;
    } catch (error) {
      throw new Error(
        "ONNX provider requires the '@huggingface/transformers' package. " +
          'Install with: npm install @huggingface/transformers',
        { cause: error }
      );
    }

    // Configure @huggingface/transformers to load from the local model directory.
    // localModelPath is the base directory; we pass the version folder name as the
    // model identifier to from_pretrained(), so it looks for:
    //   {localModelPath}/{version}/tokenizer.json
    const modelDir = cache.modelDirectory('v1');
    const parentDir = path.dirname(modelDir);
    const versionName = path.basename(modelDir);

    hfEnv.localModelPath = parentDir + path.sep;
    hfEnv.allowRemoteModels = false;

    const tokenizer = await AutoTokenizer.from_pretrained(versionName, {
      local_files_only: true,
    });

    // Load input prefix from config (empty string for the 0din fine-tuned model)
    const config = cache.loadConfig('v1');
    const inputPrefix = config.inference?.input_prefix || '';

    return new OnnxProvider(
      session,
      tokenizer,
      modelName,
      OnnxProvider.DEFAULT_DIMENSIONS,
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

    // Tokenize using @huggingface/transformers AutoTokenizer.
    // This matches Python's: tokenizer(text, padding='max_length', truncation=True,
    //                                   max_length=512, return_tensors='np')
    const encoded = this.tokenizer(prefixedText, {
      padding: 'max_length',
      truncation: true,
      max_length: OnnxProvider.MAX_SEQUENCE_LENGTH,
    });

    // encoded.input_ids and encoded.attention_mask are Tensor objects with .data (BigInt64Array)
    const inputIdsData: BigInt64Array = encoded.input_ids.data;
    const attentionMaskData: BigInt64Array = encoded.attention_mask.data;

    // token_type_ids: all zeros (XLM-RoBERTa segment IDs are always 0)
    const tokenTypeIds = new BigInt64Array(inputIdsData.length).fill(0n);

    const seqLen = inputIdsData.length;

    // Create ONNX tensors
    const ort = require('onnxruntime-node');
    const inputIdsTensor = new ort.Tensor('int64', inputIdsData, [1, seqLen]);
    const attentionMaskTensor = new ort.Tensor('int64', attentionMaskData, [1, seqLen]);
    const tokenTypeIdsTensor = new ort.Tensor('int64', tokenTypeIds, [1, seqLen]);

    // Run inference
    const results = await this.session.run({
      input_ids: inputIdsTensor,
      attention_mask: attentionMaskTensor,
      token_type_ids: tokenTypeIdsTensor,
    });

    const lastHiddenState = results[Object.keys(results)[0]].data as Float32Array;

    // Mean pooling with attention mask (matches Python/Rust implementation)
    const attentionMaskNumbers = Array.from(attentionMaskData, (v) => Number(v));
    const embedding = this.meanPool(lastHiddenState, attentionMaskNumbers, seqLen, this.dims);

    const elapsedMs = Date.now() - startTime;

    // L2-normalize and compute SHA256
    const normalized = normalizeVector(embedding);
    const sha256 = computeEmbeddingSha256(normalized);

    // Count non-padding tokens (attention_mask sum)
    const tokenCount = attentionMaskNumbers.reduce((sum, val) => sum + val, 0);

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
   * Apply mean pooling to token embeddings weighted by attention mask.
   *
   * Matches the Python implementation:
   *   embeddings = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1)
   */
  private meanPool(
    hiddenStates: Float32Array,
    attentionMask: number[],
    seqLen: number,
    hiddenSize: number
  ): number[] {
    const pooled = new Array(hiddenSize).fill(0);
    let maskSum = 0;

    for (let i = 0; i < seqLen; i++) {
      const maskVal = attentionMask[i];
      if (maskVal === 1) {
        maskSum += maskVal;
        for (let j = 0; j < hiddenSize; j++) {
          pooled[j] += hiddenStates[i * hiddenSize + j] * maskVal;
        }
      }
    }

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
