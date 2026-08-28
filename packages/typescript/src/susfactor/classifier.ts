/**
 * SusFactor classifier using a local ONNX model (onnxruntime-node).
 *
 * The ONNX graph (exported via scripts/export_susfactor_onnx.py) bakes the
 * e5-large encoder, mean-pooling, and MLP head into a single model:
 *
 *   inputs:  input_ids[1, seq] int64, attention_mask[1, seq] int64
 *   output:  logits[1, 2] float32   (softmax[1] = P(suspicious))
 *
 * The model is not bundled with the SDK; download it into the cache directory
 * (see ModelCache) before use.
 *
 * @module susfactor
 */

import * as path from "path";

import { ModelCache } from "../providers/model-cache";
import type { GetModelOptions } from "../providers/model-cache";
import { SusFactorError } from "../error";
import {
  CHUNK_STRIDE,
  ChunkedSusFactorResult,
  LABEL_SAFE,
  LABEL_SUSPICIOUS,
  MAX_CONTENT_TOKENS,
  PhaseSpan,
  SusFactorLabel,
  SusFactorResult,
} from "./types";

/**
 * Split a token-ID sequence into overlapping chunks of at most
 * {@link MAX_CONTENT_TOKENS} tokens each.
 *
 * - Sequences at or below MAX_CONTENT_TOKENS produce exactly one chunk.
 * - Adjacent chunks share CHUNK_OVERLAP tokens of context.
 * - An empty input produces one empty chunk.
 */
export function chunkTokenIds(ids: ArrayLike<bigint>): bigint[][] {
  const arr = Array.from(ids);
  if (arr.length <= MAX_CONTENT_TOKENS) {
    return [arr];
  }
  const chunks: bigint[][] = [];
  let start = 0;
  while (start < arr.length) {
    const end = Math.min(start + MAX_CONTENT_TOKENS, arr.length);
    chunks.push(arr.slice(start, end));
    if (end === arr.length) break;
    start += CHUNK_STRIDE;
  }
  return chunks;
}

/** Canonical model identifier reported in results (shared across SDKs). */
export const DEFAULT_MODEL = "0dinai/susfactor-e5-large";
/** HuggingFace repo holding the ONNX export the TS/Rust runtimes download. */
export const DEFAULT_ONNX_REPO = "0dinai/susfactor-e5-large-onnx";
export const DEFAULT_THRESHOLD = 0.5;
export const MAX_SEQUENCE_LENGTH = 512;
export const MODEL_VERSION = "susfactor-v1";

/**
 * Softmax over a 2-logit vector, returning P(class 1) = suspicious.
 */
export function softmaxSuspicious(logits: ArrayLike<number>): number {
  const a = logits[0];
  const b = logits[1];
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  return eb / (ea + eb);
}

/**
 * Map a suspicious probability to a label using the threshold.
 */
export function labelForScore(
  score: number,
  threshold: number,
): SusFactorLabel {
  return score >= threshold ? LABEL_SUSPICIOUS : LABEL_SAFE;
}

/**
 * Classifies prompts as safe vs. suspicious using SusFactor.
 *
 * Use {@link SusFactorClassifier.create} to load from the model cache. The
 * constructor takes the session/tokenizer directly (useful for testing).
 */
export class SusFactorClassifier {
  private session: any;
  private tokenizer: any;
  private modelName: string;
  private decisionThreshold: number;

  constructor(
    session: any,
    tokenizer: any,
    modelName: string,
    threshold: number = DEFAULT_THRESHOLD,
  ) {
    this.session = session;
    this.tokenizer = tokenizer;
    this.modelName = modelName;
    this.decisionThreshold = threshold;
  }

  /**
   * Load the SusFactor classifier from a local model cache.
   *
   * @param cache - ModelCache instance for locating model files.
   * @param options - model name and decision threshold.
   * @throws SusFactorError if model files are missing or deps unavailable.
   */
  static async create(
    cache: ModelCache,
    options: {
      model?: string;
      threshold?: number;
      /** HuggingFace token for downloading the gated susfactor model. */
      hfToken?: string;
      /** Base URL override for tests (e.g. local mock server). */
      baseUrl?: string;
      /** Progress callback forwarded to {@link ModelCache.downloadModel}. */
      onProgress?: GetModelOptions["onProgress"];
    } = {},
  ): Promise<SusFactorClassifier> {
    const modelName = options.model || DEFAULT_MODEL;
    const threshold = options.threshold ?? DEFAULT_THRESHOLD;

    // Auto-download model files if not already cached.
    // The susfactor model is gated on HuggingFace — a token is required.
    await cache.downloadModel(MODEL_VERSION, {
      hfToken: options.hfToken,
      baseUrl: options.baseUrl,
      onProgress: options.onProgress,
    });

    let ort: any;
    try {
      ort = require("onnxruntime-node");
    } catch (error) {
      throw new SusFactorError(
        "SusFactor requires the 'onnxruntime-node' package. " +
          "Install with: npm install onnxruntime-node",
        { cause: error },
      );
    }

    let AutoTokenizer: any;
    let hfEnv: any;
    try {
      const hf = require("@huggingface/transformers");
      AutoTokenizer = hf.AutoTokenizer;
      hfEnv = hf.env;
    } catch (error) {
      throw new SusFactorError(
        "SusFactor requires the '@huggingface/transformers' package. " +
          "Install with: npm install @huggingface/transformers",
        { cause: error },
      );
    }

    // Always load model.onnx — the graph validated in production (Heimdall via
    // the Rust SDK). model_O4.onnx is a pre-optimized variant that has never
    // been validated against the reference; using it here would produce a
    // different inference path from what Rust runs, making cross-SDK score
    // comparison undefined. If model_O4.onnx is ever separately validated, this
    // can be revisited and the golden vectors must be regenerated from that path.
    const modelDir = cache.modelDirectory(MODEL_VERSION);
    const modelPath = path.join(modelDir, "onnx", "model.onnx");
    const session = await ort.InferenceSession.create(modelPath);

    const parentDir = path.dirname(modelDir);
    const versionName = path.basename(modelDir);
    hfEnv.localModelPath = parentDir + path.sep;
    hfEnv.allowRemoteModels = false;
    const tokenizer = await AutoTokenizer.from_pretrained(versionName, {
      local_files_only: true,
    });

    return new SusFactorClassifier(session, tokenizer, modelName, threshold);
  }

  model(): string {
    return this.modelName;
  }

  threshold(): number {
    return this.decisionThreshold;
  }

  /**
   * Classify a prompt of any length.
   *
   * Prompts within {@link MAX_CONTENT_TOKENS} (510 tokens) are scored in a
   * single inference call. Longer prompts are automatically split into
   * overlapping chunks, each scored independently — callers do not need to
   * check length or call a separate method.
   *
   * Chunks are dispatched concurrently via `Promise.all`. Actual concurrency
   * depends on the ONNX Runtime session configuration; a single shared session
   * serializes inference internally. Dispatching concurrently allows the
   * runtime to schedule work efficiently.
   *
   * Each chunk is scored independently; no scores are aggregated.
   * `isSuspicious` is `true` if **any** chunk is suspicious.
   * Short prompts produce exactly one chunk.
   */
  async classify(text: string): Promise<ChunkedSusFactorResult> {
    const wallStart = Date.now();
    const offset = (t: number): number => t - wallStart;

    // Tokenize the full text without truncation.
    const tokenizeStart = Date.now();
    const encoded = this.tokenizer(text, {
      padding: false,
      truncation: false,
    });
    const allIds: bigint[] = Array.from(
      encoded.input_ids.data as BigInt64Array,
    );
    const allMask: bigint[] = Array.from(
      encoded.attention_mask.data as BigInt64Array,
    );
    const tokenizeSpan: PhaseSpan = {
      name: "tokenize",
      startMs: offset(tokenizeStart),
      durationMs: Date.now() - tokenizeStart,
    };

    const chunkPhaseStart = Date.now();
    const idChunks = chunkTokenIds(allIds);
    const chunkSpan: PhaseSpan = {
      name: "chunk",
      startMs: offset(chunkPhaseStart),
      durationMs: Date.now() - chunkPhaseStart,
    };

    const ort = require("onnxruntime-node");

    const scoreChunk = async (
      chunkIds: bigint[],
      index: number,
    ): Promise<{ result: SusFactorResult; span: PhaseSpan }> => {
      const chunkStart = Date.now();
      const chunkLen = chunkIds.length;
      const chunkMask = allMask.slice(0, chunkLen);

      const inputIdsTensor = new ort.Tensor(
        "int64",
        new BigInt64Array(chunkIds),
        [1, chunkLen],
      );
      const attentionMaskTensor = new ort.Tensor(
        "int64",
        new BigInt64Array(chunkMask),
        [1, chunkLen],
      );

      const results = await this.session.run({
        input_ids: inputIdsTensor,
        attention_mask: attentionMaskTensor,
      });

      const logitsKey =
        "logits" in results ? "logits" : Object.keys(results)[0];
      const logits = results[logitsKey].data as Float32Array;
      const score = softmaxSuspicious(logits);
      const label = labelForScore(score, this.decisionThreshold);

      const result: SusFactorResult = {
        score,
        label,
        isSuspicious: label === LABEL_SUSPICIOUS,
        model: this.modelName,
        threshold: this.decisionThreshold,
        timingMs: Date.now() - chunkStart,
      };
      const span: PhaseSpan = {
        name: "inference",
        startMs: offset(chunkStart),
        durationMs: result.timingMs,
        chunkIndex: index,
        tokenCount: chunkLen,
      };
      return { result, span };
    };

    // Run all chunks in parallel; Promise.all preserves input order.
    const scored = await Promise.all(
      idChunks.map((chunkIds, index) => scoreChunk(chunkIds, index)),
    );
    const chunkResults = scored.map((s) => s.result);
    const inferenceSpans = scored.map((s) => s.span);

    const reduceStart = Date.now();
    const isSuspicious = chunkResults.some((r) => r.isSuspicious);
    const totalTimingMs = Date.now() - wallStart;
    const reduceSpan: PhaseSpan = {
      name: "reduce",
      startMs: offset(reduceStart),
      durationMs: Date.now() - reduceStart,
    };

    return {
      chunks: chunkResults,
      isSuspicious,
      totalTimingMs,
      spans: [tokenizeSpan, chunkSpan, ...inferenceSpans, reduceSpan],
      totalTokens: allIds.length,
    };
  }

  async close(): Promise<void> {
    // ONNX session doesn't require explicit cleanup in Node.js
  }
}
