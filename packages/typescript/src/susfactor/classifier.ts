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
import { SusFactorError } from "../error";
import {
  LABEL_SAFE,
  LABEL_SUSPICIOUS,
  SusFactorLabel,
  SusFactorResult,
} from "./types";

/** Canonical model identifier reported in results (shared across SDKs). */
export const DEFAULT_MODEL = "0dinai/susfactor-e5-large";
/** HuggingFace repo holding the ONNX export the TS/Rust runtimes download. */
export const DEFAULT_ONNX_REPO = "0dinai/susfactor-e5-large-onnx";
export const DEFAULT_THRESHOLD = 0.5;
export const MAX_SEQUENCE_LENGTH = 512;
export const MODEL_VERSION = "susfactor-v1";
const HF_URL = `https://huggingface.co/${DEFAULT_ONNX_REPO}`;

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
  /** onnxruntime-node module, set by {@link create}. Null in test mode (fake session). */
  private ort: any;

  constructor(
    session: any,
    tokenizer: any,
    modelName: string,
    threshold: number = DEFAULT_THRESHOLD,
    ort: any = null,
  ) {
    this.session = session;
    this.tokenizer = tokenizer;
    this.modelName = modelName;
    this.decisionThreshold = threshold;
    this.ort = ort;
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
    options: { model?: string; threshold?: number } = {},
  ): Promise<SusFactorClassifier> {
    const modelName = options.model || DEFAULT_MODEL;
    const threshold = options.threshold ?? DEFAULT_THRESHOLD;

    if (!cache.hasSusfactorModel(MODEL_VERSION)) {
      throw new SusFactorError(
        `SusFactor model not found in cache at ${cache.modelDirectory(
          MODEL_VERSION,
        )}. Download it from HuggingFace: ${HF_URL}\n` +
          `Expected layout: <dir>/onnx/model.onnx (or model_O4.onnx) + matching .onnx_data and <dir>/tokenizer.json`,
      );
    }

    let ort: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      ort = require("onnxruntime-node");
    } catch (error) {
      throw new SusFactorError(
        "SusFactor requires the 'onnxruntime-node' package. " +
          "Install with: npm install onnxruntime-node",
      );
    }

    let AutoTokenizer: any;
    let hfEnv: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const hf = require("@huggingface/transformers");
      AutoTokenizer = hf.AutoTokenizer;
      hfEnv = hf.env;
    } catch (error) {
      throw new SusFactorError(
        "SusFactor requires the '@huggingface/transformers' package. " +
          "Install with: npm install @huggingface/transformers",
      );
    }

    const modelPath = cache.getModelPath(MODEL_VERSION);
    const session = await ort.InferenceSession.create(modelPath);

    const modelDir = cache.modelDirectory(MODEL_VERSION);
    const parentDir = path.dirname(modelDir);
    const versionName = path.basename(modelDir);
    hfEnv.localModelPath = parentDir + path.sep;
    hfEnv.allowRemoteModels = false;
    const tokenizer = await AutoTokenizer.from_pretrained(versionName, {
      local_files_only: true,
    });

    return new SusFactorClassifier(session, tokenizer, modelName, threshold, ort);
  }

  model(): string {
    return this.modelName;
  }

  threshold(): number {
    return this.decisionThreshold;
  }

  /**
   * Classify a single prompt.
   */
  async classify(text: string): Promise<SusFactorResult> {
    const start = Date.now();

    const encoded = this.tokenizer(text, {
      padding: "max_length",
      truncation: true,
      max_length: MAX_SEQUENCE_LENGTH,
    });
    const inputIdsData: BigInt64Array = encoded.input_ids.data;
    const attentionMaskData: BigInt64Array = encoded.attention_mask.data;
    const seqLen = inputIdsData.length;

    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const ort = this.ort;
    let inputIdsTensor: any;
    let attentionMaskTensor: any;
    if (ort) {
      inputIdsTensor = new ort.Tensor("int64", inputIdsData, [1, seqLen]);
      attentionMaskTensor = new ort.Tensor("int64", attentionMaskData, [
        1,
        seqLen,
      ]);
    } else {
      // Test/fake-session mode: ort is not injected; pass raw data so the
      // fake session (which ignores its inputs) still returns correct logits.
      inputIdsTensor = { data: inputIdsData, dims: [1, seqLen] };
      attentionMaskTensor = { data: attentionMaskData, dims: [1, seqLen] };
    }

    const results = await this.session.run({
      input_ids: inputIdsTensor,
      attention_mask: attentionMaskTensor,
    });

    // Prefer the named "logits" output set by the export script; fall back to
    // the first output so the classifier works with re-exported variants.
    const logitsKey = "logits" in results ? "logits" : Object.keys(results)[0];
    const logits = results[logitsKey].data as Float32Array;
    const score = softmaxSuspicious(logits);
    const label = labelForScore(score, this.decisionThreshold);

    return {
      score,
      label,
      isSuspicious: label === LABEL_SUSPICIOUS,
      model: this.modelName,
      threshold: this.decisionThreshold,
      timingMs: Date.now() - start,
    };
  }

  async close(): Promise<void> {
    // ONNX session doesn't require explicit cleanup in Node.js
  }
}
