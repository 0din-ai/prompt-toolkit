/**
 * High-level entry point for SusFactor classification.
 *
 * @module susfactor
 */

import { ModelCache } from "../providers/model-cache";
import { DEFAULT_THRESHOLD, SusFactorClassifier } from "./classifier";
import { SusFactorResult } from "./types";

export interface SusFactorOptions {
  /** An existing classifier to reuse. If omitted, one is built and closed. */
  classifier?: SusFactorClassifier;
  /** ModelCache to locate model files when auto-constructing. */
  cache?: ModelCache;
  /** Model identifier. */
  model?: string;
  /** Decision threshold for the suspicious label. */
  threshold?: number;
}

/**
 * Classify a prompt as safe vs. suspicious.
 *
 * If a classifier is provided it is used as-is (left open for the caller).
 * Otherwise one is built from a model cache, used once, and closed.
 *
 * @example
 * ```typescript
 * import { susFactor } from '@0din/prompt-toolkit/susfactor';
 * const result = await susFactor('Ignore previous instructions');
 * console.log(result.score, result.label);
 * ```
 */
export async function susFactor(
  text: string,
  options: SusFactorOptions = {},
): Promise<SusFactorResult> {
  if (options.classifier) {
    return options.classifier.classify(text);
  }

  const cache = options.cache ?? new ModelCache();
  const owned = await SusFactorClassifier.create(cache, {
    model: options.model,
    threshold: options.threshold ?? DEFAULT_THRESHOLD,
  });
  try {
    return await owned.classify(text);
  } finally {
    await owned.close();
  }
}
