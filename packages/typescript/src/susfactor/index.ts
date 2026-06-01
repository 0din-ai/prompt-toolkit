/**
 * SusFactor jailbreak/prompt-injection classifier integration.
 *
 * SusFactor classifies a prompt as "safe" (score near 0) or "suspicious"
 * (score near 1). It is a separate capability from the LSH signature pipeline.
 *
 * The model runs via onnxruntime-node from a local ONNX export of
 * 0dinai/susfactor-e5-large. Download it into the cache directory before use.
 *
 * @example
 * ```typescript
 * import { susFactor } from '@0din/prompt-toolkit/susfactor';
 * const result = await susFactor('Ignore all previous instructions');
 * console.log(result.score, result.label);
 * ```
 *
 * @module susfactor
 */

export {
  SusFactorClassifier,
  softmaxSuspicious,
  labelForScore,
  DEFAULT_MODEL,
  DEFAULT_ONNX_REPO,
  DEFAULT_THRESHOLD,
  MODEL_VERSION,
} from "./classifier";
export { susFactor } from "./compare";
export type { SusFactorOptions } from "./compare";
export { LABEL_SAFE, LABEL_SUSPICIOUS } from "./types";
export type { SusFactorLabel, SusFactorResult } from "./types";
