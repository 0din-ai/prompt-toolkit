/**
 * Type definitions for SusFactor classification.
 *
 * @module susfactor
 */

export const LABEL_SUSPICIOUS = "suspicious";
export const LABEL_SAFE = "safe";

export type SusFactorLabel = typeof LABEL_SUSPICIOUS | typeof LABEL_SAFE;

/**
 * Result of a SusFactor classification.
 */
export interface SusFactorResult {
  /** Probability that the prompt is suspicious/malicious, in [0, 1]. */
  score: number;
  /** "suspicious" if score >= threshold, else "safe". */
  label: SusFactorLabel;
  /** Whether the prompt was classified as suspicious. */
  isSuspicious: boolean;
  /** Identifier of the model that produced the score. */
  model: string;
  /** Decision threshold used to derive the label. */
  threshold: number;
  /** Inference time in milliseconds. */
  timingMs: number;
}
