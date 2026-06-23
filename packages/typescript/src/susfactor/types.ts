/**
 * Type definitions for SusFactor classification.
 *
 * @module susfactor
 */

export const LABEL_SUSPICIOUS = "suspicious";
export const LABEL_SAFE = "safe";

export type SusFactorLabel = typeof LABEL_SUSPICIOUS | typeof LABEL_SAFE;

/**
 * Maximum number of *content* tokens per inference chunk.
 *
 * The model's hard limit is 512 tokens total, but the tokenizer adds a [CLS]
 * and a [SEP] token, leaving 510 usable positions for the prompt payload.
 */
export const MAX_CONTENT_TOKENS = 510;

/**
 * Overlap between adjacent chunks in tokens.
 *
 * Each new chunk starts CHUNK_STRIDE tokens after the previous one begins,
 * keeping 50 tokens of shared context so sentence boundaries that fall near
 * a chunk edge are still scored in full context.
 */
export const CHUNK_OVERLAP = 50;

/** Number of new tokens advanced per chunk (= MAX_CONTENT_TOKENS - CHUNK_OVERLAP). */
export const CHUNK_STRIDE = MAX_CONTENT_TOKENS - CHUNK_OVERLAP; // 460

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

/**
 * Return type of {@link SusFactorClassifier.classify} for prompts of any length.
 *
 * Prompts within {@link MAX_CONTENT_TOKENS} (510 tokens) produce exactly one
 * chunk. Longer prompts are split automatically — callers never need to check
 * length or call a different method.
 *
 * Each chunk is an independent model inference; no scores are aggregated.
 *
 * ### Displaying a single score
 *
 * The previous API returned one `score` and `label` directly. With chunking,
 * there is no single canonical score. Callers that need one number for display
 * (dashboards, logs) should decide explicitly which value they want:
 *
 * ```typescript
 * // Highest suspicion across all chunks (most conservative):
 * const maxScore = Math.max(...result.chunks.map(c => c.score));
 *
 * // First chunk only (equivalent to the old score for short prompts;
 * // may miss a suspicious tail in long prompts):
 * const firstScore = result.chunks[0].score;
 * ```
 *
 * Use `isSuspicious` for security gating. A display score is a UX choice,
 * not a security one.
 */
export interface ChunkedSusFactorResult {
  /**
   * Individual result for each chunk, in order.
   *
   * Short prompts always produce exactly one entry; access `chunks[0]` for
   * the score and label in that case.
   */
  chunks: SusFactorResult[];
  /**
   * `true` if **any** chunk's label is `"suspicious"`.
   *
   * Use this field for security gating — a prompt is suspicious if any
   * portion of it is suspicious, regardless of how many chunks are safe.
   */
  isSuspicious: boolean;
  /** Total wall-clock time for all chunks (parallel), in milliseconds. */
  totalTimingMs: number;
}
