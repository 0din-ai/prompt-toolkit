/**
 * Embedding providers for generating text embeddings.
 *
 * This module provides implementations for different embedding providers:
 * - OpenAIProvider: Uses OpenAI's API (requires 'openai' package)
 * - OnnxProvider: Uses local ONNX model inference (requires 'onnxruntime-node' package)
 *
 * All providers are optional and require their respective dependencies to be installed.
 */

export {
  ModelCache,
  type DownloadProgressEvent,
  type DownloadProgressCallback,
  type GetModelOptions,
  type DownloadModelOptions,
} from './model-cache';
export { OpenAIProvider } from './openai';
export { OnnxProvider } from './onnx';
