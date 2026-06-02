/**
 * Model cache for downloading and caching ONNX models.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

/**
 * Manages local caching of ONNX models.
 *
 * The cache directory defaults to ~/.cache/signature-sdk/models/v1/ but can be
 * overridden via the SIGNATURE_SDK_MODEL_CACHE environment variable.
 */
export class ModelCache {
  private static readonly DEFAULT_CACHE_DIR = path.join(
    os.homedir(),
    '.cache',
    'signature-sdk',
    'models'
  );
  private static readonly ENV_VAR = 'SIGNATURE_SDK_MODEL_CACHE';

  private cacheDir: string;

  /**
   * Initialize model cache.
   *
   * @param cacheDir - Optional custom cache directory path
   */
  constructor(cacheDir?: string) {
    if (cacheDir) {
      this.cacheDir = cacheDir;
    } else if (process.env[ModelCache.ENV_VAR]) {
      this.cacheDir = process.env[ModelCache.ENV_VAR]!;
    } else {
      this.cacheDir = ModelCache.DEFAULT_CACHE_DIR;
    }
  }

  /**
   * Get the cache directory path.
   */
  getCacheDir(): string {
    return this.cacheDir;
  }

  /**
   * Get the directory for a specific model version.
   *
   * @param version - Model version (default: "v1")
   */
  modelDirectory(version: string = 'v1'): string {
    return path.join(this.cacheDir, version);
  }

  /**
   * Ensure the model directory exists.
   *
   * @param version - Model version (default: "v1")
   */
  ensureModelDirectory(version: string = 'v1'): string {
    const modelDir = this.modelDirectory(version);
    fs.mkdirSync(modelDir, { recursive: true });
    return modelDir;
  }

  /**
   * Check if a model version is cached locally.
   *
   * @param version - Model version (default: "v1")
   * @returns True if the model is cached, false otherwise
   */
  hasModel(version: string = 'v1'): boolean {
    const modelDir = this.modelDirectory(version);
    if (!fs.existsSync(modelDir)) {
      return false;
    }

    // Check for required files (prefer optimized model, accept either)
    const hasOptimized = fs.existsSync(
      path.join(modelDir, 'onnx', 'model_O4.onnx')
    );
    const hasUnoptimized = fs.existsSync(
      path.join(modelDir, 'onnx', 'model.onnx')
    );

    const requiredFiles = ['tokenizer.json', 'config.json'];

    return (
      (hasOptimized || hasUnoptimized) &&
      requiredFiles.every((file) => fs.existsSync(path.join(modelDir, file)))
    );
  }

  /**
   * Check if a SusFactor ONNX model version is cached locally.
   *
   * Requires a complete model pair — either the optimized pair
   * (`onnx/model_O4.onnx` + `onnx/model_O4.onnx_data`) or the unoptimized
   * pair (`onnx/model.onnx` + `onnx/model.onnx_data`) — plus `tokenizer.json`.
   * The `.onnx_data` file holds the external weights and is required for ORT
   * to load the graph.
   *
   * @param version - Model version (default: "susfactor-v1")
   */
  hasSusfactorModel(version: string = 'susfactor-v1'): boolean {
    const modelDir = this.modelDirectory(version);
    if (!fs.existsSync(modelDir)) {
      return false;
    }
    // Accept either the optimized or unoptimized pair independently — a usable
    // cache only requires one complete pair (model + matching .onnx_data).
    const hasOptimizedPair =
      fs.existsSync(path.join(modelDir, 'onnx', 'model_O4.onnx')) &&
      fs.existsSync(path.join(modelDir, 'onnx', 'model_O4.onnx_data'));
    const hasUnoptimizedPair =
      fs.existsSync(path.join(modelDir, 'onnx', 'model.onnx')) &&
      fs.existsSync(path.join(modelDir, 'onnx', 'model.onnx_data'));
    const hasTokenizer = fs.existsSync(path.join(modelDir, 'tokenizer.json'));
    return (hasOptimizedPair || hasUnoptimizedPair) && hasTokenizer;
  }

  /**
   * Get the path to the ONNX model file.
   *
   * Prefers the optimized model (model_O4.onnx) if available,
   * falls back to the unoptimized model (model.onnx).
   *
   * @param version - Model version (default: "v1")
   */
  getModelPath(version: string = 'v1'): string {
    const modelDir = this.modelDirectory(version);
    const optimizedPath = path.join(modelDir, 'onnx', 'model_O4.onnx');
    const unoptimizedPath = path.join(modelDir, 'onnx', 'model.onnx');

    // Prefer optimized model (smaller, faster inference)
    if (fs.existsSync(optimizedPath)) {
      return optimizedPath;
    }
    return unoptimizedPath;
  }

  /**
   * Get the path to the tokenizer file.
   *
   * @param version - Model version (default: "v1")
   */
  getTokenizerPath(version: string = 'v1'): string {
    return path.join(this.modelDirectory(version), 'tokenizer.json');
  }

  /**
   * Get the path to the model config file.
   *
   * @param version - Model version (default: "v1")
   */
  getConfigPath(version: string = 'v1'): string {
    return path.join(this.modelDirectory(version), 'config.json');
  }

  /**
   * Load the model configuration.
   *
   * @param version - Model version (default: "v1")
   * @returns Model configuration object
   * @throws Error if config file doesn't exist or is invalid
   */
  loadConfig(version: string = 'v1'): any {
    const configPath = this.getConfigPath(version);
    const configData = fs.readFileSync(configPath, 'utf-8');
    return JSON.parse(configData);
  }
}
