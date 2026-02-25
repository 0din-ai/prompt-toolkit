/**
 * Model cache for downloading and caching ONNX models.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

/**
 * Manages local caching of ONNX models.
 *
 * The cache directory defaults to ~/.cache/odin-sig/models/v1/ but can be
 * overridden via the ODIN_SIG_MODEL_CACHE environment variable.
 */
export class ModelCache {
  private static readonly DEFAULT_CACHE_DIR = path.join(
    os.homedir(),
    '.cache',
    'odin-sig',
    'models'
  );
  private static readonly ENV_VAR = 'ODIN_SIG_MODEL_CACHE';

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

    // Check for required files
    const requiredFiles = [
      path.join('onnx', 'model.onnx'),
      'tokenizer.json',
      'config.json',
    ];

    return requiredFiles.every((file) =>
      fs.existsSync(path.join(modelDir, file))
    );
  }

  /**
   * Get the path to the ONNX model file.
   *
   * @param version - Model version (default: "v1")
   */
  getModelPath(version: string = 'v1'): string {
    return path.join(this.modelDirectory(version), 'onnx', 'model.onnx');
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
