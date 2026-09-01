/**
 * Model cache for downloading and caching ONNX models.
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { pipeline } from 'stream/promises';
import { createWriteStream } from 'fs';
import { Readable } from 'stream';
import { ProviderError } from '../error';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Progress event emitted during a model file download.
 *
 * Suitable for both plain-text logging and structured progress UIs.
 *
 * @example Plain-text log
 * ```typescript
 * cache.getModel('org/repo', 'model.onnx', {
 *   onProgress: (e) => {
 *     if (e.done) console.log(`Downloaded ${e.file}`);
 *     else if (e.percent !== null) process.stdout.write(`\r${e.file}: ${e.percent}%`);
 *   },
 * });
 * ```
 *
 * @example Structured (progress bar library)
 * ```typescript
 * const bar = new ProgressBar(':file [:bar] :percent', { total: 100 });
 * cache.getModel('org/repo', 'model.onnx', {
 *   onProgress: (e) => bar.update((e.percent ?? 0) / 100, { file: e.file }),
 * });
 * ```
 */
export interface DownloadProgressEvent {
  /** Filename being downloaded (e.g. "onnx/model.onnx"). */
  file: string;
  /** Bytes received so far. */
  bytesDownloaded: number;
  /**
   * Total bytes expected, or `null` if the server did not send Content-Length.
   */
  totalBytes: number | null;
  /**
   * Integer 0-100 derived from bytesDownloaded / totalBytes, or `null` when
   * totalBytes is unknown.
   */
  percent: number | null;
  /**
   * `true` on the final event emitted after the file has been fully written
   * and renamed into place.
   */
  done: boolean;
}

/** Callback type for download progress. */
export type DownloadProgressCallback = (event: DownloadProgressEvent) => void;

/**
 * Options for {@link ModelCache.getModel}.
 */
export interface GetModelOptions {
  /**
   * HuggingFace API token. Required for gated models (e.g. susfactor).
   * Falls back to the `HF_TOKEN` environment variable if not provided.
   */
  hfToken?: string;
  /**
   * Override the HuggingFace base URL. Useful in tests to point at a local
   * mock server without touching the network.
   *
   * @default "https://huggingface.co"
   */
  baseUrl?: string;
  /**
   * Called as bytes arrive from the network. See {@link DownloadProgressEvent}.
   */
  onProgress?: DownloadProgressCallback;
}

/**
 * Options for {@link ModelCache.downloadModel}.
 */
export interface DownloadModelOptions extends GetModelOptions {
  /**
   * If `true`, skip the cache-presence check and force a fresh download even
   * if the model directory already contains all required files.
   *
   * @default false
   */
  force?: boolean;
}

// ---------------------------------------------------------------------------
// File manifests
// ---------------------------------------------------------------------------

/**
 * Files required for the embedding model (v1 / 0dinai/jailbreak-embeddings-base-onnx).
 *
 * The HuggingFace repo that hosts these files is
 * `0dinai/jailbreak-embeddings-base-onnx`. We always download `model.onnx`;
 * `model_O4.onnx` is an optional optimized variant that is not automatically
 * fetched.
 *
 * Unlike some ORT exports, this repo ships `model.onnx` as a single self-contained
 * file — there is no `onnx/model.onnx_data` external-weights file upstream, so it
 * is intentionally not part of the mandatory download manifest.
 */
const EMBEDDING_MODEL_FILES = [
  'onnx/model.onnx',
  'tokenizer.json',
  'config.json',
] as const;

/** HuggingFace repo for the embedding ONNX model. */
const EMBEDDING_MODEL_REPO = '0dinai/jailbreak-embeddings-base-onnx';

/**
 * Files required for the SusFactor classifier (susfactor-v1).
 *
 * These come from `0dinai/susfactor-e5-large-onnx` (gated — requires HF token).
 * `model.onnx_data` holds the external weights required by ORT.
 * `tokenizer_config.json` is required by transformers.js's
 * `AutoTokenizer.from_pretrained(..., { local_files_only: true })` to resolve
 * `tokenizer_class`; without it, tokenizer loading throws.
 */
const SUSFACTOR_MODEL_FILES = [
  'onnx/model.onnx',
  'onnx/model.onnx_data',
  'tokenizer.json',
  'tokenizer_config.json',
] as const;

/** HuggingFace repo for the SusFactor ONNX model. */
const SUSFACTOR_MODEL_REPO = '0dinai/susfactor-e5-large-onnx';

/**
 * Maps the human-readable version key used in the SDK API to the HuggingFace
 * org/repo that owns those files. This is the single source of truth for path
 * alignment between the TS and Rust SDKs.
 *
 * The Rust SDK stores files at `<cacheDir>/<org>/<repo>/...` (derived directly
 * from the model_id passed to `get_model()`). The TS SDK uses short version
 * keys in its public API ("v1", "susfactor-v1") for ergonomics; this map
 * translates them to the same on-disk layout so both SDKs share the cache.
 */
const VERSION_TO_REPO: Record<string, string> = {
  'v1': EMBEDDING_MODEL_REPO,
  'susfactor-v1': SUSFACTOR_MODEL_REPO,
};

// ---------------------------------------------------------------------------
// ModelCache
// ---------------------------------------------------------------------------

/**
 * Manages local caching of ONNX models.
 *
 * The cache directory defaults to `~/.cache/signature-sdk/models/` but can be
 * overridden via the `SIGNATURE_SDK_MODEL_CACHE` environment variable.
 *
 * @example Auto-download on first use
 * ```typescript
 * const cache = new ModelCache();
 * const provider = await OnnxProvider.create(cache); // downloads if needed
 * ```
 *
 * @example Explicit pre-download with progress
 * ```typescript
 * const cache = new ModelCache();
 * await cache.downloadModel('v1', {
 *   onProgress: (e) => console.log(`${e.file}: ${e.percent ?? '?'}%`),
 * });
 * ```
 */
export class ModelCache {
  private static readonly DEFAULT_CACHE_DIR = path.join(
    os.homedir(),
    '.cache',
    'signature-sdk',
    'models'
  );
  private static readonly ENV_VAR = 'SIGNATURE_SDK_MODEL_CACHE';
  private static readonly DEFAULT_HF_BASE_URL = 'https://huggingface.co';
  private static readonly CONNECT_TIMEOUT_MS = 30_000;

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
   * Get the on-disk directory for a model version.
   *
   * For known versions the directory is derived from the HuggingFace repo path
   * (`cacheDir/org/repo`), matching the layout the Rust SDK uses. This ensures
   * that files downloaded by the Rust SDK and files downloaded here live in the
   * same location.
   *
   * | version         | on-disk path                                          |
   * |-----------------|-------------------------------------------------------|
   * | `"v1"`          | `<cacheDir>/0dinai/jailbreak-embeddings-base-onnx`    |
   * | `"susfactor-v1"`| `<cacheDir>/0dinai/susfactor-e5-large-onnx`           |
   * | anything else   | `<cacheDir>/<version>` (legacy / custom)              |
   *
   * @param version - Model version key (default: "v1")
   */
  modelDirectory(version: string = 'v1'): string {
    const repo = VERSION_TO_REPO[version];
    return repo
      ? path.join(this.cacheDir, repo)
      : path.join(this.cacheDir, version);
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

    // Check for required files (prefer optimized model, accept either).
    // Unlike the SusFactor model, the embedding model ships as a single
    // self-contained onnx/model.onnx with no external onnx/model.onnx_data
    // file, so that file is not required here even if it happens to exist
    // on disk (e.g. leftover from an old cache).
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
   * Requires the validated model pair (`onnx/model.onnx` +
   * `onnx/model.onnx_data`) plus `tokenizer.json` and `tokenizer_config.json`.
   * This is the graph validated in production via the Rust SDK. The
   * `.onnx_data` file holds the external weights required by ORT to load the
   * graph. `tokenizer_config.json` is required by transformers.js's
   * `AutoTokenizer.from_pretrained(..., { local_files_only: true })` to
   * resolve `tokenizer_class`; without it, tokenizer loading throws.
   *
   * Note: `model_O4.onnx` is a pre-optimized variant that has not been
   * validated against the production reference; it is intentionally not accepted
   * here until separately validated.
   *
   * @param version - Model version (default: "susfactor-v1")
   */
  hasSusfactorModel(version: string = 'susfactor-v1'): boolean {
    const modelDir = this.modelDirectory(version);
    if (!fs.existsSync(modelDir)) {
      return false;
    }
    // Require the validated unoptimized pair (model.onnx + model.onnx_data).
    const hasValidatedPair =
      fs.existsSync(path.join(modelDir, 'onnx', 'model.onnx')) &&
      fs.existsSync(path.join(modelDir, 'onnx', 'model.onnx_data'));
    const hasTokenizer = fs.existsSync(path.join(modelDir, 'tokenizer.json'));
    const hasTokenizerConfig = fs.existsSync(
      path.join(modelDir, 'tokenizer_config.json'),
    );
    return hasValidatedPair && hasTokenizer && hasTokenizerConfig;
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

  // ---------------------------------------------------------------------------
  // Download API
  // ---------------------------------------------------------------------------

  /**
   * Get the local path to a model file, downloading it from HuggingFace if it
   * is not already cached.
   *
   * This is the primary building block for auto-download on first use. It
   * mirrors the Rust SDK's `ModelCache::get_model()` behaviour:
   * - Cache-hit: returns the existing path immediately, no network access.
   * - Cache-miss: streams the file from HuggingFace to a unique temp file,
   *   then atomically renames it into place.
   *
   * @param modelId  - HuggingFace model repo (e.g. `"0dinai/jailbreak-embeddings-base-onnx"`)
   * @param filename - File path within the repo (e.g. `"onnx/model.onnx"`)
   * @param options  - Token, base URL override, progress callback
   * @returns Absolute path to the cached file
   *
   * @example
   * ```typescript
   * const cache = new ModelCache();
   * const modelPath = await cache.getModel(
   *   '0dinai/jailbreak-embeddings-base-onnx',
   *   'onnx/model.onnx',
   *   { onProgress: (e) => console.log(`${e.file}: ${e.percent ?? '?'}%`) },
   * );
   * ```
   */
  async getModel(
    modelId: string,
    filename: string,
    options: GetModelOptions = {},
  ): Promise<string> {
    const destPath = path.join(this.cacheDir, modelId, filename);

    // Cache hit — no network access needed.
    if (fs.existsSync(destPath)) {
      return destPath;
    }

    await this._downloadFile(modelId, filename, destPath, options);
    return destPath;
  }

  /**
   * Download all required files for a model version.
   *
   * Known versions:
   * - `"v1"` — embedding model (`0dinai/jailbreak-embeddings-base-onnx`)
   * - `"susfactor-v1"` — SusFactor classifier (`0dinai/susfactor-e5-large-onnx`, gated)
   *
   * A HuggingFace token is required for `"susfactor-v1"`. Pass it via
   * `options.hfToken` or set the `HF_TOKEN` environment variable.
   *
   * Files are downloaded in parallel. Already-cached files are skipped
   * unless `options.force` is `true`.
   *
   * @param version - Model version identifier
   * @param options - Download options (token, baseUrl, progress, force)
   *
   * @example
   * ```typescript
   * const cache = new ModelCache();
   * await cache.downloadModel('v1', {
   *   onProgress: (e) => {
   *     if (e.done) console.log(`✓ ${e.file}`);
   *     else if (e.percent !== null) process.stdout.write(`\r${e.file} ${e.percent}%`);
   *   },
   * });
   * ```
   */
  async downloadModel(
    version: string,
    options: DownloadModelOptions = {},
  ): Promise<void> {
    const { force = false, ...getOptions } = options;

    // Skip if already fully cached (and not forced).
    if (!force) {
      const alreadyCached =
        version === 'susfactor-v1'
          ? this.hasSusfactorModel(version)
          : this.hasModel(version);
      if (alreadyCached) return;
    }

    const { repo, files } = this._manifestForVersion(version);

    // Download all files in parallel.
    await Promise.all(
      files.map((filename) => this.getModel(repo, filename, getOptions)),
    );
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Return the HuggingFace repo and required file list for a known version.
   */
  private _manifestForVersion(version: string): {
    repo: string;
    files: readonly string[];
  } {
    if (version === 'susfactor-v1') {
      return { repo: SUSFACTOR_MODEL_REPO, files: SUSFACTOR_MODEL_FILES };
    }
    if (version === 'v1') {
      return { repo: EMBEDDING_MODEL_REPO, files: EMBEDDING_MODEL_FILES };
    }
    throw new ProviderError(
      `Unknown model version "${version}". ` +
        'Known versions: "v1" (embedding), "susfactor-v1" (SusFactor classifier).',
    );
  }

  /**
   * Stream a single file from HuggingFace into the cache with an atomic
   * temp-file rename.
   *
   * Robustness properties (matching the Rust SDK):
   * 1. **Streaming** — response body is piped chunk-by-chunk; never fully
   *    buffered in memory.
   * 2. **Connect timeout** — AbortController cancels the fetch after
   *    {@link ModelCache.CONNECT_TIMEOUT_MS} ms if the server hasn't responded.
   * 3. **Atomic rename** — written to `<dest>.tmp.<pid>.<counter>.<random>`
   *    and renamed into place; concurrent callers racing to the same file are
   *    handled gracefully.
   * 4. **Partial cleanup** — temp file is removed on any error before
   *    re-throwing.
   */
  private async _downloadFile(
    modelId: string,
    filename: string,
    destPath: string,
    options: GetModelOptions,
  ): Promise<void> {
    const baseUrl =
      options.baseUrl ?? ModelCache.DEFAULT_HF_BASE_URL;
    const token =
      options.hfToken ?? process.env['HF_TOKEN'] ?? undefined;
    const onProgress = options.onProgress;

    const url = `${baseUrl}/${modelId}/resolve/main/${filename}`;

    // Ensure parent directory exists.
    fs.mkdirSync(path.dirname(destPath), { recursive: true });

    // Unique temp path — avoids concurrent-download collisions.
    const tempPath = `${destPath}.tmp.${process.pid}.${Date.now()}.${Math.random().toString(36).slice(2)}`;

    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      ModelCache.CONNECT_TIMEOUT_MS,
    );

    let response: Response;
    try {
      const headers: Record<string, string> = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;

      response = await fetch(url, {
        signal: controller.signal,
        headers,
      });
    } finally {
      clearTimeout(timeout);
    }

    if (!response.ok) {
      throw new ProviderError(
        `Failed to download ${filename} from ${modelId}: HTTP ${response.status}`,
      );
    }

    const totalBytes = response.headers.get('content-length')
      ? parseInt(response.headers.get('content-length')!, 10)
      : null;

    let bytesDownloaded = 0;

    // Progress tracking — wrapper around the raw response body stream.
    const progressTransform = async function* (
      source: AsyncIterable<Uint8Array>,
    ) {
      for await (const chunk of source) {
        yield chunk;
        bytesDownloaded += chunk.length;
        if (onProgress) {
          const percent =
            totalBytes !== null
              ? Math.round((bytesDownloaded / totalBytes) * 100)
              : null;
          onProgress({
            file: filename,
            bytesDownloaded,
            totalBytes,
            percent,
            done: false,
          });
        }
      }
    };

    // Write to temp file, then atomically rename.
    const writeStream = createWriteStream(tempPath);
    try {
      const body = response.body;
      if (!body) {
        throw new ProviderError(`Empty response body downloading ${filename}`);
      }
      // Node 18+ fetch body is a Web ReadableStream<Uint8Array>. The TypeScript
      // types for Readable.fromWeb and fetch's ReadableStream diverge slightly
      // across @types/node versions; the cast to Parameters<typeof Readable.fromWeb>[0]
      // is the narrowest safe bridge.
      const nodeReadable = Readable.fromWeb(body as Parameters<typeof Readable.fromWeb>[0]);
      await pipeline(progressTransform(nodeReadable), writeStream);
    } catch (err) {
      // Clean up partial temp file before propagating.
      try { fs.unlinkSync(tempPath); } catch { /* already gone */ }
      throw err;
    }

    // Atomic rename into final location.
    try {
      fs.renameSync(tempPath, destPath);
    } catch (err: unknown) {
      // On Windows rename throws EEXIST when the destination already exists —
      // another concurrent caller won the race. The file is already there
      // with identical bytes, so we clean up and return successfully.
      const code = (err as NodeJS.ErrnoException)?.code;
      const message = err instanceof Error ? err.message : String(err);
      if (code === 'EEXIST') {
        try { fs.unlinkSync(tempPath); } catch { /* already gone */ }
        return;
      }
      try { fs.unlinkSync(tempPath); } catch { /* already gone */ }
      throw new ProviderError(`Failed to finalise ${filename}: ${message}`);
    }

    // Final done event.
    if (onProgress) {
      onProgress({
        file: filename,
        bytesDownloaded,
        totalBytes,
        percent: totalBytes !== null ? 100 : null,
        done: true,
      });
    }
  }
}
