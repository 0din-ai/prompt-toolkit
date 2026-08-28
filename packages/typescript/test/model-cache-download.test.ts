/**
 * Tests for ModelCache download functionality.
 *
 * The download path is tested via a lightweight in-process HTTP server so no
 * real network calls are made. The atomic-rename and progress-callback paths
 * are verified in dedicated sub-tests.
 */

import * as fs from 'fs';
import * as http from 'http';
import * as os from 'os';
import * as path from 'path';
import { ModelCache } from '../src/providers/model-cache';
import type { DownloadProgressEvent } from '../src/providers/model-cache';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'model-cache-test-'));
}

function rmrf(dir: string) {
  fs.rmSync(dir, { recursive: true, force: true });
}

/** Start a minimal HTTP server serving static body for a single path. */
function startServer(
  responsePath: string,
  body: Buffer,
  statusCode = 200,
): Promise<{ url: string; close: () => Promise<void> }> {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      if (req.url === responsePath) {
        res.writeHead(statusCode, {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(body.length),
        });
        res.end(body);
      } else {
        res.writeHead(404);
        res.end();
      }
    });
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as { port: number };
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}

// ---------------------------------------------------------------------------
// getModel — cache hit (no download)
// ---------------------------------------------------------------------------

describe('ModelCache.getModel — cache hit', () => {
  let dir: string;

  beforeEach(() => { dir = makeTempDir(); });
  afterEach(() => { rmrf(dir); });

  it('returns existing path without network access', async () => {
    const cache = new ModelCache(dir);
    const fileDir = path.join(dir, 'org', 'repo');
    fs.mkdirSync(fileDir, { recursive: true });
    const filePath = path.join(fileDir, 'model.onnx');
    fs.writeFileSync(filePath, Buffer.from('cached'));

    const result = await cache.getModel('org/repo', 'model.onnx');
    expect(result).toBe(filePath);
    expect(fs.readFileSync(result).toString()).toBe('cached');
  });
});

// ---------------------------------------------------------------------------
// getModel — download
// ---------------------------------------------------------------------------

describe('ModelCache.getModel — download', () => {
  let dir: string;
  let server: { url: string; close: () => Promise<void> };
  const body = Buffer.from('onnx-model-bytes');

  beforeEach(async () => {
    dir = makeTempDir();
    server = await startServer('/org/repo/resolve/main/model.onnx', body);
  });
  afterEach(async () => {
    rmrf(dir);
    await server.close();
  });

  it('downloads and writes file to cache', async () => {
    const cache = new ModelCache(dir);
    const result = await cache.getModel('org/repo', 'model.onnx', {
      baseUrl: server.url,
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.readFileSync(result)).toEqual(body);
  });

  it('returns same path on second call (cache hit)', async () => {
    const cache = new ModelCache(dir);
    const opts = { baseUrl: server.url };
    const first = await cache.getModel('org/repo', 'model.onnx', opts);
    const second = await cache.getModel('org/repo', 'model.onnx', opts);
    expect(first).toBe(second);
  });

  it('leaves no temp files after successful download', async () => {
    const cache = new ModelCache(dir);
    await cache.getModel('org/repo', 'model.onnx', { baseUrl: server.url });

    const walk = (d: string): string[] =>
      fs.readdirSync(d).flatMap((e) => {
        const full = path.join(d, e);
        return fs.statSync(full).isDirectory() ? walk(full) : [full];
      });
    const files = walk(dir);
    const tmpFiles = files.filter((f) => f.includes('.tmp.'));
    expect(tmpFiles).toHaveLength(0);
  });

  it('fires progress events with increasing bytesDownloaded', async () => {
    const cache = new ModelCache(dir);
    const events: DownloadProgressEvent[] = [];

    await cache.getModel('org/repo', 'model.onnx', {
      baseUrl: server.url,
      onProgress: (e) => events.push(e),
    });

    expect(events.length).toBeGreaterThan(0);
    // Last event must be the done event
    expect(events[events.length - 1].done).toBe(true);
    expect(events[events.length - 1].file).toBe('model.onnx');
    // totalBytes is known (server sends Content-Length)
    expect(events[events.length - 1].totalBytes).toBe(body.length);
    expect(events[events.length - 1].bytesDownloaded).toBe(body.length);
    expect(events[events.length - 1].percent).toBe(100);
  });

  it('emits progress with null percent when Content-Length is absent', async () => {
    // Start a server without Content-Length
    const noLenServer = await new Promise<{ url: string; close: () => Promise<void> }>(
      (resolve) => {
        const srv = http.createServer((_req, res) => {
          res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
          res.end(body);
        });
        srv.listen(0, '127.0.0.1', () => {
          const { port } = srv.address() as { port: number };
          resolve({
            url: `http://127.0.0.1:${port}`,
            close: () => new Promise((r) => srv.close(() => r())),
          });
        });
      },
    );

    try {
      const cache = new ModelCache(dir);
      const events: DownloadProgressEvent[] = [];
      await cache.getModel('org/repo', 'model.onnx', {
        baseUrl: noLenServer.url,
        onProgress: (e) => events.push(e),
      });
      const intermediate = events.filter((e) => !e.done);
      // totalBytes and percent are null when Content-Length is absent
      for (const e of intermediate) {
        expect(e.totalBytes).toBeNull();
        expect(e.percent).toBeNull();
      }
    } finally {
      await noLenServer.close();
    }
  });
});

// ---------------------------------------------------------------------------
// getModel — HTTP error
// ---------------------------------------------------------------------------

describe('ModelCache.getModel — HTTP error', () => {
  let dir: string;
  let server: { url: string; close: () => Promise<void> };

  beforeEach(async () => {
    dir = makeTempDir();
    server = await startServer(
      '/org/repo/resolve/main/missing.onnx',
      Buffer.from('not found'),
      404,
    );
  });
  afterEach(async () => {
    rmrf(dir);
    await server.close();
  });

  it('throws ProviderError on non-2xx response', async () => {
    const cache = new ModelCache(dir);
    await expect(
      cache.getModel('org/repo', 'missing.onnx', { baseUrl: server.url }),
    ).rejects.toThrow('HTTP 404');
  });
});

// ---------------------------------------------------------------------------
// downloadModel — file manifest
// ---------------------------------------------------------------------------

describe('ModelCache.downloadModel', () => {
  let dir: string;

  afterEach(() => { if (dir) rmrf(dir); });

  it('already-cached version returns immediately without download', async () => {
    dir = makeTempDir();
    const cache = new ModelCache(dir);

    // Pre-populate at the HF org/repo path that modelDirectory('v1') now resolves to.
    // This mirrors the Rust SDK layout: <cacheDir>/intfloat/multilingual-e5-large/...
    const modelDir = cache.modelDirectory('v1');
    const onnxDir = path.join(modelDir, 'onnx');
    fs.mkdirSync(onnxDir, { recursive: true });
    fs.writeFileSync(path.join(onnxDir, 'model.onnx'), 'x');
    fs.writeFileSync(path.join(onnxDir, 'model.onnx_data'), 'x');
    fs.writeFileSync(path.join(modelDir, 'tokenizer.json'), '{}');
    fs.writeFileSync(path.join(modelDir, 'config.json'), '{}');

    // Should resolve immediately (cache hit), never touching the network
    await expect(
      cache.downloadModel('v1', { baseUrl: 'http://127.0.0.1:1' }),
    ).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// hasModel — v1 manifest completeness
// ---------------------------------------------------------------------------

describe("ModelCache.hasModel('v1') completeness", () => {
  let dir: string;

  afterEach(() => { if (dir) rmrf(dir); });

  it('returns false when onnx/model.onnx_data is missing', () => {
    dir = makeTempDir();
    const cache = new ModelCache(dir);

    const modelDir = cache.modelDirectory('v1');
    const onnxDir = path.join(modelDir, 'onnx');
    fs.mkdirSync(onnxDir, { recursive: true });
    fs.writeFileSync(path.join(onnxDir, 'model.onnx'), 'x');
    fs.writeFileSync(path.join(modelDir, 'tokenizer.json'), '{}');
    fs.writeFileSync(path.join(modelDir, 'config.json'), '{}');

    expect(cache.hasModel('v1')).toBe(false);
  });

  it('returns true once onnx/model.onnx_data is also present', () => {
    dir = makeTempDir();
    const cache = new ModelCache(dir);

    const modelDir = cache.modelDirectory('v1');
    const onnxDir = path.join(modelDir, 'onnx');
    fs.mkdirSync(onnxDir, { recursive: true });
    fs.writeFileSync(path.join(onnxDir, 'model.onnx'), 'x');
    fs.writeFileSync(path.join(onnxDir, 'model.onnx_data'), 'x');
    fs.writeFileSync(path.join(modelDir, 'tokenizer.json'), '{}');
    fs.writeFileSync(path.join(modelDir, 'config.json'), '{}');

    expect(cache.hasModel('v1')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// downloadModel — v1 manifest / hasModel self-consistency (regression)
// ---------------------------------------------------------------------------

/** Start a minimal HTTP server serving distinct small bodies per path. */
function startMultiRouteServer(
  routes: Record<string, Buffer>,
): Promise<{ url: string; close: () => Promise<void> }> {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const body = req.url ? routes[req.url] : undefined;
      if (body) {
        res.writeHead(200, {
          'Content-Type': 'application/octet-stream',
          'Content-Length': String(body.length),
        });
        res.end(body);
      } else {
        res.writeHead(404);
        res.end();
      }
    });
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address() as { port: number };
      resolve({
        url: `http://127.0.0.1:${port}`,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}

describe('ModelCache.downloadModel — v1 manifest self-consistency', () => {
  let dir: string;
  let server: { url: string; close: () => Promise<void> };

  beforeEach(async () => {
    dir = makeTempDir();
    server = await startMultiRouteServer({
      '/intfloat/multilingual-e5-large/resolve/main/onnx/model.onnx':
        Buffer.from('graph-bytes'),
      '/intfloat/multilingual-e5-large/resolve/main/onnx/model.onnx_data':
        Buffer.from('external-weights-bytes'),
      '/intfloat/multilingual-e5-large/resolve/main/tokenizer.json':
        Buffer.from('{}'),
      '/intfloat/multilingual-e5-large/resolve/main/config.json':
        Buffer.from('{}'),
    });
  });
  afterEach(async () => {
    rmrf(dir);
    await server.close();
  });

  it('downloads model.onnx_data and hasModel agrees the cache is complete', async () => {
    const cache = new ModelCache(dir);
    await cache.downloadModel('v1', { baseUrl: server.url });

    expect(cache.hasModel('v1')).toBe(true);
    expect(
      fs.existsSync(
        path.join(cache.modelDirectory('v1'), 'onnx', 'model.onnx_data'),
      ),
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// downloadModel — susfactor-v1 manifest / hasSusfactorModel self-consistency
// (regression)
// ---------------------------------------------------------------------------

describe('ModelCache.downloadModel — susfactor-v1 manifest self-consistency', () => {
  let dir: string;
  let server: { url: string; close: () => Promise<void> };

  beforeEach(async () => {
    dir = makeTempDir();
    server = await startMultiRouteServer({
      '/0dinai/susfactor-e5-large-onnx/resolve/main/onnx/model.onnx':
        Buffer.from('graph-bytes'),
      '/0dinai/susfactor-e5-large-onnx/resolve/main/onnx/model.onnx_data':
        Buffer.from('external-weights-bytes'),
      '/0dinai/susfactor-e5-large-onnx/resolve/main/tokenizer.json':
        Buffer.from('{}'),
      '/0dinai/susfactor-e5-large-onnx/resolve/main/tokenizer_config.json':
        Buffer.from('{}'),
    });
  });
  afterEach(async () => {
    rmrf(dir);
    await server.close();
  });

  it('downloads tokenizer_config.json and hasSusfactorModel agrees the cache is complete', async () => {
    const cache = new ModelCache(dir);
    await cache.downloadModel('susfactor-v1', { baseUrl: server.url });

    expect(cache.hasSusfactorModel('susfactor-v1')).toBe(true);
    expect(
      fs.existsSync(
        path.join(
          cache.modelDirectory('susfactor-v1'),
          'tokenizer_config.json',
        ),
      ),
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// hasSusfactorModel — susfactor-v1 manifest completeness
// ---------------------------------------------------------------------------

describe("ModelCache.hasSusfactorModel('susfactor-v1') completeness", () => {
  let dir: string;

  afterEach(() => { if (dir) rmrf(dir); });

  it('returns false when tokenizer_config.json is missing', () => {
    dir = makeTempDir();
    const cache = new ModelCache(dir);

    const modelDir = cache.modelDirectory('susfactor-v1');
    const onnxDir = path.join(modelDir, 'onnx');
    fs.mkdirSync(onnxDir, { recursive: true });
    fs.writeFileSync(path.join(onnxDir, 'model.onnx'), 'x');
    fs.writeFileSync(path.join(onnxDir, 'model.onnx_data'), 'x');
    fs.writeFileSync(path.join(modelDir, 'tokenizer.json'), '{}');

    expect(cache.hasSusfactorModel('susfactor-v1')).toBe(false);
  });

  it('returns true once tokenizer_config.json is also present', () => {
    dir = makeTempDir();
    const cache = new ModelCache(dir);

    const modelDir = cache.modelDirectory('susfactor-v1');
    const onnxDir = path.join(modelDir, 'onnx');
    fs.mkdirSync(onnxDir, { recursive: true });
    fs.writeFileSync(path.join(onnxDir, 'model.onnx'), 'x');
    fs.writeFileSync(path.join(onnxDir, 'model.onnx_data'), 'x');
    fs.writeFileSync(path.join(modelDir, 'tokenizer.json'), '{}');
    fs.writeFileSync(path.join(modelDir, 'tokenizer_config.json'), '{}');

    expect(cache.hasSusfactorModel('susfactor-v1')).toBe(true);
  });
});
