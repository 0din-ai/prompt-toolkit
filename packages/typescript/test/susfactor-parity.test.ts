/**
 * SusFactor cross-SDK parity test: TypeScript vs. the validated Rust reference.
 *
 * Loads spec/test-vectors/susfactor_vectors.json (committed golden vectors
 * generated from the Rust SDK) and asserts that the TypeScript ONNX inference
 * reproduces each score within TOLERANCE and the label exactly.
 *
 * The TypeScript SDK loads onnx/model.onnx — the same graph validated in
 * production via the Rust SDK (Heimdall). model_O4.onnx is intentionally NOT
 * used until separately validated (see packages/typescript/src/susfactor/classifier.ts).
 *
 * Skipped when:
 *   - SUSFACTOR_MODEL_DIR is not set or onnx/model.onnx is absent
 *   - Golden scores have not yet been generated (rust_score is null)
 *
 * NOTE: we intentionally do NOT run ONNX inference under jest.  jest's module
 * sandbox replaces global typed-array constructors, which breaks
 * onnxruntime-node's internal `instanceof Float32Array` checks.  This test
 * therefore marks itself as pending (todo) when running under jest and is
 * designed to be run via ts-node for full end-to-end validation.
 *
 * To run end-to-end (outside jest):
 *   SUSFACTOR_MODEL_DIR=/path/to/cache/susfactor-v1 \
 *     npx ts-node test/susfactor-parity.test.ts
 *
 * The jest describe block still compiles and type-checks in CI (providing
 * coverage of the fixture-loading + skip logic), it just skips the live
 * inference.
 */

import * as fs from 'fs';
import * as path from 'path';

/** Maximum absolute score difference vs. Rust reference. */
const TOLERANCE = 1e-3;

// ── Fixture loading ──────────────────────────────────────────────────────────

interface GoldenVector {
  name: string;
  prompt: string;
  rust_score: number;
  expected_label: 'suspicious' | 'safe';
  notes?: string;
}

function loadVectors(): GoldenVector[] {
  const fixturePath = path.resolve(
    __dirname,
    '../../..',
    'spec',
    'test-vectors',
    'susfactor_vectors.json',
  );
  if (!fs.existsSync(fixturePath)) return [];
  const doc = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
  return (doc.vectors ?? []).filter(
    (v: any) => v.rust_score !== null && v.expected_label !== null,
  );
}

const VECTORS = loadVectors();

// ── Model availability ───────────────────────────────────────────────────────

const MODEL_DIR = process.env.SUSFACTOR_MODEL_DIR;
const MODEL_AVAILABLE = Boolean(
  MODEL_DIR &&
    fs.existsSync(path.join(MODEL_DIR, 'onnx', 'model.onnx')) &&
    fs.existsSync(path.join(MODEL_DIR, 'onnx', 'model.onnx_data')) &&
    fs.existsSync(path.join(MODEL_DIR, 'tokenizer.json')),
);

// ── Jest describe (always compiles; live inference skipped under jest) ────────

const describeIfReady =
  MODEL_AVAILABLE && VECTORS.length > 0 ? describe : describe.skip;

// Live ONNX inference cannot run under jest — jest's module sandbox replaces
// global typed-array constructors, which breaks onnxruntime-node's internal
// `instanceof Float32Array` checks. We always register a todo so jest sees at
// least one item in the describe block; the ts-node entrypoint at the bottom
// of this file runs the real assertions outside jest.
describeIfReady('SusFactor parity: TypeScript vs Rust reference', () => {
  it.todo(
    'live ONNX inference runs outside jest — use: ' +
      'SUSFACTOR_MODEL_DIR=... npx ts-node test/susfactor-parity.test.ts',
  );
});

// ── Fixture contract tests (always run, no model needed) ─────────────────────

describe('SusFactor parity fixture', () => {
  it('fixture file exists at spec/test-vectors/susfactor_vectors.json', () => {
    const fixturePath = path.resolve(
      __dirname,
      '../../..',
      'spec',
      'test-vectors',
      'susfactor_vectors.json',
    );
    expect(fs.existsSync(fixturePath)).toBe(true);
  });

  it('fixture contains at least one vector with a name and prompt', () => {
    const fixturePath = path.resolve(
      __dirname,
      '../../..',
      'spec',
      'test-vectors',
      'susfactor_vectors.json',
    );
    const doc = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
    expect(Array.isArray(doc.vectors)).toBe(true);
    expect(doc.vectors.length).toBeGreaterThan(0);
    for (const v of doc.vectors) {
      expect(typeof v.name).toBe('string');
      expect(typeof v.prompt).toBe('string');
    }
  });

  it('all scored vectors have valid expected_label', () => {
    for (const v of VECTORS) {
      expect(['suspicious', 'safe']).toContain(v.expected_label);
      expect(typeof v.rust_score).toBe('number');
      expect(v.rust_score).toBeGreaterThanOrEqual(0);
      expect(v.rust_score).toBeLessThanOrEqual(1);
    }
  });
});

// ── ts-node entrypoint (runs when executed directly, not via jest) ────────────

async function runParityCheck() {
  if (!MODEL_AVAILABLE) {
    console.error(
      'SUSFACTOR_MODEL_DIR not set or model.onnx / model.onnx_data / tokenizer.json missing.',
    );
    process.exit(1);
  }
  if (VECTORS.length === 0) {
    console.error(
      'No scored golden vectors found. Run: make generate-susfactor-goldens',
    );
    process.exit(1);
  }

  const { ModelCache } = require('../src/providers/model-cache');
  const { SusFactorClassifier } = require('../src/susfactor/classifier');

  const cache = new ModelCache(MODEL_DIR);
  const clf = await SusFactorClassifier.create(cache, {});

  let failures = 0;
  for (const vec of VECTORS) {
    const result = await clf.classify(vec.prompt);
    const diff = Math.abs(result.score - vec.rust_score);
    const labelOk = result.label === vec.expected_label;
    const scoreOk = diff <= TOLERANCE;
    if (labelOk && scoreOk) {
      console.log(`  ✅  ${vec.name}: score=${result.score.toFixed(6)} (Δ=${diff.toExponential(2)})`);
    } else {
      failures++;
      console.error(
        `  ❌  ${vec.name}: score=${result.score.toFixed(6)} ` +
          `(committed=${vec.rust_score.toFixed(6)}, Δ=${diff.toExponential(2)} > tol=${TOLERANCE}) ` +
          `label=${result.label} (expected=${vec.expected_label})`,
      );
    }
  }

  await clf.close();

  if (failures > 0) {
    console.error(`\n${failures} parity failure(s). Investigate before loosening tolerance.`);
    process.exit(1);
  }
  console.log(`\n✅ All ${VECTORS.length} parity checks passed.`);
}

// Run when invoked via ts-node / node (not imported by jest)
if (require.main === module) {
  runParityCheck().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
