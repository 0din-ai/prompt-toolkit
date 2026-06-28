/**
 * Jest manual mock for onnxruntime-node.
 *
 * onnxruntime-node is an optional peer dependency and is not installed in the
 * test environment (the parity CI job runs `npm ci` without peer deps).
 * classifier.ts calls require("onnxruntime-node") at runtime inside classify()
 * to build Tensor objects; the fake sessions used in unit tests ignore those
 * inputs entirely and return pre-canned logits.
 *
 * This stub satisfies the require() call so that tests using fake sessions
 * (fakeSession / fakeTokenizer patterns) can run without the real package.
 * Tests that exercise real ONNX inference are model-gated and skipped unless
 * SUSFACTOR_MODEL_DIR is set, at which point onnxruntime-node must be
 * installed by the caller.
 *
 * Jest automatically uses __mocks__/onnxruntime-node.js for modules that are
 * not found in node_modules, so no jest.mock() call is needed in test files.
 */

'use strict';

/**
 * Minimal Tensor stub. classifier.ts constructs Tensor objects for input_ids
 * and attention_mask, then passes them to session.run(). Fake test sessions
 * ignore the tensor values and return pre-canned logits, so the stub only
 * needs to accept the constructor arguments without error.
 */
class Tensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

/**
 * InferenceSession stub — only used by SusFactorClassifier.create(), which
 * is not exercised in unit tests (they inject a fake session via the
 * constructor). Included here so that any accidental call surfaces a clear
 * "not implemented" error rather than a cryptic MODULE_NOT_FOUND.
 */
const InferenceSession = {
  create: async (_modelPath) => {
    throw new Error(
      'onnxruntime-node mock: InferenceSession.create() is not implemented. ' +
      'Install onnxruntime-node and ensure SUSFACTOR_MODEL_DIR is set for live inference.'
    );
  },
};

module.exports = { Tensor, InferenceSession };
