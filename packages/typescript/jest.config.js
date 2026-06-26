module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/test'],
  testMatch: ['**/*.test.ts'],
  collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts'],
  moduleFileExtensions: ['ts', 'js', 'json'],
  // onnxruntime-node is an optional peer dependency that is not installed by
  // `npm ci` in CI (only in devDependencies would guarantee install). The
  // classifier calls require("onnxruntime-node") at runtime inside classify()
  // to construct Tensor objects. Under Jest, live ONNX inference is
  // intentionally avoided (jest's module sandbox breaks ort's instanceof
  // Float32Array checks). All unit tests use fake sessions injected via the
  // constructor; the Tensor objects are created but ignored by the fake session.
  //
  // This mapper redirects the require to a minimal stub so tests that use fake
  // sessions can run without the real package installed. Real inference is
  // exercised via ts-node (see susfactor-parity.test.ts header).
  moduleNameMapper: {
    '^onnxruntime-node$': '<rootDir>/__mocks__/onnxruntime-node.js',
  },
  // Coverage thresholds lock in the current baseline so regressions are caught
  // in CI. Thresholds are set 2-3 points below current actuals to absorb minor
  // variance without false failures.
  //
  // Branch coverage is intentionally lower: onnx.ts and openai.ts are 0% branch
  // because they require live peer deps (onnxruntime-node, openai) that are not
  // installed in the test environment. These are covered by integration tests.
  coverageThreshold: {
    global: {
      statements: 73,
      branches: 57,
      functions: 61,
      lines: 73,
    },
  },
};
