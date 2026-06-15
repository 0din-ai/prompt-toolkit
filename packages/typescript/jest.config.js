module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/test'],
  testMatch: ['**/*.test.ts'],
  collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts'],
  moduleFileExtensions: ['ts', 'js', 'json'],
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
