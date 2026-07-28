'use strict';

const js = require('@eslint/js');
const tseslint = require('@typescript-eslint/eslint-plugin');
const globals = require('globals');

// Flat-config replacement for the legacy .eslintrc.js. eslint 10 dropped
// support for eslintrc entirely, so this reproduces the same effective
// ruleset: eslint:recommended + plugin:@typescript-eslint/recommended,
// parsed with @typescript-eslint/parser, scoped to src/ and test/.
module.exports = [
  {
    ignores: ['dist/**', 'coverage/**', 'node_modules/**'],
  },
  {
    files: ['src/**/*.ts', 'test/**/*.ts'],
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.es2015,
      },
    },
  },
  js.configs.recommended,
  ...tseslint.configs['flat/recommended'],
  {
    files: ['src/**/*.ts', 'test/**/*.ts'],
    rules: {
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      // Successor to the deprecated no-var-requires rule this repo had
      // disabled; recommended turns it on as an error.
      '@typescript-eslint/no-require-imports': 'off',
    },
  },
  {
    files: ['test/**/*.ts'],
    languageOptions: {
      globals: {
        ...globals.jest,
      },
    },
  },
];
