#!/usr/bin/env ts-node
/**
 * Interactive CLI tool: reads a prompt from stdin (supports multiline),
 * generates a V1 signature using the local ONNX model, and prints the result.
 *
 * Usage:
 *   npx ts-node examples/sign_prompt.ts
 *
 * Or pipe input directly:
 *   echo "my prompt" | npx ts-node examples/sign_prompt.ts
 *   cat prompt.txt  | npx ts-node examples/sign_prompt.ts
 *
 * Multiline input: type your prompt, then press Ctrl+D (Unix) or Ctrl+Z (Windows)
 * to signal end of input.
 */

import * as readline from 'readline';
import { signText, getSignatureString, SignatureVersion } from '../src';
import { ModelCache, OnnxProvider } from '../src/providers';

async function readPrompt(): Promise<string> {
  const isTTY = process.stdin.isTTY;

  if (isTTY) {
    process.stderr.write('Enter your prompt (multiline supported).\n');
    process.stderr.write('Press Ctrl+D when done:\n\n');
  }

  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
    terminal: false,
  });

  const lines: string[] = [];
  for await (const line of rl) {
    lines.push(line);
  }

  return lines.join('\n');
}

async function main(): Promise<void> {
  // Read the prompt
  const prompt = await readPrompt();

  if (!prompt.trim()) {
    process.stderr.write('Error: empty prompt\n');
    process.exit(1);
  }

  // Load model and sign
  process.stderr.write('\nSigning...\n');

  const cache = new ModelCache();
  if (!cache.hasModel('v1')) {
    process.stderr.write(
      `Error: ONNX model not found at ${cache.modelDirectory('v1')}\n` +
        'Download it from https://huggingface.co/0dinai/jailbreak-embeddings-large-onnx\n'
    );
    process.exit(1);
  }

  const provider = await OnnxProvider.create(cache);
  const result = await signText(prompt, { provider, version: SignatureVersion.V1 });
  await provider.close();

  const signature = getSignatureString(result);

  // Machine-readable output to stdout, status info to stderr
  process.stderr.write('\n');
  console.log(signature);

  // Verbose info to stderr so stdout is just the clean signature string
  process.stderr.write(`provider:   ${result.provider}\n`);
  process.stderr.write(`model:      ${result.model}\n`);
  process.stderr.write(`sha256:     ${result.embeddingSha256}\n`);
  process.stderr.write(`timing:     ${(result.timingMs ?? 0).toFixed(1)}ms\n`);
}

main().catch((err) => {
  process.stderr.write(`Error: ${err.message ?? err}\n`);
  process.exit(1);
});
