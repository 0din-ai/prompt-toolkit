#!/usr/bin/env ts-node
/**
 * Example demonstrating the high-level signText() API.
 *
 * This example shows how to use the convenience function that takes a text prompt
 * and returns a complete signature result in one call.
 *
 * Note: This example requires the ONNX model files to be downloaded and cached.
 * The OpenAI example requires an API key set in the OPENAI_API_KEY environment variable.
 *
 * Usage:
 *   npm install  # Install dependencies
 *   npx ts-node examples/sign_text.ts
 */

import { signText, SignatureVersion, getSignatureString, hammingDistanceHex, cosineFromHamming } from '../src';
import { ModelCache, OnnxProvider, OpenAIProvider } from '../src/providers';

async function onnxExample(): Promise<void> {
  console.log('=== ONNX Provider Example ===\n');

  try {
    // Initialize model cache and ONNX provider
    console.log('1. Initializing ONNX provider...');
    const cache = new ModelCache();
    const provider = await OnnxProvider.create(cache);
    console.log(`   Provider: ${provider.name()} (${provider.model()})\n`);

    // Example prompts
    const prompts = [
      'How do I reset my password?',
      'What is the meaning of life?',
      'Please help me with my account login issue',
    ];

    const results = [];
    for (let i = 0; i < prompts.length; i++) {
      const prompt = prompts[i];
      console.log(`${i + 1}. Signing prompt:`);
      console.log(`   "${prompt}"`);

      // Generate signature using signText()
      const result = await signText(prompt, provider, SignatureVersion.V1);

      const sigString = getSignatureString(result);
      console.log(`   Signature: ${sigString}`);
      console.log(`   Provider:  ${result.provider}`);
      console.log(`   Model:     ${result.model}`);
      console.log(`   Dimensions: ${result.dimensions}`);
      console.log(`   SHA256:    ${result.embeddingSha256}`);
      console.log(`   Timing:    ${result.timingMs.toFixed(2)}ms`);
      console.log();

      results.push(result);
    }

    // Compare two similar prompts
    console.log('=== Similarity Comparison ===\n');

    const promptA = 'How do I reset my password?';
    const promptB = 'Please help me with my account login issue';

    const resultA = results[0];
    const resultB = results[2];

    // Extract signatures for comparison
    const sigA = resultA.lsh.signatures[0].signature;
    const sigB = resultB.lsh.signatures[0].signature;

    // Compute similarity
    const distance = hammingDistanceHex(sigA, sigB);
    const similarity = cosineFromHamming(distance, 256);

    console.log(`Prompt A: "${promptA}"`);
    console.log(`Prompt B: "${promptB}"`);
    console.log();
    console.log(`Signature A: 0din-v1:${sigA}`);
    console.log(`Signature B: 0din-v1:${sigB}`);
    console.log();
    console.log(`Hamming distance: ${distance}/256 bits`);
    console.log(`Cosine similarity: ${similarity.toFixed(4)}`);
    console.log();

    if (similarity > 0.9) {
      console.log('✓ High similarity - likely duplicates');
    } else if (similarity > 0.7) {
      console.log('~ Moderate similarity - related topics');
    } else {
      console.log('✗ Low similarity - different topics');
    }

    // Clean up
    await provider.close();
  } catch (error) {
    console.error(`Error: ${error}`);
    console.log('\nMake sure the ONNX model files are present in the cache directory.');
    console.log('Install ONNX dependencies with: npm install onnxruntime-node');
  }
}

async function openaiExample(): Promise<void> {
  console.log('\n=== OpenAI Provider Example ===\n');

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.log('Skipping OpenAI example - OPENAI_API_KEY not set');
    return;
  }

  try {
    console.log('1. Initializing OpenAI provider...');
    const provider = new OpenAIProvider({ apiKey });
    console.log(`   Provider: ${provider.name()} (${provider.model()})\n`);

    // Generate signature
    const prompt = 'How do I reset my password?';
    console.log(`2. Signing prompt: "${prompt}"`);

    const result = await signText(prompt, provider, SignatureVersion.V0);

    const sigString = getSignatureString(result);
    console.log(`   Signature: ${sigString}`);
    console.log(`   Provider:  ${result.provider}`);
    console.log(`   Model:     ${result.model}`);
    console.log(`   Dimensions: ${result.dimensions}`);
    console.log(`   Tokens:    ${result.lsh.signatures[0].signature.substring(0, 16)}...`);
    console.log(`   Timing:    ${result.timingMs.toFixed(2)}ms`);
    console.log();

    // Clean up
    await provider.close();
  } catch (error) {
    console.error(`Error: ${error}`);
    console.log('\nMake sure you have installed the OpenAI package:');
    console.log('  npm install openai');
  }
}

async function main(): Promise<void> {
  await onnxExample();
  await openaiExample();
}

main().catch(console.error);
