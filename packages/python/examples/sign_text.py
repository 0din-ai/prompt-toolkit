#!/usr/bin/env python3
"""Example demonstrating the high-level sign_text() API.

This example shows how to use the convenience function that takes a text prompt
and returns a complete signature result in one call.

Note: This example requires the ONNX model files to be downloaded and cached.
The OpenAI example requires an API key set in the OPENAI_API_KEY environment variable.
"""

import asyncio
import os
import sys

# Add parent directory to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from odin_sig import (
    cosine_from_hamming,
    hamming_distance_hex,
    sign_text,
    SignatureVersion,
)


async def onnx_example():
    """Example using ONNX provider (local, no API key needed)."""
    print("=== ONNX Provider Example ===\n")

    try:
        from odin_sig.providers import ModelCache, OnnxProvider
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall ONNX dependencies with:")
        print("  pip install '0din-sig[onnx]'")
        return

    # Initialize model cache and ONNX provider
    print("1. Initializing ONNX provider...")
    cache = ModelCache()
    try:
        provider = await OnnxProvider.new(cache)
        print(f"   Provider: {provider.name()} ({provider.model()})\n")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        print("\n   Please download the model manually first.")
        return

    # Example prompts
    prompts = [
        "How do I reset my password?",
        "What is the meaning of life?",
        "Please help me with my account login issue",
    ]

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. Signing prompt:")
        print(f'   "{prompt}"')

        # Generate signature using sign_text()
        result = await sign_text(prompt, provider, SignatureVersion.V1, None)

        print(f"   Signature: {result.signature_string}")
        print(f"   Provider:  {result.provider}")
        print(f"   Model:     {result.model}")
        print(f"   Dimensions: {result.dimensions}")
        print(f"   SHA256:    {result.embedding_sha256}")
        print(f"   Timing:    {result.timing_ms:.2f}ms")
        print()

        results.append(result)

    # Compare two similar prompts
    print("=== Similarity Comparison ===\n")

    prompt_a = "How do I reset my password?"
    prompt_b = "Please help me with my account login issue"

    result_a = results[0]
    result_b = results[2]

    # Extract signatures for comparison
    sig_a = result_a.lsh.signatures[0].signature
    sig_b = result_b.lsh.signatures[0].signature

    # Compute similarity
    distance = hamming_distance_hex(sig_a, sig_b)
    similarity = cosine_from_hamming(distance, 256)

    print(f'Prompt A: "{prompt_a}"')
    print(f'Prompt B: "{prompt_b}"')
    print()
    print(f"Signature A: 0din-v1:{sig_a}")
    print(f"Signature B: 0din-v1:{sig_b}")
    print()
    print(f"Hamming distance: {distance}/256 bits")
    print(f"Cosine similarity: {similarity:.4f}")
    print()

    if similarity > 0.9:
        print("✓ High similarity - likely duplicates")
    elif similarity > 0.7:
        print("~ Moderate similarity - related topics")
    else:
        print("✗ Low similarity - different topics")

    # Clean up
    await provider.close()


async def openai_example():
    """Example using OpenAI provider (requires API key)."""
    print("\n=== OpenAI Provider Example ===\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping OpenAI example - OPENAI_API_KEY not set")
        return

    try:
        from odin_sig.providers import OpenAIProvider
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall OpenAI dependencies with:")
        print("  pip install '0din-sig[openai]'")
        return

    print("1. Initializing OpenAI provider...")
    provider = OpenAIProvider(api_key=api_key)
    print(f"   Provider: {provider.name()} ({provider.model()})\n")

    # Generate signature
    prompt = "How do I reset my password?"
    print(f'2. Signing prompt: "{prompt}"')

    result = await sign_text(prompt, provider, SignatureVersion.V0, None)

    print(f"   Signature: {result.signature_string}")
    print(f"   Provider:  {result.provider}")
    print(f"   Model:     {result.model}")
    print(f"   Dimensions: {result.dimensions}")
    print(f"   Tokens:    {result.lsh.signatures[0].signature[:16]}...")
    print(f"   Timing:    {result.timing_ms:.2f}ms")
    print()

    # Clean up
    await provider.close()


async def main():
    """Run examples."""
    await onnx_example()
    await openai_example()


if __name__ == "__main__":
    asyncio.run(main())
