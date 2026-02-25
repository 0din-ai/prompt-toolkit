"""0din-sig: Multi-language SDK for LSH signature generation.

This package provides locality-sensitive hashing (LSH) for AI prompt similarity
detection, with support for both standard LSH and Confidence Matrix LSH (CM-LSH).

Quick Start (High-Level API):
    >>> from odin_sig import sign_text, SignatureVersion
    >>> from odin_sig.providers import ModelCache, OnnxProvider
    >>>
    >>> # Initialize ONNX provider (local, no API key needed)
    >>> cache = ModelCache()
    >>> provider = await OnnxProvider.new(cache)
    >>>
    >>> # Generate signature from text
    >>> result = await sign_text(
    ...     "How do I reset my password?",
    ...     provider,
    ...     SignatureVersion.V1,
    ...     None,
    ... )
    >>> print(result.signature_string)
    0din-v1:8d000000ac854dae...

Quick Start (Low-Level API):
    >>> from odin_sig import simhash_lsh_multi, normalize_vector
    >>>
    >>> vector = [0.5, 0.5, 0.5, 0.5]
    >>> normalized = normalize_vector(vector)
    >>> families = simhash_lsh_multi(normalized)
    >>> print(families[0].signature)

Signature Versions:
    - V0: OpenAI text-embedding-3-large (1536 dimensions, API-based)
    - V1: multilingual-e5-small ONNX (384 dimensions, local)
    - Latest: Resolves to V1

Algorithm:
    SimHash via Random Hyperplane LSH (Charikar 2002):
    - Deterministic hyperplanes via SplitMix64 PRNG
    - Default: 3 families × 256 bits × 16 bands
    - Hex-encoded signatures (64 hex chars = 256 bits)
    - Hamming distance → cosine similarity via cos(π × d/n)
"""

from odin_sig.lsh import (
    LSHFamily,
    cosine_from_hamming,
    hamming_distance_hex,
    normalize_vector,
    simhash_lsh_multi,
)
from odin_sig.provider import EmbeddingProvider
from odin_sig.sign import sign_text
from odin_sig.types import (
    EmbeddingResult,
    HashAlgorithm,
    LshConfig,
    LshOutput,
    ParsedSignature,
    SignatureResult,
    SignatureVersion,
    compute_embedding_sha256,
    parse_signature_string,
    signature_string,
)

__version__ = "0.1.1"

__all__ = [
    # High-level API
    "sign_text",
    "EmbeddingProvider",
    "SignatureResult",
    # Core LSH functions
    "simhash_lsh_multi",
    "normalize_vector",
    "hamming_distance_hex",
    "cosine_from_hamming",
    "LSHFamily",
    # Types
    "SignatureVersion",
    "HashAlgorithm",
    "LshConfig",
    "LshOutput",
    "EmbeddingResult",
    "ParsedSignature",
    # Signature utilities
    "signature_string",
    "parse_signature_string",
    "compute_embedding_sha256",
]
