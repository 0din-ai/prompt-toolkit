"""Type definitions and utilities for signature generation."""

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lsh import LSHFamily


class SignatureVersion(str, Enum):
    """Signature version enumeration.

    Each version corresponds to a specific embedding model and dimensionality.
    V0 and V1 signatures are NOT comparable due to different embedding spaces.
    """

    V0 = "v0"  # OpenAI text-embedding-3-large (1536 dims)
    V1 = "v1"  # multilingual-e5-large ONNX (1024 dims)
    LATEST = "latest"  # Resolves to V1

    def resolve(self) -> "SignatureVersion":
        """Resolve 'latest' to the current version."""
        if self == SignatureVersion.LATEST:
            return SignatureVersion.V1
        return self

    def embedding_dimensions(self) -> int:
        """Get expected embedding dimensions for this version."""
        resolved = self.resolve()
        if resolved == SignatureVersion.V0:
            return 1536
        elif resolved == SignatureVersion.V1:
            return 1024
        raise ValueError(f"Unknown version: {resolved}")

    def to_algorithm(self) -> "HashAlgorithm":
        """Get hash algorithm for this version."""
        resolved = self.resolve()
        if resolved == SignatureVersion.V0:
            return HashAlgorithm.OPENAI
        elif resolved == SignatureVersion.V1:
            return HashAlgorithm.ONNX
        raise ValueError(f"Unknown version: {resolved}")

    @staticmethod
    def from_algorithm(algorithm: "HashAlgorithm") -> "SignatureVersion":
        """Get version from hash algorithm."""
        if algorithm == HashAlgorithm.OPENAI:
            return SignatureVersion.V0
        elif algorithm == HashAlgorithm.ONNX:
            return SignatureVersion.V1
        raise ValueError(f"Unknown algorithm: {algorithm}")


class HashAlgorithm(str, Enum):
    """Hash algorithm enumeration."""

    LSH = "lsh"  # Generic LSH (used with any embedding)
    OPENAI = "openai"  # OpenAI embeddings (V0, 1536 dims)
    ONNX = "onnx"  # ONNX local embeddings (V1, 1024 dims)


@dataclass
class LshConfig:
    """LSH configuration parameters."""

    families: int = 3  # Number of independent hash families
    bits: int = 256  # Number of bits per signature
    bands: int = 16  # Number of bands for LSH indexing


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embedding: list[float]
    normalized_embedding: list[float]
    normalized_embedding_sha256: str
    model: str
    dimensions: int
    token_count: int = 0
    timing_ms: float | None = None


@dataclass
class ParsedSignature:
    """Parsed signature string."""

    version: SignatureVersion
    signature: str  # Hex string (family 0 only for V0/V1)


@dataclass
class LshOutput:
    """LSH computation output."""

    config: LshConfig
    signatures: list["LSHFamily"]


@dataclass
class SignatureResult:
    """Complete signature generation result.

    This contains all metadata from the signature generation process,
    including the primary signature string, provider info, embedding hash,
    and LSH families.
    """

    signature: str  # Formatted signature string (e.g., "0din-v1:...")
    version: SignatureVersion
    prompt_preview: str
    prompt_length: int
    provider: str
    model: str
    dimensions: int
    embedding_sha256: str
    lsh: LshOutput
    timing_ms: float | None = None

    @property
    def signature_string(self) -> str:
        """Get the formatted signature string.

        Returns:
            Signature in 0din format (e.g., "0din-v1:8d000000ac854dae...")
        """
        resolved_version = self.version.resolve()
        primary_sig = self.lsh.signatures[0].signature
        return f"0din-{resolved_version.value}:{primary_sig}"


def signature_string(version: SignatureVersion, signature: str) -> str:
    """Format signature as versioned string.

    Args:
        version: Signature version (v0, v1, etc.)
        signature: Hex-encoded signature string

    Returns:
        Formatted string like "0din-v0:deadbeef..."
    """
    resolved = version.resolve()
    return f"0din-{resolved.value}:{signature}"


def parse_signature_string(s: str) -> ParsedSignature:
    """Parse versioned signature string.

    Args:
        s: Signature string like "0din-v0:deadbeef..."

    Returns:
        ParsedSignature with version and signature

    Raises:
        InvalidInputError: If format is invalid or version unsupported
    """
    from signature_sdk.error import InvalidInputError

    if not s.startswith("0din-"):
        raise InvalidInputError(f"Invalid signature prefix: {s}")

    parts = s.split(":", 1)
    if len(parts) != 2:
        raise InvalidInputError(f"Invalid signature format: {s}")

    version_str = parts[0][5:]  # Remove "0din-" prefix
    signature = parts[1]

    # Validate version
    try:
        version = SignatureVersion(version_str)
    except ValueError:
        raise InvalidInputError(f"Unsupported signature version: {version_str}")

    # Validate hex signature
    if not re.match(r"^[0-9a-f]+$", signature):
        raise InvalidInputError(f"Invalid hex signature: {signature}")

    return ParsedSignature(version=version, signature=signature)


def _compute_embedding_sha256_python(normalized_embedding: list[float]) -> str:
    """Compute SHA256 hash of normalized embedding (pure Python implementation).

    This implementation matches the canonical specification:
    1. Quantize each value to 6 decimal places: round(x * 1e6) / 1e6
    2. Serialize as JSON array: [0.001234, 0.005678, ...]
       - Space after comma
       - Whole numbers must include .0 (e.g., 1.0 not 1)
       - Preserve sign for negative zero (-0.0)
    3. Hash the JSON string representation

    The 6-decimal quantization eliminates floating-point jitter from:
    - OpenAI API non-determinism (different servers/GPUs)
    - Cross-platform float representation differences
    - Numerical precision variations

    Args:
        normalized_embedding: L2-normalized embedding vector

    Returns:
        Hex string of SHA256 hash
    """
    import math

    # Quantize to 6 decimals
    quantized = [round(x * 1_000_000) / 1_000_000 for x in normalized_embedding]

    # Format as JSON with specific rules
    json_parts = []
    for i, x in enumerate(quantized):
        # Check for negative zero (preserve sign from original)
        if x == 0.0 and math.copysign(1, normalized_embedding[i]) == -1:
            s = "-0.0"
        else:
            s = str(x)
            # Ensure whole numbers have .0
            if x == int(x) and "." not in s:
                s = f"{s}.0"
        json_parts.append(s)

    json_str = f"[{', '.join(json_parts)}]"

    return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class PromptInfo:
    """Metadata about a prompt in a comparison."""

    preview: str
    length: int
    signature: str


@dataclass
class QualityStats:
    """Quality metrics for a signature comparison."""

    absolute_error: float
    signed_error: float
    squared_error: float
    quality_rating: str


@dataclass
class ComparisonResult:
    """Result of comparing two signatures.

    Contains metadata about both prompts, distance metrics, and optional
    quality statistics.
    """

    prompt_a: PromptInfo
    prompt_b: PromptInfo
    hamming_distance: int
    cosine_similarity: float
    lsh_config: LshConfig
    quality_stats: QualityStats | None = None
    timing_ms: float | None = None


# Transparent native acceleration
# Try to use native implementation, fall back to pure Python
from signature_sdk._accel import NATIVE_AVAILABLE

if NATIVE_AVAILABLE:
    from signature_sdk._accel import compute_embedding_sha256
else:
    compute_embedding_sha256 = _compute_embedding_sha256_python
