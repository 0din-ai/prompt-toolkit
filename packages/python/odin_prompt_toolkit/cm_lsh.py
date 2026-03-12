"""Confidence Matrix LSH (CM-LSH) utilities.

This module provides an enhanced LSH implementation that combines:
- Standard LSH-TS (256-bit random hyperplane hash)
- ITQ (Iterative Quantization) for improved quantization
- Confidence matrix to weight reliable bits higher
- Isotonic calibration for accurate similarity estimates

The result is a DualHash with:
- hash_a: 512-bit signature (first 256 bits = LSH-TS compatible)
- hash_b: 512-bit confidence matrix
- Calibrated similarity function

Adapted from hybrid_cm_lsh/hybrid_cm_lsh.py to match signature_cli code style.
"""

import math
from dataclasses import dataclass, field

import numpy as np

# Re-use the existing LSH functions for consistency
from odin_prompt_toolkit.lsh import _sign_for, _splitmix64


@dataclass
class DualHash:
    """Result of CM-LSH hashing with confidence matrix."""

    hash_a: str  # 512-bit signature (128 hex chars, first 64 = LSH-TS compat)
    hash_b: str  # 512-bit confidence matrix (128 hex chars)
    bands: list[str] = field(default_factory=list)  # Band slices for LSH indexing
    bits: int = 512

    def lsh_ts_compat(self) -> str:
        """Get LSH-TS compatible 256-bit signature (first 64 hex chars)."""
        return self.hash_a[:64]


@dataclass
class ITQParams:
    """Parameters for Iterative Quantization transformation."""

    mean: np.ndarray  # Mean vector for centering
    pca: np.ndarray  # PCA projection matrix
    rotation: np.ndarray  # ITQ rotation matrix


@dataclass
class HybridParams:
    """Combined parameters for hybrid LSH+ITQ."""

    lsh_ts_hyperplanes: np.ndarray  # LSH-TS hyperplanes (256 x dims)
    itq: ITQParams  # ITQ parameters


class Calibrator:
    """Isotonic regression calibrator for similarity scores.

    Maps raw similarity scores to calibrated cosine similarity estimates
    using piecewise linear interpolation.
    """

    def __init__(
        self,
        x_thresh: np.ndarray,
        y_thresh: np.ndarray,
        x_min: float,
        x_max: float,
    ):
        """Initialize calibrator with thresholds.

        Args:
            x_thresh: X thresholds for piecewise linear function
            y_thresh: Y thresholds for piecewise linear function
            x_min: Minimum x value for clipping
            x_max: Maximum x value for clipping
        """
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh
        self.x_min = x_min
        self.x_max = x_max

    def predict(self, x: float | np.ndarray) -> float | np.ndarray:
        """Predict calibrated similarity from raw score.

        Args:
            x: Raw similarity score(s)

        Returns:
            Calibrated similarity score(s)
        """
        return np.interp(
            np.clip(x, self.x_min, self.x_max), self.x_thresh, self.y_thresh
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "x_thresh": self.x_thresh.tolist(),
            "y_thresh": self.y_thresh.tolist(),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        """Deserialize from dictionary."""
        return cls(
            x_thresh=np.array(data["x_thresh"], dtype=np.float32),
            y_thresh=np.array(data["y_thresh"], dtype=np.float32),
            x_min=data["x_min"],
            x_max=data["x_max"],
        )


class HybridCMLSH:
    """Hybrid Confidence Matrix LSH implementation.

    Combines LSH-TS and ITQ for improved similarity search:
    - LSH-TS: 256-bit random hyperplane hash (deterministic, backward compatible)
    - ITQ: 256-bit quantized projection (learned, rotation optimized)
    - Confidence: weights reliable bits higher in similarity computation
    - Calibration: maps raw scores to accurate cosine similarities
    """

    def __init__(
        self,
        params: HybridParams,
        calibrator: Calibrator,
        alpha: float = 0.65,
        family: int = 0,
    ):
        """Initialize CM-LSH with parameters.

        Args:
            params: Hybrid LSH+ITQ parameters
            calibrator: Isotonic calibrator for similarity scores
            alpha: Weight for confident bits (default: 0.65)
            family: LSH family index (default: 0)
        """
        self.params = params
        self.calibrator = calibrator
        self.alpha = alpha
        self.family = family

    def hash(self, embedding: list[float]) -> DualHash:
        """Generate CM-LSH hash from embedding.

        Args:
            embedding: Input embedding vector (will be L2-normalized)

        Returns:
            DualHash with hash_a (signature), hash_b (confidence), and bands
        """
        # Normalize embedding
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        return self._gen_hash(emb)

    def sim(self, h1: DualHash, h2: DualHash) -> float:
        """Compute calibrated similarity between two hashes.

        Args:
            h1: First hash
            h2: Second hash

        Returns:
            Calibrated cosine similarity estimate in [0, 1]
        """
        raw_sim = self._raw_sim(h1, h2)
        calibrated = self.calibrator.predict(np.array([raw_sim]))
        return (
            float(calibrated[0])
            if isinstance(calibrated, np.ndarray)
            else float(calibrated)
        )

    def cmp(self, e1: list[float], e2: list[float]) -> float:
        """Compare two embeddings via CM-LSH.

        Args:
            e1: First embedding
            e2: Second embedding

        Returns:
            Calibrated cosine similarity estimate
        """
        return self.sim(self.hash(e1), self.hash(e2))

    def is_dup(self, h1: DualHash, h2: DualHash, threshold: float = 0.85) -> bool:
        """Check if two hashes represent duplicates.

        Args:
            h1: First hash
            h2: Second hash
            threshold: Similarity threshold (default: 0.85)

        Returns:
            True if similarity >= threshold
        """
        return self.sim(h1, h2) >= threshold

    def verify_lsh_ts(self, embedding: list[float]) -> bool:
        """Verify that LSH-TS portion matches standalone LSH-TS hash.

        Args:
            embedding: Input embedding vector

        Returns:
            True if first 256 bits match standalone LSH-TS hash
        """
        h = self.hash(embedding)

        # Normalize embedding
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        # Generate standalone LSH-TS hash
        lsh_ts = _lsh_ts_hash(emb, self.family, 256)

        return h.lsh_ts_compat() == lsh_ts

    def _gen_hash(self, emb: np.ndarray) -> DualHash:
        """Generate hash from normalized embedding.

        Args:
            emb: Normalized embedding vector

        Returns:
            DualHash with signature and confidence matrix
        """
        # 1. LSH-TS projection (256 bits)
        p1 = emb @ self.params.lsh_ts_hyperplanes.T

        # 2. ITQ projection (256 bits)
        centered = emb - self.params.itq.mean
        pca_proj = centered @ self.params.itq.pca.T
        p2 = pca_proj @ self.params.itq.rotation.T

        # 3. Combine projections (512 bits total)
        proj = np.concatenate([p1, p2])

        # 4. Sign bits (hash_a)
        signs = proj > 0

        # 5. Confidence bits (hash_b)
        # Use 45th percentile as threshold
        conf_thresh = np.percentile(np.abs(proj), 45)
        conf_bits = np.abs(proj) > conf_thresh

        # 6. Pack into hex strings
        hash_a = _pack_bits(signs)
        hash_b = _pack_bits(conf_bits)

        # 7. Split into bands (64 bands for LSH indexing)
        band_len = len(hash_a) // 64
        bands = [hash_a[i : i + band_len] for i in range(0, len(hash_a), band_len)][:64]

        return DualHash(hash_a=hash_a, hash_b=hash_b, bands=bands, bits=512)

    def _raw_sim(self, h1: DualHash, h2: DualHash) -> float:
        """Compute raw similarity before calibration.

        Args:
            h1: First hash
            h2: Second hash

        Returns:
            Raw similarity score
        """
        a1 = _unpack_bits(h1.hash_a)
        a2 = _unpack_bits(h2.hash_a)
        b1 = _unpack_bits(h1.hash_b)
        b2 = _unpack_bits(h2.hash_b)

        # Compute agreement and confidence overlap
        agree = a1 == a2
        both_conf = b1 & b2

        # Weighted similarity: alpha * (confident agreement) + (1-alpha) * (overall agreement)
        if both_conf.any():
            conf_agree_rate = agree[both_conf].mean()
            overall_agree_rate = agree.mean()
            return self.alpha * conf_agree_rate + (1 - self.alpha) * overall_agree_rate
        else:
            return agree.mean()


def gen_hyperplanes(family: int, bits: int, dims: int) -> np.ndarray:
    """Generate deterministic random hyperplanes for LSH.

    Args:
        family: Hash family index
        bits: Number of bits (hyperplanes)
        dims: Dimensionality of input vectors

    Returns:
        Matrix of shape (bits, dims) with +1/-1 entries
    """
    hp = np.zeros((bits, dims), dtype=np.float32)
    for b in range(bits):
        for d in range(dims):
            hp[b, d] = float(_sign_for(family, b, d))
    return hp


def _lsh_ts_hash(emb: np.ndarray, family: int, bits: int) -> str:
    """Generate standalone LSH-TS hash (for verification).

    Args:
        emb: Normalized embedding vector
        family: Hash family index
        bits: Number of bits

    Returns:
        Hex string of length bits/4
    """
    bool_bits = []
    for b in range(bits):
        dot = sum(emb[d] * _sign_for(family, b, d) for d in range(len(emb)))
        bool_bits.append(dot > 0)

    return _pack_bits(np.array(bool_bits))


def _pack_bits(bits: np.ndarray) -> str:
    """Pack boolean array into hex string.

    Args:
        bits: Boolean array

    Returns:
        Hex string (4 bits per character)
    """
    hex_chars = []
    for i in range(0, len(bits), 4):
        n = (
            (8 if bits[i] else 0)
            + (4 if bits[i + 1] else 0)
            + (2 if bits[i + 2] else 0)
            + (1 if bits[i + 3] else 0)
        )
        hex_chars.append(format(n, "x"))
    return "".join(hex_chars)


def _unpack_bits(hex_str: str) -> np.ndarray:
    """Unpack hex string into boolean array.

    Args:
        hex_str: Hex string

    Returns:
        Boolean array
    """
    bits = []
    for c in hex_str:
        n = int(c, 16)
        bits.extend([n & 8 != 0, n & 4 != 0, n & 2 != 0, n & 1 != 0])
    return np.array(bits, dtype=bool)


# ============================================================================
# Training functions (for completeness - not used in CLI)
# ============================================================================


def train_itq(embeddings: list[list[float]], bits: int, iterations: int) -> ITQParams:
    """Train ITQ parameters from embeddings.

    Args:
        embeddings: List of embedding vectors
        bits: Number of output bits
        iterations: Number of ITQ iterations

    Returns:
        Trained ITQ parameters
    """
    embs = np.array(embeddings, dtype=np.float32)

    # Center embeddings
    mean = embs.mean(axis=0)
    centered = embs - mean

    # PCA
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:bits]
    pca = eigvecs[:, idx].T

    # Project to PCA space
    proj = centered @ pca.T

    # Iterative quantization
    rotation = np.eye(bits, dtype=np.float32)
    for _ in range(iterations):
        # Quantize with current rotation
        rotated = proj @ rotation.T
        quantized = np.sign(rotated)

        # Update rotation via SVD
        u, _, vt = np.linalg.svd(quantized.T @ proj)
        rotation = (vt.T @ u.T).astype(np.float32)

    return ITQParams(
        mean=mean.astype(np.float32), pca=pca.astype(np.float32), rotation=rotation
    )


def create_default_cm_lsh(dimensions: int, family: int = 0) -> HybridCMLSH:
    """Create a default CM-LSH instance without training.

    This creates a minimal CM-LSH configuration using:
    - Random hyperplanes for LSH-TS (deterministic based on family)
    - Identity transformations for ITQ (no learned rotation)
    - Linear calibrator (no adjustment)

    Note: Always produces 512-bit output (256 from LSH-TS + 256 from ITQ).
    For dimensions < 256, ITQ output is padded with zeros.

    Args:
        dimensions: Embedding dimensionality
        family: LSH family index (default: 0)

    Returns:
        HybridCMLSH instance with default parameters
    """
    # Generate LSH-TS hyperplanes (always 256 bits)
    lsh_hp = gen_hyperplanes(family, 256, dimensions)

    # Create identity ITQ parameters (always 256 bits output)
    # If dimensions < 256, we'll pad the output
    itq_dims = min(256, dimensions)
    mean = np.zeros(dimensions, dtype=np.float32)

    # PCA: identity for first itq_dims dimensions
    # Output is always 256 dimensions (padded if needed)
    pca = np.zeros((256, dimensions), dtype=np.float32)
    for i in range(itq_dims):
        if i < dimensions:
            pca[i, i] = 1.0

    rotation = np.eye(256, dtype=np.float32)

    itq = ITQParams(mean=mean, pca=pca, rotation=rotation)

    # Create params
    params = HybridParams(lsh_ts_hyperplanes=lsh_hp, itq=itq)

    # Create linear calibrator (identity function)
    calibrator = Calibrator(
        x_thresh=np.array([0.0, 1.0], dtype=np.float32),
        y_thresh=np.array([0.0, 1.0], dtype=np.float32),
        x_min=0.0,
        x_max=1.0,
    )

    return HybridCMLSH(params=params, calibrator=calibrator, alpha=0.65, family=family)
