"""Hash algorithm implementations."""

from signature_sdk.hashers.lsh import SimHashLsh
from signature_sdk.hasher import Hasher
from signature_sdk.types import HashAlgorithm


def get_hasher(algorithm: HashAlgorithm) -> Hasher:
    """Get a hasher instance by algorithm.

    Args:
        algorithm: Hash algorithm to use

    Returns:
        Hasher instance for the specified algorithm

    Raises:
        ValueError: If the algorithm is not supported

    Example:
        >>> from signature_sdk import get_hasher, HashAlgorithm
        >>> hasher = get_hasher(HashAlgorithm.LSH)
        >>> print(hasher.name())
        'lsh'
    """
    if algorithm == HashAlgorithm.LSH:
        return SimHashLsh()
    raise ValueError(f"Unknown algorithm: {algorithm}")


__all__ = ["get_hasher", "SimHashLsh"]
