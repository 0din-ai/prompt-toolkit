"""Hash algorithm implementations."""

from odin_prompt_toolkit.hasher import Hasher
from odin_prompt_toolkit.hashers.lsh import SimHashLsh
from odin_prompt_toolkit.types import HashAlgorithm


def get_hasher(algorithm: HashAlgorithm) -> Hasher:
    """Get a hasher instance by algorithm.

    Args:
        algorithm: Hash algorithm to use

    Returns:
        Hasher instance for the specified algorithm

    Raises:
        ValueError: If the algorithm is not supported

    Example:
        >>> from odin_prompt_toolkit import get_hasher, HashAlgorithm
        >>> hasher = get_hasher(HashAlgorithm.LSH)
        >>> print(hasher.name())
        'lsh'
    """
    if algorithm == HashAlgorithm.LSH:
        return SimHashLsh()
    raise ValueError(f"Unknown algorithm: {algorithm}")


__all__ = ["get_hasher", "SimHashLsh"]
