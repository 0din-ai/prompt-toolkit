"""Abstract hasher interface for hash algorithm implementations."""

from typing import Protocol

from odin_sig.lsh import LSHFamily
from odin_sig.types import LshConfig


class Hasher(Protocol):
    """Abstract interface for hash algorithm implementations.

    Each hasher takes a normalized embedding vector and LSH configuration,
    and produces LSH signatures suitable for similarity matching.

    Example:
        >>> from odin_sig import get_hasher, HashAlgorithm, LshConfig
        >>> from odin_sig.lsh import normalize_vector
        >>>
        >>> hasher = get_hasher(HashAlgorithm.LSH)
        >>> vector = normalize_vector([1.0, 2.0, 3.0])
        >>> config = LshConfig(families=3, bits=256, bands=16)
        >>> families = hasher.compute(vector, config)
    """

    def name(self) -> str:
        """Algorithm name (e.g., 'lsh', 'cm-lsh').

        Returns:
            Algorithm identifier string
        """
        ...

    def compute(self, embedding: list[float], config: LshConfig) -> list[LSHFamily]:
        """Compute LSH signatures from a normalized embedding vector.

        Args:
            embedding: L2-normalized embedding vector
            config: LSH configuration parameters

        Returns:
            List of LSH families, one per family index
        """
        ...


__all__ = ["Hasher"]
