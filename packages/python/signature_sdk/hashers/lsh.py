"""SimHash LSH implementation."""

from signature_sdk.lsh import LSHFamily, simhash_lsh_multi
from signature_sdk.types import LshConfig


class SimHashLsh:
    """SimHash LSH implementation (current default algorithm).

    Uses deterministic random hyperplane LSH with SplitMix64-based
    hyperplane generation. This is the canonical LSH implementation
    used for V0 and V1 signatures.

    Example:
        >>> from signature_sdk.hashers import SimHashLsh
        >>> from signature_sdk.types import LshConfig
        >>> from signature_sdk.lsh import normalize_vector
        >>>
        >>> hasher = SimHashLsh()
        >>> vector = normalize_vector([1.0, 2.0, 3.0])
        >>> config = LshConfig(families=3, bits=256, bands=16)
        >>> families = hasher.compute(vector, config)
        >>> print(hasher.name())
        'lsh'
    """

    def name(self) -> str:
        """Return algorithm name.

        Returns:
            'lsh'
        """
        return "lsh"

    def compute(self, embedding: list[float], config: LshConfig) -> list[LSHFamily]:
        """Compute LSH signatures from normalized embedding.

        Delegates to the simhash_lsh_multi function with the provided
        configuration parameters.

        Args:
            embedding: L2-normalized embedding vector
            config: LSH configuration parameters

        Returns:
            List of LSH families, one per family index
        """
        return simhash_lsh_multi(
            embedding,
            families=config.families,
            bits=config.bits,
            bands=config.bands,
        )


__all__ = ["SimHashLsh"]
