"""Transparent native acceleration layer.

This module attempts to import the native Rust extension (odin_prompt_toolkit_native)
and exposes its functions. If the native extension is not available, it
sets NATIVE_AVAILABLE to False and the pure Python implementations are used.

The native extension can be installed via:
    pip install odin-prompt-toolkit-native

Or by building from source:
    cd packages/python-native
    maturin develop --release

To force pure-Python mode even when the native extension is installed:
    export ODIN_PROMPT_TOOLKIT_NO_NATIVE=1
"""

import os

# Check if user wants to force pure-Python mode
_force_no_native = os.environ.get("ODIN_PROMPT_TOOLKIT_NO_NATIVE", "").lower() in (
    "1",
    "true",
    "yes",
)

# Try to import native extension (unless explicitly disabled)
if _force_no_native:
    NATIVE_AVAILABLE = False

    # Fallback to None - pure Python implementations will be used
    NativeLshFamily = None
    simhash_lsh_multi = None
    normalize_vector = None
    hamming_distance_hex = None
    cosine_from_hamming = None
    compute_embedding_sha256 = None

else:
    try:
        from odin_prompt_toolkit_native import (
            LshFamily as _NativeLshFamily,
            compute_embedding_sha256 as _native_compute_embedding_sha256,
            cosine_from_hamming as _native_cosine_from_hamming,
            hamming_distance_hex as _native_hamming_distance_hex,
            normalize_vector as _native_normalize_vector,
            simhash_lsh_multi as _native_simhash_lsh_multi,
        )

        NATIVE_AVAILABLE = True

        # The native LshFamily is a PyO3 pyclass, but it's duck-type compatible
        # with the Python dataclass (has same attributes: family, bits, signature, bands)
        NativeLshFamily = _NativeLshFamily

        # Export native functions directly
        simhash_lsh_multi = _native_simhash_lsh_multi
        normalize_vector = _native_normalize_vector
        hamming_distance_hex = _native_hamming_distance_hex
        cosine_from_hamming = _native_cosine_from_hamming
        compute_embedding_sha256 = _native_compute_embedding_sha256

    except ImportError:
        NATIVE_AVAILABLE = False

        # Fallback to None - pure Python implementations will be used
        NativeLshFamily = None
        simhash_lsh_multi = None
        normalize_vector = None
        hamming_distance_hex = None
        cosine_from_hamming = None
        compute_embedding_sha256 = None


__all__ = [
    "NATIVE_AVAILABLE",
    "NativeLshFamily",
    "simhash_lsh_multi",
    "normalize_vector",
    "hamming_distance_hex",
    "cosine_from_hamming",
    "compute_embedding_sha256",
]
