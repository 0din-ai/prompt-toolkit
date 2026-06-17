"""Model cache for downloading and caching ONNX models."""

import json
import os
from pathlib import Path
from typing import Optional


class ModelCache:
    """Manages local caching of ONNX models.

    The cache directory defaults to ~/.cache/signature-sdk/models/v1/ but can be
    overridden via the SIGNATURE_SDK_MODEL_CACHE environment variable.

    Args:
        cache_dir: Optional custom cache directory path

    Example:
        >>> cache = ModelCache()
        >>> model_dir = cache.model_directory("v1")
        >>> print(f"Models cached at: {model_dir}")
    """

    DEFAULT_CACHE_DIR = "~/.cache/signature-sdk/models"
    ENV_VAR = "SIGNATURE_SDK_MODEL_CACHE"

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model cache.

        Args:
            cache_dir: Optional custom cache directory path
        """
        if cache_dir:
            self._cache_dir = Path(cache_dir).expanduser()
        elif self.ENV_VAR in os.environ:
            self._cache_dir = Path(os.environ[self.ENV_VAR]).expanduser()
        else:
            self._cache_dir = Path(self.DEFAULT_CACHE_DIR).expanduser()

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self._cache_dir

    def model_directory(self, version: str = "v1") -> Path:
        """Get the directory for a specific model version.

        Args:
            version: Model version (default: "v1")

        Returns:
            Path to the model version directory
        """
        return self._cache_dir / version

    def ensure_model_directory(self, version: str = "v1") -> Path:
        """Ensure the model directory exists.

        Args:
            version: Model version (default: "v1")

        Returns:
            Path to the model version directory
        """
        model_dir = self.model_directory(version)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def has_model(self, version: str = "v1") -> bool:
        """Check if a model version is cached locally.

        Args:
            version: Model version (default: "v1")

        Returns:
            True if the model is cached, False otherwise
        """
        model_dir = self.model_directory(version)
        if not model_dir.exists():
            return False

        # Check for required files (prefer optimized model, accept either)
        has_optimized = (model_dir / "onnx" / "model_O4.onnx").exists()
        has_unoptimized = (model_dir / "onnx" / "model.onnx").exists()

        required_files = [
            "tokenizer.json",
            "config.json",
        ]

        return (has_optimized or has_unoptimized) and all(
            (model_dir / f).exists() for f in required_files
        )

    def get_model_path(self, version: str = "v1") -> Path:
        """Get the path to the ONNX model file.

        Prefers the optimized model (model_O4.onnx) if available,
        falls back to the unoptimized model (model.onnx).

        Args:
            version: Model version (default: "v1")

        Returns:
            Path to the ONNX model file
        """
        model_dir = self.model_directory(version)
        optimized_path = model_dir / "onnx" / "model_O4.onnx"
        unoptimized_path = model_dir / "onnx" / "model.onnx"

        # Prefer optimized model (smaller, faster inference)
        if optimized_path.exists():
            return optimized_path
        return unoptimized_path

    def get_tokenizer_path(self, version: str = "v1") -> Path:
        """Get the path to the tokenizer file.

        Args:
            version: Model version (default: "v1")

        Returns:
            Path to the tokenizer JSON file
        """
        return self.model_directory(version) / "tokenizer.json"

    def get_config_path(self, version: str = "v1") -> Path:
        """Get the path to the model config file.

        Args:
            version: Model version (default: "v1")

        Returns:
            Path to the config JSON file
        """
        return self.model_directory(version) / "config.json"

    def load_config(self, version: str = "v1") -> dict:
        """Load the model configuration.

        Args:
            version: Model version (default: "v1")

        Returns:
            Model configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.get_config_path(version)
        with open(config_path) as f:
            return json.load(f)


# --- SusFactor model-cache helpers -----------------------------------------
#
# The SusFactor classifier uses a different on-disk layout from the ONNX
# embedding models (a HuggingFace ``encoder/`` directory plus a separate
# ``head.pt``), so it gets dedicated helpers rather than overloading the
# ONNX-specific ModelCache methods.

# Files required for a usable SusFactor model, relative to the version dir.
SUSFACTOR_REQUIRED_FILES = (
    "encoder/config.json",
    "encoder/model.safetensors",
    "encoder/tokenizer.json",
    "head.pt",
)

# Files required for a usable SusFactor ONNX model, relative to the version dir.
# The ONNX graph bakes encoder + mean-pool + head into a single model.onnx file.
# Tokenizer files live at the version root (same layout as the ONNX embedding models).
# The model is published to 0dinai/susfactor-e5-large-onnx on HuggingFace.
SUSFACTOR_ONNX_REQUIRED_FILES = (
    "onnx/model.onnx",
    "tokenizer.json",
)

HF_URL_SUSFACTOR_ONNX = "https://huggingface.co/0dinai/susfactor-e5-large-onnx"


def susfactor_model_dir(cache: ModelCache, version: str = "susfactor-v1") -> Path:
    """Return the cache directory for a SusFactor model version."""
    return cache.model_directory(version)


def susfactor_model_files_present(cache: ModelCache, version: str = "susfactor-v1") -> bool:
    """Check whether all required SusFactor model files are cached locally."""
    model_dir = susfactor_model_dir(cache, version)
    if not model_dir.exists():
        return False
    return all((model_dir / rel).exists() for rel in SUSFACTOR_REQUIRED_FILES)


def susfactor_onnx_files_present(cache: ModelCache, version: str = "susfactor-v1") -> bool:
    """Check whether the SusFactor ONNX model files are cached locally.

    Only ``onnx/model.onnx`` and ``tokenizer.json`` are required::

        <version>/
            onnx/
                model.onnx        # required: full graph (encoder + mean-pool + MLP head)
                model.onnx_data   # optional: companion weights for large exports;
                                  #           ONNX Runtime loads it automatically
            tokenizer.json        # required

    Additional tokenizer files (``tokenizer_config.json``,
    ``special_tokens_map.json``, ``sentencepiece.bpe.model``) are loaded on
    demand by the HuggingFace tokenizer but are not checked here.
    """
    model_dir = susfactor_model_dir(cache, version)
    if not model_dir.exists():
        return False
    return all((model_dir / rel).exists() for rel in SUSFACTOR_ONNX_REQUIRED_FILES)


def susfactor_onnx_model_path(cache: ModelCache, version: str = "susfactor-v1") -> Path:
    """Return the path to the SusFactor ONNX model file."""
    return susfactor_model_dir(cache, version) / "onnx" / "model.onnx"
