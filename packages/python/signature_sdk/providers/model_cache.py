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

    DEFAULT_CACHE_DIR = "~/.cache/odin-sig/models"
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

        # Check for required files
        required_files = [
            "onnx/model.onnx",
            "tokenizer.json",
            "config.json",
        ]

        return all((model_dir / f).exists() for f in required_files)

    def get_model_path(self, version: str = "v1") -> Path:
        """Get the path to the ONNX model file.

        Args:
            version: Model version (default: "v1")

        Returns:
            Path to the ONNX model file
        """
        return self.model_directory(version) / "onnx" / "model.onnx"

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
