"""Error types for odin-prompt-toolkit operations."""


class SigError(Exception):
    """Base exception for odin-prompt-toolkit operations.

    All exceptions raised by the odin-prompt-toolkit library inherit from this base class,
    making it easy to catch all library-specific errors.

    Example:
        >>> try:
        ...     # some odin-prompt-toolkit operation
        ...     pass
        ... except SigError as e:
        ...     print(f"Signature operation failed: {e}")
    """


class ConfigError(SigError):
    """Configuration error.

    Raised when LSH configuration parameters are invalid or incompatible.

    Example:
        >>> from odin_prompt_toolkit import LshConfig
        >>> # Invalid configuration would raise ConfigError
    """


class ProviderError(SigError):
    """Embedding provider error.

    Raised when an embedding provider fails to generate embeddings,
    such as API failures, authentication errors, or network issues.

    Example:
        >>> # API call failure would raise ProviderError
    """


class ModelError(SigError):
    """Model loading or inference error.

    Raised when ONNX model loading fails or inference encounters an error.

    Example:
        >>> # Model file not found would raise ModelError
    """


class InvalidInputError(SigError):
    """Invalid input data.

    Raised when input data doesn't meet requirements, such as empty text,
    invalid embedding dimensions, or malformed signature strings.

    Example:
        >>> from odin_prompt_toolkit import parse_signature_string
        >>> try:
        ...     parse_signature_string("invalid")
        ... except InvalidInputError as e:
        ...     print(f"Invalid signature: {e}")
    """


class ThreatFeedError(SigError):
    """Base exception for threat feed operations.

    Raised for any error related to fetching, caching, or querying
    the 0din threat feed.
    """


class ThreatFeedApiError(ThreatFeedError):
    """Threat feed API error.

    Raised when HTTP requests to the 0din threat feed API fail,
    such as authentication errors, network issues, or bad responses.

    Attributes:
        status_code: HTTP status code if available, None for network errors.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ThreatFeedCacheError(ThreatFeedError):
    """Threat feed cache error.

    Raised when cache I/O operations fail, such as corrupt cache files,
    write failures, or schema version mismatches.
    """


class SusFactorError(SigError):
    """SusFactor classification error.

    Raised when the SusFactor jailbreak classifier fails to load its model
    files or run inference (e.g., missing model cache, tokenizer or head
    weights not found, or inference failure).
    """


__all__ = [
    "SigError",
    "ConfigError",
    "ProviderError",
    "ModelError",
    "InvalidInputError",
    "ThreatFeedError",
    "ThreatFeedApiError",
    "ThreatFeedCacheError",
    "SusFactorError",
]
