"""Error types for signature-sdk operations."""


class SigError(Exception):
    """Base exception for signature-sdk operations.

    All exceptions raised by the signature-sdk library inherit from this base class,
    making it easy to catch all library-specific errors.

    Example:
        >>> try:
        ...     # some signature-sdk operation
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


__all__ = [
    "SigError",
    "ConfigError",
    "ProviderError",
    "ModelError",
    "InvalidInputError",
]
