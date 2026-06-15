"""ONNX embedding provider implementation using onnxruntime."""

import time
from typing import Optional

import numpy as np

from ..lsh import normalize_vector
from ..types import EmbeddingResult, compute_embedding_sha256
from .model_cache import ModelCache

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "ONNX provider requires the 'onnxruntime' package. "
        "Install with: pip install 'odin-prompt-toolkit[onnx]'"
    ) from e

try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(
        "ONNX provider requires the 'transformers' package. "
        "Install with: pip install 'odin-prompt-toolkit[onnx]'"
    ) from e


class OnnxProvider:
    """ONNX embedding provider using local model inference.

    This provider uses the 0dinai/0din-jailbreak-embeddings-small model by default,
    which produces 1024-dimensional embeddings suitable for multilingual text similarity.

    The model is automatically downloaded from HuggingFace on first use and cached locally.

    Args:
        cache: ModelCache instance for managing model files
        model: Model name or local path (default: "0dinai/0din-jailbreak-embeddings-small")
        name: Provider name (default: "onnx")

    Example:
        >>> cache = ModelCache()
        >>> provider = await OnnxProvider.new(cache)
        >>> result = await provider.generate_embedding("Hello, world!")
        >>> print(f"Generated {result.dimensions}-dimensional embedding")
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-large"
    DEFAULT_DIMENSIONS = 1024

    def __init__(
        self,
        session: ort.InferenceSession,
        tokenizer: AutoTokenizer,
        model_name: str,
        dimensions: int,
        name: str,
    ):
        """Initialize ONNX provider (use `new()` class method instead)."""
        self._session = session
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._dimensions = dimensions
        self._name = name

    @classmethod
    async def new(
        cls,
        cache: ModelCache,
        model: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "OnnxProvider":
        """Create a new ONNX provider instance.

        Args:
            cache: ModelCache instance
            model: Model name or path (default: intfloat/multilingual-e5-small)
            name: Provider name (default: "onnx")

        Returns:
            Initialized OnnxProvider

        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If model loading fails
        """
        model_name = model or cls.DEFAULT_MODEL
        provider_name = name or "onnx"

        # Check if model is cached
        if not cache.has_model("v1"):
            raise FileNotFoundError(
                f"Model not found in cache at {cache.model_directory('v1')}. "
                "Please download the model manually from HuggingFace: "
                "https://huggingface.co/intfloat/multilingual-e5-small"
            )

        # Load ONNX model
        model_path = str(cache.get_model_path("v1"))
        session = ort.InferenceSession(model_path)

        # Load tokenizer
        # Suppress false-positive Mistral regex warning for XLMRoberta tokenizer
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
            tokenizer_path = str(cache.get_tokenizer_path("v1"))
            tokenizer = AutoTokenizer.from_pretrained(
                cache.model_directory("v1"),
                local_files_only=True,
            )

        # Get dimensions from model output
        output_shape = session.get_outputs()[0].shape
        dimensions = output_shape[-1] if output_shape else cls.DEFAULT_DIMENSIONS

        return cls(
            session=session,
            tokenizer=tokenizer,
            model_name=model_name,
            dimensions=dimensions,
            name=provider_name,
        )

    def name(self) -> str:
        """Get provider name."""
        return self._name

    def model(self) -> str:
        """Get model identifier."""
        return self._model_name

    def dimensions(self) -> int:
        """Get embedding dimensionality."""
        return self._dimensions

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for the given text.

        Args:
            text: Input text to embed

        Returns:
            EmbeddingResult with embedding, normalized embedding, SHA256, etc.

        Raises:
            RuntimeError: If inference fails
        """
        start_time = time.time()

        # Tokenize input
        inputs = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        # Run inference — include token_type_ids if the model requires it.
        # XLM-RoBERTa tokenizers don't produce token_type_ids, but the ONNX
        # model may still expect them; supply a zero tensor in that case.
        input_ids = inputs["input_ids"].astype(np.int64)
        onnx_inputs = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        required_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in required_names:
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            else:
                onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]  # Shape: [batch_size, seq_len, hidden_size]

        # Mean pooling with attention mask
        attention_mask = inputs["attention_mask"]
        attention_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

        # Mask out padding tokens
        masked_hidden = last_hidden_state * attention_mask_expanded

        # Sum over sequence length
        sum_hidden = np.sum(masked_hidden, axis=1)

        # Divide by number of non-padding tokens
        sum_mask = np.sum(attention_mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # Avoid division by zero

        # Mean pooled embedding
        mean_pooled = sum_hidden / sum_mask
        embedding = mean_pooled[0].tolist()  # First (and only) batch item

        elapsed_ms = (time.time() - start_time) * 1000

        # Normalize and compute SHA256
        normalized = normalize_vector(embedding)
        sha256 = compute_embedding_sha256(normalized)

        # Count tokens (approximate)
        token_count = int(np.sum(attention_mask))

        return EmbeddingResult(
            embedding=embedding,
            normalized_embedding=normalized,
            normalized_embedding_sha256=sha256,
            model=self._model_name,
            dimensions=len(embedding),
            token_count=token_count,
            timing_ms=elapsed_ms,
        )

    async def close(self) -> None:
        """Close the provider and clean up resources."""
        # ONNX sessions don't require explicit cleanup
        pass
