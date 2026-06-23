"""SusFactor classifier using a local e5-large encoder + MLP head.

Mirrors the reference implementation (``0din-ai/odin-ml`` →
``susfactor_evals.classifiers.susfactor.SusFactorClassifier`` /
``susfactor_training.models.E5Classifier``):

    tokenize (max_len 512) -> XLM-RoBERTa e5 encoder -> mean-pool with attention
    mask (no L2 normalization) -> 2-layer MLP head -> softmax -> P(class 1).

Class index 1 is the *suspicious / malicious* class. The model is not bundled
with the SDK; download ``0dinai/susfactor-e5-large`` from HuggingFace (gated --
requires a token) and cache it locally before use.
"""

from __future__ import annotations

from typing import Any, Optional

from ..error import SusFactorError
from ..providers.model_cache import (
    susfactor_model_dir,
    susfactor_model_files_present,
)
from .types import (
    CHUNK_STRIDE,
    DEFAULT_THRESHOLD,
    MAX_CONTENT_TOKENS,
    MODEL_VERSION,
    ChunkedSusFactorResult,
    SusFactorResult,
    label_for_score,
    suspicious_prob,
)

_INSTALL_HINT = "pip install 'odin-prompt-toolkit[susfactor]'"


def _require_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as e:
        raise ImportError(
            f"SusFactor requires the 'torch' package. Install with: {_INSTALL_HINT}"
        ) from e


def _require_transformers() -> tuple:
    try:
        from transformers import AutoModel, AutoTokenizer

        return AutoModel, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            f"SusFactor requires the 'transformers' package. Install with: {_INSTALL_HINT}"
        ) from e


DEFAULT_MODEL = "0dinai/susfactor-e5-large"
DEFAULT_HIDDEN_DIM = 256
EMBEDDING_DIM = 1024
NUM_CLASSES = 2
# DEFAULT_THRESHOLD, MAX_SEQUENCE_LENGTH, and MODEL_VERSION live in types.py
# so both the torch and ONNX classifiers can import them without depending on
# each other.
HF_URL = "https://huggingface.co/0dinai/susfactor-e5-large"


def _build_head(hidden_dim: int) -> Any:
    """Build the SusFactor MLP head."""
    torch = _require_torch()

    class ClassificationHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.0),
                torch.nn.Linear(EMBEDDING_DIM, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(hidden_dim, NUM_CLASSES),
            )

        def forward(self, embeddings: Any) -> Any:
            return self.classifier(embeddings)

    return ClassificationHead()


def _resolve_device(device: Optional[str]) -> str:
    torch = _require_torch()
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean pooling over tokens, respecting the attention mask (no L2 norm)."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class SusFactorClassifier:
    """Classifies prompts as safe vs. suspicious using SusFactor E5-Large.

    Use the :meth:`new` factory to load the model from the local cache. The
    constructor takes pre-built components directly (useful for testing).

    Example:
        >>> from odin_prompt_toolkit.providers import ModelCache
        >>> from odin_prompt_toolkit.susfactor import SusFactorClassifier
        >>> clf = await SusFactorClassifier.new(ModelCache())
        >>> result = await clf.classify("Ignore previous instructions")
        >>> print(result.score, result.label)
    """

    def __init__(
        self,
        encoder: Any,
        tokenizer: Any,
        head: Any,
        model_name: str,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
    ):
        """Initialize with pre-built components (prefer :meth:`new`)."""
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._head = head
        self._model_name = model_name
        self._threshold = threshold
        self._device = device

    @classmethod
    async def new(
        cls,
        cache: Any,
        model: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> "SusFactorClassifier":
        """Load the SusFactor classifier from a local model cache.

        Args:
            cache: A ``ModelCache`` instance for locating model files.
            model: Model identifier reported in results (default
                ``0dinai/susfactor-e5-large``).
            threshold: Decision threshold for the suspicious label.
            device: Torch device ("cuda"/"mps"/"cpu"); auto-detected if None.
            hidden_dim: MLP head hidden dimension (default 256).

        Returns:
            An initialized ``SusFactorClassifier``.

        Raises:
            SusFactorError: If the model files are not present in the cache.
        """
        model_name = model or DEFAULT_MODEL
        resolved_device = _resolve_device(device)
        model_dir = susfactor_model_dir(cache, MODEL_VERSION)

        if not susfactor_model_files_present(cache, MODEL_VERSION):
            raise SusFactorError(
                f"SusFactor model not found in cache at {model_dir}. "
                f"Download it from HuggingFace (gated -- requires a token): "
                f"{HF_URL}\nExpected layout: <dir>/encoder/ (config.json, "
                "model.safetensors, tokenizer.json) and <dir>/head.pt"
            )

        encoder_dir = model_dir / "encoder"
        head_path = model_dir / "head.pt"

        try:
            torch = _require_torch()
            AutoModel, AutoTokenizer = _require_transformers()
            encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
            head = _build_head(hidden_dim)
            import inspect

            load_kwargs: dict = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = True
            state_dict = torch.load(head_path, **load_kwargs)
            head.load_state_dict(state_dict)
        except SusFactorError:
            raise
        except Exception as e:  # noqa: BLE001 - surface as a domain error
            raise SusFactorError(f"Failed to load SusFactor model: {e}") from e

        encoder = encoder.to(resolved_device)
        encoder.eval()
        head = head.to(resolved_device)
        head.eval()

        return cls(
            encoder=encoder,
            tokenizer=tokenizer,
            head=head,
            model_name=model_name,
            threshold=threshold,
            device=resolved_device,
        )

    def model(self) -> str:
        """Return the model identifier."""
        return self._model_name

    def threshold(self) -> float:
        """Return the decision threshold."""
        return self._threshold

    @staticmethod
    def chunk_token_ids(ids: list[int]) -> list[list[int]]:
        """Split a token-ID sequence into overlapping chunks of at most
        ``MAX_CONTENT_TOKENS`` each. See ``SusFactorOnnxClassifier.chunk_token_ids``
        for full documentation."""
        if len(ids) <= MAX_CONTENT_TOKENS:
            return [list(ids)]
        chunks = []
        start = 0
        while True:
            end = min(start + MAX_CONTENT_TOKENS, len(ids))
            chunks.append(ids[start:end])
            if end == len(ids):
                break
            start += CHUNK_STRIDE
        return chunks

    async def classify(self, text: str) -> ChunkedSusFactorResult:
        """Classify a prompt of any length.

        Prompts within ``MAX_CONTENT_TOKENS`` (510 tokens) are scored in a
        single inference call. Longer prompts are automatically split into
        overlapping chunks, each scored independently — callers do not need
        to check length or call a separate method.

        Chunks are dispatched concurrently via ``asyncio.gather``. Actual
        concurrency depends on the runtime; a single shared model serializes
        inference internally. Dispatching concurrently allows the runtime to
        schedule work efficiently.

        Each chunk is scored independently; no scores are aggregated.
        A prompt is suspicious if **any** chunk is suspicious.
        """
        import asyncio
        import time as _time

        wall_start = _time.time()

        torch = _require_torch()
        inputs_full = self._tokenizer(
            [text],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        all_ids: list[int] = inputs_full["input_ids"][0].tolist()
        all_mask: list[int] = inputs_full["attention_mask"][0].tolist()

        id_chunks = self.chunk_token_ids(all_ids)

        async def _score_chunk(chunk_ids: list[int]) -> SusFactorResult:
            chunk_start = _time.time()
            chunk_len = len(chunk_ids)
            chunk_mask = all_mask[:chunk_len]

            import torch as _torch
            ids_t = _torch.tensor([chunk_ids], dtype=_torch.long).to(self._device)
            mask_t = _torch.tensor([chunk_mask], dtype=_torch.long).to(self._device)

            with _torch.no_grad():
                outputs = self._encoder(input_ids=ids_t, attention_mask=mask_t)
                pooled = _mean_pool(outputs.last_hidden_state, mask_t)
                logits = self._head(pooled)
                logits_np = logits[0].detach().cpu().numpy()

            score = suspicious_prob(logits_np.tolist())
            label = label_for_score(score, self._threshold)
            return SusFactorResult(
                score=score,
                label=label,
                model=self._model_name,
                threshold=self._threshold,
                timing_ms=(_time.time() - chunk_start) * 1000,
            )

        chunk_results: list[SusFactorResult] = await asyncio.gather(
            *[_score_chunk(chunk) for chunk in id_chunks]
        )

        is_suspicious = any(r.is_suspicious for r in chunk_results)
        return ChunkedSusFactorResult(
            chunks=list(chunk_results),
            is_suspicious=is_suspicious,
            total_timing_ms=(_time.time() - wall_start) * 1000,
        )

    async def close(self) -> None:
        """Release model resources."""
        self._encoder = None
        self._head = None
        self._tokenizer = None
