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

import time
from typing import Any, Optional

from ..error import SusFactorError
from ..providers.model_cache import (
    susfactor_model_dir,
    susfactor_model_files_present,
)
from .types import LABEL_SAFE, LABEL_SUSPICIOUS, SusFactorResult, label_for_score, suspicious_prob

_INSTALL_HINT = "pip install 'odin-prompt-toolkit[susfactor]'"


def _require_torch() -> "torch":  # type: ignore[name-defined]
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
DEFAULT_THRESHOLD = 0.5
DEFAULT_HIDDEN_DIM = 256
EMBEDDING_DIM = 1024
NUM_CLASSES = 2
MAX_SEQUENCE_LENGTH = 512
MODEL_VERSION = "susfactor-v1"
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

        if not susfactor_model_files_present(cache, MODEL_VERSION):
            model_dir = susfactor_model_dir(cache, MODEL_VERSION)
            raise SusFactorError(
                f"SusFactor model not found in cache at {model_dir}. "
                f"Download it from HuggingFace (gated -- requires a token): "
                f"{HF_URL}\nExpected layout: <dir>/encoder/ (config.json, "
                "model.safetensors, tokenizer.json) and <dir>/head.pt"
            )

        model_dir = susfactor_model_dir(cache, MODEL_VERSION)
        encoder_dir = model_dir / "encoder"
        head_path = model_dir / "head.pt"

        try:
            torch = _require_torch()
            AutoModel, AutoTokenizer = _require_transformers()
            encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
            head = _build_head(hidden_dim)
            state_dict = torch.load(head_path, map_location="cpu", weights_only=True)
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

    async def classify(self, text: str) -> SusFactorResult:
        """Classify a single prompt.

        Args:
            text: The prompt to classify.

        Returns:
            A ``SusFactorResult`` with the suspicious probability and label.

        Raises:
            SusFactorError: If inference fails.
        """
        start = time.time()
        try:
            inputs = self._tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
            ).to(self._device)

            torch = _require_torch()
            with torch.no_grad():
                outputs = self._encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                logits = self._head(pooled)
                logits_np = logits[0].detach().cpu().numpy()
        except SusFactorError:
            raise
        except Exception as e:  # noqa: BLE001 - surface as a domain error
            raise SusFactorError(f"SusFactor inference failed: {e}") from e

        score = suspicious_prob(logits_np.tolist())
        label = label_for_score(score, self._threshold)
        elapsed_ms = (time.time() - start) * 1000

        return SusFactorResult(
            score=score,
            label=label,
            model=self._model_name,
            threshold=self._threshold,
            timing_ms=elapsed_ms,
        )

    async def close(self) -> None:
        """Release model resources."""
        self._encoder = None
        self._head = None
        self._tokenizer = None
