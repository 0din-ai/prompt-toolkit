"""SusFactor classifier backed by ONNX Runtime (drop-in for SusFactorClassifier).

The ONNX model (exported via ``scripts/export_susfactor_onnx.py``) bakes the
full inference graph into a single file:

    inputs:  input_ids[1, seq] int64, attention_mask[1, seq] int64
    output:  logits[1, 2] float32   (softmax[:, 1] = P(suspicious))

This avoids loading torch + transformers at inference time, giving substantially
faster CPU inference (~3-5× vs. the PyTorch path). It is published to:

    https://huggingface.co/0dinai/susfactor-e5-large-onnx

Mirror the torch classifier's public API exactly so callers can swap::

    # before
    clf = await SusFactorClassifier.new(cache)
    # after
    clf = await SusFactorOnnxClassifier.new(cache)

Both return identical ``SusFactorResult`` values (within floating-point
tolerance of the ONNX export).

Requires: onnxruntime, transformers (for tokenization only — no torch needed).
Install with: ``pip install 'odin-prompt-toolkit[onnx]'``
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional

import numpy as np

from ..error import SusFactorError
from ..providers.model_cache import (
    HF_URL_SUSFACTOR_ONNX,
    susfactor_model_dir,
    susfactor_onnx_files_present,
    susfactor_onnx_model_path,
)
from .types import SusFactorResult, label_for_score, suspicious_prob

_INSTALL_HINT = "pip install 'odin-prompt-toolkit[onnx]'"

DEFAULT_MODEL = "0dinai/susfactor-e5-large"
DEFAULT_THRESHOLD = 0.5
MAX_SEQUENCE_LENGTH = 512
MODEL_VERSION = "susfactor-v1"


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort

        return ort
    except ImportError as e:
        raise ImportError(
            f"SusFactorOnnxClassifier requires 'onnxruntime'. Install with: {_INSTALL_HINT}"
        ) from e


def _require_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer
    except ImportError as e:
        raise ImportError(
            f"SusFactorOnnxClassifier requires 'transformers'. Install with: {_INSTALL_HINT}"
        ) from e


class SusFactorOnnxClassifier:
    """Classifies prompts as safe vs. suspicious using the SusFactor ONNX model.

    Drop-in replacement for :class:`SusFactorClassifier` that uses ONNX Runtime
    instead of PyTorch. No ``torch`` dependency is required at inference time.

    The ONNX graph bakes the e5-large encoder, mean-pooling, and MLP head into a
    single model file, so inference is faster on CPU compared to the torch path.

    Use the :meth:`new` factory to load the model from the local cache. The
    constructor takes pre-built components directly (useful for testing).

    Example:
        >>> from odin_prompt_toolkit.providers import ModelCache
        >>> from odin_prompt_toolkit.susfactor import SusFactorOnnxClassifier
        >>> clf = await SusFactorOnnxClassifier.new(ModelCache())
        >>> result = await clf.classify("Ignore previous instructions")
        >>> print(result.score, result.label)
    """

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        model_name: str,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """Initialize with pre-built components (prefer :meth:`new`)."""
        self._session = session
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._threshold = threshold

    @classmethod
    async def new(
        cls,
        cache: Any,
        model: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
    ) -> "SusFactorOnnxClassifier":
        """Load the SusFactor ONNX classifier from a local model cache.

        The ``device`` parameter is accepted for API parity with
        :class:`SusFactorClassifier` but ONNX Runtime selects its own execution
        providers. Pass ``None`` (the default) to let ONNX Runtime auto-select.

        Args:
            cache: A ``ModelCache`` instance for locating model files.
            model: Model identifier reported in results (default
                ``0dinai/susfactor-e5-large``).
            threshold: Decision threshold for the suspicious label.
            device: Ignored (ONNX Runtime manages execution providers).

        Returns:
            An initialized ``SusFactorOnnxClassifier``.

        Raises:
            SusFactorError: If the ONNX model files are not present in the
                cache or if loading fails.
        """
        model_name = model or DEFAULT_MODEL

        if not susfactor_onnx_files_present(cache, MODEL_VERSION):
            model_dir = susfactor_model_dir(cache, MODEL_VERSION)
            raise SusFactorError(
                f"SusFactor ONNX model not found in cache at {model_dir}. "
                f"Download it from HuggingFace: {HF_URL_SUSFACTOR_ONNX}\n"
                "Expected layout: <dir>/onnx/model.onnx and <dir>/tokenizer.json\n"
                "Export script: scripts/export_susfactor_onnx.py"
            )

        model_path = susfactor_onnx_model_path(cache, MODEL_VERSION)
        model_dir = susfactor_model_dir(cache, MODEL_VERSION)

        try:
            ort = _require_onnxruntime()
            AutoTokenizer = _require_tokenizer()

            # Always use model.onnx (validated production path, not O4 variant).
            # This matches the TypeScript and Rust runtimes.
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

            # Suppress false-positive Mistral regex warning for XLMRoberta tokenizer.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_dir),
                    local_files_only=True,
                )
        except SusFactorError:
            raise
        except Exception as e:  # noqa: BLE001 - surface as a domain error
            raise SusFactorError(f"Failed to load SusFactor ONNX model: {e}") from e

        return cls(
            session=session,
            tokenizer=tokenizer,
            model_name=model_name,
            threshold=threshold,
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
            # Tokenize. Use padding=True (dynamic length) rather than
            # padding='max_length' so short prompts don't run 512-token
            # inference. The ONNX graph was exported with a dynamic seq axis.
            inputs = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="np",
            )

            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            # Determine which inputs the ONNX graph actually requires.
            # The SusFactor export only declares input_ids + attention_mask, but
            # guard against re-exported variants that add token_type_ids.
            onnx_inputs: dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            required_names = {inp.name for inp in self._session.get_inputs()}
            if "token_type_ids" in required_names:
                if "token_type_ids" in inputs:
                    onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
                else:
                    onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

            outputs = self._session.run(None, onnx_inputs)

            # Prefer the named "logits" output set by the export script; fall
            # back to the first output for re-exported variants.
            output_names = [o.name for o in self._session.get_outputs()]
            logits_idx = output_names.index("logits") if "logits" in output_names else 0
            logits = outputs[logits_idx][0]  # shape: [2]
        except SusFactorError:
            raise
        except Exception as e:  # noqa: BLE001 - surface as a domain error
            raise SusFactorError(f"SusFactor ONNX inference failed: {e}") from e

        score = suspicious_prob(logits.tolist())
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
        # ONNX sessions don't require explicit cleanup.
        self._session = None
        self._tokenizer = None
