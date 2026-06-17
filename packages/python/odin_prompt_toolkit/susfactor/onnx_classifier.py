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
)
from .types import (
    DEFAULT_THRESHOLD,
    MAX_SEQUENCE_LENGTH,
    MODEL_VERSION,
    SusFactorResult,
    label_for_score,
    suspicious_prob,
)

_INSTALL_HINT = "pip install 'odin-prompt-toolkit[onnx]'"

# The ONNX model is published separately from the torch weights.
# Reporting this repo in result.model lets callers distinguish the backend.
# DEFAULT_THRESHOLD, MAX_SEQUENCE_LENGTH, and MODEL_VERSION live in types.py.
DEFAULT_MODEL = "0dinai/susfactor-e5-large-onnx"


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
                ``0dinai/susfactor-e5-large-onnx``).
            threshold: Decision threshold for the suspicious label.
            device: Accepted for API parity with :class:`SusFactorClassifier`
                but not used — ONNX Runtime selects execution providers
                automatically based on what is installed.  GPU acceleration
                requires ``onnxruntime-gpu``; pass ``device`` to a future
                ``providers`` parameter if you need explicit control.

        Returns:
            An initialized ``SusFactorOnnxClassifier``.

        Raises:
            SusFactorError: If the ONNX model files are not present in the
                cache or if loading fails.
        """
        model_name = model or DEFAULT_MODEL

        # Validate optional dependencies before touching the filesystem.
        # Keeping these outside the try/except ensures ImportError propagates
        # with its own install-hint message rather than being swallowed into a
        # generic SusFactorError("Failed to load ... No module named ...").
        ort = _require_onnxruntime()
        AutoTokenizer = _require_tokenizer()

        # Resolve the model directory once and reuse it throughout new() so
        # that susfactor_model_dir() is not called redundantly.
        model_dir = susfactor_model_dir(cache, MODEL_VERSION)
        model_path = model_dir / "onnx" / "model.onnx"

        if not susfactor_onnx_files_present(cache, MODEL_VERSION):
            raise SusFactorError(
                f"SusFactor ONNX model not found in cache at {model_dir}. "
                f"Download it from HuggingFace: {HF_URL_SUSFACTOR_ONNX}\n"
                "Expected layout: <dir>/onnx/model.onnx and <dir>/tokenizer.json\n"
                "Export script: scripts/export_susfactor_onnx.py"
            )

        try:
            # Always use model.onnx (validated production path, not O4 variant).
            # This matches the TypeScript and Rust runtimes.
            # Let ONNX Runtime auto-select providers from what is installed
            # (CPUExecutionProvider always available; CUDAExecutionProvider /
            # CoreMLExecutionProvider picked up automatically with onnxruntime-gpu
            # or onnxruntime on macOS).  Do not hard-code CPUExecutionProvider
            # as that would silently disable GPU even when onnxruntime-gpu is
            # installed.
            session = ort.InferenceSession(str(model_path))

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
            raise SusFactorError(
                f"Failed to load SusFactor ONNX model from {model_path}: {e}"
            ) from e

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
            SusFactorError: If the classifier has been closed or inference fails.
        """
        if self._session is None:
            raise SusFactorError(
                "classify() called on a closed SusFactorOnnxClassifier"
            )
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
            onnx_inputs: dict[str, np.ndarray] = {
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

            # Use the named "logits" output set by the export script.  Fall back
            # to the first output for re-exported variants, but warn so
            # unexpected model variants are visible rather than silently wrong.
            output_names = [o.name for o in self._session.get_outputs()]
            if "logits" in output_names:
                logits_idx = output_names.index("logits")
            else:
                warnings.warn(
                    f"SusFactorOnnxClassifier: 'logits' output not found in "
                    f"model outputs {output_names}; falling back to index 0. "
                    "Re-export with scripts/export_susfactor_onnx.py to fix.",
                    stacklevel=2,
                )
                logits_idx = 0
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
