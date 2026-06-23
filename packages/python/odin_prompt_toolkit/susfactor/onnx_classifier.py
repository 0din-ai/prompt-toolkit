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
    CHUNK_OVERLAP,
    CHUNK_STRIDE,
    ChunkedSusFactorResult,
    DEFAULT_THRESHOLD,
    MAX_CONTENT_TOKENS,
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

    @staticmethod
    def chunk_token_ids(ids: list[int]) -> list[list[int]]:
        """Split a token-ID sequence into overlapping chunks of at most
        ``MAX_CONTENT_TOKENS`` tokens each.

        - Sequences at or below ``MAX_CONTENT_TOKENS`` produce exactly one chunk
          (identical to the input).
        - Adjacent chunks share ``CHUNK_OVERLAP`` tokens of context so that
          sentence boundaries near a chunk edge are still scored in full context.
        - An empty input produces one empty chunk.

        Args:
            ids: Raw token IDs (not including special tokens added by the
                tokenizer — the caller is responsible for providing the payload
                tokens only).

        Returns:
            A list of token-ID lists, each of length ≤ ``MAX_CONTENT_TOKENS``.
        """
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
        overlapping chunks scored in parallel — callers do not need to check
        length or call a separate method.

        Each chunk is scored independently; no scores are aggregated.
        A prompt is suspicious if **any** chunk is suspicious.

        Args:
            text: The prompt to classify (any length).

        Returns:
            A :class:`ChunkedSusFactorResult` with one entry per chunk.
            Short prompts produce exactly one chunk.

        Raises:
            SusFactorError: If the classifier has been closed or inference fails.
        """
        import asyncio
        import time as _time

        if self._session is None:
            from ..error import SusFactorError
            raise SusFactorError(
                "classify() called on a closed SusFactorOnnxClassifier"
            )

        wall_start = _time.time()

        # Tokenize the full text without truncation — we handle chunking ourselves.
        inputs = self._tokenizer(
            text,
            padding=False,
            truncation=False,
            return_tensors="np",
        )
        all_ids: list[int] = inputs["input_ids"][0].tolist()
        all_mask: list[int] = inputs["attention_mask"][0].tolist()

        # Chunk on token IDs.
        id_chunks = self.chunk_token_ids(all_ids)

        # Build a coroutine per chunk that runs the ONNX session directly
        # (no extra tokenization — we pass pre-built token arrays).
        async def _score_chunk(chunk_ids: list[int]) -> SusFactorResult:
            chunk_start = _time.time()
            chunk_len = len(chunk_ids)
            chunk_mask = all_mask[:chunk_len]

            import numpy as np
            ids_arr = np.array([chunk_ids], dtype=np.int64)
            mask_arr = np.array([chunk_mask], dtype=np.int64)

            onnx_inputs: dict = {
                "input_ids": ids_arr,
                "attention_mask": mask_arr,
            }
            required_names = {inp.name for inp in self._session.get_inputs()}
            if "token_type_ids" in required_names:
                onnx_inputs["token_type_ids"] = np.zeros_like(ids_arr)

            outputs = self._session.run(None, onnx_inputs)
            output_names = [o.name for o in self._session.get_outputs()]
            logits_idx = output_names.index("logits") if "logits" in output_names else 0
            logits = outputs[logits_idx][0]

            score = suspicious_prob(logits.tolist())
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
        total_timing_ms = (_time.time() - wall_start) * 1000

        return ChunkedSusFactorResult(
            chunks=list(chunk_results),
            is_suspicious=is_suspicious,
            total_timing_ms=total_timing_ms,
        )

    async def close(self) -> None:
        """Release model resources."""
        # ONNX sessions don't require explicit cleanup.
        self._session = None
        self._tokenizer = None
