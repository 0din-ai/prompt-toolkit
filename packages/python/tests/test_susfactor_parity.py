"""SusFactor cross-SDK parity test: Python vs. the validated Rust reference.

Loads ``spec/test-vectors/susfactor_vectors.json`` (committed golden vectors
generated from the Rust SDK) and asserts that Python reproduces each score
within TOLERANCE and the label exactly.

If any score differs by more than TOLERANCE, this is a bug — not a number to
loosen.  The most likely culprits are the hand-reconstructed MLP head
(``_build_head``) or the ``_mean_pool`` implementation diverging from the baked
ONNX graph.

Running
-------
With the model cached locally::

    SIGNATURE_SDK_MODEL_CACHE=/path/to/cache pytest tests/test_susfactor_parity.py -v

The test is skipped (not failed) when:
  - ``SIGNATURE_SDK_MODEL_CACHE`` is not set / model files absent
  - ``torch`` or ``transformers`` are not installed
  - Golden scores have not yet been generated (``rust_score`` is null)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# ── Availability guards ──────────────────────────────────────────────────────

TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)

from odin_prompt_toolkit.providers.model_cache import (  # noqa: E402
    ModelCache,
    susfactor_model_files_present,
)

MODEL_AVAILABLE = TORCH_AVAILABLE and susfactor_model_files_present(
    ModelCache(), "susfactor-v1"
)

# ── Fixture loading ──────────────────────────────────────────────────────────

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "spec"
    / "test-vectors"
    / "susfactor_vectors.json"
)

# Score tolerance: maximum absolute difference between Python and Rust scores.
# Start strict.  If this fails, investigate the head/pooling, not the number.
TOLERANCE = 1e-3


def _load_vectors() -> list[dict]:
    """Load the committed golden vectors, skipping unscored entries."""
    if not FIXTURE_PATH.exists():
        return []
    with FIXTURE_PATH.open() as f:
        doc = json.load(f)
    return [
        v
        for v in doc.get("vectors", [])
        if v.get("rust_score") is not None and v.get("expected_label") is not None
    ]


_VECTORS = _load_vectors()

# ── Skip conditions ──────────────────────────────────────────────────────────

_skip_reason = None
if not TORCH_AVAILABLE:
    _skip_reason = "requires torch + transformers (pip install 'odin-prompt-toolkit[susfactor]')"
elif not MODEL_AVAILABLE:
    _skip_reason = "SusFactor model not cached (set SIGNATURE_SDK_MODEL_CACHE)"
elif not _VECTORS:
    _skip_reason = (
        "no scored golden vectors found — run `make generate-susfactor-goldens` first"
    )

pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
async def classifier():
    """Load the real SusFactor model once for the whole module."""
    from odin_prompt_toolkit.susfactor.classifier import SusFactorClassifier

    clf = await SusFactorClassifier.new(ModelCache())
    yield clf
    await clf.close()


# ── Parity tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "vec",
    _VECTORS,
    ids=[v["name"] for v in _VECTORS],
)
@pytest.mark.asyncio
async def test_score_matches_rust_reference(vec: dict, classifier) -> None:
    """Python score must be within TOLERANCE of the committed Rust score."""
    chunked = await classifier.classify(vec["prompt"])
    # Parity vectors are short prompts — exactly one chunk.
    assert len(chunked.chunks) == 1, (
        f"[{vec['name']}] expected 1 chunk for short parity prompt, "
        f"got {len(chunked.chunks)}"
    )
    result = chunked.chunks[0]

    rust_score: float = vec["rust_score"]
    expected_label: str = vec["expected_label"]

    assert result.label == expected_label, (
        f"[{vec['name']}] label mismatch: got {result.label!r}, "
        f"expected {expected_label!r} (rust_score={rust_score:.6f})"
    )

    diff = abs(result.score - rust_score)
    assert diff <= TOLERANCE, (
        f"[{vec['name']}] score drift: Python={result.score:.6f} "
        f"Rust={rust_score:.6f} diff={diff:.2e} > tolerance={TOLERANCE:.0e}\n"
        f"  prompt: {vec['prompt'][:80]!r}\n"
        f"  This is likely a divergence in _build_head or _mean_pool. "
        f"Investigate before loosening the tolerance."
    )
