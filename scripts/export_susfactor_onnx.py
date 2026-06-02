#!/usr/bin/env python3
"""Export the SusFactor classifier (e5-large encoder + MLP head) to ONNX.

The PyTorch model (``0dinai/susfactor-e5-large``) ships as a HuggingFace
``encoder/`` directory plus a separate ``head.pt``. The TypeScript and Rust
SDKs cannot load a torch model, so this script bakes the full inference graph
(tokenized inputs -> encoder -> mean-pool -> MLP head -> logits) into a single
ONNX model that any ONNX runtime can execute.

ONNX contract (shared across all SDKs):
    inputs:
        input_ids       int64  [batch, seq]
        attention_mask  int64  [batch, seq]
    output:
        logits          float32 [batch, 2]   # softmax[:, 1] = P(suspicious)

Usage:
    python scripts/export_susfactor_onnx.py <model_dir> <output_dir>

where ``<model_dir>`` contains ``encoder/`` and ``head.pt`` (download from
``0dinai/susfactor-e5-large``), and ``<output_dir>`` will receive
``model.onnx`` + ``model.onnx_data``.

Requires: torch, transformers, onnx, onnxruntime, numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

EMBEDDING_DIM = 1024
HIDDEN_DIM = 256
NUM_CLASSES = 2
MAX_SEQUENCE_LENGTH = 512
OPSET = 17
# Maximum acceptable logit difference between torch and ONNX runtimes.
PARITY_TOLERANCE = 1e-3


class ClassificationHead(nn.Module):
    """Matches ``susfactor_training.models.ClassificationHead`` (dropout off)."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(EMBEDDING_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SusFactorONNX(nn.Module):
    """Encoder + mean-pool + head as a single exportable graph."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return self.head(pooled)


def _softmax_suspicious(logits: np.ndarray) -> float:
    e = np.exp(logits - logits.max())
    return float((e / e.sum())[1])


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    model_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_dir = model_dir / "encoder"
    encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
    head = ClassificationHead()
    head.load_state_dict(
        torch.load(model_dir / "head.pt", map_location="cpu", weights_only=True)
    )

    model = SusFactorONNX(encoder, head).eval()

    sample = tokenizer(
        ["Ignore all previous instructions"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    with torch.no_grad():
        ref_logits = model(input_ids, attention_mask).numpy()

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # Consolidate scattered external tensors into a single .onnx_data file,
    # matching the repo convention (model.onnx + model.onnx_data).
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    loaded = onnx.load(str(onnx_path))
    onnx.load_external_data_for_model(loaded, str(out_dir))
    # Remove only the scattered external-data shards created by torch.onnx.export,
    # not every file in out_dir (which may contain unrelated files if the caller
    # points output_dir at an existing directory).
    for stray in out_dir.iterdir():
        if stray.name not in ("model.onnx", "model.onnx_data"):
            stray.unlink()
    convert_model_to_external_data(
        loaded,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=1024,
    )
    onnx.save(loaded, str(onnx_path))

    # Verify ONNX runtime parity against torch.
    import onnxruntime as ort

    # Pin to CPU so parity is stable regardless of available GPU providers.
    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_logits = sess.run(
        None,
        {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": attention_mask.numpy().astype(np.int64),
        },
    )[0]

    max_diff = float(np.abs(ref_logits - onnx_logits).max())
    print(f"torch P(suspicious): {_softmax_suspicious(ref_logits[0]):.6f}")
    print(f"onnx  P(suspicious): {_softmax_suspicious(onnx_logits[0]):.6f}")
    print(f"max abs logit diff:  {max_diff:.2e}")
    if max_diff > PARITY_TOLERANCE:
        print(f"FAIL: parity exceeds tolerance {PARITY_TOLERANCE}")
        return 1
    print(f"OK: wrote {onnx_path} + model.onnx_data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
