# Proposal: susfactor-vertex

**Status**: Proposed  
**Issue**: #27  
**Target**: v0.8.0  
**Branch**: feat/susfactor-vertex-ai

## Summary

Add a Vertex AI serving backend for the SusFactor classifier so the ~2 GB ONNX model no longer needs to be shipped in application pods. Selection is by configuration; the caller-facing contract (`classify()` → `ChunkedSusFactorResult`) is unchanged.

## Problem

`SusFactorClassifier` loads the ~2 GB SusFactor model into every application pod via an init-container download + `emptyDir`. This creates OOM risk, slow startup, and model-lifecycle pain. Kubernetes memory reservations must account for the model footprint in every replica.

## Solution

Keep tokenization, chunking, softmax, and labeling client-side in Rust (byte-identical to today). Delegate only the ONNX graph execution (`input_ids`, `attention_mask` → `logits[1, 2]`) to a Vertex AI Triton endpoint via the `rawPredict` REST API.

## Key Design Decisions

1. **Extract `susfactor::common`**: All shared logic (tokenize, chunk, softmax, label, reduce) moved to a new `common.rs`. Both backends use it verbatim — they cannot diverge.
2. **`SusFactorProvider` trait**: Analogous to `EmbeddingProvider`. Both `OnnxSusFactor` (renamed from `SusFactorClassifier`) and `VertexSusFactor` implement it.
3. **`gcp_auth = "0.12"`**: Minimal crate for ADC + GKE Workload Identity metadata server tokens. Built-in caching, `Send + Sync`, no interactive OAuth baggage.
4. **Shadow mode**: Wrapper that runs both backends concurrently, returns the ONNX result, and emits a structured divergence signal (score delta, label mismatch, `is_suspicious` mismatch). No automatic fallback in `vertex` mode.
5. **Feature isolation**: New `susfactor-vertex` Cargo feature must compile without `ort`/`ndarray`.

## Spec reference

`spec/SUSFACTOR-VERTEX.md` (v1.0.0)
