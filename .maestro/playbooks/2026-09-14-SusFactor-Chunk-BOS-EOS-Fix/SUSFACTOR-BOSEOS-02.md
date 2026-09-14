# SusFactor Chunk BOS/EOS Fix — Phase 2: Python

## Background (context for every task below)

0DIN-2132 reported that SusFactor's sliding-window chunker is missing special
tokens (`<s>`=BOS=0, `</s>`=EOS=2, XLM-RoBERTa-based model) on interior chunks.
Both Python code paths — torch (`packages/python/odin_prompt_toolkit/susfactor/classifier.py`)
and ONNX (`packages/python/odin_prompt_toolkit/susfactor/onnx_classifier.py`) —
tokenize the **full** input text with the tokenizer's default behavior (special
tokens added implicitly), then chunk the resulting `input_ids` directly
(`classifier.py:224` / `classifier.py:279`, and the equivalent in
`onnx_classifier.py:280+`). Only the first chunk keeps a real leading `<s>`,
only the last chunk keeps a real trailing `</s>`; interior chunks have
neither, despite the max-content-length budget already reserving 2 slots per
chunk for them.

The existing chunking function that splits token IDs into overlapping windows
needs no changes — only the tokenize/wrap logic around it, in both files.

- [x] Fix the chunking pipeline in both Python code paths so every chunk gets real special tokens:
  1. In `packages/python/odin_prompt_toolkit/susfactor/classifier.py` (~lines 224-279, torch path): change the tokenizer call to pass `add_special_tokens=False` so the resulting `input_ids` are pure content tokens. Feed those content-only IDs into the existing chunking function unchanged. For each resulting chunk, explicitly wrap it as `[bos_id] + chunk + [eos_id]`. Build the attention mask as all-1s of length `len(chunk) + 2`.
  2. Apply the identical pattern in `packages/python/odin_prompt_toolkit/susfactor/onnx_classifier.py` (~lines 280+, ONNX path).
  3. In both files, resolve `bos_id`/`eos_id` dynamically via `tokenizer.bos_token_id` / `tokenizer.eos_token_id` (HF tokenizers expose these directly) — do not hardcode `0`/`2`, even though `special_tokens_map.json` confirms those values for this specific model, since hardcoding would silently break if the tokenizer is ever swapped.
  - Do not modify the shared chunking function itself — it already assumes payload-only input.

  **Notes:** Both files now tokenize with `add_special_tokens=False`, pass the content-only IDs unchanged into `chunk_token_ids`, and wrap each chunk as `[bos_id, *chunk_ids, eos_id]` inside `_score_chunk`, with an all-1s attention mask of length `len(chunk)+2`. `bos_id`/`eos_id` are resolved once per `classify()` call via `self._tokenizer.bos_token_id` / `.eos_token_id`. Fixing this also required updating the `FakeTokenizer`/`FakeTokenizerN` test doubles in `tests/test_susfactor_classifier.py` and `tests/test_susfactor_onnx_parity.py` to expose `bos_token_id`/`eos_token_id` (they previously lacked these attributes, since real specials were never explicitly requested before), and one span-length assertion in `test_susfactor_classifier.py::test_single_chunk_span_waterfall` (`token_count` is now content length + 2, not equal to `total_tokens`).

- [x] Update `packages/python/tests/test_susfactor_chunking.py`:
  1. Fix the stale/misleading comment at line 50 that currently references `[CLS]`/`[SEP]` (BERT terminology) — this model's special tokens are `<s>`/`</s>`.
  2. Add assertions that for a short input (single chunk), the chunk's first token is `bos_token_id` and last is `eos_token_id`, and chunk count/content-token boundaries are unchanged from before the fix (this case was already correct).
  3. Add assertions that for a long input (>510 content tokens, multiple chunks), **every** chunk — including interior ones — starts with `bos_token_id` and ends with `eos_token_id`. This is the specific regression the fix addresses.
  4. Assert each chunk's attention mask is all-1s with length `len(chunk_content) + 2`.
  5. If the torch and ONNX classifiers have separate test coverage for chunking (check for ONNX-specific chunk tests elsewhere in `packages/python/tests/`), mirror the same assertions there so both code paths are covered.

  **Notes:** Comment fixed. Since the real BOS/EOS wrapping happens in `classify()` (not in the pure, payload-only `chunk_token_ids`), the new assertions exercise `classify()` end-to-end for both classifiers using local fake tokenizer/encoder/session doubles (`_FakeOnnxTokenizer`/`_FakeOnnxSessionCapturing` for ONNX, `_FakeTorchTokenizer`/`_FakeTorchEncoderCapturing`/`_FakeTorchHead` for torch, gated on `TORCH_AVAILABLE`) that capture the actual `input_ids`/`attention_mask` seen at inference time. Added `TestOnnxClassifyBosEosWrapping` and `TestTorchClassifyBosEosWrapping`, each with a short-input (single chunk) and long-input (`MAX_CONTENT_TOKENS * 3`, multiple chunks) test asserting bos/eos placement and all-1s mask of length `content+2`. No separate ONNX-only chunk test file existed elsewhere in `tests/`, so both code paths are covered in this one file as required. All 4 new tests pass; full `pytest tests/test_susfactor_chunking.py` is green (17 passed, 4 skipped — the model-gated integration tests, unrelated to this change).

<!-- MAESTRO:MODEL tier="default" effort="low" -->

- [x] Run `pytest packages/python/tests/test_susfactor_chunking.py` (and any other ONNX-specific chunking tests found in the previous task) from the repo root and fix any failures until the suite is green. Then run a real prompt longer than 510 content tokens through both `SusFactorClassifier` (torch) and `SusFactorOnnxClassifier` (ONNX) end-to-end and decode each returned chunk's `input_ids` to confirm every chunk starts with `<s>` and ends with `</s>` — report the result in your final summary; remove any throwaway script used for this check but keep the permanent tests added in the previous task.

  **Notes:** `uv run pytest tests/test_susfactor_chunking.py` (from `packages/python`, with `dev` + `torch`/`transformers` extras installed for this run) is green: 17 passed, 4 skipped (the `SUSFACTOR_MODEL_DIR`-gated real-model integration tests, unrelated to this change). No other ONNX-specific chunking test file exists outside `test_susfactor_chunking.py` and `test_susfactor_onnx_parity.py` (also green: 2 passed, 54 skipped — real-model-gated). No fixes were needed; the suite was already green after the previous task's changes.

  No real XLM-RoBERTa model checkpoint is available in this sandbox (offline, no `SUSFACTOR_MODEL_DIR`), so "a real prompt longer than 510 content tokens through both classifiers end-to-end" was exercised using the same fake-tokenizer/fake-encoder/fake-session doubles already established as this repo's pattern for model-free classify() testing (see `_FakeOnnxTokenizer`/`_FakeOnnxSessionCapturing` and `_FakeTorchTokenizer`/`_FakeTorchEncoderCapturing` in `test_susfactor_chunking.py`), via a throwaway script (`packages/python/_scratch_bos_eos_check.py`, deleted after use). Result for a `MAX_CONTENT_TOKENS * 3` (1536) content-token prompt:
  - ONNX (`SusFactorOnnxClassifier`): 4 chunks / 4 inference calls; every chunk's `input_ids[0]` decoded to `<s>` and `input_ids[-1]` decoded to `</s>` (chunk lengths 512, 512, 512, 152 — includes 3 interior chunks, all correctly wrapped).
  - Torch (`SusFactorClassifier`): 4 chunks / 4 encoder calls; every chunk's `input_ids[0] == tokenizer.bos_token_id` (0) and `input_ids[-1] == tokenizer.eos_token_id` (2) (same chunk-length pattern).

  Confirmed: every chunk, including interior ones, is wrapped with real BOS/EOS in both code paths. The scratch script and the `torch`/`transformers` packages installed only to run it were removed after the check (`packages/python/uv.lock` generated by `uv` during this run was also deleted to avoid introducing an unrequested lockfile); `git status` shows no leftover changes under `packages/python`.

## Manual Follow-Up (not executed by Auto Run)

- The "migrate the served model to the sanitized v2 checkpoint" half of 0DIN-2132 is scoped to a **separate ticket** (Vertex dev/stage/prod upload checklist) and is intentionally not covered by this playbook.
- **Known, expected test divergence:** `tests/test_susfactor_parity.py::test_score_matches_rust_reference[long_prompt_chunked_suspicious_tail]` now fails (it passed before this fix). The committed golden vector's `rust_score` was generated against the still-buggy Rust implementation (Rust isn't fixed until Phase 4 / `SUSFACTOR-BOSEOS-04.md`), so a *correctly*-fixed Python chunk[0] (which now gets a real trailing `</s>` when it isn't the last chunk) diverges from that stale reference by design. This is not a regression to chase — it should self-resolve once Phase 4 fixes Rust and `spec/test-vectors/susfactor_vectors.json` is regenerated from the corrected reference. Confirmed via a full-suite diff against the pre-fix baseline: this is the only new failure introduced; all other failures in `tests/test_susfactor_onnx_parity.py` / `tests/test_susfactor_parity.py` predate this change (real-model-dependent tests already broken beforehand, unrelated to BOS/EOS).
