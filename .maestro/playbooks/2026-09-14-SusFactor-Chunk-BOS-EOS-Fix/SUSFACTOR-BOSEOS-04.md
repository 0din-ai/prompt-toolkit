# SusFactor Chunk BOS/EOS Fix — Phase 4: Rust

## Background (context for every task below)

0DIN-2132 reported that SusFactor's sliding-window chunker is missing special
tokens (`<s>`=BOS=0, `</s>`=EOS=2, XLM-RoBERTa-based model) on interior chunks.
The Go, Python, and TypeScript SDKs all had the same bug: tokenizing the full
input **with** special tokens added, then slicing that flat array into
overlapping chunks, so only the first/last chunk kept real `<s>`/`</s>` and
interior chunks had neither. Rust has the same bug shape, but inline —
`packages/rust/src/susfactor/common.rs` and `packages/rust/src/susfactor/onnx.rs`
implement the tokenize-then-chunk logic directly rather than through a
separate `chunk.go`/`chunk_token_ids`-style module, so there is no isolated
splitting function to leave untouched — the fix has to be located and applied
inline.

- [x] Locate and fix the equivalent tokenize-and-chunk logic in `packages/rust/src/susfactor/common.rs` and `packages/rust/src/susfactor/onnx.rs`:
  1. Find where these files tokenize the input text and then slice the resulting token-ID array into overlapping sliding-window chunks (mirrors the pattern fixed in Go's `classifier.go`, Python's `classifier.py`/`onnx_classifier.py`, and TypeScript's `classifier.ts`).
  2. Change the tokenize call to not add special tokens automatically (equivalent of `add_special_tokens=false`), so the chunking step receives pure content-token IDs.
  3. Keep the sliding-window splitting math itself unchanged — only the tokenize/wrap logic around it needs fixing.
  4. For each resulting chunk, explicitly wrap it as `[bos_id] + chunk + [eos_id]`.
  5. Build the attention mask as all-1s of length `chunk.len() + 2`.
  6. Resolve `bos_id`/`eos_id` dynamically from the tokenizer binding in use (check its API for a `token_to_id("<s>")`-style lookup or equivalent) rather than hardcoding `0`/`2`, even though `special_tokens_map.json` confirms those values for this specific model, since hardcoding would silently break if the tokenizer is ever swapped.
  7. If `common.rs` and `onnx.rs` share the tokenize/chunk logic through a common helper, fix it once in the shared location; if the logic is duplicated between the two files, fix both.

  **Notes:** The shared logic lives entirely in `packages/rust/src/susfactor/common.rs` — `onnx.rs` and `vertex.rs` (the two backends the task description didn't name but which both call the exact same helpers) never duplicate it, so the fix landed once, in `common.rs`:
  - `tokenize_full` now calls `tokenizer.encode(text, false)` (was `true`) and returns pure content `Vec<i64>` (the attention-mask return value was dropped — it was always trivially all-1s and unused after the fix, matching what the Python/TS fixes did).
  - Added `resolve_special_token_ids(tokenizer) -> Result<(i64, i64)>`, which looks up `"<s>"`/`"</s>"` via `Tokenizer::token_to_id` and errors (`SigError::Model`) if either is missing — no hardcoded `0`/`2`.
  - Replaced `chunk_token_ids_with_mask(ids, mask)` with `chunk_token_ids_with_special_tokens(ids, bos_id, eos_id)`, which calls the original `chunk_token_ids` splitting function **unchanged** and then wraps each resulting window as `[bos_id, ...chunk, eos_id]` with an all-1s mask of `chunk.len() + 2`.
  - `onnx.rs` and `vertex.rs`'s `classify()` methods were updated identically: call `tokenize_full`, then `resolve_special_token_ids`, then `chunk_token_ids_with_special_tokens` instead of the old `chunk_token_ids_with_mask`.
  - Updated the now-stale `ChunkedSusFactorResult::total_tokens` doc comment in `types.rs` (it referenced `[CLS]`/`[SEP]`, which was never accurate for this XLM-RoBERTa model, and is now definitely wrong since `total_tokens` is pure content length pre-wrap).
  - No images were associated with this task.

- [x] Add test coverage for BOS/EOS placement in Rust's SusFactor chunking. Check whether `packages/rust/` already has a chunking test file analogous to `packages/go/susfactor/chunk_test.go` or `packages/python/tests/test_susfactor_chunking.py`; if one exists, add to it, otherwise create a new test module colocated with `common.rs` (Rust convention: `#[cfg(test)] mod tests` in the same file, or `packages/rust/tests/susfactor_chunking.rs` if the project uses external test files — follow whichever pattern the existing Rust test suite already uses). Assert:
  - For a short input that produces a single chunk: the chunk's first token equals `bos_id` and its last token equals `eos_id`, and chunk count/content-token boundaries match the pre-fix behavior for this case (it was already correct).
  - For a long input (>510 content tokens) that produces multiple chunks: **every** chunk — including interior ones — starts with `bos_id` and ends with `eos_id`. This is the specific regression the fix addresses.
  - Each chunk's attention mask is all-1s with length `chunk.len() + 2`.

  **Notes:** Rust's existing convention is `#[cfg(test)] mod tests` colocated in the source file (no external `tests/susfactor_chunking.rs`-style file existed or was warranted), so the new tests were added to `common.rs`'s existing test module, next to the pre-existing "Chunking logic tests — pure, no model required" section:
  - `chunk_token_ids_with_special_tokens_single_chunk_wraps_bos_eos` — single-chunk case, asserts BOS/EOS at the boundaries, content-token boundaries unchanged, and mask is all-1s of length `ids.len() + 2`.
  - `chunk_token_ids_with_special_tokens_every_chunk_wraps_bos_eos` — `MAX_CONTENT_TOKENS * 3` input producing multiple chunks; asserts **every** chunk (interior included) starts with `bos_id`/ends with `eos_id` and has a matching all-1s mask. This is the direct regression test for the bug.
  - `chunk_token_ids_with_special_tokens_content_matches_unwrapped_chunking` — cross-checks the wrapped output's inner content against plain `chunk_token_ids` output to confirm the splitting math itself is untouched.
  - `resolve_special_token_ids_finds_bos_and_eos` — model-gated (skips without `SUSFACTOR_MODEL_DIR`), confirms dynamic resolution returns `0`/`2` for the real tokenizer, matching `special_tokens_map.json` without hardcoding those values in the implementation.
  - `cargo test --lib --features susfactor` and `--features susfactor-vertex` both pass (22 and 31 susfactor-scoped tests respectively, including `SUSFACTOR_MODEL_DIR`-gated tests since that env var was set in this environment). `cargo fmt --check` and `cargo clippy --all-features -- -D warnings` are both clean, and the `susfactor-vertex` feature-tree check (`cargo tree --features susfactor-vertex --no-default-features | grep -E "ort|ndarray"`) returns empty as required by `AGENTS.md`.
  - No images were associated with this task.

<!-- MAESTRO:MODEL tier="default" effort="low" -->

- [ ] Run `cargo test` scoped to the susfactor module (e.g. `cargo test --package <rust-package-name> susfactor` — check `packages/rust/Cargo.toml` for the exact package name) from the repo root and fix any failures until the suite is green. Then run a real prompt longer than 510 content tokens through the classifier end-to-end and confirm every returned chunk's token IDs start with `bos_id` and end with `eos_id` — report the result in your final summary; remove any throwaway script used for this check but keep the permanent test added in the previous task.

## Manual Follow-Up (not executed by Auto Run)

- The "migrate the served model to the sanitized v2 checkpoint" half of 0DIN-2132 is scoped to a **separate ticket** (Vertex dev/stage/prod upload checklist) and is intentionally not covered by this playbook.
- Once all 4 SDKs are fixed, consider a final cross-SDK sanity pass: run the same long test prompt through all four classifiers and confirm they agree on chunk boundaries and BOS/EOS placement (not automated here since it spans multiple language toolchains in one step).
