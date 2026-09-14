# SusFactor Chunk BOS/EOS Fix — Phase 3: TypeScript

## Background (context for every task below)

0DIN-2132 reported that SusFactor's sliding-window chunker is missing special
tokens (`<s>`=BOS=0, `</s>`=EOS=2, XLM-RoBERTa-based model) on interior chunks.
`packages/typescript/src/susfactor/classifier.ts` (~lines 218-221) tokenizes
the **full** input text with special tokens added implicitly, then chunks the
resulting IDs directly (chunking happens at `classifier.ts:40` / `:235`).
Only the first chunk keeps a real leading `<s>`, only the last chunk keeps a
real trailing `</s>`; interior chunks have neither, despite the max-content-
length budget already reserving 2 slots per chunk for them.

The existing `chunkTokenIds` function needs no changes — only the
tokenize/wrap logic around it in `classifier.ts`.

- [x] In `packages/typescript/src/susfactor/classifier.ts`, fix the chunking pipeline so every chunk gets real special tokens instead of only the first/last:
  1. Change the tokenize call at ~lines 218-221 to pass `addSpecialTokens: false` so the resulting IDs are pure content tokens.
  2. Feed the content-only IDs into the existing `chunkTokenIds` function unchanged (called at ~line 235) — it already assumes payload-only input.
  3. For each resulting chunk, explicitly wrap it as `[bosId, ...chunk, eosId]`.
  4. Build the attention mask as all-1s of length `chunk.length + 2`.
  5. Resolve `bosId`/`eosId` dynamically from the tokenizer (check what the TS tokenizer binding exposes — a `tokenToId("<s>")`-style lookup or equivalent) rather than hardcoding `0`/`2`, even though `special_tokens_map.json` confirms those values for this specific model, since hardcoding would silently break if the tokenizer is ever swapped.
  - Do not modify `chunkTokenIds` itself.

  > Implementation notes: tokenized with `add_special_tokens: false`; `allMask`
  > (derived from the special-token-inclusive encoding) was removed since the
  > mask is now always all-1s of the wrapped length. `bosId`/`eosId` are
  > resolved via `this.tokenizer.bos_token_id` / `.eos_token_id` — confirmed by
  > inspecting `@huggingface/transformers@4.2.0`'s `tokenization_utils.js`,
  > where `PreTrainedTokenizer` sets these as plain numeric properties from the
  > tokenizer config (same field names as the Python fix's
  > `tokenizer.bos_token_id`/`eos_token_id`), and `XLMRobertaTokenizer` has no
  > override. Each chunk is wrapped as `[bosId, ...chunk, eosId]` with an
  > all-1s mask of `chunk.length + 2`.

- [x] Add test coverage in `packages/typescript/test/susfactor-chunking.test.ts` asserting:
  - For a short input that produces a single chunk: the chunk's first token equals `bosId` and its last token equals `eosId`, and chunk count/content-token boundaries are unchanged from before the fix (this case was already correct).
  - For a long input (>510 content tokens) that produces multiple chunks: **every** chunk — including interior ones — starts with `bosId` and ends with `eosId`. This is the specific regression the fix addresses.
  - Each chunk's attention mask is all-1s with length `chunk.length + 2`.

  > Implementation notes: added a `recordingSession` fake session that
  > captures the `input_ids`/`attention_mask` tensors passed to `session.run`
  > for every chunk, plus a `fakeTokenizerForLength` that exposes
  > `bos_token_id`/`eos_token_id` (mirroring the real tokenizer's fields) so
  > the new "BOS/EOS wrapping per chunk" describe block can assert
  > first/last-token and mask-length invariants across all chunks, and that
  > the IDs are read dynamically from the tokenizer (tested with non-default
  > IDs 111/222) rather than hardcoded. Also had to add `bos_token_id: 0` /
  > `eos_token_id: 2` to the other pre-existing fake tokenizers in
  > `susfactor.test.ts`, `susfactor-create.test.ts`, and
  > `root-exports-susfactor.test.ts` — those call `classify()` for real and
  > started failing (`Cannot convert undefined to a BigInt`) once
  > `classify()` began reading `bos_token_id`/`eos_token_id` off the
  > tokenizer. Verified with `npx jest susfactor` (5 suites, 50 passed, 5
  > skipped — model-gated) and `npm run build`/`lint`/`format`, all clean.

<!-- MAESTRO:MODEL tier="default" effort="low" -->

- [x] Run the TypeScript test suite for `packages/typescript/test/susfactor-chunking.test.ts` (check `packages/typescript/package.json` for the exact test command, e.g. `npm test` or `vitest run susfactor-chunking`) and fix any failures until it's green. Then run a real prompt longer than 510 content tokens through the classifier end-to-end and decode each returned chunk's token IDs to confirm every chunk starts with `<s>` and ends with `</s>` — report the result in your final summary; remove any throwaway script used for this check but keep the permanent test added in the previous task.

  > Implementation notes: `npx jest susfactor-chunking` was already green
  > (no failures to fix) — 24 passed, 3 skipped (model-gated). Ran the
  > full `npx jest susfactor` suite too: 5 suites, 50 passed, 5 skipped,
  > and `npm run build` (tsc) clean.
  >
  > For the real end-to-end check, found a cached real model at
  > `/Users/sgolub/.cache/signature-sdk/models/susfactor-v1` (matches the
  > `SUSFACTOR_MODEL_DIR` layout the tests expect: `onnx/model.onnx` +
  > `onnx/model.onnx_data` + `tokenizer.json`). `onnxruntime-node`,
  > `@huggingface/transformers`, and `sharp` are peer deps not present in
  > `packages/typescript/node_modules`, so they were installed
  > temporarily (not saved to `package.json`/lockfile) to run this
  > one-off check. Wrote a throwaway `scratch-e2e-boseos.ts` that: loads
  > `SusFactorClassifier.create()` against the real cache, builds a
  > 600-word prompt (>510 content tokens), monkey-patches
  > `session.run` to capture the exact `input_ids`/`attention_mask`
  > tensors sent to the model for every chunk, then asserts on the
  > captured tensors directly (not just on `chunkTokenIds` output).
  > Result: 3 chunks (lengths 512/512/485 including the wrapped BOS/EOS),
  > every chunk's `input_ids` starts with `0` (`<s>`) and ends with `2`
  > (`</s>`), and every attention mask is all-1s of length
  > `chunk.length`. Scores: `[0.257, 0.200, 0.315]`, all `safe`. Deleted
  > `scratch-e2e-boseos.ts` afterward — no permanent script left behind
  > (the permanent regression coverage is the test file from the
  > previous task, untouched). The temporarily-installed peer deps live
  > only in the local, gitignored `node_modules/` and were not added to
  > `package.json` or `package-lock.json`.

## Manual Follow-Up (not executed by Auto Run)

- The "migrate the served model to the sanitized v2 checkpoint" half of 0DIN-2132 is scoped to a **separate ticket** (Vertex dev/stage/prod upload checklist) and is intentionally not covered by this playbook.
