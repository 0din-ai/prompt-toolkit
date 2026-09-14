# SusFactor Chunk BOS/EOS Fix — Phase 1: Go

## Background (context for every task below)

0DIN-2132 reported that SusFactor's sliding-window chunker is missing special
tokens (`<s>`=BOS=0, `</s>`=EOS=2, XLM-RoBERTa-based model) on interior chunks.
Root cause: `packages/go/susfactor/classifier.go` tokenizes the **full** input
text *with* special tokens added by the tokenizer, then slices that one flat
ID array into overlapping chunks via `ChunkTokenIDs` (`chunk.go:15`) — even
though `chunk.go`'s own docstring says the caller must supply payload-only
IDs. Result: only chunk 0 keeps a real leading `<s>`, only the last chunk
keeps a real trailing `</s>`; every interior chunk has neither, despite
`MaxContentTokens = MaxSequenceLength - 2` (`types.go:14-24`) already
reserving exactly 2 slots per chunk for them. The attention mask is also
built as `allMask[:chunkLen]` (always sliced from position 0) — currently
harmless only because there is never any padding in these inputs, but it is
the same "flat array sliced naively" bug and should be replaced, not kept.

`ChunkTokenIDs` itself is correct and needs no changes — only the caller
around it.

- [x] In `packages/go/susfactor/classifier.go`, fix the chunking pipeline so every chunk gets real special tokens instead of only the first/last:
  1. Change the tokenize call at ~line 257 (`c.tokenizer.EncodeWithOptions(text, true, ...)`) to pass `false` for `addSpecialTokens`, so `allIDs` becomes pure content-token IDs with no BOS/EOS mixed in.
  2. Leave the call to `ChunkTokenIDs(allIDs)` (~line 269) unchanged — it already expects payload-only IDs per its docstring in `chunk.go`, and now receives them correctly.
  3. In the per-chunk loop that builds inference inputs (~lines 268-317), explicitly wrap each chunk: `[bosID] + chunk + [eosID]` instead of relying on tokens that came from the original tokenize call.
  4. Replace the attention-mask construction `allMask[:chunkLen]` with a freshly-built all-1s mask of length `len(chunk) + 2` (accounting for the two added special tokens).
  5. Resolve `bosID` and `eosID` dynamically from the tokenizer rather than hardcoding `0`/`2` — check whether Go's `daulet/tokenizers` binding (used by `c.tokenizer`) exposes a `TokenToID("<s>")`/`TokenToID("</s>")`-style lookup or equivalent vocab lookup, and use it. If no such lookup exists on the binding, read the two IDs once from the tokenizer's own config/vocab at load time (not a hardcoded literal in the chunking loop) and document why in a short comment.
  - Do not modify `chunk.go` — its splitting logic is already correct for payload-only input.

  **Done:** `c.tokenizer.EncodeWithOptions(text, false)` now yields content-only `allIDs`, unchanged into `ChunkTokenIDs`. Added a new `buildChunkInput(chunkIDs, bosID, eosID)` helper that wraps every chunk as `[bosID] + chunk + [eosID]` and builds a fresh all-1s mask sized `len(chunk)+2` — replacing the `allMask[:chunkLen]` slice entirely. Confirmed `github.com/daulet/tokenizers@v1.27.0`'s `Tokenizer` type exposes no `TokenToID`/vocab-lookup method (only `Encode`/`Decode`/`VocabSize`), so per the fallback instruction, `bosID`/`eosID` are now resolved once at load time in `loadTokenizerNoTruncation` (new `resolveBOSEOS` helper) by reading the `added_tokens` table already present in the parsed `tokenizer.json` (the same doc that's edited to strip `truncation`), matching entries with `content == "<s>"` / `"</s>"`. Values are stored on `SusFactorClassifier` as `bosID`/`eosID` fields instead of being hardcoded in the chunking loop. `chunk.go` was not modified. `gofmt` and `go vet ./...` are clean.

- [x] Add test coverage in `packages/go/susfactor/chunk_test.go` (or a new `_test.go` file alongside `classifier.go` if per-chunk special-token placement can't be tested via `ChunkTokenIDs` alone, since that function only sees content IDs) asserting:
  - For a short input that produces a single chunk: the chunk's first token equals `bosID` and its last token equals `eosID`, and the chunk count/content-token boundaries match what the pre-fix code already produced correctly (this case was already right; assert it stays right).
  - For a long input (>510 content tokens) that produces multiple chunks: **every** chunk (including interior ones) starts with `bosID` and ends with `eosID` — this is the specific regression this fix addresses, since before the fix only the first/last chunk had real special tokens.
  - The attention mask for each chunk is all-1s with length `len(chunk_content) + 2`.

  **Done:** Added `packages/go/susfactor/special_tokens_test.go` (new file, since this exercises `buildChunkInput` in `classifier.go`, not `ChunkTokenIDs`). `TestBuildChunkInputSpecialTokens` covers both the short single-chunk case and a 1024-content-token case (3 chunks via `ChunkTokenIDs`), asserting every chunk — including interior chunk 1 — starts with `bosID`, ends with `eosID`, preserves content in between, and has an all-1s mask of length `len(content)+2`. Also added `TestLoadTokenizerResolvesBOSEOS`, which loads the real bundled `models/v1/tokenizer.json` (no `SUSFACTOR_MODEL_DIR`/ONNX model needed) and asserts the resolved IDs match the known XLM-RoBERTa vocab (`<s>=0`, `</s>=2`). Updated the one existing caller of `loadTokenizerNoTruncation` (`tokenizer_truncation_test.go`) for the new 4-return-value signature. Full `packages/go/susfactor` suite passes (`go test ./susfactor/... -v`); model-gated tests skip as before since `SUSFACTOR_MODEL_DIR` is unset in this environment.

<!-- MAESTRO:MODEL tier="default" effort="low" -->

- [ ] Run `go test ./packages/go/susfactor/...` from the repo root and fix any failures until the suite is green. Then write a short Go program or test (can be a `_test.go` with `t.Skip` removed temporarily, or a one-off `go run`) that classifies a real prompt longer than 510 content tokens through the public `Classify` API and decodes each returned chunk's `input_ids` to confirm every chunk literally starts with `<s>` and ends with `</s>` when decoded — report the result in your final summary; remove any throwaway script you created for this check but keep the permanent test added in the previous task.

## Manual Follow-Up (not executed by Auto Run)

- The "migrate the served model to the sanitized v2 checkpoint" half of 0DIN-2132 is scoped to a **separate ticket** (Vertex dev/stage/prod upload checklist) and is intentionally not covered by this playbook.
