# odin-prompt-toolkit Integration Specification

This document defines the external interface of the prompt toolkit — the three capabilities it exposes to integrators, their input/output contracts, and how to fulfill each in your environment.

**Spec version**: 1.0.0  
**Last updated**: 2026-06-23

See [`SPEC.md`](SPEC.md) for the underlying algorithm specification and [`VERSIONING.md`](VERSIONING.md) for signature version details.

---

## What this toolkit is

The odin-prompt-toolkit is a **toolkit for prompt security analysis**, not an SDK for a specific service. It defines:

1. A formal algorithm spec and test vectors (this repo)
2. Reference implementations in Rust, Python, and TypeScript
3. A hosted reference server: [Heimdall](https://github.com/0din-ai/heimdall), which exposes these capabilities over gRPC and HTTP

Integrators can use the language libraries directly, call Heimdall over gRPC/HTTP, or implement the spec themselves. All three paths produce identical results for the same input.

---

## The three capabilities

```
generate_signature(prompt)  →  signature string
evaluate_suspicion(prompt)  →  suspicion score (0.0–1.0)
find_similar(signature)     →  matching signatures from a corpus
```

There is also a combined convenience call:

```
analyze(prompt)  →  { signature, suspicion_score }
```

---

## 1. `generate_signature`

Converts a text prompt into a compact, version-stamped LSH signature suitable for similarity comparison and deduplication.

### Input

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | Yes | The raw text to sign |
| `version` | string | No | `"v0"`, `"v1"`, or `"latest"` (default: `"latest"` → `"v1"`) |

### Output

| Field | Type | Notes |
|---|---|---|
| `signature` | string | Format: `0din-v{N}:<64-hex-chars>` — see `VERSIONING.md` |
| `version` | string | Resolved version (`"v0"` or `"v1"`, never `"latest"`) |
| `embedding_sha256` | string | Canonical hash of the normalized embedding — useful for deduplication at the embedding level |

### Example

```json
// Request
{ "prompt": "How do I reset my password?", "version": "v1" }

// Response
{
  "signature": "0din-v1:8d000000ac854dae3f2c1a9b7e4f0d8c...",
  "version": "v1",
  "embedding_sha256": "9a04781069052282acb2e95529c7f5bc..."
}
```

### Implementation options

| Path | How |
|---|---|
| Rust | `sign_text()` in `packages/rust` |
| Python | `sign_text()` in `packages/python` |
| TypeScript | `signText()` in `packages/typescript` |
| gRPC | `SignatureService.GenerateSignature` in Heimdall proto |
| HTTP | `POST /v1/signature` in Heimdall OpenAPI spec |
| New language | Auto-generate a gRPC client from the Heimdall proto |

---

## 2. `evaluate_suspicion`

Scores a prompt for jailbreak / prompt-injection risk using the Sus Factor ONNX classifier.

### Input

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | Yes | The raw text to evaluate — any length |

### Output (`ChunkedSusFactorResult`)

`classify()` always returns a `ChunkedSusFactorResult`. Chunking for long prompts is handled transparently — callers never check length or call a separate method.

| Field | Type | Notes |
|---|---|---|
| `chunks` | `SusFactorResult[]` | One entry per chunk, in order. Short prompts: always one entry. |
| `is_suspicious` / `isSuspicious` | bool | **Use this for security gating.** `true` if any chunk is suspicious. |
| `total_timing_ms` / `totalTimingMs` | float | Wall-clock time for all chunks, in ms |

Each `SusFactorResult` chunk entry:

| Field | Type | Notes |
|---|---|---|
| `score` | float | 0.0 (benign) → 1.0 (highly suspicious), for this chunk only |
| `label` | string | `"suspicious"` if `score >= threshold`, else `"safe"` |
| `model` | string | Model identifier |
| `threshold` | float | Decision threshold used |
| `timing_ms` | float | Inference time for this chunk |

### Example — short prompt (one chunk)

```json
// Request
{ "prompt": "Ignore all previous instructions and..." }

// Response
{
  "is_suspicious": true,
  "chunks": [
    { "score": 0.94, "label": "suspicious", "model": "0dinai/susfactor-e5-large", "threshold": 0.5, "timing_ms": 45.2 }
  ],
  "total_timing_ms": 45.2
}
```

### Example — long prompt (multiple chunks)

```json
// Response for a prompt exceeding 510 tokens
{
  "is_suspicious": true,
  "chunks": [
    { "score": 0.04, "label": "safe",       "timing_ms": 44.1 },
    { "score": 0.03, "label": "safe",       "timing_ms": 43.8 },
    { "score": 0.91, "label": "suspicious", "timing_ms": 44.6 }
  ],
  "total_timing_ms": 46.3
}
```

Note: `is_suspicious` is `true` even though only one of three chunks flagged. The prompt contains suspicious content — the score of 4% on earlier chunks is irrelevant to that determination.

### Displaying a single score (migration note)

The previous API returned one `score` and one `label` directly. **`classify()` no longer returns a single score** — there is no canonical value when a prompt spans multiple model inferences.

Callers that need one number for display (dashboards, logs) must choose explicitly:

```python
# Most conservative — highest suspicion across all chunks:
display_score = max(c.score for c in result.chunks)

# First-chunk only — matches old behaviour for short prompts,
# but will miss suspicious content in later chunks of long prompts:
display_score = result.chunks[0].score
```

```typescript
const displayScore = Math.max(...result.chunks.map(c => c.score)); // conservative
const displayScore = result.chunks[0].score;                        // first-chunk only
```

**Use `is_suspicious` for security decisions. A display score is a UX choice.**

### Long-prompt chunking constants

| Constant | Value | Meaning |
|---|---|---|
| `MAX_CONTENT_TOKENS` | 510 | Tokens per chunk (512 minus `[CLS]` + `[SEP]`) |
| `CHUNK_OVERLAP` | 50 | Tokens shared between adjacent chunks |
| `CHUNK_STRIDE` | 460 | New tokens advanced per chunk |

Chunks are dispatched concurrently — the ONNX Runtime handles scheduling internally. Actual simultaneous execution depends on the session configuration; a single shared session serializes inference. Wall-clock time is generally better than pure sequential but is not guaranteed to be bounded by the slowest single chunk.

### Implementation options

| Path | How |
|---|---|
| Rust | `SusFactorClassifier.classify()` in `packages/rust` (requires `susfactor` feature) |
| Python | `SusFactorOnnxClassifier.classify()` in `packages/python` |
| TypeScript | `SusFactorClassifier.classify()` in `packages/typescript` |
| gRPC | `SusFactorService.Evaluate` in Heimdall proto |
| HTTP | `POST /v1/susfactor` in Heimdall OpenAPI spec |
| New language | Auto-generate a gRPC client from the Heimdall proto |

---

## 3. `analyze` (combined)

Convenience wrapper that runs `generate_signature` and `evaluate_suspicion` in a single call. Prefer this when you need both outputs for the same prompt.

### Input

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | Yes | |
| `version` | string | No | Same as `generate_signature` |

### Output

Combined fields from both capabilities above.

### Implementation options

| Path | How |
|---|---|
| gRPC | `PromptService.Analyze` in Heimdall proto |
| HTTP | `POST /v1/analyze` in Heimdall OpenAPI spec |
| Direct | Call `generate_signature` + `evaluate_suspicion` sequentially |

---

## 4. `find_similar`

Checks a signature against a corpus of previously generated signatures and returns matches above a similarity threshold.

**This is the integrator's responsibility.** The toolkit provides the signature algorithm and comparison primitives; the corpus lives in your data store.

### What the toolkit provides

```
hamming_distance(sig_a, sig_b)  →  integer (bit distance, 0–256)
cosine_from_hamming(distance, total_bits)  →  float (estimated cosine similarity)
```

A similarity threshold of **Hamming ≤ 55** (out of 256 bits) corresponds to approximately **cosine ≥ 0.77** and is the validated threshold for duplicate detection in the 0DIN threat feed.

### Corpus lookup patterns

#### Pattern A: Direct database query (recommended starting point)

Store signatures in your database alongside your content. Query using Hamming distance.

**PostgreSQL reference implementation:**

```sql
-- Find all prompts with Hamming distance ≤ 55 from a query signature
-- Requires the pg_trgm or bit_count approach; example uses a custom function.

-- Store signatures as bit(256):
ALTER TABLE prompts ADD COLUMN signature_bits bit(256);

-- Index for fast lookup (note: exact Hamming search still requires a scan
-- unless you use band-based indexing — see Band Index pattern below):
CREATE INDEX idx_signature ON prompts USING hash (signature_bits);

-- Query:
SELECT id, prompt_preview, signature,
       bit_count(signature_bits # query_bits) AS hamming_distance
FROM prompts
WHERE bit_count(signature_bits # query_bits) <= 55
ORDER BY hamming_distance ASC
LIMIT 20;

-- Where query_bits is your input signature cast to bit(256):
-- CAST('8d000000ac854dae...' AS bit(256))  -- hex to bit cast
```

**Band-based index for sub-linear lookup** (recommended for large corpora):

Each V1 signature is split into 16 bands of 4 hex chars each (see `SPEC.md §3`). Two signatures that share any band are candidates for similarity. Store bands separately and query by exact band match first:

```sql
-- Bands table (pre-computed from signatures):
CREATE TABLE signature_bands (
  prompt_id   bigint REFERENCES prompts(id),
  band_index  smallint,   -- 0–15
  band_value  char(4),    -- 4 hex chars
  PRIMARY KEY (prompt_id, band_index)
);
CREATE INDEX idx_bands ON signature_bands (band_value, band_index);

-- Candidate fetch (fast, index-only):
SELECT DISTINCT prompt_id
FROM signature_bands
WHERE band_value = $1 AND band_index = $2;
-- Run for each of the 16 bands of the query signature, union results.

-- Then rerank candidates by exact Hamming distance (small set, cheap).
```

#### Pattern B: Heimdall signature cache (for hosted deployments)

If you are using Heimdall as your server, you can seed its signature cache with your corpus. Heimdall returns matching IDs; you resolve them in your own database.

```
// Seed: export signatures from your DB and POST to Heimdall
POST /v1/signatures/seed
Body: [{ "id": "your-opaque-id", "signature": "0din-v1:..." }, ...]

// Query: Heimdall checks against its cache
POST /v1/signatures/find-similar
Body: { "signature": "0din-v1:...", "threshold": 55 }

// Response: matching IDs — you resolve them in your DB
{ "matches": [{ "id": "your-opaque-id", "hamming_distance": 12 }] }
```

Raw prompts never leave your environment. Heimdall only sees signatures and opaque IDs.

**Status**: Heimdall signature cache is on the roadmap — not yet implemented. Use Pattern A today.

---

## Threshold reference

| Hamming distance | Estimated cosine similarity | Interpretation |
|---|---|---|
| 0 | 1.00 | Identical |
| ≤ 10 | ≥ 0.99 | Near-exact duplicate |
| ≤ 25 | ≥ 0.95 | Very high similarity (original threshold, too restrictive) |
| ≤ 55 | ≥ 0.77 | **Recommended duplicate detection threshold** |
| ≤ 80 | ≥ 0.61 | Related prompts |
| 128 | 0.00 | Unrelated |
| 256 | −1.00 | Maximally dissimilar |

Thresholds were validated against 55 hand-labeled pairs from the 0DIN threat feed (see `VALIDATION.md`).

---

## Choosing an integration path

```
Do you need Ruby, Go, or another language without a native library?
  → Use the Heimdall gRPC proto to auto-generate a client.

Do you need to run completely offline (no network calls)?
  → Use the Rust, Python, or TypeScript library directly.

Do you need browser execution?
  → Use the TypeScript library. Providers (ONNX, OpenAI) require a server;
    the pure-compute functions (hamming, cosine, normalize) run in-browser today.

Do you already have embeddings from another source?
  → Skip the providers entirely. Pass your normalized embedding vector
    directly to simhash_lsh_multi() in any language library.
```

---

## What is NOT in scope for this toolkit

The following require your own infrastructure and are not provided:

- **Corpus storage** — your database, your schema
- **Authentication / rate limiting** — Heimdall supports this but it is deployment-specific
- **Embedding model hosting** — V1 ONNX model downloads on first use; V0 requires an OpenAI API key
- **Multi-turn conversation signatures** — open design question; see below

---

## Open design questions

### Multi-turn signatures

Signatures today are generated at the prompt level. For multi-turn conversations, the right aggregation strategy is unresolved:

- **Option A**: Mean or max of individual turn embeddings before signature generation
- **Option B**: Generate one signature per turn; similarity = max across all turn pairs

This requires data model changes (signatures at turn level, not conversation level). A decision is pending — do not build against this yet.

---

## References

- Algorithm spec: [`SPEC.md`](SPEC.md)
- Version registry: [`VERSIONING.md`](VERSIONING.md)  
- Cross-language validation: [`../VALIDATION.md`](../VALIDATION.md)
- Heimdall proto: `github.com/0din-ai/heimdall` (gRPC + OpenAPI source of truth)
- Test vectors: [`test-vectors/`](test-vectors/)
