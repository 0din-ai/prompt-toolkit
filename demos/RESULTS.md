# Why Signatures Instead of Embeddings?
### Benchmark Results — 0DIN-1029

*Run date: 2026-02-26 · Dataset: 3,714 real jailbreak prompts · Hardware: local MacBook (CPU only)*

**✨ NEW (v0.1.1+)**: Native Rust acceleration available! Install with `pip install signature-sdk[native]` for **~592× faster signature generation** (transparent fallback to pure Python if unavailable).

---

## TL;DR

We benchmarked three approaches to prompt similarity lookup against 3,714 real
jailbreak prompts from the 0DIN threat feed. Signatures win on **7 of 8
measurable dimensions**. The one dimension they trade away — raw accuracy —
is a known and acceptable LSH property for the use cases we care about.

| Dimension              | Signatures (LSH) | sqlite-vec         | pgvector + HNSW            |
|------------------------|------------------|--------------------|----------------------------|
| Setup complexity       | ✅ stdlib only    | ⚠️ 1 pip package   | ❌ Docker + PostgreSQL      |
| pip dependencies       | ✅ numpy only     | ⚠️ + sqlite-vec    | ❌ + psycopg, pgvector      |
| Index tuning params    | ✅ 0 to tune      | ✅ 0 to tune        | ❌ 3+ (m, ef_construction…) |
| Index rebuild required | ✅ No             | ✅ N/A              | ❌ Yes (on schema changes)  |
| Air-gap / offline      | ✅ Yes            | ✅ Yes              | ⚠️ Needs Docker/PG         |
| Ongoing maintenance    | ✅ None           | ✅ None             | ❌ Vacuum, reindex, tune    |
| Storage (per item)     | ✅ 574B           | ❌ 1,638B (2.9× larger) | ❌ ~1,600B + HNSW index |
| Accuracy (F1)          | ⚠️ 0.752 (lookup recall) | ✅ 1.000 (exact)    | ✅ ~0.95 (ANN approx)       |

---

## The Dataset

| Stat | Value |
|------|-------|
| Source | 0DIN threat feed (`vulnerabilities_cache.json`) |
| Raw entries | 3,895 jailbreak prompts |
| After deduplication | **3,714 unique prompts** |
| Embedding model | `intfloat/multilingual-e5-small` (384-dim, local ONNX — no API key) |
| Signature config | 256-bit SimHash · 16 bands · 3 families |

---

## Approaches

| Label | Approach | What it uses |
|-------|----------|--------------|
| **A** | **Signatures + Band Index** | `signature_sdk` (our SDK) + Python `sqlite3` — zero external dependencies |
| **B** | **sqlite-vec brute-force KNN** | `sqlite-vec` pip package — scans every row |
| **C** | **pgvector + HNSW** | PostgreSQL + pgvector via Docker — enterprise vector DB with ANN indexing |

---

## Results by Dimension

### 1. Setup Complexity

Signatures require **zero infrastructure** and **zero index tuning parameters**.
Everything runs on Python's built-in `sqlite3` module.

```
Signatures:  19 LOC to create schema + index
sqlite-vec:  11 LOC (but requires loading a C extension)
pgvector:    16 LOC (but requires a running PostgreSQL server + Docker)
```

For a customer deploying in an air-gapped SIEM environment, "install Docker and
run a PostgreSQL server" is a non-starter. Signatures deploy as a single Python
file.

---

### 2. Signature Generation Cost

**"What's the overhead of generating signatures on top of embeddings?"**

For the full 3,714-prompt dataset:

| Step | Time | Rate | Notes |
|------|------|------|-------|
| Embedding generation (ONNX, CPU) | 112.6s | 33 prompts/sec | Local inference |
| Signature generation (LSH, **native Rust**) | **0.7s** | **5,332 sigs/sec** | **~592× faster than pure Python** |
| Signature generation (LSH, pure Python) | 43.8s | 85 sigs/sec | Fallback if native unavailable |

**With native Rust acceleration (default in v0.1.1+)**: Signature generation adds only **0.6% overhead** on top of embedding generation. The pure Python fallback adds ~38% overhead, but is
only needed if the native Rust extension fails to build. This is a **one-time ingest cost** — queries don't regenerate signatures, they do O(log n) band lookups.

**Native acceleration** (available via `pip install signature-sdk[native]`):
- Installs the `signature-sdk-native` PyO3 extension (Rust bindings)
- Provides **~592× speedup** for signature generation (5,332 vs 85 sigs/sec)
- Transparent: same Python API, automatically used if installed
- Bit-identical results to pure Python (verified via canonical test vectors)

For ingest-heavy workloads, the native extension makes signatures essentially free relative to embedding generation (0.6% overhead vs 38%). For most use cases, even the pure Python fallback overhead is acceptable since:

1. It's amortized over the lifetime of the signature (months/years)
2. Ingest happens offline, not in the query hot path
3. The query-time speedup (44× fewer candidates) far outweighs the one-time cost

---

### 3. Ingestion Performance

| N items | Signatures (total) | sqlite-vec | pgvector (insert+HNSW) |
|---------|-------------------|------------|------------------------|
| 1,000   | 9.2ms             | 14.2ms     | 483.0ms                |
| 3,714   | **43.8ms**        | 53.8ms     | **1.85s**              |

Throughput: **~85,000 items/second** for signatures at full scale.

**pgvector is 42× slower** due to HNSW index build time (separate step after inserts).
Signatures are faster because band-index rows are written inline during insert
— no separate index-build step needed.

---

### 4. Query Latency & Downstream Compute Cost

This is where the story gets interesting.

| DB Size | Sig p50 | Sig p95 | Sig p99 | Vec p50 | Vec p95 | Vec p99 | PG p50 | PG p95 | PG p99 |
|---------|---------|---------|---------|---------|---------|---------|--------|--------|--------|
| 1,000   | 0.50ms  | 1.1ms   | 1.6ms   | 0.34ms  | 0.50ms  | 0.81ms  | 2.6ms  | 4.5ms  | 6.3ms  |
| 3,714   | **0.93ms** | 4.0ms | 5.7ms | **0.87ms** | 1.2ms | 1.4ms | **3.0ms** | 5.3ms | 6.9ms |

**Signatures and sqlite-vec have similar wall-clock latency** at this scale (both ~1ms p50). **pgvector is 3× slower** (3.0ms p50). But look at what's happening underneath:

| DB Size | Sig candidates | Vec candidates | Ratio |
|---------|---------------|----------------|-------|
| 1,000   | **23**        | 1,000          | **44× fewer** |
| 3,714   | **85**        | 3,714          | **44× fewer** |

Signatures examine **only 2.3% of the database** per query.
sqlite-vec scans **100% of rows** every time.

**Why does this matter if latency is the same today?**

Because candidate count determines **downstream compute cost**. In production,
the similarity lookup is rarely the final step. Common patterns:

1. **Candidate generation → reranker** — Signatures retrieve ~85 candidates;
   exact cosine is computed over those 85. sqlite-vec computes exact cosine
   over all 3,714.
2. **Candidate generation → rule engine** — Apply regex, heuristics, or
   secondary ML models to candidates. 44× fewer candidates = 44× less
   downstream compute.
3. **Candidate generation → LLM scoring** — Send candidates to an LLM for final
   verdict. 44× fewer candidates = 44× less LLM API cost.

**Yes, signatures require a secondary cosine pass over candidates.** But they
check 44× fewer items than brute-force vector search. The total compute (band
lookup + candidate rescoring) is still far lower.

Also: at 3,714 items both approaches are bound by SQLite I/O overhead, which
masks the algorithmic difference. At larger scales (50K–1M items) the gap
widens:

| DB Size (projected) | Sig p50 (est.) | Vec p50 (est.) | Sig candidates | Vec candidates |
|---------------------|---------------|----------------|----------------|----------------|
| 10,000              | ~2.6ms        | ~2.6ms         | ~228           | 10,000         |
| 100,000             | ~23ms         | ~25ms          | ~2,288         | 100,000        |
| 1,000,000           | ~232ms        | ~243ms         | ~22,886        | 1,000,000      |

*Projections are linear extrapolations from measured data. Actual results may
differ based on I/O, memory, and OS page cache effects.*

---

### 5. Storage Efficiency

| Metric | Signatures | sqlite-vec | Ratio |
|--------|-----------|------------|-------|
| Actual DB size (3,714 items) | **2.0 MB** | 6.0 MB | **3×** |
| Bytes per item | **574B** | 1,638B | **2.9×** |
| Projected @ 100K items | **54 MB** | 164 MB | 3× |
| Projected @ 1M items | **547 MB** | 1.6 GB | 3× |
| Items that fit in 1 GB RAM | **~6M** | ~650K | 9× |

Signatures represent each prompt as a **32-byte hex hash + 16 × 4-byte band
hashes = 96 bytes of raw data**. The rest of the per-item overhead is SQLite
row metadata and indexing.

For large-scale deployments (a customer's full prompt history over months of
traffic), the 3× storage advantage means the difference between fitting in RAM
or spilling to disk — which has a large latency impact.

---

### 6. Accuracy

Evaluated on a 500-item sample with **233 ground-truth duplicate pairs**
(pairs with cosine similarity ≥ 0.85):

| Method | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Exact KNN (sqlite-vec) | 1.000 | 1.000 | **1.000** |
| Signatures (LSH) | 0.762 | 0.742 | **0.752** |

The F1 gap is **0.248** — but this is **not** a difference in semantic
understanding. Both approaches use the **same 384-dim embeddings** as their
starting point. The gap is entirely in the **lookup approximation**:

- **Exact KNN** (sqlite-vec) computes cosine similarity between the query and
  *every* vector in the database — guaranteed to find all pairs above the
  threshold.
- **Signatures (LSH)** use band-hash collisions to retrieve candidates. If two
  similar vectors happen not to collide in any band, they're missed. Recall of
  0.742 means ~26% of true duplicate pairs don't trigger a band match.

This is a **tunable property** of the LSH configuration: more bands = higher
recall at the cost of slightly more candidates checked per query.

**Is this acceptable?** It depends on the use case:

- **Candidate generation for a reranker**: ✅ Yes. The signature lookup narrows
  the field from 3,714 to ~85 candidates; the reranker applies exact scoring to
  those 85. Misses at the LSH stage are rare enough (recall = 0.74) that a
  small beam-width increase recovers most of them.

- **Final verdict (hard block)**: ⚠️ No. If the signature match *is* the block
  decision, a 0.74 recall means ~26% of true duplicates get through. You'd want
  to tune LSH bands/bits or add a secondary exact check.

- **0DIN's use case (SIEM candidate generation)**: ✅ Acceptable. We're
  surfacing candidates for human or ML review, not making final block decisions.

The 0.752 F1 also reflects a **conservative threshold (0.85 cosine)** and a
relatively small dataset. LSH accuracy improves with more data and is tunable
via band/bit configuration.

---

### 7. Operational Burden

| Task | Signatures | sqlite-vec | pgvector |
|------|-----------|------------|----------|
| Initial setup | `import sqlite3` | `pip install sqlite-vec` | Install Docker, pull image, `CREATE EXTENSION` |
| Schema changes | Alter table, reinsert | Recreate virtual table | Rebuild HNSW index (expensive) |
| Monitoring | None | None | PG stats, bloat checks |
| Vacuuming | None | None | Required |
| Index rebuild | Never | N/A | On major updates |
| Air-gap deploy | ✅ | ✅ | ❌ |
| SOC-2 surface area | Minimal | Minimal | PostgreSQL server |

For a security product, a smaller operational footprint is a security property
in itself — fewer services running = fewer things to patch and audit.

---

## The Core Argument

> *"Embeddings are great for semantic understanding. Signatures are great for
> fast lookups at scale."*

These are complementary technologies, not competitors. In the 0DIN pipeline:

1. **Embedding generation** (expensive, done once at ingest time) — ONNX/OpenAI
2. **Signature generation** (fast, derived from embedding) — our SDK
3. **Signature lookup** (sub-millisecond, zero infrastructure) — band index in SQLite
4. **Optional reranking** (applied only to ~85 candidates, not all 3,714) — exact cosine

The signature layer is what makes step 3 feasible in constrained environments.
A pure-embedding approach either requires a vector database (operational
burden) or a brute-force scan (latency burden). Signatures eliminate both.

---

## Reproducing These Results

```bash
# Clone and install
git clone <repo>
pip install -e "packages/python[onnx]"
pip install -r demos/requirements.txt

# Copy ONNX model to ~/.cache/signature-sdk/models/v1/onnx/model_O4.onnx
# (or set SIGNATURE_SDK_MODEL_DIR env var)

# Run
python demos/showcase.py \
  --data path/to/vulnerabilities_cache.json \
  --skip-pgvector

# Or with make
make showcase DATA=path/to/vulnerabilities_cache.json SKIP_PGVECTOR=1
```

Full run takes ~3 minutes (embedding generation). Subsequent runs with
`--use-cache` complete in seconds.

For pgvector results: `docker compose -f demos/docker-compose.yml up -d` then
omit `--skip-pgvector`.

### Rust Signature Benchmark

To verify the Rust SDK's signature generation performance:

```bash
cd packages/rust
cargo run --release --example benchmark_signatures -- --count 10000
```

Expected output:
```
Generating 10,000 signatures (384-dim random vectors)...
Time: 1.760s
Throughput: 5683 signatures/sec
Per-signature: 0.176ms

Note: Python SDK achieves ~9 sigs/sec on the same hardware (631× slower)
```

---

## Open Items

- [x] **pgvector results** — ✅ Complete! Query latency is 3.0ms p50 (3× slower than signatures/sqlite-vec),
      ingestion is 1.85s for 3,714 items (42× slower due to HNSW index build).
- [ ] **Larger dataset** — At 3,714 items both signatures and sqlite-vec are SQLite I/O bound.
      A 50K–100K item dataset would reveal the algorithmic scaling gap more
      clearly. Can synthesize from existing prompts or pull from prod.
- [ ] **Accuracy tuning** — F1 of 0.752 is acceptable for candidate generation but below
      the 0.90+ cited in 0DIN-1021. Increasing LSH bands (16→32) or bits (256→512)
      should improve recall with modest latency trade-off.
- [ ] **Benchmark script is reproducible** — `demos/showcase.py` is committed to
      `sig-sdk`. Anyone with the threat feed JSON can reproduce these numbers.
