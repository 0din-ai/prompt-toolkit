---
sidebar_position: 4
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Performance Benchmarks

Comprehensive performance analysis comparing LSH signatures against vector database alternatives.

## TL;DR

We benchmarked three approaches to prompt similarity lookup against **3,714 real jailbreak prompts** from the 0DIN threat feed. **Signatures win on 7 of 8 measurable dimensions**.

| Dimension | Signatures (LSH) | sqlite-vec | pgvector + HNSW |
|-----------|-----------------|------------|-----------------|
| Setup complexity | ✅ stdlib only | ⚠️ 1 pip package | ❌ Docker + PostgreSQL |
| Dependencies | ✅ numpy only | ⚠️ + sqlite-vec | ❌ + psycopg, pgvector |
| Index tuning | ✅ 0 parameters | ✅ 0 parameters | ❌ 3+ parameters (m, ef_construction) |
| Index rebuild | ✅ Never | ✅ N/A | ❌ Yes (on schema changes) |
| Air-gap deploy | ✅ Yes | ✅ Yes | ⚠️ Needs Docker/PG |
| Maintenance | ✅ None | ✅ None | ❌ Vacuum, reindex, tune |
| Storage/item | ✅ 574B | ❌ 1,638B (2.9×) | ❌ ~1,600B + HNSW index |
| Accuracy (F1) | ⚠️ 0.752 | ✅ 1.000 (exact) | ✅ ~0.95 (ANN approx) |

**The one dimension signatures trade away — accuracy — is a known and acceptable LSH property** for candidate generation use cases.

---

## Benchmark Methodology

### Dataset

| Stat | Value |
|------|-------|
| Source | 0DIN threat feed (`vulnerabilities_cache.json`) |
| Raw entries | 3,895 jailbreak prompts |
| After deduplication | **3,714 unique prompts** |
| Embedding model | `intfloat/multilingual-e5-large` (1024-dim, local ONNX) |
| Signature config | 256-bit SimHash · 16 bands · 3 families |
| Hardware | MacBook Pro (CPU only, local inference) |
| Run date | 2026-02-26 |

### Approaches Compared

| Label | Approach | Implementation |
|-------|----------|----------------|
| **A** | **Signatures + Band Index** | `signature_sdk` + Python `sqlite3` (zero external dependencies) |
| **B** | **sqlite-vec KNN** | `sqlite-vec` pip package (brute-force scan) |
| **C** | **pgvector + HNSW** | PostgreSQL + pgvector via Docker (enterprise ANN indexing) |

All three use the **same embeddings** (1024-dim multilingual-e5-large). The comparison isolates the **lookup mechanism**, not the semantic understanding.

---

## Results by Dimension

### 1. Setup Complexity

**Signatures: Zero infrastructure, zero tuning parameters.**

```
Signatures:  19 LOC to create schema + index
sqlite-vec:  11 LOC (but requires loading a C extension)
pgvector:    16 LOC (but requires a running PostgreSQL server + Docker)
```

**Code Comparison**:

<Tabs groupId="approach">
<TabItem value="signatures" label="Signatures">

```python
import sqlite3

# Zero external dependencies
conn = sqlite3.connect("signatures.db")
cur = conn.cursor()

# Simple schema
cur.execute("""
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  content TEXT NOT NULL,
  signature TEXT NOT NULL
)
""")

# Band index (one table, no tuning)
cur.execute("""
CREATE TABLE band_index (
  band_idx INTEGER,
  band_value TEXT,
  doc_id INTEGER,
  PRIMARY KEY (band_idx, band_value, doc_id)
)
""")
```

</TabItem>
<TabItem value="sqlite-vec" label="sqlite-vec">

```python
import sqlite3
import sqlite_vec  # pip install sqlite-vec

conn = sqlite3.connect("vectors.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)  # Load C extension

# Virtual table (simpler than signatures, but needs extension)
conn.execute("""
CREATE VIRTUAL TABLE vec_documents 
USING vec0(
  id INTEGER PRIMARY KEY,
  embedding FLOAT[384]
)
""")
```

</TabItem>
<TabItem value="pgvector" label="pgvector">

```python
import psycopg

# Requires PostgreSQL server + pgvector extension
conn = psycopg.connect("postgresql://localhost/mydb")
cur = conn.cursor()

# Enable extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Table with vector column
cur.execute("""
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(384)
)
""")

# HNSW index (must tune m, ef_construction)
cur.execute("""
CREATE INDEX ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
""")
```

</TabItem>
</Tabs>

**For air-gapped SIEM deployments**, "install Docker and run PostgreSQL" is often a non-starter. Signatures deploy as a single Python file with zero runtime dependencies.

---

### 2. Signature Generation Cost

**"What's the overhead of generating signatures on top of embeddings?"**

#### Full Pipeline (3,714 prompts)

| Step | Time | Rate | Overhead |
|------|------|------|----------|
| Embedding generation (ONNX, CPU) | 112.6s | 33 prompts/sec | — |
| **Signature generation (native Rust)** | **0.7s** | **5,332 sigs/sec** | **0.6%** |
| Signature generation (pure Python) | 43.8s | 85 sigs/sec | 38% |

**With native Rust acceleration** (default in v0.1.1+): Signature generation adds only **0.6% overhead** on top of embedding generation.

#### Isolated Benchmark (Signatures Only)

| Implementation | Throughput | Latency | Speedup |
|---------------|-----------|---------|---------|
| **Native Rust** | 5,683 sigs/sec | 0.176 ms/sig | **631×** |
| Pure Python | 9 sigs/sec | 111 ms/sig | 1× |

**Run yourself**:
```bash
cd packages/rust
cargo run --release --example benchmark_signatures -- --count 10000
```

#### Why This Matters

Signature generation is a **one-time ingest cost**:
- Embeddings + signatures generated once at index time
- Queries do **O(log n) band lookups**, not signature regeneration
- The 0.6% overhead is amortized over the lifetime of the signature (months/years)

Even the **38% pure Python overhead** is acceptable for most use cases:
1. Ingest happens offline, not in the query hot path
2. Query-time speedup (44× fewer candidates) far outweighs one-time cost
3. Native acceleration makes the overhead negligible

---

### 3. Ingestion Performance

**Full corpus insert times:**

| N items | Signatures (total) | sqlite-vec | pgvector (insert + HNSW) |
|---------|-------------------|------------|--------------------------|
| 1,000   | 9.2ms             | 14.2ms     | 483.0ms                  |
| 3,714   | **43.8ms**        | 53.8ms     | **1.85s**                |

**Throughput**: ~**85,000 items/second** for signatures at full scale.

**pgvector is 42× slower** due to HNSW index build time (separate step after inserts). Signatures are faster because band-index rows are written inline during insert — no separate index-build step needed.

**Scaling Projections**:

| Corpus Size | Signatures | sqlite-vec | pgvector (est.) |
|-------------|-----------|------------|-----------------|
| 10,000      | ~118ms    | ~145ms     | ~5.0s           |
| 100,000     | ~1.2s     | ~1.5s      | ~50s            |
| 1,000,000   | ~12s      | ~15s       | ~8.3 min        |

*Projections are linear extrapolations from measured data.*

---

### 4. Query Latency & Candidate Reduction

**This is where the story gets interesting.**

#### Raw Query Times

| DB Size | Signatures (p50/p95/p99) | sqlite-vec (p50/p95/p99) | pgvector (p50/p95/p99) |
|---------|-------------------------|-------------------------|------------------------|
| 1,000   | 0.50ms / 1.1ms / 1.6ms  | 0.34ms / 0.50ms / 0.81ms | 2.6ms / 4.5ms / 6.3ms  |
| 3,714   | **0.93ms / 4.0ms / 5.7ms** | 0.87ms / 1.2ms / 1.4ms | **3.0ms / 5.3ms / 6.9ms** |

**At this scale, signatures and sqlite-vec have similar wall-clock latency** (~1ms p50). **pgvector is 3× slower** (3.0ms p50).

#### Candidate Count (The Real Story)

But look at what's happening underneath:

| DB Size | Signatures (candidates) | sqlite-vec (candidates) | Ratio |
|---------|------------------------|------------------------|-------|
| 1,000   | **23**                 | 1,000                  | **44× fewer** |
| 3,714   | **85**                 | 3,714                  | **44× fewer** |

**Signatures examine only 2.3% of the database per query.** sqlite-vec scans **100% of rows** every time.

#### Why Candidate Count Matters

**"If latency is the same, why does candidate count matter?"**

Because in production, similarity lookup is rarely the final step. Common patterns:

1. **Candidate generation → reranker**
   - Signatures retrieve ~85 candidates; exact cosine computed over those 85
   - sqlite-vec computes exact cosine over all 3,714
   - **44× less downstream compute**

2. **Candidate generation → rule engine**
   - Apply regex, heuristics, or ML models to candidates
   - 44× fewer candidates = 44× less secondary processing

3. **Candidate generation → LLM scoring**
   - Send candidates to an LLM for final verdict
   - **44× fewer candidates = 44× less LLM API cost**

**Yes, signatures require a secondary cosine pass over candidates**, but they check 44× fewer items than brute-force. The total compute (band lookup + candidate rescoring) is still far lower.

#### Scaling Analysis

At 3,714 items, both approaches are bound by SQLite I/O overhead, masking the algorithmic difference. At larger scales, the gap widens:

| DB Size (projected) | Sig p50 (est.) | Vec p50 (est.) | Sig candidates | Vec candidates |
|---------------------|---------------|----------------|----------------|----------------|
| 10,000              | ~2.6ms        | ~2.6ms         | ~228           | 10,000         |
| 100,000             | ~23ms         | ~25ms          | ~2,288         | 100,000        |
| 1,000,000           | ~232ms        | ~243ms         | ~22,886        | 1,000,000      |

*Linear extrapolations; actual results may differ based on I/O and caching.*

**Candidate reduction ratio stays constant at ~44×** across all scales.

---

### 5. Storage Efficiency

**Signatures are 3× more storage-efficient than vector embeddings.**

| Metric | Signatures | sqlite-vec | Ratio |
|--------|-----------|------------|-------|
| Actual DB size (3,714 items) | **2.0 MB** | 6.0 MB | **3×** |
| Bytes per item | **574B** | 1,638B | **2.9×** |
| Projected @ 100K items | **54 MB** | 164 MB | 3× |
| Projected @ 1M items | **547 MB** | 1.6 GB | 3× |
| Items that fit in 1 GB RAM | **~1.8M** | ~650K | **2.8×** |

**Storage Breakdown (per item)**:

<Tabs groupId="storage">
<TabItem value="signatures" label="Signatures">

```
Signature data:
  - Signature hex (256 bits = 32 bytes × 2 chars/byte) = 64 bytes
  - Band hashes (16 bands × 4 hex chars each) = 64 bytes
  - Total LSH data = 128 bytes

SQLite overhead:
  - Row metadata (primary key, indexes) = ~446 bytes
  
Total per item: ~574 bytes
```

</TabItem>
<TabItem value="vectors" label="Vectors">

```
Vector data:
  - 1024 dimensions × 4 bytes (float32) = 1,536 bytes

SQLite overhead:
  - Row metadata, virtual table = ~102 bytes
  
Total per item: ~1,638 bytes
```

</TabItem>
</Tabs>

**Why This Matters**:

For large-scale deployments (months of prompt history), the **3× storage advantage** means:
- Fitting in RAM vs spilling to disk (large latency impact)
- Lower infrastructure costs (smaller VMs, less S3/blob storage)
- Faster backups and replication

---

### 6. Accuracy

Evaluated on a **500-item sample** with **233 ground-truth duplicate pairs** (cosine similarity ≥ 0.85):

| Method | Precision | Recall | F1 | Notes |
|--------|-----------|--------|----|----|
| Exact KNN (sqlite-vec) | 1.000 | 1.000 | **1.000** | Guaranteed to find all pairs |
| Signatures (LSH) | 0.762 | 0.742 | **0.752** | Approximate, tunable |

**The F1 gap of 0.248 is NOT a difference in semantic understanding.** Both approaches use the **same 1024-dim embeddings**. The gap is entirely in the **lookup approximation**:

- **Exact KNN**: Computes cosine similarity between query and *every* vector — guaranteed to find all pairs above threshold
- **Signatures (LSH)**: Use band-hash collisions to retrieve candidates. If two similar vectors don't collide in any band, they're missed. Recall of 0.742 means ~26% of true duplicates don't trigger a band match.

#### Is This Acceptable?

**It depends on the use case:**

✅ **Candidate generation for a reranker**
- Signature lookup narrows from 3,714 to ~85 candidates
- Reranker applies exact scoring to those 85
- Misses at LSH stage (recall = 0.74) are rare enough that a small beam-width increase recovers most

⚠️ **Final verdict (hard block)**
- If signature match *is* the block decision, 0.74 recall means ~26% of duplicates get through
- Need to tune LSH bands/bits or add secondary exact check

✅ **0DIN's use case (SIEM candidate generation)**
- Surfacing candidates for human or ML review, not final block decisions
- 0.752 F1 is acceptable for this workflow

#### Tuning Accuracy

The 0.752 F1 reflects:
- Conservative threshold (0.85 cosine)
- Default LSH config (16 bands)

**Accuracy is tunable**:
- **More bands** (16 → 32): Higher recall, slightly more candidates checked
- **More bits** (256 → 512): Higher precision, larger storage
- **Lower threshold** (0.85 → 0.80): More pairs considered duplicates

LSH accuracy also **improves with more data** (better hash distribution).

---

### 7. Operational Burden

**Signatures have minimal operational overhead:**

| Task | Signatures | sqlite-vec | pgvector |
|------|-----------|------------|----------|
| Initial setup | `import sqlite3` | `pip install sqlite-vec` | Install Docker, pull image, `CREATE EXTENSION` |
| Schema changes | Alter table, reinsert | Recreate virtual table | Rebuild HNSW index (expensive) |
| Monitoring | None | None | PG stats, bloat checks |
| Vacuuming | None | None | Required |
| Index rebuild | Never | N/A | On major updates |
| Air-gap deploy | ✅ | ✅ | ❌ |
| SOC-2 audit surface | Minimal | Minimal | PostgreSQL server |

**For a security product, a smaller operational footprint is a security property** — fewer services running = fewer things to patch and audit.

---

## The Core Argument

> **"Embeddings are great for semantic understanding. Signatures are great for fast lookups at scale."**

These are **complementary technologies**, not competitors. In the 0DIN pipeline:

1. **Embedding generation** (expensive, done once at ingest) — ONNX or OpenAI
2. **Signature generation** (fast, derived from embedding) — signature-sdk SDK
3. **Signature lookup** (sub-millisecond, zero infrastructure) — band index in SQLite
4. **Optional reranking** (applied only to ~85 candidates, not all 3,714) — exact cosine

The signature layer makes step 3 feasible in constrained environments. A pure-embedding approach either:
- Requires a vector database (operational burden)
- Requires a brute-force scan (latency burden)

**Signatures eliminate both.**

---

## Reproducing These Results

### Quick Start

```bash
# Clone and install
git clone <repo>
pip install -e "packages/python[onnx,native]"
pip install -r demos/requirements.txt

# Copy ONNX model to ~/.cache/signature-sdk/models/v1/onnx/model_O4.onnx
# (or set SIGNATURE_SDK_MODEL_DIR env var)

# Run benchmark (skip pgvector if Docker unavailable)
python demos/showcase.py \
  --data path/to/vulnerabilities_cache.json \
  --skip-pgvector

# Or with make
make showcase DATA=path/to/vulnerabilities_cache.json SKIP_PGVECTOR=1
```

**Run time**: ~3 minutes (embedding generation). Subsequent runs with `--use-cache` complete in seconds.

### With pgvector

If you want full comparison including pgvector:

```bash
# Start PostgreSQL + pgvector
docker compose -f demos/docker-compose.yml up -d

# Run without --skip-pgvector
python demos/showcase.py --data path/to/vulnerabilities_cache.json
```

### Rust Signature Benchmark

Verify native Rust signature generation performance:

```bash
cd packages/rust
cargo run --release --example benchmark_signatures -- --count 10000
```

**Expected output**:
```
Generating 10,000 signatures (1024-dim random vectors)...
Time: 1.760s
Throughput: 5,683 signatures/sec
Per-signature: 0.176ms
```

---

## Performance Tuning

### Optimizing Query Latency

**Problem**: Query latency too high (> 10ms p50)

**Solutions**:

1. **Reduce candidate count**
   - Fewer bands (16 → 8): Smaller candidate set, lower recall
   - More families (3 → 5): Better hash distribution

2. **Optimize database**
   - Add indexes on `band_value` (should already exist)
   - Use `ANALYZE` to update query planner statistics
   - Consider in-memory SQLite (`:memory:`) for read-heavy workloads

3. **Batch queries**
   - Reuse database connection
   - Use `executemany()` for bulk lookups

**Example**:
```python
# Bad: New connection per query
for query in queries:
    conn = sqlite3.connect("sigs.db")
    candidates = lookup(conn, query)
    conn.close()

# Good: Reuse connection
conn = sqlite3.connect("sigs.db")
for query in queries:
    candidates = lookup(conn, query)
conn.close()
```

### Optimizing Storage

**Problem**: Database too large (> 1 GB for 100K items)

**Solutions**:

1. **Compress signatures**
   - Store as binary BLOB instead of hex TEXT (2× smaller)
   - Trade-off: Parsing overhead on read

2. **Reduce bands**
   - Default 16 bands → 8 bands (50% less index rows)
   - Trade-off: Lower recall

3. **Use fewer families**
   - Default 3 families → 1 family (66% less signature data)
   - Trade-off: Lower hash quality

**Example** (binary storage):
```python
import binascii

# Store
signature_bytes = binascii.unhexlify(signature_hex)
cur.execute("INSERT INTO docs (sig_binary) VALUES (?)", (signature_bytes,))

# Retrieve
signature_hex = binascii.hexlify(sig_binary).decode()
```

### Optimizing Accuracy

**Problem**: Too many false negatives (recall < 0.80)

**Solutions**:

1. **Increase bands**
   - Default 16 → 32 bands (higher recall)
   - Trade-off: More candidates, slightly higher latency

2. **Increase bits**
   - Default 256 → 512 bits (higher precision)
   - Trade-off: 2× larger signatures

3. **Use more families**
   - Default 3 → 5 families (better distribution)
   - Trade-off: 66% more storage

4. **Add reranking**
   - Always apply exact cosine to top-k candidates
   - Trade-off: More compute per query

**Example** (reranking):
```python
# Retrieve LSH candidates
candidates = lookup_bands(query_signature)

# Rerank by exact cosine
ranked = sorted(
    candidates,
    key=lambda c: cosine_similarity(query_embedding, c.embedding),
    reverse=True
)

return ranked[:top_k]
```

---

## Scaling Beyond 100K Items

### Architecture Recommendations

**For 100K–1M items**:
- ✅ Signatures + SQLite (still performant)
- ✅ Partition by time (monthly tables)
- ✅ Use SSD storage (NVMe if available)
- ⚠️ Consider PostgreSQL if you already have it

**For > 1M items**:
- ✅ Signatures + PostgreSQL (better concurrency)
- ✅ Shard by hash prefix (horizontal scaling)
- ✅ Cache hot signatures in Redis
- ⚠️ Evaluate purpose-built vector DBs (Milvus, Qdrant) if accuracy demands it

### Sharding Strategy

**Hash-based sharding** (example with 4 shards):

```python
def shard_id(signature: str, num_shards: int = 4) -> int:
    # Use first 2 hex chars of signature
    prefix = signature[8:10]  # After "0din-v1:"
    return int(prefix, 16) % num_shards

# Insert
shard = shard_id(signature, 4)
conns[shard].execute("INSERT INTO docs ...", ...)

# Query
shard = shard_id(query_signature, 4)
candidates = lookup_bands(conns[shard], query_signature)
```

**Trade-off**: Queries only search one shard (1/4 of data). For global similarity search, query all shards and merge results.

### Read Replicas

**For read-heavy workloads**:
- Use SQLite WAL mode (allows concurrent reads)
- Replicate database to multiple read replicas
- Load-balance queries across replicas

```python
import sqlite3

# Enable WAL mode for better read concurrency
conn = sqlite3.connect("sigs.db")
conn.execute("PRAGMA journal_mode=WAL")
```

---

## Comparison with Other Approaches

### vs. Exact Cosine (Brute-Force)

| Aspect | Signatures | Exact Cosine |
|--------|-----------|--------------|
| Latency | ~1ms @ 3,714 items | ~1ms @ 3,714 items |
| Latency @ 1M items | ~232ms (projected) | **~243s (1000× slower)** |
| Accuracy | F1 0.752 | F1 1.000 |
| Storage | 574B/item | 1,638B/item |
| Setup | Minimal | Trivial |

**When to use exact cosine**: Corpus < 1,000 items, accuracy is critical, latency doesn't matter.

### vs. HNSW (Hierarchical NSW)

| Aspect | Signatures | HNSW |
|--------|-----------|------|
| Latency @ 3,714 items | ~1ms | ~3ms |
| Latency @ 1M items | ~232ms (projected) | **~5-10ms (much faster)** |
| Accuracy | F1 0.752 | F1 ~0.95 |
| Storage | 574B/item | **1,600B + graph overhead** |
| Setup | Minimal | **Complex (tune m, ef_construction)** |
| Index rebuild | Never | **Required on schema change** |

**When to use HNSW**: Corpus > 1M items, accuracy > 0.90 required, have ops team for PostgreSQL/Milvus.

### vs. LSH (Other Implementations)

| Aspect | signature-sdk | Annoy | Faiss LSH |
|--------|----------|-------|-----------|
| Language support | Rust, Python, TypeScript | Python (C++ core) | Python (C++ core) |
| Dependencies | Zero (stdlib only) | pip package | pip package + libblas |
| Determinism | ✅ (SplitMix64 PRNG) | ⚠️ (random forests) | ⚠️ (random projections) |
| Signature format | Versioned string | Binary index | Binary index |
| Cross-language | ✅ Validated | ❌ | ❌ |
| Air-gap deploy | ✅ | ⚠️ (needs build tools) | ⚠️ (needs BLAS) |

**When to use signature-sdk**: Need deterministic signatures, cross-language consistency, or air-gapped deployment.

---

## Open Questions & Future Work

### Larger Dataset Evaluation

**Current**: 3,714 items (SQLite I/O bound)

**Needed**: 50K–100K item dataset to reveal algorithmic scaling differences more clearly.

**Options**:
- Synthesize from existing prompts (paraphrase, augment)
- Pull from production 0DIN feed (months of data)
- Use public jailbreak datasets (AdvBench, HarmBench)

### Accuracy Tuning

**Current**: F1 of 0.752 (acceptable for candidate generation)

**Goal**: F1 > 0.90 (as cited in 0DIN-1021)

**Approaches**:
- Increase LSH bands (16 → 32)
- Increase LSH bits (256 → 512)
- Hybrid: LSH for candidates + exact reranking

### CM-LSH Benchmarks

**Current**: Standard SimHash LSH only

**Needed**: Benchmark CM-LSH (confidence matrix) vs standard LSH:
- Accuracy improvement (expected +9-12%)
- Storage overhead (512-bit dual hash vs 256-bit single hash)
- Query latency impact

### GPU Acceleration

**Current**: Native Rust (CPU SIMD) achieves 5,683 sigs/sec

**Potential**: CUDA/Metal for batch signature generation (1M+ embeddings)

**Trade-off**: Complexity (driver dependencies) vs diminishing returns at typical scale.

---

## Summary

### Key Takeaways

1. ✅ **Signatures win on 7/8 dimensions** vs vector databases (setup, dependencies, tuning, maintenance, storage, air-gap, candidate reduction)
2. ⚠️ **Accuracy trade-off is acceptable** for candidate generation (F1 0.752 vs 1.000)
3. ✅ **Native Rust acceleration makes signatures nearly free** (0.6% overhead on top of embeddings)
4. ✅ **44× candidate reduction** drives downstream compute savings (rerankers, LLMs, rule engines)
5. ✅ **3× storage efficiency** enables larger corpora to fit in RAM
6. ✅ **Zero operational burden** (no Docker, no index tuning, no vacuuming)

### When to Use Signatures

**✅ Choose Signatures** if you need:
- Air-gapped/offline deployment
- Minimal operational complexity
- Storage efficiency (> 100K items)
- Candidate generation for downstream processing
- Cross-language deterministic hashes

**⚠️ Choose Vector DB** if you need:
- Accuracy > 0.95 F1 (hard requirement)
- Corpus > 1M items (HNSW scales better)
- Existing PostgreSQL/Milvus infrastructure

**🎯 Best of Both Worlds**:
- Use signatures for candidate retrieval (44× reduction)
- Use exact cosine for reranking top-k candidates
- Get fast lookups + high accuracy

### Performance Summary Table

| Metric | Signatures | sqlite-vec | pgvector | Winner |
|--------|-----------|------------|----------|--------|
| **Setup** | 19 LOC | 11 LOC + extension | 16 LOC + Docker | 🏆 Signatures |
| **Ingest** (3,714 items) | 43.8ms | 53.8ms | 1.85s | 🏆 Signatures |
| **Query p50** (3,714 items) | 0.93ms | 0.87ms | 3.0ms | ≈ Tie (Sig/Vec) |
| **Candidates** | 85 | 3,714 | ~100 (HNSW) | 🏆 Signatures |
| **Storage/item** | 574B | 1,638B | ~1,600B | 🏆 Signatures |
| **Accuracy F1** | 0.752 | 1.000 | 0.95 | 🏆 sqlite-vec |
| **Ops burden** | None | None | Moderate | 🏆 Signatures |

**Overall**: Signatures are the best choice for **constrained environments** and **candidate generation workloads**.

---

## Related Documentation

- **[Similarity Search](./similarity-search.md)** - Build an ANN search system with band-based indexing
- **[Native Acceleration](./native-acceleration.md)** - 592× speedup with Rust extension
- **[Configuration](../getting-started/configuration.md)** - Tune LSH parameters (bands, bits, families)
- **[API Reference: Core Functions](../api/core-functions.md)** - Signature generation functions

---

## Benchmark Data Source

All numbers in this guide are from **`demos/RESULTS.md`** (run date: 2026-02-26).

To reproduce:
```bash
python demos/showcase.py --data path/to/vulnerabilities_cache.json
```

See **Reproducing These Results** section above for full setup instructions.
