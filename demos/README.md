# Signature Capabilities Showcase

**"Why signatures instead of embeddings?"** — a reproducible, data-driven answer.

This benchmark compares three approaches to prompt similarity lookup across six
real-world dimensions, using a real threat-feed dataset on a local laptop.

## What It Measures

| Dimension | What you see |
|-----------|-------------|
| **Setup Complexity** | Infrastructure, dependencies, lines of code |
| **Ingestion** | Insert throughput, index build time |
| **Query Latency** | p50 / p95 / p99 at 1K, 5K, 10K, full dataset |
| **Storage** | Bytes on disk, projected to 1M prompts |
| **Accuracy** | Precision / recall / F1 on known duplicate pairs |
| **Operational Burden** | Maintenance, tuning, failure modes |

## Approaches Compared

| Label | Approach | Technology |
|-------|----------|------------|
| **A** | Signatures + Band Index | `odin_sig` + `sqlite3` (stdlib — zero deps) |
| **B** | sqlite-vec (brute-force KNN) | `sqlite-vec` pip package |
| **C** | pgvector + HNSW | `pgvector` via Docker |

---

## Prerequisites

### 1. Python 3.10+

```bash
python --version   # must be 3.10+
```

### 2. Install the sig-sdk Python package (with ONNX provider)

From the repo root:

```bash
pip install -e "packages/python[onnx]"
```

This pulls in `numpy`, `onnxruntime`, and `huggingface-hub`.  
The ONNX model (`all-MiniLM-L6-v2`, 384-dim) is downloaded on first use and
cached in `~/.cache/huggingface/hub/`.

### 3. Install showcase dependencies

```bash
pip install -r demos/requirements.txt
```

### 4. Docker (optional — for pgvector comparison)

Only needed if you want to include the pgvector benchmark:

```bash
docker compose -f demos/docker-compose.yml up -d
```

This starts a PostgreSQL + pgvector container on **port 5433** (chosen to avoid
conflicts with any local PostgreSQL on the default 5432).

To stop it afterwards:

```bash
docker compose -f demos/docker-compose.yml down
```

### 5. Threat-feed data

You need a JSON file of jailbreak/threat prompts.  Accepted formats:

- Array of strings: `["prompt1", "prompt2", ...]`
- Array of objects: `[{"prompt": "..."}, ...]`  (keys: `prompt`, `text`, `description`, `content`)
- Object with a wrapped array: `{"data": [...]}` or `{"prompts": [...]}`

Place your file anywhere; you pass the path with `--data`.

---

## Running the Showcase

### Quick start (all phases, full dataset)

```bash
python demos/showcase.py --data path/to/threat-feed.json
```

### Use the Makefile shortcut

```bash
make showcase DATA=path/to/threat-feed.json
```

### Subsequent runs — skip embedding generation

Embeddings and signatures are cached in `demos/cache/` after the first run:

```bash
python demos/showcase.py --data path/to/threat-feed.json --use-cache
```

### Limit dataset size for quick iteration

```bash
python demos/showcase.py --data path/to/threat-feed.json --limit 2000
```

### Skip pgvector if Docker isn't running

```bash
python demos/showcase.py --data path/to/threat-feed.json --skip-pgvector
```

### Run a single phase

```bash
python demos/showcase.py --data path/to/threat-feed.json --phase setup
python demos/showcase.py --data path/to/threat-feed.json --phase ingest
python demos/showcase.py --data path/to/threat-feed.json --phase query
python demos/showcase.py --data path/to/threat-feed.json --phase storage
python demos/showcase.py --data path/to/threat-feed.json --phase accuracy
python demos/showcase.py --data path/to/threat-feed.json --phase summary
```

### All CLI flags

```
--data PATH          Path to threat-feed JSON (required)
--limit N            Use only the first N prompts
--use-cache          Load embeddings/signatures from demos/cache/ if available
--skip-pgvector      Skip the pgvector benchmark (if Docker isn't running)
--phase PHASE        Run one phase: setup|ingest|query|storage|accuracy|summary|all
--n-queries N        Number of queries for latency benchmarks (default: 200)
```

---

## Expected Output

```
══════════════════════════════════════════════════════════════════════
  PHASE 0: Data Preparation
══════════════════════════════════════════════════════════════════════

  Loading: /path/to/threat-feed.json
  Generating embeddings (ONNX, 384-dim, local — no API key)...
  [████████████████████████████████░░░░░░░░]  80.0% (8000/10000)
  Generating LSH signatures (256-bit, 16 bands)...
  Saving cache...

  ✓ Ready: 10,000 prompts  |  embeddings: (10000, 384)  |  signatures: 10000

══════════════════════════════════════════════════════════════════════
  PHASE 1: Setup Complexity
══════════════════════════════════════════════════════════════════════

  Approach A — Signatures + Band Index
    Dependencies : 0 (Python stdlib only)
    Infrastructure: None
    Index build  : O(n) band hash
    Lines of code: ~80

  Approach B — sqlite-vec (brute-force KNN)
    Dependencies : 1 pip package (sqlite-vec)
    Infrastructure: None
    Search       : O(n) brute-force
    Lines of code: ~60

  Approach C — pgvector + HNSW
    Dependencies : psycopg, pgvector (pip), Docker
    Infrastructure: PostgreSQL server + pgvector extension
    Index build  : HNSW (O(n log n))
    Lines of code: ~90

══════════════════════════════════════════════════════════════════════
  PHASE 2: Ingestion
══════════════════════════════════════════════════════════════════════
  ...
══════════════════════════════════════════════════════════════════════
  PHASE 3: Query Latency
══════════════════════════════════════════════════════════════════════
  ...
══════════════════════════════════════════════════════════════════════
  PHASE 6: Summary
══════════════════════════════════════════════════════════════════════
  ...
  ✓ Showcase complete.
```

---

## Cache Files

After the first run, `demos/cache/` contains:

| File | Contents |
|------|----------|
| `embeddings.npz` | NumPy archive of 384-dim embeddings (one per prompt) |
| `signatures.json` | LSH signatures as hex strings |
| `signatures.db` | SQLite band-index DB (Approach A) |
| `sqlite_vec.db` | sqlite-vec vector DB (Approach B) |

These are `.gitignore`d (large binary files).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'odin_sig'`**  
Run `pip install -e "packages/python[onnx]"` from the repo root first.

**`ModuleNotFoundError: No module named 'onnxruntime'`**  
The ONNX provider is an optional extra: `pip install -e "packages/python[onnx]"`.

**pgvector connection refused**  
Start Docker: `docker compose -f demos/docker-compose.yml up -d`  
Or skip it: `--skip-pgvector`

**Embedding generation is slow**  
Normal on first run. Expected throughput: ~100–400 prompts/sec on CPU.
Subsequent runs with `--use-cache` are nearly instant.

**`JSONDecodeError` on load**  
Check your threat-feed file is valid JSON: `python -c "import json; json.load(open('file.json'))"`.
