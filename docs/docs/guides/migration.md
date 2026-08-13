---
sidebar_position: 5
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Migration Guide

Migrate from legacy systems (heimdall, thor, research) to the unified odin-prompt-toolkit SDK, or upgrade from V0 to V1 signatures.

## Overview

The odin-prompt-toolkit SDK consolidates three legacy implementations into a unified multi-language SDK:

| Legacy System | Language | New Package | Status |
|---------------|----------|-------------|--------|
| **heimdall-core** | Rust | `odin-prompt-toolkit` (crate) | Canonical implementation |
| **thor** | TypeScript | `@0din/prompt-toolkit` (npm) | Feature parity |
| **research/signature_cli** | Python | `odin-prompt-toolkit` (PyPI) | Feature parity + native acceleration |

**Key improvements**:
- ✅ Unified API across all languages
- ✅ Versioned signature format (`0din-v1:...`)
- ✅ Native Rust acceleration for Python (592× faster)
- ✅ Comprehensive documentation and test coverage
- ✅ Cross-language validation (384 tests)

---

## Migrating from Legacy Systems

### From heimdall-core (Rust)

**Before** (heimdall-core internal module):
```rust
use heimdall_core::lsh::{simhash_lsh_multi, normalize_vector};
use heimdall_core::types::{LshConfig, SignatureVersion};
use heimdall_core::provider::EmbeddingProvider;
use heimdall_core::providers::openai::OpenAIProvider;

let vector = vec![0.5, 0.5, 0.5, 0.5];
let normalized = normalize_vector(&vector);
let families = simhash_lsh_multi(&normalized, 3, 256, 16);
```

**After** (odin-prompt-toolkit standalone crate):
```rust
use odin_prompt_toolkit::{simhash_lsh_multi, normalize_vector};
use odin_prompt_toolkit::types::{LshConfig, SignatureVersion};
use odin_prompt_toolkit::provider::EmbeddingProvider;
use odin_prompt_toolkit::providers::openai::OpenAIProvider;

let vector = vec![0.5, 0.5, 0.5, 0.5];
let normalized = normalize_vector(&vector);
let families = simhash_lsh_multi(&normalized, 3, 256, 16);
```

**Changes**:
- ✅ **No API changes** — functions, types, and signatures are identical
- ✅ Update `Cargo.toml`: `heimdall_core` → `odin-prompt-toolkit`
- ✅ Update imports: `heimdall_core::` → `odin_prompt_toolkit::`
- ⚠️ Feature flags renamed: No changes (still `openai`, `onnx`, `cm-lsh`)

**Migration steps**:
1. Add to `Cargo.toml`:
   ```toml
   [dependencies]
   odin-prompt-toolkit = { version = "0.1", features = ["openai", "onnx"] }
   ```
2. Find-replace imports: `use heimdall_core::` → `use odin_prompt_toolkit::`
3. Run tests to verify behavior unchanged
4. (Optional) Remove internal `heimdall-core` module once migration complete

---

### From thor (TypeScript)

**Before** (thor internal module):
```typescript
import { simhashLshMulti, normalizeVector } from './lsh';
import { LshConfig, SignatureVersion } from './types';

const vector = [0.5, 0.5, 0.5, 0.5];
const normalized = normalizeVector(vector);
const families = simhashLshMulti(normalized, 3, 256, 16);
```

**After** (@0din/prompt-toolkit npm package):
```typescript
import { simhashLshMulti, normalizeVector } from '@0din/prompt-toolkit';
import { LshConfig, SignatureVersion } from '@0din/prompt-toolkit';

const vector = [0.5, 0.5, 0.5, 0.5];
const normalized = normalizeVector(vector);
const families = simhashLshMulti(normalized, 3, 256, 16);
```

**Changes**:
- ✅ **No API changes** — functions and types are identical
- ✅ Centralized imports from `@0din/prompt-toolkit` (no relative paths)
- ✅ TypeScript declarations included (full IntelliSense support)
- ⚠️ Signature format now versioned (`0din-v1:...`)

**Migration steps**:
1. Install package:
   ```bash
   npm install @0din/prompt-toolkit
   ```
2. Update imports:
   ```typescript
   // Before
   import { ... } from './lsh';
   import { ... } from './types';
   
   // After
   import { ... } from '@0din/prompt-toolkit';
   ```
3. Update signature parsing (if using raw hex):
   ```typescript
   // Before: raw hex strings
   const signature = "8d000000ac854dae...";
   
   // After: versioned format
   import { signatureString } from '@0din/prompt-toolkit';
   const signature = signatureString(families[0], SignatureVersion.V1);
   // Output: "0din-v1:8d000000ac854dae..."
   ```
4. Run tests to verify behavior unchanged
5. (Optional) Remove internal LSH module once migration complete

---

### From research/signature_cli (Python)

**Before** (research internal module):
```python
from src.signature_cli.lsh import simhash_lsh_multi, normalize_vector
from src.signature_cli.types import LshConfig, SignatureVersion

vector = [0.5, 0.5, 0.5, 0.5]
normalized = normalize_vector(vector)
families = simhash_lsh_multi(normalized, families=3, bits=256, bands=16)
```

**After** (odin-prompt-toolkit PyPI package):
```python
from odin_prompt_toolkit import simhash_lsh_multi, normalize_vector
from odin_prompt_toolkit.types import LshConfig, SignatureVersion

vector = [0.5, 0.5, 0.5, 0.5]
normalized = normalize_vector(vector)
families = simhash_lsh_multi(normalized, families=3, bits=256, bands=16)
```

**Changes**:
- ✅ **No API changes** — functions and types are identical
- ✅ Native Rust acceleration (592× faster signature generation)
- ✅ Keyword-only arguments for safety: `simhash_lsh_multi(vector, *, families=3, ...)`
- ⚠️ Signature format now versioned (`0din-v1:...`)

**Migration steps**:
1. Install package (with native acceleration):
   ```bash
   pip install '0din-prompt-toolkit[native,onnx]'
   ```
2. Update imports:
   ```python
   # Before
   from src.signature_cli.lsh import ...
   from src.signature_cli.types import ...
   
   # After
   from odin_prompt_toolkit import ...
   from odin_prompt_toolkit.types import ...
   ```
3. Update function calls to use keyword args:
   ```python
   # Before (positional args)
   families = simhash_lsh_multi(vector, 3, 256, 16)
   
   # After (keyword-only)
   families = simhash_lsh_multi(vector, families=3, bits=256, bands=16)
   ```
4. Update signature format handling:
   ```python
   # Before: raw hex strings
   signature = "8d000000ac854dae..."
   
   # After: versioned format
   from odin_prompt_toolkit import signature_string
   signature = signature_string(families[0], SignatureVersion.V1)
   # Output: "0din-v1:8d000000ac854dae..."
   ```
5. Verify native acceleration is active:
   ```python
   from odin_prompt_toolkit import NATIVE_AVAILABLE
   print(f"Native: {NATIVE_AVAILABLE}")  # Should be True
   ```
6. Run tests to verify behavior unchanged
7. (Optional) Remove internal signature_cli module once migration complete

---

## Breaking Changes Summary

### API Changes

| Legacy | New SDK | Change |
|--------|---------|--------|
| Raw hex signatures | Versioned strings (`0din-v1:...`) | **Breaking**: Must update parsers |
| `HeimdallError` (Rust) | `SigError` | **Breaking**: Update error handling |
| Positional args (Python) | Keyword-only args | **Breaking**: Update call sites |
| Internal modules | Public packages | Update import paths |

### Non-Breaking Improvements

- ✅ Core algorithms unchanged (bit-identical signatures)
- ✅ Function names unchanged
- ✅ Type definitions unchanged (semantically)
- ✅ Test vectors validated across all implementations

### Removed from SDK (Intentionally)

The following **stay in their respective applications** (not part of the SDK):

**From heimdall**:
- ❌ Server code (REST/gRPC endpoints)
- ❌ CLI binary
- ❌ Proto definitions
- ❌ Service orchestration layer

**From thor**:
- ❌ Application-specific business logic
- ❌ UI components
- ❌ Database schemas

**From research**:
- ❌ CLI interface (`cli.py`)
- ❌ Intent classification (`core/intent/`)
- ❌ Multi-provider orchestration (`EmbeddingManager`)

**Rationale**: The SDK provides **library functionality only**. Application-specific code stays in the original projects.

---

## Migrating from V0 to V1 Signatures

### When to Migrate

**Reasons to migrate from V0 (OpenAI) to V1 (ONNX)**:
- ✅ **Cost reduction**: Eliminate OpenAI API charges (~$0.13 per 1M tokens)
- ✅ **Privacy requirements**: Keep embeddings local (no data sent to OpenAI)
- ✅ **Offline deployment**: No internet access required
- ✅ **Throughput**: Local ONNX inference is faster for batch processing

**Reasons to stay on V0**:
- ⚠️ **Accuracy**: OpenAI text-embedding-3-large (1536-dim) may have higher quality than 0din-jailbreak-embeddings-small (1024-dim) for English text
- ⚠️ **Investment**: Already have a large corpus of V0 signatures indexed

### Migration Process

⚠️ **CRITICAL**: V0 and V1 signatures are **NOT comparable**. They use different embedding spaces (1536-dim vs 1024-dim) and models.

**You cannot do gradual migration.** Choose one version and regenerate all signatures.

#### Step 1: Prepare

1. **Back up existing signatures**:
   ```sql
   -- Create backup table
   CREATE TABLE documents_v0_backup AS SELECT * FROM documents;
   ```

2. **Estimate migration time**:
   - V1 embedding generation: ~33 prompts/sec (ONNX, CPU)
   - V1 signature generation: ~5,332 sigs/sec (native Rust)
   - **Bottleneck**: Embedding generation (~30ms per prompt)
   
   Example: 100,000 prompts ≈ 50 minutes (embedding) + 19 seconds (signatures) ≈ **51 minutes total**

3. **Set up V1 provider**:
   <Tabs groupId="language">
   <TabItem value="rust" label="Rust">
   
   ```rust
   use odin_prompt_toolkit::providers::onnx::OnnxProvider;
   use odin_prompt_toolkit::types::SignatureVersion;
   
   let provider = OnnxProvider::default()?; // Auto-downloads model
   ```
   
   </TabItem>
   <TabItem value="python" label="Python">
   
   ```python
   from odin_prompt_toolkit.providers.onnx import get_onnx_provider
   from odin_prompt_toolkit.types import SignatureVersion
   
   provider = get_onnx_provider()  # Auto-downloads model
   ```
   
   </TabItem>
   <TabItem value="typescript" label="TypeScript">
   
   ```typescript
   import { getOnnxProvider } from '@0din/prompt-toolkit/providers/onnx';
   import { SignatureVersion } from '@0din/prompt-toolkit';
   
   const provider = await getOnnxProvider(); // Auto-downloads model
   ```
   
   </TabItem>
   </Tabs>

#### Step 2: Regenerate Signatures

**Batch regeneration script** (Python example):

```python
from odin_prompt_toolkit import sign_text
from odin_prompt_toolkit.types import SignatureVersion
from odin_prompt_toolkit.providers.onnx import get_onnx_provider
import sqlite3

# Initialize provider once (reuse connection)
provider = get_onnx_provider()

# Connect to database
conn = sqlite3.connect("documents.db")
cur = conn.cursor()

# Fetch all documents
cur.execute("SELECT id, content FROM documents")
documents = cur.fetchall()

# Regenerate signatures in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    
    for doc_id, content in batch:
        # Generate V1 signature
        result = sign_text(
            content,
            version=SignatureVersion.V1,
            provider=provider
        )
        
        # Update database
        cur.execute(
            "UPDATE documents SET signature = ? WHERE id = ?",
            (result.signature, doc_id)
        )
    
    # Commit batch
    conn.commit()
    print(f"Processed {i+len(batch)}/{len(documents)}")

conn.close()
```

**Parallel processing** (for faster migration):

```python
from concurrent.futures import ThreadPoolExecutor
from odin_prompt_toolkit import sign_text
from odin_prompt_toolkit.types import SignatureVersion

def regenerate_signature(content):
    result = sign_text(content, version=SignatureVersion.V1)
    return result.signature

# Process in parallel (use num_workers = CPU count)
with ThreadPoolExecutor(max_workers=8) as executor:
    signatures = list(executor.map(
        regenerate_signature,
        [doc[1] for doc in documents]
    ))
```

#### Step 3: Update Database Schema

If storing version in a separate column:

```sql
-- Add version column
ALTER TABLE documents ADD COLUMN version TEXT DEFAULT 'v1';

-- Update existing rows
UPDATE documents SET version = 'v1';

-- Add index for version queries
CREATE INDEX idx_documents_version ON documents(version);
```

#### Step 4: Update Queries

**Before** (V0, raw hex):
```python
# Query with raw hex signature
query_signature = "a3f9c2e1b8d4f7a2..."
candidates = lookup_similar(query_signature)
```

**After** (V1, versioned string):
```python
from odin_prompt_toolkit import parse_signature_string

# Query with versioned signature
query_signature = "0din-v1:7f2c8a9d3e1b5f4c..."
parsed = parse_signature_string(query_signature)
candidates = lookup_similar(parsed.signature)  # Extract hex part
```

#### Step 5: Validate Migration

**Validation script**:

```python
from odin_prompt_toolkit import sign_text, parse_signature_string
from odin_prompt_toolkit.types import SignatureVersion

# Test sample documents
test_docs = [
    "Hello world",
    "Sample jailbreak prompt",
    "Normal user query"
]

for doc in test_docs:
    # Generate V1 signature
    result = sign_text(doc, version=SignatureVersion.V1)
    
    # Verify format
    parsed = parse_signature_string(result.signature)
    assert parsed.version == SignatureVersion.V1
    assert parsed.algorithm == "lsh"
    assert len(parsed.signature) == 64  # 256 bits = 64 hex chars
    
    print(f"✅ {doc[:30]}... → {result.signature[:20]}...")
```

#### Step 6: Deprecate V0

Once migration is complete and validated:

1. **Remove V0 signatures** from database:
   ```sql
   DELETE FROM documents_v0_backup;
   ```

2. **Update code** to only use V1:
   ```python
   # Remove version parameter (defaults to Latest → V1)
   result = sign_text(content)  # Implicitly uses V1
   ```

3. **Update documentation** to reference V1 only

---

## Migrating from Legacy Formats

### Raw Hex Strings (No Version Prefix)

If you have **raw hex signatures** without the `0din-` prefix:

**Before**:
```
8d000000ac854dae0000000000000000000000000000000000000000000000000000000000000000
```

**After**:
```
0din-v1:8d000000ac854dae0000000000000000000000000000000000000000000000000000000000000000
```

**Migration script**:

```python
import sqlite3
from odin_prompt_toolkit import parse_signature_string
from odin_prompt_toolkit.types import SignatureVersion

conn = sqlite3.connect("documents.db")
cur = conn.cursor()

# Determine version (based on your knowledge of the data)
# If you used OpenAI text-embedding-3-large → V0
# If you used ONNX 0din-jailbreak-embeddings-small → V1
target_version = SignatureVersion.V1

# Fetch all raw hex signatures
cur.execute("SELECT id, signature FROM documents")
for doc_id, raw_hex in cur.fetchall():
    # Add version prefix
    versioned = f"0din-v{target_version.value}:{raw_hex}"
    
    # Validate format
    try:
        parsed = parse_signature_string(versioned)
        
        # Update database
        cur.execute(
            "UPDATE documents SET signature = ? WHERE id = ?",
            (versioned, doc_id)
        )
    except Exception as e:
        print(f"⚠️ Failed to migrate doc {doc_id}: {e}")

conn.commit()
conn.close()
```

### Custom Signature Formats

If you have **custom formats** (e.g., `sig:abc123` or `hash_abc123`):

1. **Determine version**: Based on embedding model used (see version registry in `spec/VERSIONING.md`)
2. **Extract hex signature**: Remove custom prefix/suffix
3. **Validate length**: Should be 64 hex chars (256 bits)
4. **Add 0din prefix**: `0din-v{N}:<hex>`
5. **Validate**: Use `parse_signature_string()` to verify format

**Example**:
```python
import re
from odin_prompt_toolkit import parse_signature_string

def migrate_custom_format(custom_sig: str) -> str:
    # Example: "sig:8d000000ac854dae..." → "0din-v1:8d000000ac854dae..."
    
    # Extract hex part
    match = re.match(r"sig:([0-9a-f]{64})", custom_sig)
    if not match:
        raise ValueError(f"Invalid format: {custom_sig}")
    
    hex_signature = match.group(1)
    
    # Add version prefix (assume V1)
    versioned = f"0din-v1:{hex_signature}"
    
    # Validate
    parse_signature_string(versioned)  # Raises InvalidInputError if invalid
    
    return versioned
```

---

## Database Schema Migration

### SQLite

**Before** (V0, raw hex):
```sql
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  content TEXT NOT NULL,
  signature TEXT NOT NULL  -- Raw hex: "8d000000..."
);
```

**After** (V1, versioned):
```sql
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  content TEXT NOT NULL,
  signature TEXT NOT NULL  -- Versioned: "0din-v1:8d000000..."
);

-- Add band index for similarity search
CREATE TABLE band_index (
  band_idx INTEGER,
  band_value TEXT,
  doc_id INTEGER,
  PRIMARY KEY (band_idx, band_value, doc_id),
  FOREIGN KEY (doc_id) REFERENCES documents(id)
);
```

**Migration**:
```sql
-- Add temporary column
ALTER TABLE documents ADD COLUMN signature_v1 TEXT;

-- Populate with versioned format (via application code)
-- UPDATE documents SET signature_v1 = '0din-v1:' || signature;

-- Verify all rows migrated
SELECT COUNT(*) FROM documents WHERE signature_v1 IS NULL;

-- Drop old column
ALTER TABLE documents DROP COLUMN signature;

-- Rename new column
ALTER TABLE documents RENAME COLUMN signature_v1 TO signature;
```

### PostgreSQL

**Before** (V0):
```sql
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  signature TEXT NOT NULL
);
```

**After** (V1):
```sql
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  signature TEXT NOT NULL CHECK (signature ~ '^0din-v[0-9]+:[0-9a-f]{64}$')
);

-- Add band index
CREATE TABLE band_index (
  band_idx INTEGER,
  band_value TEXT,
  doc_id INTEGER,
  PRIMARY KEY (band_idx, band_value, doc_id),
  FOREIGN KEY (doc_id) REFERENCES documents(id)
);

CREATE INDEX idx_band_index_lookup ON band_index(band_idx, band_value);
```

**Migration**:
```sql
BEGIN;

-- Add check constraint for new format
ALTER TABLE documents ADD CONSTRAINT signature_format
  CHECK (signature ~ '^0din-v[0-9]+:[0-9a-f]{64}$');

-- Update via application code (see Python example above)

COMMIT;
```

---

## Compatibility Matrix

### Cross-Version Comparison

| Operation | V0 ↔ V0 | V1 ↔ V1 | V0 ↔ V1 |
|-----------|---------|---------|---------|
| Hamming distance | ✅ Valid | ✅ Valid | ❌ **Invalid** |
| Cosine similarity | ✅ Valid | ✅ Valid | ❌ **Invalid** |
| Band lookup | ✅ Valid | ✅ Valid | ❌ **Invalid** |
| Signature parsing | ✅ Valid | ✅ Valid | ✅ Valid (different versions) |

⚠️ **Never compare V0 and V1 signatures** — they represent different embedding spaces.

### Cross-Language Compatibility

| From | To | Compatible? | Notes |
|------|----|-----------|----|
| Rust V1 | Python V1 | ✅ Yes | Bit-identical signatures |
| Rust V1 | TypeScript V1 | ✅ Yes | Bit-identical signatures |
| Python V1 | TypeScript V1 | ✅ Yes | Bit-identical signatures |
| Rust V0 | Python V0 | ✅ Yes | Bit-identical signatures |
| Legacy heimdall | odin-prompt-toolkit Rust | ✅ Yes | Same algorithm |
| Legacy thor | @0din/prompt-toolkit | ✅ Yes | Same algorithm |
| Legacy research | odin-prompt-toolkit Python | ✅ Yes | Same algorithm (592× faster with native) |

**Validation**: See `docs/docs/concepts/cross-language.md` for test vector validation methodology.

---

## Testing After Migration

### Smoke Tests

**1. Signature generation works**:
```python
from odin_prompt_toolkit import sign_text
result = sign_text("Hello world")
print(result.signature)  # Should start with "0din-v1:"
```

**2. Parsing works**:
```python
from odin_prompt_toolkit import parse_signature_string
parsed = parse_signature_string(result.signature)
assert parsed.version.value == 1
```

**3. Similarity search works**:
```python
from odin_prompt_toolkit import simhash_lsh_multi, hamming_distance_hex

families1 = simhash_lsh_multi([0.5] * 384)
families2 = simhash_lsh_multi([0.51] * 384)

distance = hamming_distance_hex(families1[0].signature, families2[0].signature)
assert 0 <= distance <= 256  # Valid Hamming distance
```

### Regression Tests

**Compare legacy vs SDK output** (should be identical):

```python
# Legacy (research)
from src.signature_cli.lsh import simhash_lsh_multi as legacy_simhash

# New SDK
from odin_prompt_toolkit import simhash_lsh_multi

# Test vector
vector = [0.5] * 384

# Generate both
legacy = legacy_simhash(vector, families=3, bits=256, bands=16)
new = simhash_lsh_multi(vector, families=3, bits=256, bands=16)

# Compare (hex signatures should match)
for i in range(3):
    assert legacy[i].signature == new[i].signature
    assert legacy[i].bands == new[i].bands
```

### Performance Benchmarks

**Verify native acceleration** (Python only):

```python
from odin_prompt_toolkit import NATIVE_AVAILABLE, simhash_lsh_multi
import time

assert NATIVE_AVAILABLE, "Native extension not installed!"

# Benchmark
vector = [0.5] * 384
iterations = 1000

start = time.perf_counter()
for _ in range(iterations):
    simhash_lsh_multi(vector)
elapsed = time.perf_counter() - start

throughput = iterations / elapsed
print(f"Throughput: {throughput:.0f} sigs/sec")

# Should be ~5,000-6,000 sigs/sec (native)
# vs ~8-10 sigs/sec (pure Python)
assert throughput > 1000, "Native acceleration not working!"
```

---

## Rollback Plan

If migration fails or causes issues:

### Step 1: Restore from Backup

```sql
-- Drop new table
DROP TABLE documents;

-- Restore from backup
CREATE TABLE documents AS SELECT * FROM documents_v0_backup;
```

### Step 2: Revert Code Changes

```bash
# Git revert to pre-migration commit
git revert HEAD

# Or restore from backup
git checkout <pre-migration-commit> -- src/
```

### Step 3: Verify Rollback

```python
# Test with V0 signatures
from odin_prompt_toolkit import parse_signature_string

# Should still parse V0 format
v0_sig = "0din-v0:a3f9c2e1b8d4f7a2..."
parsed = parse_signature_string(v0_sig)
assert parsed.version.value == 0
```

---

## Support & Resources

### Documentation

- **[API Reference](../api/core-functions.md)** - Complete function reference
- **[Configuration Guide](../getting-started/configuration.md)** - LSH parameters and tuning
- **[Performance Guide](./performance.md)** - Benchmarks and optimization
- **[Cross-Language Validation](../concepts/cross-language.md)** - Consistency verification

### Example Code

- **Rust**: `packages/rust/examples/`
- **Python**: `packages/python/examples/`
- **TypeScript**: `packages/typescript/examples/`

### Community

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share use cases
- **Slack/Discord**: (Link TBD)

---

## Summary

### Quick Migration Checklist

**From legacy systems**:
- [ ] Install new package (`odin-prompt-toolkit`, `@0din/prompt-toolkit`, or `odin-prompt-toolkit`)
- [ ] Update imports (remove relative paths, use package name)
- [ ] Update signature format (add `0din-v1:` prefix)
- [ ] Update function calls (keyword args in Python)
- [ ] Run tests (verify behavior unchanged)
- [ ] Remove old internal modules

**From V0 to V1**:
- [ ] Back up V0 signatures
- [ ] Install ONNX model for V1 (`~/.cache/odin-prompt-toolkit/models/v1/`)
- [ ] Regenerate all signatures (cannot mix V0 and V1)
- [ ] Update database schema (versioned format)
- [ ] Update queries (parse versioned strings)
- [ ] Validate migration (smoke tests + regression)
- [ ] Deprecate V0 signatures

**Key Takeaways**:
1. ✅ **Legacy → SDK migration is straightforward** (mostly import path changes)
2. ⚠️ **V0 → V1 migration requires full regeneration** (incompatible embedding spaces)
3. ✅ **Native Rust acceleration makes Python 592× faster** (install `odin-prompt-toolkit[native]`)
4. ✅ **All implementations validated** (109 tests, bit-identical signatures)
5. ✅ **Rollback plan available** (keep backups, use version control)

**Estimated migration time**:
- Legacy → SDK: **1-2 hours** (mostly find-replace)
- V0 → V1 (10K docs): **~5 minutes** (embedding bottleneck)
- V0 → V1 (100K docs): **~51 minutes**
- V0 → V1 (1M docs): **~8.5 hours**

**Need help?** Open a GitHub issue or check the documentation links above.
