---
sidebar_position: 2
---

# Similarity Search with LSH

Build an efficient approximate nearest neighbor (ANN) search system using band-based LSH indexing.

## Overview

LSH enables **sublinear-time similarity search**: instead of comparing a query against all documents in the corpus (O(n)), LSH allows querying via hash buckets (O(log n) to O(√n)).

**Key Idea:** Similar documents hash to the same buckets, allowing fast candidate retrieval.

### Use Cases

- **Semantic search**: Find prompts similar to a user query
- **Duplicate detection**: Identify near-duplicate content
- **Recommendation**: Suggest similar items
- **Clustering**: Group similar documents

---

## Algorithm

LSH similarity search operates in three phases:

### 1. Indexing Phase (Offline)

Hash all corpus documents and build an inverted index from bands to document IDs:

```
For each document:
  1. Generate embedding
  2. Compute LSH signature (3 families × 256 bits)
  3. Extract bands (16 bands × 4 hex chars each)
  4. Insert into band index: band_value → [doc_ids]
```

**Example:**
```
Document 42:
  Family 0: 8d000000ac854dae...
  Bands: ["8d00", "0000", "ac85", "4dae", ...]
  
Index updates:
  band_index["8d00"][0] → append(42)
  band_index["0000"][0] → append(42)
  band_index["ac85"][0] → append(42)
  ...
```

### 2. Query Phase (Online)

Hash the query and retrieve candidate documents from matching buckets:

```
1. Generate query embedding
2. Compute query LSH signature
3. Extract query bands
4. Lookup candidates: candidates = Union(band_index[band_value])
5. Deduplicate candidate set
```

**Example:**
```
Query bands: ["8d00", "0000", "ac85", ...]

Candidates:
  band_index["8d00"][0] → [42, 105, 213]
  band_index["0000"][0] → [42, 87, 156]
  band_index["ac85"][0] → [42, 199]
  
Candidates (union): {42, 87, 105, 156, 199, 213}
```

### 3. Ranking Phase

Compute exact similarity for candidates and rank by distance:

```
1. For each candidate:
     - Compute Hamming distance (query_sig, candidate_sig)
     - Estimate cosine similarity from Hamming distance
2. Sort by similarity (descending)
3. Return top-k results
```

---

## Database Schema

### PostgreSQL

```sql
-- Documents table
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  signature TEXT NOT NULL,  -- "0din-v1:8d000000..."
  version VARCHAR(10) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Band index (inverted index)
CREATE TABLE lsh_bands (
  family INT NOT NULL,       -- 0-2 for default config
  band_index INT NOT NULL,   -- 0-15 for default config
  band_value VARCHAR(8) NOT NULL,  -- 4 hex chars
  document_id INT NOT NULL REFERENCES documents(id),
  PRIMARY KEY (family, band_index, band_value, document_id)
);

-- Index for fast candidate lookup
CREATE INDEX idx_lsh_bands_lookup 
ON lsh_bands(family, band_index, band_value);

-- Index for signature extraction
CREATE INDEX idx_documents_signature 
ON documents(signature);
```

### SQLite

```sql
-- Documents table (same structure as PostgreSQL)
CREATE TABLE documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  signature TEXT NOT NULL,
  version VARCHAR(10) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Band index
CREATE TABLE lsh_bands (
  family INTEGER NOT NULL,
  band_index INTEGER NOT NULL,
  band_value VARCHAR(8) NOT NULL,
  document_id INTEGER NOT NULL REFERENCES documents(id),
  PRIMARY KEY (family, band_index, band_value, document_id)
);

CREATE INDEX idx_lsh_bands_lookup 
ON lsh_bands(family, band_index, band_value);
```

---

## Implementation Examples

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

### Indexing Documents

<Tabs groupId="language">
<TabItem value="python" label="Python">

```python
import asyncio
import asyncpg
from odin_sig import sign_text, SignatureVersion, parse_signature_string
from odin_sig.providers import ModelCache, OnnxProvider

async def index_document(pool, text: str, provider):
    """Index a single document."""
    # Generate signature
    result = await sign_text(text, provider=provider, version=SignatureVersion.V1)
    
    async with pool.acquire() as conn:
        # Insert document
        doc_id = await conn.fetchval(
            "INSERT INTO documents (content, signature, version) "
            "VALUES ($1, $2, $3) RETURNING id",
            text, result.signature_string, "v1"
        )
        
        # Extract and index bands
        for family in result.lsh.families:
            for band_idx, band_value in enumerate(family.bands):
                await conn.execute(
                    "INSERT INTO lsh_bands (family, band_index, band_value, document_id) "
                    "VALUES ($1, $2, $3, $4)",
                    family.family, band_idx, band_value, doc_id
                )
        
        return doc_id

async def index_corpus(texts: list[str]):
    """Index entire corpus."""
    # Setup
    pool = await asyncpg.create_pool(database="similarity_search")
    cache = ModelCache()
    provider = await OnnxProvider.new(cache)
    
    # Index all documents
    doc_ids = []
    for text in texts:
        doc_id = await index_document(pool, text, provider)
        doc_ids.append(doc_id)
        print(f"Indexed document {doc_id}")
    
    await provider.close()
    await pool.close()
    
    return doc_ids
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import { Pool } from 'pg';
import { signText, SignatureVersion, parseSignatureString } from '@0din/sig';
import { ModelCache, OnnxProvider } from '@0din/sig/providers';

async function indexDocument(
  pool: Pool,
  text: string,
  provider: OnnxProvider
): Promise<number> {
  // Generate signature
  const result = await signText(text, provider, SignatureVersion.V1);
  
  const client = await pool.connect();
  try {
    // Insert document
    const res = await client.query(
      'INSERT INTO documents (content, signature, version) VALUES ($1, $2, $3) RETURNING id',
      [text, result.signatureString, 'v1']
    );
    const docId = res.rows[0].id;
    
    // Extract and index bands
    for (const family of result.lsh.families) {
      for (let bandIdx = 0; bandIdx < family.bands.length; bandIdx++) {
        await client.query(
          'INSERT INTO lsh_bands (family, band_index, band_value, document_id) VALUES ($1, $2, $3, $4)',
          [family.family, bandIdx, family.bands[bandIdx], docId]
        );
      }
    }
    
    return docId;
  } finally {
    client.release();
  }
}

async function indexCorpus(texts: string[]): Promise<number[]> {
  const pool = new Pool({ database: 'similarity_search' });
  const cache = new ModelCache();
  const provider = await OnnxProvider.create(cache);
  
  const docIds: number[] = [];
  for (const text of texts) {
    const docId = await indexDocument(pool, text, provider);
    docIds.push(docId);
    console.log(`Indexed document ${docId}`);
  }
  
  await provider.close();
  await pool.end();
  
  return docIds;
}
```

</TabItem>
</Tabs>

### Querying for Similar Documents

<Tabs groupId="language">
<TabItem value="python" label="Python">

```python
from odin_sig import hamming_distance_hex, cosine_from_hamming

async def search_similar(pool, query: str, provider, top_k: int = 10, threshold: float = 0.7):
    """Find top-k similar documents to query."""
    # Generate query signature
    result = await sign_text(query, provider=provider, version=SignatureVersion.V1)
    parsed = parse_signature_string(result.signature_string)
    query_sig = parsed.signature
    
    async with pool.acquire() as conn:
        # Retrieve candidates from band index (using first family only)
        family = result.lsh.families[0]
        
        # Build query for all bands
        placeholders = ','.join(f'${i+1}' for i in range(len(family.bands)))
        candidate_query = f"""
            SELECT DISTINCT d.id, d.content, d.signature
            FROM lsh_bands b
            JOIN documents d ON b.document_id = d.id
            WHERE b.family = 0 
              AND b.band_value IN ({placeholders})
              AND d.version = 'v1'
        """
        
        candidates = await conn.fetch(candidate_query, *family.bands)
        
        # Rank candidates by Hamming distance
        ranked = []
        for candidate in candidates:
            candidate_parsed = parse_signature_string(candidate['signature'])
            distance = hamming_distance_hex(query_sig, candidate_parsed.signature)
            similarity = cosine_from_hamming(distance, bits=256)
            
            if similarity >= threshold:
                ranked.append({
                    'id': candidate['id'],
                    'content': candidate['content'],
                    'similarity': similarity,
                    'hamming_distance': distance,
                })
        
        # Sort by similarity (descending) and return top-k
        ranked.sort(key=lambda x: x['similarity'], reverse=True)
        return ranked[:top_k]

# Usage
results = await search_similar(pool, "How do I reset my password?", provider)
for r in results:
    print(f"[{r['similarity']:.3f}] {r['content']}")
```

</TabItem>
</Tabs>

---

## Optimization Strategies

### 1. Multi-Family Querying

Use multiple hash families to increase recall (reduces false negatives):

```python
async def search_multi_family(pool, query: str, provider, families_to_use: int = 3):
    """Query using multiple families (OR logic)."""
    result = await sign_text(query, provider=provider)
    
    all_candidates = set()
    
    # Collect candidates from each family
    for family in result.lsh.families[:families_to_use]:
        placeholders = ','.join(f'${i+1}' for i in range(len(family.bands)))
        query = f"""
            SELECT DISTINCT document_id
            FROM lsh_bands
            WHERE family = {family.family}
              AND band_value IN ({placeholders})
        """
        candidates = await conn.fetch(query, *family.bands)
        all_candidates.update(c['document_id'] for c in candidates)
    
    # Rank combined candidate set
    # ... (same ranking logic as before)
```

**Tradeoff:** Higher recall but more candidates to rank.

### 2. Partial Band Matching

Require matches in k out of n bands (reduces candidate set):

```sql
-- Require at least 3 band matches (out of 16)
SELECT document_id, COUNT(*) as match_count
FROM lsh_bands
WHERE family = 0 
  AND band_value IN ('8d00', '0000', 'ac85', ...)
GROUP BY document_id
HAVING COUNT(*) >= 3
ORDER BY match_count DESC;
```

**Tradeoff:** Lower recall but faster queries.

### 3. Tiered Querying

Query progressively more families if not enough results:

```python
async def tiered_search(pool, query: str, provider, min_results: int = 10):
    """Start with 1 family, expand if needed."""
    result = await sign_text(query, provider=provider)
    
    for num_families in [1, 2, 3]:
        candidates = await get_candidates(pool, result, num_families)
        ranked = rank_candidates(query_sig, candidates)
        
        if len(ranked) >= min_results:
            return ranked[:min_results]
    
    # If still not enough, return what we have
    return ranked
```

### 4. Caching Query Signatures

Cache signatures for frequently repeated queries:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_query_signature(query: str):
    return sign_text(query, provider=provider)
```

---

## Performance Characteristics

### Index Size

| Corpus Size | Documents | Bands (3 families × 16) | Index Rows | Storage |
|-------------|-----------|------------------------|------------|---------|
| Small | 1K | 48 per doc | 48K | ~5 MB |
| Medium | 100K | 48 per doc | 4.8M | ~500 MB |
| Large | 10M | 48 per doc | 480M | ~50 GB |

**Note:** Band index size scales linearly with corpus size.

### Query Performance

**Candidate retrieval:** O(k × m) where k = bands matched, m = avg docs per bucket
- Typical: 100-1000 candidates for 1M corpus
- Database index scan: 10-50ms

**Ranking:** O(n × b) where n = candidates, b = bits
- Hamming distance: very fast (bit operations)
- Typical: 1-5ms for 1000 candidates

**Total query time:** ~50-100ms for 1M corpus

### Precision/Recall Tradeoffs

| Configuration | Recall | Candidates | Query Time |
|---------------|--------|------------|------------|
| 1 family × 16 bands | 60-70% | 100-200 | Fast |
| 2 families × 16 bands | 75-85% | 200-500 | Medium |
| 3 families × 16 bands | 85-95% | 500-1000 | Slower |
| Exact search (brute force) | 100% | All | Very slow |

---

## Production Considerations

### Batch Indexing

Index documents in batches for better performance:

```python
async def batch_index(pool, texts: list[str], provider, batch_size: int = 100):
    """Index documents in batches."""
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        async with pool.acquire() as conn:
            async with conn.transaction():
                for text in batch:
                    # ... index logic
                    pass
        
        print(f"Indexed batch {i//batch_size + 1}")
```

### Index Maintenance

**Incremental updates:**
```python
async def update_document(pool, doc_id: int, new_text: str, provider):
    """Update document and re-index."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Delete old bands
            await conn.execute(
                "DELETE FROM lsh_bands WHERE document_id = $1",
                doc_id
            )
            
            # Re-generate signature and bands
            result = await sign_text(new_text, provider=provider)
            
            # Update document
            await conn.execute(
                "UPDATE documents SET content = $1, signature = $2 WHERE id = $3",
                new_text, result.signature_string, doc_id
            )
            
            # Re-index bands
            # ... (same as indexing)
```

### Monitoring

Track key metrics:
- Average candidates per query
- Hamming distance distribution
- Query latency (p50, p95, p99)
- Index size growth

---

## See Also

- [Duplicate Detection](./duplicate-detection) - Similar use case
- [LSH Overview](../concepts/lsh-overview) - Algorithm fundamentals
- [Configuration](../getting-started/configuration) - Tuning parameters
- [Performance Guide](./performance) - Optimization strategies
