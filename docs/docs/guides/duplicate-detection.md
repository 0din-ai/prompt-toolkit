---
sidebar_position: 1
---

# Duplicate Detection

Build an efficient duplicate detector using LSH band-based indexing.

## Overview

Traditional duplicate detection requires **O(n²) comparisons**. LSH reduces this to **O(n)** through band-based candidate generation.

## Algorithm

1. **Hash** all documents into signatures with bands
2. **Index** documents by band values
3. **Query** candidates sharing any band
4. **Verify** candidates with Hamming distance

## Implementation

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs groupId="language">
  <TabItem value="python" label="Python">

```python
from collections import defaultdict
from odin_prompt_toolkit import (
    simhash_lsh_multi, normalize_vector,
    hamming_distance_hex, cosine_from_hamming
)

# Step 1: Generate signatures for all documents
documents = [
    [1.0, 1.0, 1.0, 1.0],      # Doc 0
    [1.0, 0.95, 1.05, 1.0],    # Doc 1 (similar to 0)
    [0.98, 1.02, 1.0, 1.01],   # Doc 2 (similar to 0, 1)
    [0.0, 1.0, 0.0, 1.0],      # Doc 3 (different)
]

signatures = [
    simhash_lsh_multi(normalize_vector(doc))
    for doc in documents
]

# Step 2: Build band index
band_index = defaultdict(list)

for doc_id, sig in enumerate(signatures):
    family = sig[0]  # Use first family
    for band_idx, band_value in enumerate(family.bands):
        key = (band_idx, band_value)
        band_index[key].append(doc_id)

# Step 3: Find candidate pairs
candidates = set()

for docs in band_index.values():
    if len(docs) > 1:
        # Multiple documents share this band
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                pair = tuple(sorted([docs[i], docs[j]]))
                candidates.add(pair)

print(f"Found {len(candidates)} candidate pairs")

# Step 4: Verify with Hamming distance
threshold = 0.85
duplicates = []

for id1, id2 in candidates:
    sig1 = signatures[id1][0].signature
    sig2 = signatures[id2][0].signature
    
    hamming = hamming_distance_hex(sig1, sig2)
    similarity = cosine_from_hamming(hamming, 256)
    
    if similarity >= threshold:
        duplicates.append((id1, id2, similarity))

# Sort by similarity
duplicates.sort(key=lambda x: x[2], reverse=True)

print(f"\nDuplicates (similarity >= {threshold}):")
for id1, id2, sim in duplicates:
    print(f"  Doc {id1} <-> Doc {id2}: {sim:.4f}")
```

  </TabItem>
</Tabs>

## Output

```
Found 2 candidate pairs

Duplicates (similarity >= 0.85):
  Doc 0 <-> Doc 1: 0.8876
```

## Performance Analysis

### Without LSH

```python
# Naive approach: compare all pairs
comparisons = 0
for i in range(n):
    for j in range(i + 1, n):
        similarity = cosine_similarity(docs[i], docs[j])
        comparisons += 1

# For 1M documents: 500 billion comparisons!
```

### With LSH

```python
# Band-based approach
comparisons = len(candidates)  # Only verify candidates

# For 1M documents: ~1,000 comparisons (0.0002% of naive)
```

## Tuning Parameters

### Threshold

- **Higher (0.9+)**: Only very similar duplicates
- **Medium (0.7-0.9)**: Moderate similarity
- **Lower (0.5-0.7)**: Loose matches

### Bands

- **More bands (32)**: Higher recall, more candidates
- **Default (16)**: Balanced
- **Fewer bands (8)**: Higher precision, fewer candidates

### Families

Using multiple families increases recall:

```python
# Combine candidates from all families
for family in sig:
    for band_idx, band_value in enumerate(family.bands):
        band_index[key].append(doc_id)
```

## Database Integration

Store signatures and bands in a database:

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    content TEXT,
    signature TEXT,
    family0_band0 TEXT,
    family0_band1 TEXT,
    -- ... more bands
);

CREATE INDEX idx_band0 ON documents(family0_band0);
CREATE INDEX idx_band1 ON documents(family0_band1);
```

Query for candidates:

```sql
SELECT DISTINCT d.id 
FROM documents d
WHERE d.family0_band0 = ? 
   OR d.family0_band1 = ?
   -- ... more bands
```

## Production Considerations

### Batch Processing

Process documents in batches to reduce memory:

```python
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    process_batch(batch)
```

### Incremental Updates

Add new documents without rebuilding entire index:

```python
def add_document(doc_id, signature):
    for band_idx, band_value in enumerate(signature.bands):
        band_index[(band_idx, band_value)].append(doc_id)
```

### False Positives

Band matching generates candidates, some are false positives:

```
Precision = True Duplicates / All Candidates
```

Always verify candidates with Hamming distance.

## Next Steps

- [Similarity Search](./similarity-search) — Build ANN search
- [LSH Overview](../concepts/lsh-overview) — Deep dive into algorithm
- [API Reference](../api/core-functions) — Complete API docs
