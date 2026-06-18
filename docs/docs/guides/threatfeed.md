---
sidebar_position: 6
---

# Threat Feed Integration

The threat feed integration lets you compare prompt signatures against the 0DIN portal's known threat intelligence — a curated database of jailbreaks, prompt injections, and adversarial prompts.

## How It Works

1. **Sync** — Fetch detection signatures from the 0DIN API and cache them locally with a band index
2. **Sign** — Generate an LSH signature for the prompt you want to check
3. **Query** — Find cached signatures that are similar to your query using band-indexed LSH lookup
4. **Act** — Inspect matches (title, severity, security boundary) and decide

The query is fast — O(candidates) rather than O(total entries) — because the band index filters to likely matches before computing Hamming distance.

## Authentication

You need a 0DIN API token. The client reads it from:

1. Explicit `api_token` / `apiToken` constructor argument
2. `ODIN_THREATFEED_API_TOKEN` environment variable
3. `ODIN_API_TOKEN` environment variable

## Python

### Install

```bash
pip install "odin-prompt-toolkit[threatfeed] @ git+https://github.com/0din-ai/prompt-toolkit#subdirectory=packages/python"
```

### Full Example

```python
import asyncio
from odin_prompt_toolkit import sign_text, SignatureVersion
from odin_prompt_toolkit.providers import ModelCache, OnnxProvider
from odin_prompt_toolkit.threatfeed import (
    ThreatFeedClient,
    ThreatFeedCache,
    compare_to_threatfeed,
)

async def main():
    # --- One-time setup: sync threat feed ---
    client = ThreatFeedClient(api_token="your-token")
    cache = ThreatFeedCache(version=SignatureVersion.V1)
    result = await cache.sync(client, full=True)
    print(f"Synced {result.total} signatures")

    # --- At query time ---
    # Generate LSH signature for the incoming prompt
    model_cache = ModelCache()
    provider = await OnnxProvider.new(model_cache)
    sig_result = await sign_text("Ignore all previous instructions", provider)

    # Compare against threat feed
    matches = compare_to_threatfeed(sig_result, cache, threshold=0.85)
    for m in matches:
        print(f"[{m.severity}] {m.title} — similarity: {m.cosine_similarity:.3f}")

    await provider.close()

asyncio.run(main())
```

### Incremental Sync

After the initial full sync, use incremental sync to fetch only new/updated entries:

```python
cache = ThreatFeedCache(version=SignatureVersion.V1)
cache.load()  # Load existing cache from disk

result = await cache.sync(client)  # full=False by default
print(f"Added {result.added}, updated {result.updated}")
```

### `ThreatFeedClient`

```python
ThreatFeedClient(
    api_token: str | None = None,   # Falls back to env vars
    base_url: str | None = None,    # Default: "https://0din.ai"
    per_page: int = 100,
)
```

| Method | Description |
|---|---|
| `fetch_all(since=None)` | Fetch all entries, paginating. `since` is an ISO8601 timestamp for incremental fetches. |
| `fetch_one(uuid)` | Fetch a single entry by UUID. |

### `ThreatFeedCache`

```python
ThreatFeedCache(
    version: SignatureVersion,       # V0 or V1
    cache_dir: str | Path | None = None,  # Default: ~/.odin-prompt-toolkit/threatfeed/
    bands: int = 16,
)
```

| Method / Property | Description |
|---|---|
| `load()` | Load cache from disk. Returns `True` if found. |
| `save()` | Write cache to disk (atomic). |
| `sync(client, full=False)` | Fetch from API and save. Returns `SyncResult`. |
| `query(signature, threshold=0.85, max_results=10)` | Find similar signatures. Returns `list[ThreatMatch]`. |
| `entry_count` | Number of entries currently loaded. |
| `last_synced` | ISO8601 timestamp of last sync. |
| `entries` | All `CachedSignature` entries. |

Cache directory can also be set via `ODIN_PROMPT_TOOLKIT_THREATFEED_CACHE` environment variable.

### `compare_to_threatfeed()`

```python
def compare_to_threatfeed(
    result: SignatureResult,
    cache: ThreatFeedCache,
    threshold: float = 0.85,
    max_results: int = 10,
) -> list[ThreatMatch]
```

Convenience wrapper that extracts the primary signature (family 0) from a `SignatureResult` and calls `cache.query()`.

### `ThreatMatch`

```python
@dataclass
class ThreatMatch:
    uuid: str
    title: str
    severity: str               # e.g. "critical", "high", "medium", "low"
    security_boundary: str      # e.g. "system", "user"
    signature: str              # The matching threat signature (hex)
    hamming_distance: int       # Bit-level distance from query
    cosine_similarity: float    # Estimated cosine similarity [0, 1]
```

---

## TypeScript

### Install

```bash
npm install github:0din-ai/prompt-toolkit#main
```

### Full Example

```typescript
import { signText, SignatureVersion } from '@0din/odin-prompt-toolkit';
import { ModelCache, OnnxProvider } from '@0din/odin-prompt-toolkit/providers';
import {
  ThreatFeedClient,
  ThreatFeedCache,
  compareToThreatfeed,
} from '@0din/odin-prompt-toolkit/threatfeed';

// --- One-time setup ---
const client = new ThreatFeedClient({ apiToken: 'your-token' });
const feedCache = new ThreatFeedCache({ version: SignatureVersion.V1 });
const syncResult = await feedCache.sync(client, { full: true });
console.log(`Synced ${syncResult.total} signatures`);

// --- At query time ---
const modelCache = new ModelCache();
const provider = await OnnxProvider.create(modelCache);
const sigResult = await signText('Ignore all previous instructions', provider);

const matches = compareToThreatfeed(sigResult, feedCache, { threshold: 0.85 });
for (const m of matches) {
  console.log(`[${m.severity}] ${m.title} — ${m.cosineSimilarity.toFixed(3)}`);
}

await provider.close();
```

### `ThreatFeedClient`

```typescript
new ThreatFeedClient({
  apiToken?: string,    // Falls back to ODIN_THREATFEED_API_TOKEN / ODIN_API_TOKEN env vars
  baseUrl?: string,     // Default: "https://0din.ai"
  perPage?: number,     // Default: 100
})
```

| Method | Description |
|---|---|
| `fetchAll(options?)` | Fetch all pages. `options.since` for incremental. |
| `fetchOne(uuid)` | Fetch single entry by UUID. |

### `ThreatFeedCache`

```typescript
new ThreatFeedCache({
  version: SignatureVersion,
  cacheDir?: string,    // Default: ~/.odin-prompt-toolkit/threatfeed/
  bands?: number,       // Default: 16
})
```

| Method / Property | Description |
|---|---|
| `load()` | Load from disk. Returns `true` if found. |
| `save()` | Write to disk. |
| `sync(client, options?)` | Fetch and save. `options.full` for full resync. |
| `query(signature, options?)` | Find similar signatures. |
| `entryCount` | Number of loaded entries. |
| `lastSynced` | ISO8601 timestamp. |

### `compareToThreatfeed()`

```typescript
function compareToThreatfeed(
  result: SignatureResult,
  cache: ThreatFeedCache,
  options?: { threshold?: number; maxResults?: number }
): ThreatMatch[]
```

---

## Choosing a Threshold

The `threshold` parameter controls the trade-off between recall (finding more potential matches) and precision (avoiding false positives).

| Threshold | Behavior |
|---|---|
| `0.95+` | Near-exact matches only. Very low false positives. May miss paraphrased attacks. |
| `0.85` (default) | Catches most semantic variants of known attacks. Recommended starting point. |
| `0.70` | Broad matching. Higher recall, more review required. |

The cosine similarity is estimated from Hamming distance via `cos(π × d/n)`, where `d` is the Hamming distance and `n` is 256 (total bits).

## Persisting the Cache

Both clients persist the cache to disk automatically on `sync()`. The default location is `~/.odin-prompt-toolkit/threatfeed/cache-{version}.json`.

On application startup, call `cache.load()` to restore the persisted index — this avoids a full API roundtrip on every run:

```python
cache = ThreatFeedCache(version=SignatureVersion.V1)
loaded = cache.load()
if not loaded or should_refresh:
    await cache.sync(client, full=not loaded)
```

## Next Steps

- **[SusFactor Classifier](../concepts/susfactor)** — Direct jailbreak scoring without generating a signature
- **[LSH Overview](../concepts/lsh-overview)** — How signatures enable fast similarity lookup
