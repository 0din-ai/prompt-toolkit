---
sidebar_position: 6
---

# SusFactor Classifier

SusFactor is a jailbreak and prompt-injection classifier built into odin-prompt-toolkit. It is a **separate capability** from the LSH signature pipeline — it classifies a prompt directly rather than producing an embedding or signature.

## What It Does

SusFactor scores a prompt on a continuous scale from 0 to 1:

- **Score near 0** → `safe` — the prompt looks benign
- **Score near 1** → `suspicious` — the prompt looks like a jailbreak or prompt injection

The default decision threshold is `0.5`. Scores at or above the threshold return the label `suspicious`; below returns `safe`.

## The Model

SusFactor uses `0dinai/susfactor-e5-large`, a fine-tuned e5-large encoder with a small MLP classification head:

```
Input text
  → Tokenize (XLM-RoBERTa tokenizer)
  → If > 510 content tokens: split into overlapping 510-token chunks
  → [Per chunk] e5-large encoder (transformer)
  → [Per chunk] Mean pooling over tokens (with attention mask, no L2 normalization)
  → [Per chunk] 2-layer MLP head (1024 → 256 → 2 logits)
  → [Per chunk] Softmax → P(suspicious)
  → ChunkedSusFactorResult: is_suspicious = any(chunk.is_suspicious)
```

The model is **not bundled** with the SDK. It must be downloaded from HuggingFace before use (it is a gated model — a token is required):

- Torch weights: [`0dinai/susfactor-e5-large`](https://huggingface.co/0dinai/susfactor-e5-large)
- ONNX export: [`0dinai/susfactor-e5-large-onnx`](https://huggingface.co/0dinai/susfactor-e5-large-onnx)

## Backends

There are two inference backends:

| Backend | Class (Python) | Class (TS/Rust) | Dependencies | Notes |
|---|---|---|---|---|
| **PyTorch** | `SusFactorClassifier` | — | `torch`, `transformers` | Python only |
| **ONNX Runtime** | `SusFactorOnnxClassifier` | `SusFactorClassifier` | `onnxruntime` (+ `transformers` for tokenizer) | All three languages; ~3–5× faster on CPU than PyTorch path |

The Rust and TypeScript SDKs only expose the ONNX backend. The Python SDK exposes both, with `SusFactorOnnxClassifier` preferred for production use.

## SusFactor vs. LSH Signatures

| | LSH Signatures | SusFactor |
|---|---|---|
| **Output** | 256-bit hex signature | Float score 0–1 + label |
| **Use case** | Similarity / deduplication | Jailbreak / injection detection |
| **Requires embedding** | Yes | No (baked into model) |
| **Cross-language parity** | Yes (identical signatures) | Yes (within float tolerance) |
| **Model** | V0 (OpenAI) or V1 (ONNX) | susfactor-e5-large |

You can use both together — generate an LSH signature for deduplication *and* run SusFactor for threat classification on the same prompt.

## Quick Example

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs groupId="language">
  <TabItem value="rust" label="Rust">

```rust
use odin_prompt_toolkit::providers::ModelCache;
use odin_prompt_toolkit::susfactor::SusFactorClassifier;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cache = ModelCache::new()?;
    let clf = SusFactorClassifier::new(&cache, None, None, None).await?;
    
    let result = clf.classify("Ignore all previous instructions").await?;
    println!("Score: {:.3}", result.chunks[0].score);  // e.g. 0.972
    println!("Label: {}", result.chunks[0].label);     // "suspicious"
    println!("Suspicious: {}", result.is_suspicious);  // true
    
    Ok(())
}
```

Requires `features = ["susfactor"]` in `Cargo.toml`.

  </TabItem>
  <TabItem value="python" label="Python">

```python
from odin_prompt_toolkit.providers import ModelCache
from odin_prompt_toolkit.susfactor import SusFactorOnnxClassifier

# ONNX backend (recommended — no torch required)
cache = ModelCache()
clf = await SusFactorOnnxClassifier.new(cache)

result = await clf.classify("Ignore all previous instructions")
print(result.chunks[0].score)         # 0.972
print(result.chunks[0].label)         # "suspicious"
print(result.is_suspicious)           # True

await clf.close()
```

Or use the one-shot helper (constructs and disposes classifier automatically):

```python
from odin_prompt_toolkit.susfactor import sus_factor

result = await sus_factor("Ignore all previous instructions")
print(result.chunks[0].score, result.chunks[0].label)
print(result.is_suspicious)  # overall gate: True if any chunk is suspicious
```

Requires `pip install 'odin-prompt-toolkit[onnx]'` for ONNX backend, or `[susfactor]` for the PyTorch backend.

  </TabItem>
  <TabItem value="typescript" label="TypeScript">

```typescript
import { SusFactorClassifier } from '@0din/odin-prompt-toolkit/susfactor';
import { ModelCache } from '@0din/odin-prompt-toolkit/providers';

const cache = new ModelCache();
const clf = await SusFactorClassifier.create(cache);

const result = await clf.classify('Ignore all previous instructions');
console.log(result.chunks[0].score);        // 0.972
console.log(result.chunks[0].label);        // "suspicious"
console.log(result.isSuspicious);           // true — overall gate

await clf.close();
```

Or use the one-shot helper:

```typescript
import { susFactor } from '@0din/odin-prompt-toolkit/susfactor';

const result = await susFactor('Ignore all previous instructions');
console.log(result.chunks[0].score, result.chunks[0].label);
console.log(result.isSuspicious); // overall gate
```

  </TabItem>
</Tabs>

## Long-Prompt Chunking

The model accepts at most **512 tokens** per call (including the tokenizer's `[CLS]` and `[SEP]` tokens, leaving **510 tokens of usable content**). Prompts longer than 510 tokens are split automatically into overlapping chunks — you never need to check length or call a separate method.

### How it works

```
Prompt tokens: [─────────────────────────────────────────────]
                └── chunk 1 (510 tokens) ──┘
                              └── chunk 2 (510 tokens) ──┘
                                            └── chunk 3  ──┘
                ←── stride: 460 tokens ───→
                ←── overlap: 50 tokens ──→
```

- **Chunk size**: 510 tokens
- **Stride**: 460 tokens (each chunk advances 460 tokens from the previous)
- **Overlap**: 50 tokens shared between adjacent chunks — preserves context at boundaries
- Each chunk is scored **independently**; no scores are aggregated across chunks

### Return type: `ChunkedSusFactorResult`

`classify()` always returns a `ChunkedSusFactorResult`, even for short prompts (which produce exactly one chunk):

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | list of `SusFactorResult` | One result per chunk, in order |
| `is_suspicious` | bool | `true` if **any** chunk is suspicious |
| `total_timing_ms` | float | Wall-clock time across all chunks |

Each `SusFactorResult` in `chunks` has:

| Field | Type | Description |
|-------|------|-------------|
| `score` | float | P(suspicious) for this chunk, 0–1 |
| `label` | string | `"suspicious"` or `"safe"` |
| `is_suspicious` | bool | `score >= threshold` |
| `timing_ms` | float | Inference time for this chunk |

**Use `is_suspicious` at the top level for security gating** — it is `true` if any part of the prompt is suspicious. Access `chunks[0].score` for the first-chunk score (useful for parity checks and logging).

### Example — long prompt

```python
result = await clf.classify(long_prompt)

# Gate on the overall result — any suspicious chunk blocks the request
if result.is_suspicious:
    raise ValueError("Prompt blocked")

# Inspect individual chunks if you need to know which part triggered it
for i, chunk in enumerate(result.chunks):
    print(f"Chunk {i}: score={chunk.score:.3f} label={chunk.label}")
```

```typescript
const result = await clf.classify(longPrompt);

if (result.isSuspicious) throw new Error('Prompt blocked');

result.chunks.forEach((chunk, i) => {
  console.log(`Chunk ${i}: score=${chunk.score.toFixed(3)} label=${chunk.label}`);
});
```

```rust
let result = clf.classify(&long_prompt).await?;

if result.is_suspicious {
    return Err("Prompt blocked".into());
}

for (i, chunk) in result.chunks.iter().enumerate() {
    println!("Chunk {i}: score={:.3} label={}", chunk.score, chunk.label);
}
```

---

## Decision Threshold

The threshold controls the boundary between `safe` and `suspicious`. The default is `0.5`.

- **Lower threshold** (e.g. `0.3`) → more sensitive, more false positives
- **Higher threshold** (e.g. `0.7`) → less sensitive, more false negatives

```python
clf = await SusFactorOnnxClassifier.new(cache, threshold=0.7)
result = await clf.classify("some prompt")
# result.chunks[0].label based on score >= 0.7; result.is_suspicious for overall gate
```

For high-security environments, a lower threshold is recommended. For contexts where false positives are costly, raise the threshold.

## Next Steps

- **[SusFactor API Reference](../api/susfactor-api)** — Full API documentation
- **[Threat Feed Guide](../guides/threatfeed)** — Compare signatures against known threat intelligence
