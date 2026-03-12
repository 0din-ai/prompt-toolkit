---
sidebar_position: 2
---

# Signature Versions

odin-prompt-toolkit supports two signature versions with different embedding providers.

## Version Comparison

| Feature | V0 | V1 |
|---------|----|----|
| **Provider** | OpenAI API | Local ONNX |
| **Model** | text-embedding-3-large | multilingual-e5-large |
| **Dimensions** | 1536 | 1024 |
| **API Key** | Required | Not required |
| **Latency** | ~100-500ms | ~10-50ms |
| **Cost** | $0.13 per 1M tokens | Free |
| **Quality** | Excellent | Good |

## V0: OpenAI Embeddings

**Use when**:
- You need the highest quality embeddings
- You're already using OpenAI for other tasks
- Latency is acceptable

**Setup**:
```bash
export OPENAI_API_KEY=your-key-here
```

**Format**: `0din-v0:<hex_signature>`

## V1: ONNX Embeddings

**Use when**:
- You want local, API-free operation
- You need low latency
- You have many embeddings to generate

**Setup**: No configuration required. Model downloads automatically (~470MB) on first use.

**Format**: `0din-v1:<hex_signature>`

## Compatibility

:::danger Not Comparable
V0 and V1 signatures use **different embedding spaces** (1536 vs 1024 dimensions) and are **not comparable**. Always compare signatures with the same version.
:::

**Do NOT**:
```python
sig_v0 = "0din-v0:a3f9c2e1..."
sig_v1 = "0din-v1:7f2c8a9d..."
similarity = compare(sig_v0, sig_v1)  # ❌ WRONG!
```

**Do**:
```python
sig_v1_a = "0din-v1:7f2c8a9d..."
sig_v1_b = "0din-v1:8d000000..."
similarity = compare(sig_v1_a, sig_v1_b)  # ✅ Correct
```

## Migration

When migrating between versions, you must **regenerate all signatures** with the new embedding provider.

## Next Steps

- [Embedding Providers](./embedding-providers) — Deep dive into providers
- [Configuration](../getting-started/configuration) — Setup guide
