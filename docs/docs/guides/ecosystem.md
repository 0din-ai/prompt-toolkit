---
sidebar_label: Ecosystem
---

# Ecosystem

Projects and integrations built with `odin-prompt-toolkit`.

---

## Security Integrations

### litellm-shield

**Repository:** [0din-ai/litellm-shield](https://github.com/0din-ai/litellm-shield)  
**Language:** Python  
**Toolkit packages used:** Python SDK (`odin-prompt-toolkit`), SusFactor classifier

A [LiteLLM](https://docs.litellm.ai/) guardrail that runs the 0DIN SusFactor classifier in-process to detect jailbreak and prompt-injection attempts across user messages, assistant responses, and tool-call arguments. No network hop — fully offline once the ONNX model is provisioned.

**Recommended rollout path:** Shadow → Flag → Block

```yaml
guardrails:
  - guardrail_name: "susfactor"
    litellm_params:
      guardrail: susfactor_guardrail.SusFactorGuardrail
      mode: "pre_call"       # pre_call | during_call | post_call
      enforcement: "block"   # shadow | flag | block
      threshold: 0.5
      fail_open: true
```

**Key characteristics:**
- Runs the SusFactor e5-large encoder + MLP head as a single ONNX graph (~16ms P50 on CPU)[^hw]
- Three enforcement modes: `shadow` (log only), `flag` (allow + annotate via `X-SusFactor-Decision` header), `block` (reject with 400)
- Three call positions: `pre_call` (prompt never reaches model), `during_call` (parallel with LLM), `post_call` (scan model output)
- Structured observability via `StandardLoggingGuardrailInformation` — Langfuse, Datadog, and OpenTelemetry compatible
- End-to-end latency penalty ≈ 0ms in `during_call` parallel mode (ONNX backend)

---

### openclaw-shield

**Repository:** [0din-ai/openclaw-shield](https://github.com/0din-ai/openclaw-shield)  
**Language:** TypeScript  
**Toolkit packages used:** TypeScript SDK (`@0din/prompt-toolkit`), LSH signatures, threat feed

A reference implementation and production plugin for [OpenClaw](https://github.com/openclaw-ai/openclaw) (an open-source AI agent runtime) that demonstrates how to integrate the prompt security toolkit at five distinct agent lifecycle hooks. Addresses the harder problem: **injection through tool results** (emails, web pages, API responses, documents) — not just user input.

**Detection pipeline (three phases):**

1. **Text normalization** — strips invisible Unicode, normalizes homoglyphs (Cyrillic а → a), decodes HTML entities
2. **Pattern matching** — 49 substring patterns across 7 jailbreak categories; sub-microsecond, always active
3. **LSH signature similarity** — ~788 signatures from 0DIN threat feed, backed by `@0din/prompt-toolkit` with `jailbreak-embeddings-large` (1024d ONNX, XLM-RoBERTa large); sliding window for documents

**Five integration points:**

| Hook | What it protects |
|---|---|
| `before_agent_run` | User prompt — direct jailbreak attempts |
| `before_tool_call` | Tool parameters — agent relaying a malicious query |
| `after_tool_call` | Tool results — poisoned external data (emails, web pages, docs) |
| `tool_result_persist` | Session history — prevents malicious content persisting across turns |
| `message_sending` | Outgoing replies — surfaces threats to the user |

```json
{
  "plugins": {
    "entries": {
      "openclaw-shield": {
        "enabled": true,
        "config": {
          "defaultAction": "warn",
          "patternMatchingEnabled": true,
          "signatureDetectionEnabled": true,
          "similarityThresholdWarn": 0.75,
          "similarityThresholdBlock": 0.85,
          "threatFeedEnabled": true
        }
      }
    }
  }
}
```

**Performance:**[^hw]

| Stage | Latency |
|---|---|
| Text normalization + pattern matching | < 0.01 ms |
| Band-index lookup (match) | ~0.14 ms |
| ONNX embedding (warm) | ~150–250 ms |
| Full pipeline (no ONNX match) | < 0.01 ms |
| Sliding window (1 KB document) | ~600–1000 ms |

---

### bedrock-shield

**Repository:** [0din-ai/bedrock-shield](https://github.com/0din-ai/bedrock-shield)  
**Language:** Python  
**Toolkit packages used:** Python SDK (`odin-prompt-toolkit`), SusFactor classifier

A drop-in replacement or complement for AWS Bedrock Runtime guardrail APIs. Wraps the 0DIN SusFactor ONNX classifier with a response shape compatible with `InvokeGuardrailChecks` and `ApplyGuardrail` — so teams already using Bedrock Guardrails can swap in SusFactor with no integration pattern change.

**Three integration modes:**

| Mode | Description | IAM required |
|---|---|---|
| Replace `InvokeGuardrailChecks` | Swap `client.invoke_guardrail_checks(...)` for `SusFactorChecks().invoke_guardrail_checks(...)` — same request/response shape | None |
| Stack with `ApplyGuardrail` | Runs both AWS guardrail + SusFactor in parallel, merges results (most-severe-wins) | `bedrock:ApplyGuardrail` |
| Standalone | Pure local inference — no AWS account or IAM required | None |

```python
from bedrock_shield import SusFactorChecks

checker = SusFactorChecks(threshold=0.5)
response = await checker.invoke_guardrail_checks(
    messages=[{"role": "user", "content": [{"type": "text", "text": user_input}]}],
    checks={"promptAttack": {"categories": [{"category": "JAILBREAK"}]}},
)
pa = response["results"]["promptAttack"]["results"][0]
if pa["severityScore"] >= 0.4:
    raise ValueError("Jailbreak attempt detected")
```

**Key characteristics:**
- Response shape is wire-compatible with Bedrock Runtime — no caller-side changes needed
- `severityScore` bucketed to nearest 0.2 step (`{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`) for `InvokeGuardrailChecks`; `confidence` string enum (`NONE / LOW / MEDIUM / HIGH`) for `ApplyGuardrail`
- `block` / `flag` / `shadow` enforcement modes with `fail_open` support
- Async core with `*_sync` wrappers for non-async boto3 callers

**Performance (ONNX backend):**[^hw]

| Backend | P50 latency |
|---|---|
| ONNX (default) | ~20–50 ms |
| Torch | ~150–400 ms |
| AWS `InvokeGuardrailChecks` | ~200–500 ms (network) |

---

## Add Your Project

Using `odin-prompt-toolkit` in production? Open a PR or issue on [GitHub](https://github.com/0din-ai/prompt-toolkit) to add it here.

[^hw]: Measured on Apple M4 Pro, CPU only.
