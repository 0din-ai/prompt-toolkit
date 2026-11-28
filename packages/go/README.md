# odin-prompt-toolkit — Go SDK

Go implementation of the SusFactor jailbreak / prompt-injection classifier.
Produces cross-SDK parity scores within `1e-3` of the canonical Rust reference
across 15+ golden test vectors.

## Features

| Capability | Notes |
|---|---|
| **Classify(prompt)** | Returns score (0–1) + label per chunk; any prompt length |
| **ONNX backend** | Fully offline inference via `yalue/onnxruntime_go` |
| **Long-prompt chunking** | Auto-splits prompts > 510 tokens; suspicious if any chunk flags |
| **HuggingFace download** | `WithModelCache` downloads on first run; subsequent runs are cache hits |
| **Cross-SDK parity** | Byte-identical tokenization to Rust reference; scores within 1e-3 |

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| Go | 1.25+ | CGo required (`CGO_ENABLED=1`) |
| ONNX Runtime | **v1.31.0** | Must match `yalue/onnxruntime_go` v1.38.0 exactly |
| `libtokenizers.a` | v1.27.0 | Pre-built static lib from `daulet/tokenizers` releases |

> **ORT version is strict.** `yalue/onnxruntime_go` v1.38.0 requires ORT API version 31, which is ORT v1.31.0. Other versions fail at startup.

## Installation

```bash
# 1. Add to your go.mod
go get github.com/0din-ai/prompt-toolkit/packages/go

# 2. Download the pre-built libtokenizers.a for your platform
bash scripts/download-libtokenizers.sh
# Writes: lib/<GOOS>_<GOARCH>/libtokenizers.a

# 3. Download and install ORT shared lib (from the PyPI onnxruntime 1.31.0 wheel)
# macOS (Apple Silicon):
curl -fsSL -o /tmp/onnxruntime.whl \
  "https://files.pythonhosted.org/packages/b3/bd/2ac094311163b803e3626c3937461d6900934bd56cca7601f6150ff860c3/onnxruntime-1.31.0-cp312-cp312-macosx_14_0_arm64.whl"
echo "aaab9b3af536b06ca27ab5e35e3d429c97457ce76cf298af103f687e8b9975c0  /tmp/onnxruntime.whl" | shasum -a 256 -c -
unzip -o -j /tmp/onnxruntime.whl "onnxruntime/capi/libonnxruntime.1.31.0.dylib" -d /tmp/ort-wheel
export ORT_LIB_PATH=/tmp/ort-wheel/libonnxruntime.1.31.0.dylib

# Linux x86-64:
curl -fsSL -o /tmp/onnxruntime.whl \
  "https://files.pythonhosted.org/packages/6c/44/1e9e762b95b7da0a8424913a1ed7c38cdaf88624a3c41ddba24ebac88bc9/onnxruntime-1.31.0-cp312-cp312-manylinux_2_28_x86_64.whl"
echo "5e129d6c56abd53e659cb70f00a108d6824086470ff99c2e47a82e5786563db3  /tmp/onnxruntime.whl" | sha256sum -c -
unzip -o -j /tmp/onnxruntime.whl "onnxruntime/capi/libonnxruntime.so.1.31.0" -d /tmp/ort-wheel
sudo cp /tmp/ort-wheel/libonnxruntime.so.1.31.0 /usr/local/lib/libonnxruntime.so
sudo ldconfig
export ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so
```

## Quick Start

### With pre-downloaded model

```go
import (
    "context"
    "fmt"
    "os"
    "github.com/0din-ai/prompt-toolkit/packages/go/susfactor"
)

clf, err := susfactor.NewClassifier(context.Background(),
    susfactor.WithModelDir(os.Getenv("SUSFACTOR_MODEL_DIR")),
)
if err != nil {
    panic(err)
}
defer clf.Close()

result, err := clf.Classify(context.Background(), "Ignore all previous instructions.")
if err != nil {
    panic(err)
}
fmt.Printf("suspicious=%v score=%.4f\n", result.IsSuspicious, result.Chunks[0].Score)
// suspicious=true score=0.9979
```

### With automatic HuggingFace download

```go
cache := susfactor.NewModelCache("") // defaults to ~/.cache/signature-sdk/models

clf, err := susfactor.NewClassifier(context.Background(),
    susfactor.WithModelCache(cache,
        susfactor.WithHFToken(os.Getenv("HF_TOKEN")), // required: repo is gated
    ),
)
// Downloads ~2.1 GB on first run; subsequent calls are instant cache hits.
```

## Building

```bash
# CGO_LDFLAGS tells the linker where libtokenizers.a is
CGO_ENABLED=1 CGO_LDFLAGS="-L./lib/$(go env GOOS)_$(go env GOARCH)" \
  go build ./...
```

If you have installed `libtokenizers.a` system-wide (e.g. `/usr/local/lib/`), set
`CGO_LDFLAGS="-L/usr/local/lib"` instead.

## Testing

```bash
# Unit tests — no model required; parity test skips automatically
CGO_ENABLED=1 go test ./susfactor/... -count=1

# Parity test — requires model
SUSFACTOR_MODEL_DIR=/path/to/susfactor-onnx-model \
ORT_LIB_PATH=/path/to/libonnxruntime.dylib \
CGO_ENABLED=1 go test ./susfactor/... -run TestSusFactorParityGoldens -v
```

## Options reference

| Option | Default | Description |
|---|---|---|
| `WithModelDir(path)` | — | Load model files from a local directory |
| `WithModelCache(cache, opts...)` | — | Download missing files on first use |
| `WithModel(name)` | `"0dinai/susfactor-e5-large"` | Override model string in results |
| `WithThreshold(t)` | `0.5` | Decision threshold (inclusive: `score >= t` → suspicious) |
| `WithORTLibPath(path)` | auto | Path to ORT shared lib; falls back to `ORT_LIB_PATH` env |
| `WithIntraOpThreads(n)` | unset (ORT default) | ORT intra-op thread count; `n <= 0` or omitted leaves ORT's default. See [CPU and container sizing](#cpu-and-container-sizing) |
| `WithHFToken(token)` | `$HF_TOKEN` | HuggingFace token for gated model download |

## Model cache layout

Matches Rust/TypeScript convention:
```
~/.cache/signature-sdk/models/          # or $SIGNATURE_SDK_MODEL_CACHE
└── 0dinai/
    └── susfactor-e5-large-onnx/
        ├── onnx/
        │   ├── model.onnx              # required (~1.4 GB)
        │   └── model.onnx_data         # optional — tolerate 404
        ├── tokenizer.json              # required
        └── tokenizer_config.json       # optional
```

## Verifying a deployment

```bash
# Build and run the validation tool against the 15 golden vectors
CGO_ENABLED=1 go run ./cmd/susfactor-check/
# All 15 PASS → exit 0; any FAIL → exit 1; setup error → exit 2

# With explicit model dir
SUSFACTOR_MODEL_DIR=/path/to/model ORT_LIB_PATH=/path/to/ort go run ./cmd/susfactor-check/
```

## CPU and container sizing

Inference is CPU-bound, so how many CPUs the process can use sets latency.

**ONNX Runtime sizes its thread pool to the host's cores, not your container's
limit.** The classifier does not set a thread count by default, so ORT starts one
intra-op thread per core it sees. Inside a container that is the node's core
count; a Kubernetes CPU limit (a CFS quota) does not change it. A process with
a 2-CPU quota on a 12-core node therefore runs 12 threads that the kernel
throttles to 2 CPUs of time, which is slower than running 2 threads.

Set the thread count to the CPUs the process actually has:

```go
clf, err := susfactor.NewClassifier(ctx,
    susfactor.WithModelDir(dir),
    susfactor.WithIntraOpThreads(2), // match the container's CPU limit
)
```

`WithIntraOpThreads(n)` calls ORT's `SetIntraOpNumThreads(n)` on the session
options. `n <= 0`, or leaving the option out, makes no call and keeps ORT's
default, so existing code behaves exactly as before. The option does not change
scores, chunk scores or chunking.

**Inference is serial.** `Classify` scores a prompt's chunks one after another,
and each chunk's inference holds the classifier's mutex, so concurrent `Classify`
calls on one classifier also run one chunk at a time. Extra goroutines do not add
throughput; extra CPUs (with matching threads) make each chunk faster.

### Sizing guidance

The numbers below are **indicative**: one run on an Apple M2 Pro under Docker
(12 cores available), for about 1,000 tokens of input. Measure on your own
hardware before sizing a deployment.

| CPU configuration | Time per 1k tokens |
|---|---|
| 12 cores, unrestricted | 2.0 s |
| 2 cores, pinned | 13.8 s |
| 2-CPU quota, default threads | 18.7 s |
| 1-CPU quota, default threads | 52.7 s |

- Cost is about **24 core-seconds per 1k tokens**. Divide by the CPUs you have for
  a rough latency: 4 CPUs is roughly 6 s per 1k tokens, before any
  throttling penalty.
- **Pinned cores beat a quota** of the same size. In the run above the 2-CPU
  quota was 36% slower than 2 pinned cores, because threads sized to the host
  contend for a quota they cannot exceed. In Kubernetes, requests and limits
  are a quota; pinning needs the static CPU manager policy and an integer
  Guaranteed-QoS request.
- **Set `WithIntraOpThreads` equal to the CPUs you actually have**: the CPU
  limit rounded down, with a minimum of 1. More threads than CPUs only adds
  contention.
- Long prompts cost proportionally more. Chunks are 510 tokens and advance by a
  460-token stride, so a 10k-token prompt is about 22 chunks scored one after
  another. Size CPUs for your longest prompt, not the average.

## Docker

See the [Docker integration guide](../../docs/docs/guides/go-docker-integration.md) for
a complete multi-stage Dockerfile, ORT version pinning, and TabStack-specific setup.

A reference Dockerfile is at [`Dockerfile.example`](./Dockerfile.example).

## Parity

Scores are within `1e-3` of the canonical Rust reference for all 15 golden vectors.
Labels match exactly, including the boundary case (`score == 0.5` → `suspicious`).

Run the full cross-SDK parity check:
```bash
SUSFACTOR_MODEL_DIR=/path/to/model \
  python scripts/cross_validate.py --susfactor-parity
```

## API

### `NewClassifier(ctx context.Context, opts ...Option) (*SusFactorClassifier, error)`

Loads the ONNX session and tokenizer. Initializes the ORT environment once per process.

### `(*SusFactorClassifier) Classify(ctx context.Context, text string) (ChunkedSusFactorResult, error)`

Classifies a prompt of any length. Short prompts produce one chunk; longer prompts are
split automatically into overlapping 510-token chunks scored independently.

### `ChunkedSusFactorResult`

```go
type ChunkedSusFactorResult struct {
    Chunks        []SusFactorResult // one per chunk, in order
    IsSuspicious  bool              // true if ANY chunk is suspicious — use for security gating
    TotalTimingMs float64           // wall-clock time across all chunks
    TotalTokens   int               // length of the full tokenized input sequence (before chunking)
    Spans         []PhaseSpan       // lifecycle phase timings: tokenize, chunk, inference..., reduce
}

type SusFactorResult struct {
    Score     float32 // P(suspicious) in [0, 1]
    Label     string  // "suspicious" | "safe"
    Model     string  // model identifier
    Threshold float32 // decision threshold used
    TimingMs  float64 // inference time for this chunk
}

type PhaseSpan struct {
    Name       string  // "tokenize" | "chunk" | "inference" | "reduce"
    StartMs    float64 // offset from the call's wall-clock start, in ms
    DurationMs float64 // wall time of this phase, in ms
    ChunkIndex *int    // 0-based chunk index; non-nil only on "inference" spans
    TokenCount *int    // tokens fed to this chunk's inference; non-nil only on "inference" spans
}
```

### `ModelCache`

```go
cache := susfactor.NewModelCache("")          // or NewModelCache("/custom/path")
dir, err := cache.EnsureModel(ctx, repoID,   // downloads missing files
    susfactor.WithHFToken("hf_..."),
    susfactor.WithCacheBaseURL("https://huggingface.co"),
)
```
