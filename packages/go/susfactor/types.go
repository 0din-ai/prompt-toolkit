// Package susfactor provides a Go implementation of the SusFactor jailbreak
// classifier, backed by an ONNX model (encoder + mean-pool + MLP head). It
// mirrors the public API of the Python and TypeScript SDKs and produces
// cross-SDK parity scores within 1e-3 of the canonical Rust implementation.
package susfactor

// Label constants returned by Classify.
const (
	LabelSuspicious = "suspicious"
	LabelSafe       = "safe"
)

// Inference constants — must match Python/Rust/TypeScript exactly.
const (
	MaxSequenceLength = 512
	// MaxContentTokens is the maximum number of payload tokens per chunk.
	// The model accepts 512 tokens total; the tokenizer adds [CLS] and [SEP],
	// leaving 510 positions for prompt content.
	MaxContentTokens = MaxSequenceLength - 2 // 510
	// ChunkOverlap is the number of tokens shared between adjacent chunks.
	ChunkOverlap = 50
	// ChunkStride is the number of new tokens advanced per chunk.
	ChunkStride = MaxContentTokens - ChunkOverlap // 460

	ModelVersion      = "susfactor-v1"
	DefaultModel      = "0dinai/susfactor-e5-large"
	DefaultOnnxRepo   = "0dinai/susfactor-e5-large-onnx"
	OnnxModelFile     = "onnx/model.onnx"
	OnnxModelDataFile = "onnx/model.onnx_data"

	DefaultThreshold float32 = 0.5
)

// SusFactorResult holds the classification result for a single chunk.
type SusFactorResult struct {
	// Score is the probability that the chunk is suspicious, in [0, 1].
	Score float32
	// Label is "suspicious" if Score >= Threshold, else "safe".
	Label string
	// Model is the identifier of the model that produced this score.
	Model string
	// Threshold is the decision threshold used to derive Label.
	Threshold float32
	// TimingMs is the inference time for this chunk in milliseconds.
	TimingMs float64
}

// IsSuspicious reports whether the chunk was classified as suspicious.
func (r SusFactorResult) IsSuspicious() bool {
	return r.Label == LabelSuspicious
}

// PhaseSpan is one timed phase in the lifecycle of a Classify call.
type PhaseSpan struct {
	// Name is the phase: "tokenize", "chunk", "inference", or "reduce".
	Name string
	// StartMs is the offset from the call's wall-clock start, in ms.
	StartMs float64
	// DurationMs is the wall time of this phase, in ms.
	DurationMs float64
	// ChunkIndex is the 0-based chunk index; non-nil only on "inference" spans.
	ChunkIndex *int
}

// ChunkedSusFactorResult is the return type of Classify for any prompt length.
//
// Short prompts (≤ MaxContentTokens tokens) produce exactly one chunk.
// Longer prompts are split automatically into overlapping chunks scored
// independently. The caller never needs to check length or call a separate
// method — Classify handles it transparently.
type ChunkedSusFactorResult struct {
	// Chunks holds one SusFactorResult per chunk, in order.
	Chunks []SusFactorResult
	// IsSuspicious is true if any chunk's label is "suspicious".
	// Use this field for security gating.
	IsSuspicious bool
	// TotalTimingMs is the wall-clock time for all chunks in milliseconds.
	TotalTimingMs float64
	// Spans holds the ordered lifecycle phase timings for the call:
	// one "tokenize" span, one "chunk" span, one "inference" span per
	// chunk (in order, each carrying its ChunkIndex), and one "reduce"
	// span. StartMs values share the single wall-clock baseline captured
	// at the start of the call. The gap between TotalTimingMs and the sum
	// of span durations is scheduling/assembly overhead and is intentional.
	Spans []PhaseSpan
}
