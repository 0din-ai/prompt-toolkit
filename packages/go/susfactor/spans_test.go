package susfactor

import (
	"math"
	"testing"
)

// assertSpanShape validates the lifecycle-span contract on a
// ChunkedSusFactorResult: ordering, per-phase invariants, and one "inference"
// span per chunk carrying a unique 0-based ChunkIndex matching its position.
// Durations are nondeterministic, so only shape/finiteness/sign are checked.
func assertSpanShape(t *testing.T, result ChunkedSusFactorResult) {
	t.Helper()

	spans := result.Spans
	if len(spans) == 0 {
		t.Fatal("spans is empty")
	}

	n := len(result.Chunks)
	if want := n + 3; len(spans) != want {
		t.Fatalf("span count = %d, want %d (tokenize + chunk + %d inference + reduce)",
			len(spans), want, n)
	}

	if spans[0].Name != "tokenize" {
		t.Errorf("first span name = %q, want \"tokenize\"", spans[0].Name)
	}
	if spans[1].Name != "chunk" {
		t.Errorf("second span name = %q, want \"chunk\"", spans[1].Name)
	}
	if last := spans[len(spans)-1]; last.Name != "reduce" {
		t.Errorf("last span name = %q, want \"reduce\"", last.Name)
	}

	// Non-inference spans must not carry a ChunkIndex.
	for _, name := range []string{"tokenize", "chunk", "reduce"} {
		for _, s := range spans {
			if s.Name == name && s.ChunkIndex != nil {
				t.Errorf("%q span has non-nil ChunkIndex", name)
			}
		}
	}

	// Inference spans: exactly n, in order, with ChunkIndex == position 0..n-1.
	inferenceCount := 0
	seen := make(map[int]bool, n)
	for i := range n {
		s := spans[2+i] // inference spans occupy indices [2, 2+n)
		if s.Name != "inference" {
			t.Fatalf("span[%d] name = %q, want \"inference\"", 2+i, s.Name)
		}
		inferenceCount++
		if s.ChunkIndex == nil {
			t.Fatalf("inference span at position %d has nil ChunkIndex", i)
		}
		if got := *s.ChunkIndex; got != i {
			t.Errorf("inference span at position %d has ChunkIndex %d, want %d", i, got, i)
		}
		if seen[*s.ChunkIndex] {
			t.Errorf("duplicate ChunkIndex %d", *s.ChunkIndex)
		}
		seen[*s.ChunkIndex] = true
	}
	if inferenceCount != n {
		t.Errorf("inference span count = %d, want %d (== chunk count)", inferenceCount, n)
	}

	// Durations finite/non-negative; start offsets non-negative.
	for i, s := range spans {
		if s.StartMs < 0 {
			t.Errorf("span[%d] (%q) StartMs = %v, want >= 0", i, s.Name, s.StartMs)
		}
		if s.DurationMs < 0 || math.IsNaN(s.DurationMs) || math.IsInf(s.DurationMs, 0) {
			t.Errorf("span[%d] (%q) DurationMs = %v, want finite >= 0", i, s.Name, s.DurationMs)
		}
		if math.IsNaN(s.StartMs) || math.IsInf(s.StartMs, 0) {
			t.Errorf("span[%d] (%q) StartMs = %v, want finite", i, s.Name, s.StartMs)
		}
	}
}

// assembleResult builds a ChunkedSusFactorResult with n chunks whose Spans are
// laid out exactly as Classify assembles them: tokenize, chunk, one inference
// span per chunk (ChunkIndex = position), reduce. Used to validate the span
// contract without a live ONNX model.
func assembleResult(n int) ChunkedSusFactorResult {
	chunks := make([]SusFactorResult, n)
	spans := make([]PhaseSpan, 0, n+3)
	spans = append(spans,
		PhaseSpan{Name: "tokenize", StartMs: 0, DurationMs: 0.5},
		PhaseSpan{Name: "chunk", StartMs: 0.5, DurationMs: 0.25},
	)
	for i := range n {
		idx := i
		chunks[i] = SusFactorResult{Score: 0.1, Label: LabelSafe, Model: DefaultModel, Threshold: DefaultThreshold, TimingMs: 1.0}
		spans = append(spans, PhaseSpan{
			Name:       "inference",
			StartMs:    0.75 + float64(i),
			DurationMs: 1.0,
			ChunkIndex: &idx,
		})
	}
	spans = append(spans, PhaseSpan{Name: "reduce", StartMs: 0.75 + float64(n), DurationMs: 0.1})
	return ChunkedSusFactorResult{Chunks: chunks, IsSuspicious: false, TotalTimingMs: 2.5, Spans: spans}
}

func TestPhaseSpanShape(t *testing.T) {
	for _, n := range []int{1, 2, 5} {
		n := n
		t.Run("chunks", func(t *testing.T) {
			assertSpanShape(t, assembleResult(n))
		})
	}
}
