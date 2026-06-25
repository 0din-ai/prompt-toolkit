package susfactor

import (
	"math"
	"testing"
)

func TestSuspiciousProb(t *testing.T) {
	tests := []struct {
		name    string
		logits  [2]float32
		want    float64 // expected as float64 for comparison
		maxDiff float64
	}{
		{
			name:    "equal logits → 0.5 exactly",
			logits:  [2]float32{0, 0},
			want:    0.5,
			maxDiff: 1e-7,
		},
		{
			name:    "large logit[0] → near 0",
			logits:  [2]float32{10, 0},
			want:    1.0 / (1.0 + math.Exp(10)), // ≈ 4.54e-5
			maxDiff: 1e-7,
		},
		{
			name:    "large logit[1] → near 1",
			logits:  [2]float32{0, 10},
			want:    math.Exp(10) / (1.0 + math.Exp(10)), // ≈ 0.99995
			maxDiff: 1e-7,
		},
		{
			name:    "extreme values no overflow",
			logits:  [2]float32{-100, 100},
			want:    1.0,
			maxDiff: 1e-6,
		},
		{
			name:    "no NaN on extreme negative",
			logits:  [2]float32{100, -100},
			want:    0.0,
			maxDiff: 1e-6,
		},
		// Values from ORT smoke test — canonical jailbreak
		{
			name:    "canonical jailbreak logits",
			logits:  [2]float32{-2.856093, 3.290639},
			want:    0.9978640,
			maxDiff: 1e-5,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SuspiciousProb(tc.logits)

			// Must not be NaN or Inf
			if math.IsNaN(float64(got)) {
				t.Fatalf("SuspiciousProb returned NaN for logits %v", tc.logits)
			}
			if math.IsInf(float64(got), 0) {
				t.Fatalf("SuspiciousProb returned Inf for logits %v", tc.logits)
			}

			diff := math.Abs(float64(got) - tc.want)
			if diff > tc.maxDiff {
				t.Errorf("SuspiciousProb(%v) = %.10f, want %.10f (diff %.2e > %.2e)",
					tc.logits, got, tc.want, diff, tc.maxDiff)
			}
		})
	}
}

func TestLabelForScore(t *testing.T) {
	tests := []struct {
		name      string
		score     float32
		threshold float32
		want      string
	}{
		{"at threshold → suspicious (inclusive)", 0.5, 0.5, LabelSuspicious},
		{"below threshold → safe", 0.4999, 0.5, LabelSafe},
		{"zero → safe", 0.0, 0.5, LabelSafe},
		{"one → suspicious", 1.0, 0.5, LabelSuspicious},
		{"custom threshold 0.7 below → safe", 0.69, 0.7, LabelSafe},
		{"custom threshold 0.7 at → suspicious", 0.70, 0.7, LabelSuspicious},
		{"custom threshold 0.7 above → suspicious", 0.71, 0.7, LabelSuspicious},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := LabelForScore(tc.score, tc.threshold)
			if got != tc.want {
				t.Errorf("LabelForScore(%.4f, %.4f) = %q, want %q",
					tc.score, tc.threshold, got, tc.want)
			}
		})
	}
}
