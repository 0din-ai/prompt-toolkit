package susfactor

import "math"

// SuspiciousProb computes P(class=suspicious) from a pair of raw logits using
// a numerically stable 2-class softmax. This matches the Python reference:
//
//	m  = max(logits[0], logits[1])
//	e0 = exp(logits[0] - m)
//	e1 = exp(logits[1] - m)
//	return e1 / (e0 + e1)
//
// Class index 1 is the suspicious class, consistent across all SDKs.
func SuspiciousProb(logits [2]float32) float32 {
	l0, l1 := float64(logits[0]), float64(logits[1])
	m := math.Max(l0, l1)
	e0 := math.Exp(l0 - m)
	e1 := math.Exp(l1 - m)
	return float32(e1 / (e0 + e1))
}

// LabelForScore maps a suspicious probability score to a label using threshold.
// The comparison is inclusive: score >= threshold → "suspicious".
func LabelForScore(score, threshold float32) string {
	if score >= threshold {
		return LabelSuspicious
	}
	return LabelSafe
}
