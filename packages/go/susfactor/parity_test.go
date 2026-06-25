package susfactor

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// goldenVector mirrors spec/test-vectors/susfactor_vectors.json entry.
type goldenVector struct {
	Name          string   `json:"name"`
	Prompt        string   `json:"prompt"`
	ExpectedLabel string   `json:"expected_label"`
	RustScore     *float64 `json:"rust_score"`
}

type goldenFile struct {
	ScoreTolerance float64        `json:"score_tolerance"`
	Threshold      float64        `json:"threshold"`
	Vectors        []goldenVector `json:"vectors"`
}

func loadGoldens(t *testing.T) goldenFile {
	t.Helper()
	// Resolve path relative to this file: ../../spec/test-vectors/susfactor_vectors.json
	_, thisFile, _, _ := runtime.Caller(0)
	// From packages/go/susfactor/, go up 3 levels to repo root
	root := filepath.Join(filepath.Dir(thisFile), "..", "..", "..")
	path := filepath.Join(root, "spec", "test-vectors", "susfactor_vectors.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden vectors: %v (looked at %s)", err, path)
	}
	var gf goldenFile
	if err := json.Unmarshal(data, &gf); err != nil {
		t.Fatalf("parse golden vectors: %v", err)
	}
	return gf
}

// TestSusFactorParityGoldens runs the 15 golden-vector prompts through the
// Go SusFactor ONNX classifier and asserts scores are within 1e-3 of the
// canonical Rust reference and labels match exactly.
//
// Requires:
//
//	SUSFACTOR_MODEL_DIR — path to directory containing onnx/model.onnx + tokenizer.json
//	ORT_LIB_PATH       — path to libonnxruntime.dylib/.so (defaults to common brew/tmp paths)
//
// Skips automatically when SUSFACTOR_MODEL_DIR is not set.
func TestSusFactorParityGoldens(t *testing.T) {
	modelDir := os.Getenv("SUSFACTOR_MODEL_DIR")
	if modelDir == "" {
		t.Skip("SUSFACTOR_MODEL_DIR not set — skipping parity test")
	}

	clf, err := NewClassifier(
		WithModelDir(modelDir),
	)
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer clf.Close()

	gf := loadGoldens(t)

	for _, v := range gf.Vectors {
		if v.RustScore == nil || v.ExpectedLabel == "" {
			continue // skip unscored entries
		}
		v := v // capture
		t.Run(v.Name, func(t *testing.T) {
			result, err := clf.Classify(context.Background(), v.Prompt)
			if err != nil {
				t.Fatalf("Classify: %v", err)
			}
			if len(result.Chunks) == 0 {
				t.Fatal("no chunks returned")
			}

			// For all current goldens the prompt fits in one chunk.
			// We validate the first chunk score and the overall label.
			chunk := result.Chunks[0]

			diff := math.Abs(float64(chunk.Score) - *v.RustScore)
			if diff > gf.ScoreTolerance {
				t.Errorf("score mismatch: got %.8f, want %.8f (diff=%.2e > tol=%.0e)",
					chunk.Score, *v.RustScore, diff, gf.ScoreTolerance)
			}
			if chunk.Label != v.ExpectedLabel {
				t.Errorf("label mismatch: got %q, want %q (score=%.6f)",
					chunk.Label, v.ExpectedLabel, chunk.Score)
			}

			// Overall IsSuspicious must agree with any-chunk rule.
			wantSuspicious := v.ExpectedLabel == LabelSuspicious
			if result.IsSuspicious != wantSuspicious {
				t.Errorf("IsSuspicious=%v, want %v", result.IsSuspicious, wantSuspicious)
			}
		})
	}
}
