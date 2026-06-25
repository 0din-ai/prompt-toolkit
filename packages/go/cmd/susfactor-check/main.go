// susfactor-check runs the 15 golden parity vectors through the local
// SusFactor ONNX model and exits 0 on pass, 1 on failure.
//
// Use this to verify that a Docker image or deployment environment is
// correctly set up before serving live traffic.
//
// Usage:
//
//	SUSFACTOR_MODEL_DIR=/path/to/model \
//	ORT_LIB_PATH=/path/to/libonnxruntime.dylib \
//	susfactor-check
//
// Exit codes:
//
//	0 — all vectors passed
//	1 — one or more vectors failed (scores or labels)
//	2 — setup error (missing model dir, ORT init failure, etc.)
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

	"github.com/0din-ai/prompt-toolkit/packages/go/susfactor"
)

const (
	tolerance = 1e-3
	exitSetup = 2
	exitFail  = 1
	exitPass  = 0
)

type goldenVector struct {
	Name          string   `json:"name"`
	Prompt        string   `json:"prompt"`
	ExpectedLabel string   `json:"expected_label"`
	RustScore     *float64 `json:"rust_score"`
}

type goldenFile struct {
	ScoreTolerance float64        `json:"score_tolerance"`
	Vectors        []goldenVector `json:"vectors"`
}

func main() {
	os.Exit(run())
}

func run() int {
	modelDir := os.Getenv("SUSFACTOR_MODEL_DIR")
	if modelDir == "" {
		fmt.Fprintln(os.Stderr, "error: SUSFACTOR_MODEL_DIR not set")
		return exitSetup
	}

	clf, err := susfactor.NewClassifier(context.Background(),
		susfactor.WithModelDir(modelDir),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: load classifier: %v\n", err)
		return exitSetup
	}
	defer clf.Close()

	vectors, tol, err := loadVectors()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: load golden vectors: %v\n", err)
		return exitSetup
	}

	fmt.Printf("susfactor-check: running %d golden vectors (tolerance %.0e)\n", len(vectors), tol)

	passed, failed := 0, 0
	for _, v := range vectors {
		if v.RustScore == nil || v.ExpectedLabel == "" {
			continue
		}
		result, err := clf.Classify(context.Background(), v.Prompt)
		if err != nil {
			fmt.Printf("  FAIL %-45s  error: %v\n", v.Name, err)
			failed++
			continue
		}
		chunk0 := result.Chunks[0]
		diff := math.Abs(float64(chunk0.Score) - *v.RustScore)
		scoreOK := diff <= tol

		// Use IsSuspicious (any-chunk) as the canonical label gate, matching
		// parity_test.go. For single-chunk prompts this is equivalent to
		// chunk[0].Label == expected_label; for multi-chunk prompts it correctly
		// catches a suspicious tail in a later chunk.
		wantSuspicious := v.ExpectedLabel == "suspicious"
		labelOK := result.IsSuspicious == wantSuspicious

		if scoreOK && labelOK {
			fmt.Printf("  PASS %-45s  chunks=%d score=%.6f label=%s\n",
				v.Name, len(result.Chunks), chunk0.Score, v.ExpectedLabel)
			passed++
		} else {
			gotLabel := "safe"
			if result.IsSuspicious {
				gotLabel = "suspicious"
			}
			fmt.Printf("  FAIL %-45s  chunks=%d score=%.6f (want %.6f, diff=%.2e) label=%s (want %s)\n",
				v.Name, len(result.Chunks), chunk0.Score, *v.RustScore, diff, gotLabel, v.ExpectedLabel)
			failed++
		}
	}

	fmt.Printf("\nResults: %d passed, %d failed\n", passed, failed)
	if failed > 0 {
		return exitFail
	}
	return exitPass
}

// loadVectors finds the golden vectors JSON relative to this binary's location,
// or via SUSFACTOR_VECTORS_PATH env override.
func loadVectors() ([]goldenVector, float64, error) {
	path := os.Getenv("SUSFACTOR_VECTORS_PATH")
	if path == "" {
		// Attempt to resolve relative to the source file path baked in at
		// compile time. This works with `go run` in-tree, but the baked path
		// does not exist on a deployment machine where the source is absent.
		// We stat the candidate before committing to it so that installed
		// binaries get a clear actionable error rather than a confusing
		// "no such file" pointing at a developer's working tree.
		_, thisFile, _, ok := runtime.Caller(0)
		if ok {
			// packages/go/cmd/susfactor-check/main.go → repo root is 4 levels up
			candidate := filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "..",
				"spec", "test-vectors", "susfactor_vectors.json")
			if _, err := os.Stat(candidate); err == nil {
				path = candidate
			}
		}
	}

	if path == "" {
		return nil, 0, fmt.Errorf(
			"SUSFACTOR_VECTORS_PATH is not set and source path is unavailable " +
				"(binary was built outside the repo tree); " +
				"set SUSFACTOR_VECTORS_PATH to the absolute path of susfactor_vectors.json")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("read %s: %w", path, err)
	}

	var gf goldenFile
	if err := json.Unmarshal(data, &gf); err != nil {
		return nil, 0, fmt.Errorf("parse: %w", err)
	}

	tol := gf.ScoreTolerance
	if tol == 0 {
		tol = tolerance
	}
	return gf.Vectors, tol, nil
}
