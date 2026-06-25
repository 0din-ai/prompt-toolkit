package susfactor

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

// ortOnce guards ORT environment initialization — InitializeEnvironment must
// be called exactly once per process.
var (
	ortOnce    sync.Once
	ortInitErr error
)

func initORT(libPath string) error {
	ortOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// defaultORTLibPath returns the well-known ORT shared library path for the
// current platform, used when ORT_LIB_PATH is not set.
func defaultORTLibPath() string {
	switch runtime.GOOS {
	case "darwin":
		if _, err := os.Stat("/opt/homebrew/lib/libonnxruntime.dylib"); err == nil {
			return "/opt/homebrew/lib/libonnxruntime.dylib"
		}
		return "/usr/local/lib/libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

// config holds resolved constructor options.
type config struct {
	modelDir   string
	model      string
	threshold  float32
	ortLibPath string
}

// Option configures a SusFactorClassifier.
type Option func(*config)

// WithModelDir sets the path to the local model directory, which must contain:
//
//	onnx/model.onnx
//	tokenizer.json
func WithModelDir(dir string) Option {
	return func(c *config) { c.modelDir = dir }
}

// WithModel overrides the model identifier string reported in results.
// Defaults to DefaultModel ("0dinai/susfactor-e5-large").
func WithModel(name string) Option {
	return func(c *config) { c.model = name }
}

// WithThreshold sets the decision threshold. Score >= threshold → "suspicious".
// Defaults to DefaultThreshold (0.5).
func WithThreshold(t float32) Option {
	return func(c *config) { c.threshold = t }
}

// WithORTLibPath sets the path to the ONNX Runtime shared library.
// Defaults to ORT_LIB_PATH env var, then platform-specific well-known paths.
func WithORTLibPath(path string) Option {
	return func(c *config) { c.ortLibPath = path }
}

// SusFactorClassifier classifies prompts as safe or suspicious using the
// SusFactor ONNX model. Safe to use from multiple goroutines; inference is
// serialized via a mutex (a single DynamicAdvancedSession serializes calls).
//
// Create with NewClassifier; release resources with Close.
type SusFactorClassifier struct {
	dynSession           *ort.DynamicAdvancedSession
	tokenizer            *tokenizers.Tokenizer
	model                string
	threshold            float32
	requiresTokenTypeIDs bool
	inputNames           []string
	mu                   sync.Mutex
}

// NewClassifier creates a SusFactorClassifier from local model files.
func NewClassifier(opts ...Option) (*SusFactorClassifier, error) {
	cfg := &config{
		model:     DefaultModel,
		threshold: DefaultThreshold,
	}
	for _, o := range opts {
		o(cfg)
	}

	if cfg.modelDir == "" {
		return nil, newError("model directory required: use WithModelDir or WithModelCache")
	}
	if cfg.ortLibPath == "" {
		cfg.ortLibPath = os.Getenv("ORT_LIB_PATH")
	}
	if cfg.ortLibPath == "" {
		cfg.ortLibPath = defaultORTLibPath()
	}

	modelPath := filepath.Join(cfg.modelDir, OnnxModelFile)
	tokPath := filepath.Join(cfg.modelDir, "tokenizer.json")
	if _, err := os.Stat(modelPath); err != nil {
		return nil, newError("ONNX model not found at %s: %v", modelPath, err)
	}
	if _, err := os.Stat(tokPath); err != nil {
		return nil, newError("tokenizer.json not found at %s: %v", tokPath, err)
	}

	if err := initORT(cfg.ortLibPath); err != nil {
		return nil, newError("ORT initialization failed: %v", err)
	}

	sessionOpts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, newError("create session options: %v", err)
	}
	defer sessionOpts.Destroy()
	if err := sessionOpts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, newError("set graph optimization: %v", err)
	}

	// Probe whether the graph requires token_type_ids.
	requiresTypeIDs, inputNames, err := probeInputNames(modelPath, sessionOpts)
	if err != nil {
		return nil, newError("probe model inputs: %v", err)
	}

	// Create DynamicAdvancedSession — accepts variable-length inputs at Run time.
	dynSession, err := ort.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		[]string{"logits"},
		sessionOpts,
	)
	if err != nil {
		return nil, newError("create ONNX session: %v", err)
	}

	tk, err := tokenizers.FromFile(tokPath)
	if err != nil {
		dynSession.Destroy()
		return nil, newError("load tokenizer: %v", err)
	}

	return &SusFactorClassifier{
		dynSession:           dynSession,
		tokenizer:            tk,
		model:                cfg.model,
		threshold:            cfg.threshold,
		requiresTokenTypeIDs: requiresTypeIDs,
		inputNames:           inputNames,
	}, nil
}

// Classify scores a prompt of any length. Prompts within MaxContentTokens
// (510 tokens) are scored in a single inference call. Longer prompts are
// split into overlapping chunks scored independently.
//
// A prompt is suspicious if any chunk scores at or above the threshold.
func (c *SusFactorClassifier) Classify(ctx context.Context, text string) (ChunkedSusFactorResult, error) {
	if c.dynSession == nil {
		return ChunkedSusFactorResult{}, newError("Classify called on a closed SusFactorClassifier")
	}

	wallStart := time.Now()

	// Tokenize full text, no truncation, with special tokens ([CLS]/[SEP]).
	enc := c.tokenizer.EncodeWithOptions(text, true,
		tokenizers.WithReturnAttentionMask(),
	)
	allIDs := u32ToI64(enc.IDs)
	allMask := u32ToI64(enc.AttentionMask)

	idChunks := ChunkTokenIDs(allIDs)

	results := make([]SusFactorResult, len(idChunks))
	for i, chunkIDs := range idChunks {
		select {
		case <-ctx.Done():
			return ChunkedSusFactorResult{}, ctx.Err()
		default:
		}

		chunkStart := time.Now()
		chunkLen := len(chunkIDs)

		// EXACT: chunk_mask = allMask[:chunkLen]
		// Reuse the leading mask values (all 1s for non-padded input).
		// This matches Python/Rust/TypeScript exactly.
		chunkMask := allMask[:chunkLen]

		logits, err := c.runInference(chunkIDs, chunkMask)
		if err != nil {
			return ChunkedSusFactorResult{}, fmt.Errorf("chunk %d inference: %w", i, err)
		}

		score := SuspiciousProb(logits)
		label := LabelForScore(score, c.threshold)
		results[i] = SusFactorResult{
			Score:     score,
			Label:     label,
			Model:     c.model,
			Threshold: c.threshold,
			TimingMs:  float64(time.Since(chunkStart).Microseconds()) / 1000.0,
		}
	}

	isSuspicious := false
	for _, r := range results {
		if r.IsSuspicious() {
			isSuspicious = true
			break
		}
	}

	return ChunkedSusFactorResult{
		Chunks:        results,
		IsSuspicious:  isSuspicious,
		TotalTimingMs: float64(time.Since(wallStart).Microseconds()) / 1000.0,
	}, nil
}

// runInference feeds one chunk through the ONNX session and returns raw logits.
// Serialized via mu.
func (c *SusFactorClassifier) runInference(ids, mask []int64) ([2]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	shape := ort.NewShape(1, int64(len(ids)))

	idTensor, err := ort.NewTensor(shape, ids)
	if err != nil {
		return [2]float32{}, fmt.Errorf("id tensor: %w", err)
	}
	defer idTensor.Destroy()

	maskTensor, err := ort.NewTensor(shape, mask)
	if err != nil {
		return [2]float32{}, fmt.Errorf("mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	inputs := []ort.Value{idTensor, maskTensor}

	if c.requiresTokenTypeIDs {
		zeros := make([]int64, len(ids))
		zeroTensor, err := ort.NewTensor(shape, zeros)
		if err != nil {
			return [2]float32{}, fmt.Errorf("type id tensor: %w", err)
		}
		defer zeroTensor.Destroy()
		inputs = append(inputs, zeroTensor)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 2))
	if err != nil {
		return [2]float32{}, fmt.Errorf("output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	if err := c.dynSession.Run(inputs, []ort.Value{outputTensor}); err != nil {
		return [2]float32{}, fmt.Errorf("ORT run: %w", err)
	}

	data := outputTensor.GetData()
	return [2]float32{data[0], data[1]}, nil
}

// Close releases all model resources.
func (c *SusFactorClassifier) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tokenizer != nil {
		c.tokenizer.Close()
		c.tokenizer = nil
	}
	if c.dynSession != nil {
		c.dynSession.Destroy()
		c.dynSession = nil
	}
	return nil
}

// ---------- helpers ----------

func u32ToI64(in []uint32) []int64 {
	out := make([]int64, len(in))
	for i, v := range in {
		out[i] = int64(v)
	}
	return out
}

// probeInputNames checks whether the model requires token_type_ids by trying
// to create a temporary session with just input_ids + attention_mask.
func probeInputNames(modelPath string, opts *ort.SessionOptions) (requiresTypeIDs bool, names []string, err error) {
	dummy := []int64{0}
	shape := ort.NewShape(1, 1)

	idT, err := ort.NewTensor(shape, dummy)
	if err != nil {
		return false, nil, err
	}
	defer idT.Destroy()

	maskT, err := ort.NewTensor(shape, dummy)
	if err != nil {
		return false, nil, err
	}
	defer maskT.Destroy()

	logitsT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 2))
	if err != nil {
		return false, nil, err
	}
	defer logitsT.Destroy()

	// Try without token_type_ids.
	s, probeErr := ort.NewAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		[]ort.Value{idT, maskT},
		[]ort.Value{logitsT},
		opts,
	)
	if probeErr == nil {
		s.Destroy()
		return false, []string{"input_ids", "attention_mask"}, nil
	}

	// Try with token_type_ids.
	typeT, err := ort.NewTensor(shape, dummy)
	if err != nil {
		return false, nil, err
	}
	defer typeT.Destroy()

	s, probeErr = ort.NewAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"logits"},
		[]ort.Value{idT, maskT, typeT},
		[]ort.Value{logitsT},
		opts,
	)
	if probeErr == nil {
		s.Destroy()
		return true, []string{"input_ids", "attention_mask", "token_type_ids"}, nil
	}

	return false, nil, fmt.Errorf("cannot determine model inputs: %w", probeErr)
}
