package susfactor

import (
	"context"
	"encoding/json"
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
//
// ORT registers a global ONNX Runtime environment on first call. The shared
// library path passed here wins for the lifetime of the process; subsequent
// calls to NewClassifier with a different WithORTLibPath will silently use
// whatever path was set first. Design accordingly: use the same ORT build for
// all classifiers in a process, and set ORT_LIB_PATH or WithORTLibPath
// consistently before the first NewClassifier call.
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
	modelCache *ModelCache
	cacheOpts  []CacheOption
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
//
// ORT is initialized once per process; the first call wins. See the package-
// level note on ortOnce for implications.
func WithORTLibPath(path string) Option {
	return func(c *config) { c.ortLibPath = path }
}

// WithModelCache configures the classifier to download missing model files
// from HuggingFace before loading. When combined with WithHFToken (via the
// cache options), gated repos are accessible.
//
// WithModelCache and WithModelDir are mutually exclusive; WithModelCache takes
// precedence when both are provided (the cache resolves the model directory).
func WithModelCache(cache *ModelCache, cacheOpts ...CacheOption) Option {
	return func(c *config) {
		c.modelCache = cache
		c.cacheOpts = cacheOpts
	}
}

// SusFactorClassifier classifies prompts as safe or suspicious using the
// SusFactor ONNX model. Safe to use from multiple goroutines; inference is
// serialized via a mutex (a single DynamicAdvancedSession handles one call at
// a time).
//
// Create with NewClassifier; release resources with Close.
type SusFactorClassifier struct {
	dynSession *ort.DynamicAdvancedSession
	tokenizer  *tokenizers.Tokenizer
	model      string
	threshold  float32
	// closed tracks whether Close has been called; guarded by mu.
	closed bool
	mu     sync.Mutex
}

// NewClassifier creates a SusFactorClassifier from local model files.
//
// ctx is used for any HuggingFace downloads triggered by WithModelCache.
// Pass context.Background() for startup code; pass a request context when
// calling from a handler that has a deadline.
func NewClassifier(ctx context.Context, opts ...Option) (*SusFactorClassifier, error) {
	cfg := &config{
		model:     DefaultModel,
		threshold: DefaultThreshold,
	}
	for _, o := range opts {
		o(cfg)
	}

	// Resolve model directory: WithModelCache takes precedence over WithModelDir.
	if cfg.modelCache != nil {
		dir, err := cfg.modelCache.EnsureModel(ctx, DefaultOnnxRepo, cfg.cacheOpts...)
		if err != nil {
			return nil, newError("ensure model: %v", err)
		}
		cfg.modelDir = dir
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

	// The validated SusFactor ONNX graph always uses exactly these two inputs.
	// token_type_ids is not required by this model; we avoid the double-load
	// that a probe session would impose (~2 GB for this model's external weights).
	// If a future export adds token_type_ids, add it back here.
	inputNames := []string{"input_ids", "attention_mask"}

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

	tk, err := loadTokenizerNoTruncation(tokPath)
	if err != nil {
		dynSession.Destroy()
		return nil, newError("load tokenizer: %v", err)
	}

	return &SusFactorClassifier{
		dynSession: dynSession,
		tokenizer:  tk,
		model:      cfg.model,
		threshold:  cfg.threshold,
	}, nil
}

// loadTokenizerNoTruncation loads a tokenizer from a tokenizer.json file with
// any embedded truncation disabled.
//
// The bundled tokenizer.json sets truncation.max_length = 512, which would
// silently cut every prompt to 512 tokens before chunking runs — bypassing
// long-prompt chunking and dropping content past the limit. We tokenize the
// full input and window it ourselves, so the truncation directive is stripped
// from the tokenizer definition before it is loaded.
func loadTokenizerNoTruncation(path string) (*tokenizers.Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	if _, ok := doc["truncation"]; ok {
		doc["truncation"] = json.RawMessage("null")
		if raw, err = json.Marshal(doc); err != nil {
			return nil, err
		}
	}
	return tokenizers.FromBytes(raw)
}

// Classify scores a prompt of any length. Prompts within MaxContentTokens
// (510 tokens) are scored in a single inference call. Longer prompts are
// split into overlapping chunks scored independently.
//
// A prompt is suspicious if any chunk scores at or above the threshold.
func (c *SusFactorClassifier) Classify(ctx context.Context, text string) (ChunkedSusFactorResult, error) {
	// Guard against use-after-Close under the lock to avoid a data race with
	// Close setting dynSession to nil.
	c.mu.Lock()
	closed := c.closed
	c.mu.Unlock()
	if closed {
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

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 2))
	if err != nil {
		return [2]float32{}, fmt.Errorf("output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	if err := c.dynSession.Run(
		[]ort.Value{idTensor, maskTensor},
		[]ort.Value{outputTensor},
	); err != nil {
		return [2]float32{}, fmt.Errorf("ORT run: %w", err)
	}

	data := outputTensor.GetData()
	return [2]float32{data[0], data[1]}, nil
}

// Close releases all model resources. The classifier must not be used after Close.
func (c *SusFactorClassifier) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
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
