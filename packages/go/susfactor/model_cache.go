package susfactor

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
)

// ErrNotFound is returned by downloadFile when the server returns HTTP 404.
// Callers use this to distinguish optional files (tolerate) from required ones (fail).
var ErrNotFound = errors.New("file not found on server (HTTP 404)")

// ErrUnauthorized is returned by downloadFile when the server returns HTTP 401
// or 403. For optional files in gated repos this is treated the same as
// ErrNotFound — the file may exist but the token lacks access or the file is
// not present in the gated manifest.
var ErrUnauthorized = errors.New("server returned 401/403 — check HF_TOKEN")

// tmpCounter ensures unique temp file names even when two goroutines race
// within the same nanosecond (possible on macOS with coarse clock resolution).
var tmpCounter atomic.Int64

// hfHTTPClient is shared across all downloads. It uses a 30s dial/TLS timeout
// but no overall transfer timeout, since model files can be several GB and
// transfer speed varies widely. Cancellation is via the request context.
var hfHTTPClient = &http.Client{
	Transport: &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   30 * time.Second,
		ResponseHeaderTimeout: 60 * time.Second,
	},
}

// defaultCacheDir returns the default model cache root.
// Matches Rust/TS convention: $SIGNATURE_SDK_MODEL_CACHE or ~/.cache/signature-sdk/models
func defaultCacheDir() string {
	if dir := os.Getenv("SIGNATURE_SDK_MODEL_CACHE"); dir != "" {
		return dir
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "signature-sdk", "models")
	}
	return filepath.Join(home, ".cache", "signature-sdk", "models")
}

// ModelCache manages local storage of SusFactor model files.
// The cache layout matches the Rust/TypeScript convention:
//
//	<root>/<org>/<repo>/onnx/model.onnx
//	<root>/<org>/<repo>/tokenizer.json
//
// Multiple goroutines may call methods on ModelCache concurrently.
type ModelCache struct {
	root string
}

// NewModelCache returns a ModelCache rooted at dir.
// If dir is empty, defaults to $SIGNATURE_SDK_MODEL_CACHE or ~/.cache/signature-sdk/models.
func NewModelCache(dir string) *ModelCache {
	if dir == "" {
		dir = defaultCacheDir()
	}
	return &ModelCache{root: dir}
}

// ModelDir returns the local directory for a given HuggingFace repo ID.
// The repo ID is split on "/" to form the subdirectory path:
//
//	"0dinai/susfactor-e5-large-onnx" → <root>/0dinai/susfactor-e5-large-onnx
func (c *ModelCache) ModelDir(repoID string) string {
	parts := strings.SplitN(repoID, "/", 2)
	switch len(parts) {
	case 2:
		return filepath.Join(c.root, parts[0], parts[1])
	default:
		return filepath.Join(c.root, repoID)
	}
}

// Required files that must be present for the classifier to load.
var susfactorRequiredFiles = []string{
	"onnx/model.onnx",
	"tokenizer.json",
}

// Optional files — tolerate HTTP 404 / local absence without error.
var susfactorOptionalFiles = []string{
	"onnx/model.onnx_data",
	"tokenizer_config.json",
}

// HasSusFactorModel reports whether all required model files are present in
// the cache for repoID.
func (c *ModelCache) HasSusFactorModel(repoID string) bool {
	dir := c.ModelDir(repoID)
	for _, rel := range susfactorRequiredFiles {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if _, err := os.Stat(p); err != nil {
			return false
		}
	}
	return true
}

// cacheOptions holds resolved options for EnsureModel.
type cacheOptions struct {
	hfToken string
	baseURL string
}

// CacheOption configures EnsureModel behaviour.
type CacheOption func(*cacheOptions)

// WithHFToken sets the HuggingFace bearer token used to download gated models.
// Falls back to the HF_TOKEN environment variable if not set.
func WithHFToken(token string) CacheOption {
	return func(o *cacheOptions) { o.hfToken = token }
}

// WithCacheBaseURL overrides the HuggingFace base URL. Intended for testing.
// Default: "https://huggingface.co".
func WithCacheBaseURL(url string) CacheOption {
	return func(o *cacheOptions) { o.baseURL = url }
}

// EnsureModel downloads any missing model files for repoID from HuggingFace
// and returns the local model directory path.
//
// Required files (onnx/model.onnx, tokenizer.json) must be successfully
// downloaded or already present; missing required files return an error.
// Optional files (onnx/model.onnx_data, tokenizer_config.json) that return
// HTTP 404 are silently skipped.
//
// Downloads are atomic: files are written to a temp path first, then renamed
// into place. Concurrent calls for the same file are safe — the last writer wins
// and no partial files are left on disk.
func (c *ModelCache) EnsureModel(ctx context.Context, repoID string, opts ...CacheOption) (string, error) {
	cfg := &cacheOptions{
		baseURL: "https://huggingface.co",
	}
	for _, o := range opts {
		o(cfg)
	}
	if cfg.hfToken == "" {
		cfg.hfToken = os.Getenv("HF_TOKEN")
	}

	dir := c.ModelDir(repoID)

	// Required files — any download failure is fatal.
	for _, rel := range susfactorRequiredFiles {
		if err := c.downloadFile(ctx, repoID, rel, cfg.hfToken, cfg.baseURL); err != nil {
			return "", newError("download required file %q: %v", rel, err)
		}
	}

	// Optional files — tolerate 404 and auth errors (401/403).
	// A gated repo may return 401 for optional files that exist but aren't
	// accessible with the provided token, or that simply aren't present in the
	// gated manifest. Either way, the classifier can still run without them.
	for _, rel := range susfactorOptionalFiles {
		if err := c.downloadFile(ctx, repoID, rel, cfg.hfToken, cfg.baseURL); err != nil {
			if errors.Is(err, ErrNotFound) || errors.Is(err, ErrUnauthorized) {
				continue // absent or inaccessible; that's fine for optional files
			}
			return "", newError("download optional file %q: %v", rel, err)
		}
	}

	return dir, nil
}

// downloadFile fetches <baseURL>/<repoID>/resolve/main/<filename> and writes
// it atomically to <cache>/<repoID>/<filename>. Returns nil on cache hit
// (file already present). Returns ErrNotFound on HTTP 404.
//
// The HuggingFace base URL is provided as a parameter to allow test servers.
func (c *ModelCache) downloadFile(ctx context.Context, repoID, filename, hfToken, baseURL string) error {
	destPath := filepath.Join(c.ModelDir(repoID), filepath.FromSlash(filename))

	// Cache hit — return immediately without network access.
	if _, err := os.Stat(destPath); err == nil {
		return nil
	}

	// Ensure parent directory exists.
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("create cache dir: %w", err)
	}

	// Build request URL.
	url := fmt.Sprintf("%s/%s/resolve/main/%s", strings.TrimRight(baseURL, "/"), repoID, filename)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}
	req.Header.Set("User-Agent", "odin-prompt-toolkit-go/0.1.0")

	resp, err := hfHTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusNotFound:
		return ErrNotFound
	case http.StatusUnauthorized, http.StatusForbidden:
		return ErrUnauthorized
	case http.StatusOK:
		// fall through to download
	default:
		return fmt.Errorf("server returned %d for %s", resp.StatusCode, url)
	}

	// Stream to a unique temp file, then atomically rename into place.
	// Include a per-process atomic counter so concurrent goroutines downloading
	// the same file always get distinct temp names, even at the same nanosecond.
	tmp := fmt.Sprintf("%s.tmp.%d.%d.%d",
		destPath, os.Getpid(), time.Now().UnixNano(), tmpCounter.Add(1))
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}

	_, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()

	if copyErr != nil || closeErr != nil {
		os.Remove(tmp) // best-effort cleanup
		if copyErr != nil {
			return fmt.Errorf("write temp file: %w", copyErr)
		}
		return fmt.Errorf("close temp file: %w", closeErr)
	}

	// Atomic rename. On Windows and some platforms rename can fail with
	// EEXIST if the destination already exists (written by a racing goroutine);
	// treat that as a successful race-loss.
	if err := os.Rename(tmp, destPath); err != nil {
		os.Remove(tmp)
		if isExistErr(err) {
			return nil // another goroutine won the race; file is present
		}
		return fmt.Errorf("rename into place: %w", err)
	}
	return nil
}

// isExistErr reports whether err indicates the destination already exists
// (EEXIST / ERROR_ALREADY_EXISTS). Used to handle concurrent download races.
func isExistErr(err error) bool {
	return os.IsExist(err) || errors.Is(err, os.ErrExist)
}


