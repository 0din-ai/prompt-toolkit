package susfactor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLoadTokenizerDisablesTruncation is a regression for the bug where the
// bundled tokenizer.json truncated every prompt to 512 tokens before chunking,
// silently dropping long-prompt content. Model-gated: skips when
// SUSFACTOR_MODEL_DIR is unset.
func TestLoadTokenizerDisablesTruncation(t *testing.T) {
	dir := os.Getenv("SUSFACTOR_MODEL_DIR")
	if dir == "" {
		t.Skip("SUSFACTOR_MODEL_DIR unset; skipping truncation regression")
	}
	tk, bosID, eosID, err := loadTokenizerNoTruncation(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	defer tk.Close()
	if bosID == eosID {
		t.Fatalf("bosID and eosID must differ, got both = %d", bosID)
	}

	long := strings.Repeat("The quarterly business review covered revenue and churn. ", 150)
	ids, _ := tk.Encode(long, true)
	if len(ids) <= MaxContentTokens {
		t.Fatalf("truncation not disabled: got %d tokens (<= %d)", len(ids), MaxContentTokens)
	}
}
