package susfactor

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestLoadTokenizerResolvesBOSEOS asserts that bosID/eosID are resolved
// dynamically from the tokenizer.json's added_tokens table (XLM-RoBERTa
// vocab: <s>=0, </s>=2), not hardcoded in the chunking loop. Model-gated:
// skips when SUSFACTOR_MODEL_DIR is unset (no tokenizer.json is checked
// into the repo).
func TestLoadTokenizerResolvesBOSEOS(t *testing.T) {
	dir := os.Getenv("SUSFACTOR_MODEL_DIR")
	if dir == "" {
		t.Skip("SUSFACTOR_MODEL_DIR not set")
	}
	tokPath := filepath.Join(dir, "tokenizer.json")

	tk, bosID, eosID, err := loadTokenizerNoTruncation(tokPath)
	if err != nil {
		t.Fatalf("loadTokenizerNoTruncation: %v", err)
	}
	defer tk.Close()

	if bosID != 0 {
		t.Errorf("bosID = %d, want 0 (<s>)", bosID)
	}
	if eosID != 2 {
		t.Errorf("eosID = %d, want 2 (</s>)", eosID)
	}
}

// TestBuildChunkInputSpecialTokens is a regression for 0DIN-2132: the chunker
// tokenized the full input WITH special tokens added, then sliced that flat
// array into overlapping chunks — so only chunk 0 kept a real leading <s> and
// only the last chunk kept a real trailing </s>. Every interior chunk had
// neither. buildChunkInput now wraps each chunk independently, so every chunk
// (not just the first/last) must carry its own bosID/eosID and an all-1s mask
// sized to match.
func TestBuildChunkInputSpecialTokens(t *testing.T) {
	const bosID, eosID = int64(-1), int64(-2)

	assertChunk := func(t *testing.T, contentIDs []int64) {
		t.Helper()
		ids, mask := buildChunkInput(contentIDs, bosID, eosID)

		wantLen := len(contentIDs) + 2
		if len(ids) != wantLen {
			t.Fatalf("len(ids) = %d, want %d", len(ids), wantLen)
		}
		if ids[0] != bosID {
			t.Errorf("ids[0] = %d, want bosID %d", ids[0], bosID)
		}
		if ids[len(ids)-1] != eosID {
			t.Errorf("ids[last] = %d, want eosID %d", ids[len(ids)-1], eosID)
		}
		for i, want := range contentIDs {
			if got := ids[i+1]; got != want {
				t.Errorf("ids[%d] = %d, want content id %d", i+1, got, want)
			}
		}

		if len(mask) != wantLen {
			t.Fatalf("len(mask) = %d, want %d", len(mask), wantLen)
		}
		for i, m := range mask {
			if m != 1 {
				t.Errorf("mask[%d] = %d, want 1", i, m)
			}
		}
	}

	t.Run("short input, single chunk", func(t *testing.T) {
		content := makeSeq(50)
		idChunks := ChunkTokenIDs(content)
		if len(idChunks) != 1 {
			t.Fatalf("want 1 chunk for short input, got %d", len(idChunks))
		}
		assertChunk(t, idChunks[0])
	})

	t.Run("long input, every chunk (including interior) gets real special tokens", func(t *testing.T) {
		// > 510 content tokens forces multiple chunks (chunk.go: MaxContentTokens=510).
		content := makeSeq(1024)
		idChunks := ChunkTokenIDs(content)
		if len(idChunks) < 3 {
			t.Fatalf("want >= 3 chunks for 1024 content tokens, got %d", len(idChunks))
		}
		for i, chunk := range idChunks {
			chunk := chunk
			t.Run(fmt.Sprintf("chunk_%d", i), func(t *testing.T) {
				assertChunk(t, chunk)
			})
		}
	})
}
