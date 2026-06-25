package susfactor

import (
	"testing"
)

func TestChunkTokenIDs(t *testing.T) {
	t.Run("empty slice → one empty chunk", func(t *testing.T) {
		chunks := ChunkTokenIDs([]int64{})
		if len(chunks) != 1 {
			t.Fatalf("want 1 chunk, got %d", len(chunks))
		}
		if len(chunks[0]) != 0 {
			t.Errorf("want empty chunk, got len=%d", len(chunks[0]))
		}
	})

	t.Run("single token → one chunk", func(t *testing.T) {
		chunks := ChunkTokenIDs([]int64{42})
		if len(chunks) != 1 {
			t.Fatalf("want 1 chunk, got %d", len(chunks))
		}
		if chunks[0][0] != 42 {
			t.Errorf("want [42], got %v", chunks[0])
		}
	})

	t.Run("exactly MaxContentTokens → one chunk", func(t *testing.T) {
		ids := makeSeq(MaxContentTokens)
		chunks := ChunkTokenIDs(ids)
		if len(chunks) != 1 {
			t.Fatalf("want 1 chunk, got %d", len(chunks))
		}
		if len(chunks[0]) != MaxContentTokens {
			t.Errorf("want len=%d, got %d", MaxContentTokens, len(chunks[0]))
		}
	})

	t.Run("MaxContentTokens+1 → two chunks with overlap", func(t *testing.T) {
		ids := makeSeq(MaxContentTokens + 1)
		chunks := ChunkTokenIDs(ids)
		if len(chunks) != 2 {
			t.Fatalf("want 2 chunks, got %d", len(chunks))
		}
		// chunk 0: [0 : 510]
		if len(chunks[0]) != MaxContentTokens {
			t.Errorf("chunk 0 len=%d, want %d", len(chunks[0]), MaxContentTokens)
		}
		// chunk 1: [460 : 511]
		wantStart := ChunkStride
		wantLen := (MaxContentTokens + 1) - wantStart
		if len(chunks[1]) != wantLen {
			t.Errorf("chunk 1 len=%d, want %d", len(chunks[1]), wantLen)
		}
		// verify overlap: last ChunkOverlap tokens of chunk0 == first ChunkOverlap of chunk1
		for i := 0; i < ChunkOverlap; i++ {
			c0val := chunks[0][MaxContentTokens-ChunkOverlap+i]
			c1val := chunks[1][i]
			if c0val != c1val {
				t.Errorf("overlap mismatch at offset %d: chunk0=%d chunk1=%d", i, c0val, c1val)
			}
		}
	})

	t.Run("len=1024 chunk boundaries", func(t *testing.T) {
		ids := makeSeq(1024)
		chunks := ChunkTokenIDs(ids)

		// chunk 0: [0:510]
		assertChunkBounds(t, chunks, 0, ids, 0, 510)
		// chunk 1: [460:970]
		assertChunkBounds(t, chunks, 1, ids, 460, 970)
		// chunk 2: [920:1024]
		assertChunkBounds(t, chunks, 2, ids, 920, 1024)

		if len(chunks) != 3 {
			t.Errorf("want 3 chunks, got %d", len(chunks))
		}
	})

	t.Run("no chunk exceeds MaxContentTokens", func(t *testing.T) {
		for _, n := range []int{0, 1, 509, 510, 511, 1000, 2048} {
			ids := makeSeq(n)
			chunks := ChunkTokenIDs(ids)
			for i, c := range chunks {
				if len(c) > MaxContentTokens {
					t.Errorf("n=%d chunk[%d] len=%d > MaxContentTokens=%d",
						n, i, len(c), MaxContentTokens)
				}
			}
		}
	})

	t.Run("adjacent chunks share exactly ChunkOverlap tokens", func(t *testing.T) {
		ids := makeSeq(2048)
		chunks := ChunkTokenIDs(ids)
		for i := 1; i < len(chunks); i++ {
			prev := chunks[i-1]
			curr := chunks[i]
			overlapLen := ChunkOverlap
			if len(curr) < overlapLen {
				overlapLen = len(curr)
			}
			for j := 0; j < overlapLen; j++ {
				prevVal := prev[len(prev)-overlapLen+j]
				currVal := curr[j]
				if prevVal != currVal {
					t.Errorf("chunks[%d/%d] overlap mismatch at j=%d: prev=%d curr=%d",
						i-1, i, j, prevVal, currVal)
				}
			}
		}
	})
}

// makeSeq returns [0, 1, 2, ..., n-1] as int64.
func makeSeq(n int) []int64 {
	s := make([]int64, n)
	for i := range s {
		s[i] = int64(i)
	}
	return s
}

// assertChunkBounds checks that chunks[idx] equals ids[start:end].
func assertChunkBounds(t *testing.T, chunks [][]int64, idx int, ids []int64, start, end int) {
	t.Helper()
	if idx >= len(chunks) {
		t.Fatalf("chunk index %d out of range (len=%d)", idx, len(chunks))
	}
	want := ids[start:end]
	got := chunks[idx]
	if len(got) != len(want) {
		t.Errorf("chunk[%d] len=%d, want %d (ids[%d:%d])", idx, len(got), len(want), start, end)
		return
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("chunk[%d][%d] = %d, want %d", idx, i, got[i], want[i])
		}
	}
}
