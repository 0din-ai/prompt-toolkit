package susfactor

// ChunkTokenIDs splits a token-ID sequence into overlapping chunks of at most
// MaxContentTokens tokens each. This mirrors the Python/Rust/TypeScript
// implementations exactly:
//
//   - Sequences at or below MaxContentTokens produce exactly one chunk
//     (identical to the input, including empty).
//   - Adjacent chunks share ChunkOverlap tokens so that sentence boundaries
//     near a chunk edge are scored with full context.
//   - An empty input produces one empty chunk.
//
// The caller is responsible for providing the payload token IDs (not including
// special tokens added by the tokenizer).
func ChunkTokenIDs(ids []int64) [][]int64 {
	if len(ids) <= MaxContentTokens {
		// Single chunk — copy to avoid aliasing the caller's slice.
		chunk := make([]int64, len(ids))
		copy(chunk, ids)
		return [][]int64{chunk}
	}

	var chunks [][]int64
	start := 0
	for {
		end := start + MaxContentTokens
		if end > len(ids) {
			end = len(ids)
		}
		chunk := make([]int64, end-start)
		copy(chunk, ids[start:end])
		chunks = append(chunks, chunk)
		if end == len(ids) {
			break
		}
		start += ChunkStride
	}
	return chunks
}
