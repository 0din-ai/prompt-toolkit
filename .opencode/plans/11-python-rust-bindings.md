# Phase 11: Python/Rust Hybrid Bindings (PyO3 + Maturin)

**Linear ticket**: 0DIN-1029 (continuation)
**Goal**: Give the Python SDK a ~627× speedup on signature generation by transparently calling Rust for the hot-path LSH functions.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  User code: from odin_sig import simhash_lsh_multi  │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  odin_sig/__init__.py     │
         │  (import dispatcher)      │
         └─────────────┬─────────────┘
                       │
            ┌──────────▼──────────┐
            │  try:               │
            │    from ._native    │──── YES ──▶ Rust (PyO3) ⚡
            │  except ImportError │
            │    from .lsh        │──── NO ───▶ Pure Python 🐍
            └─────────────────────┘
```

**Key principle**: Zero API change. Users who `pip install 0din-sig` get pure Python. Users who `pip install 0din-sig[native]` get Rust acceleration. The public API is identical either way.

## Scope: 5 Functions

| Function | Python location | Rust source | Why it matters |
|----------|----------------|-------------|----------------|
| `simhash_lsh_multi` | `lsh.py:58` | `lsh.rs:24` | 99% of CPU time (294,912 iterations/sig) |
| `normalize_vector` | `lsh.py:192` | `lsh.rs:126` | Called before every signature |
| `hamming_distance_hex` | `lsh.py:137` | `lsh.rs:87` | Every comparison query |
| `cosine_from_hamming` | `lsh.py:170` | `lsh.rs:113` | Every comparison query |
| `compute_embedding_sha256` | `types.py:186` | `lsh.rs:147` | Canonical hash for dedup |

**Out of scope**: CM-LSH, providers, sign_text, type definitions, async code.

## Crate Structure

New crate at `packages/python-native/`:

```
packages/python-native/
├── Cargo.toml          # PyO3 + maturin config, depends on odin-sig (path dep)
├── pyproject.toml      # maturin build backend
├── src/
│   └── lib.rs          # #[pymodule] with 5 #[pyfunction]s + 2 #[pyclass]es
└── README.md           # Build instructions
```

**Why a separate crate** (not a feature flag on `packages/rust/`):
- The existing `odin-sig` crate is a pure library published to crates.io
- PyO3 adds a `cdylib` target and heavy build deps — shouldn't pollute the library
- Maturin expects to own the crate's build process
- Clean separation: `odin-sig` = Rust library, `odin-sig-python` = Python extension

**The native crate depends on `odin-sig` as a path dependency** and just wraps its functions with PyO3 decorators. Zero logic duplication.

## Implementation Plan

### Phase 11a: Scaffold the native crate (low risk)

1. **Create `packages/python-native/Cargo.toml`**
   - `[lib] name = "odin_sig_native"`, `crate-type = ["cdylib"]`
   - Dependencies: `pyo3 = { version = "0.23", features = ["extension-module"] }`, `odin-sig = { path = "../rust", default-features = false }`
   - No ONNX/OpenAI features needed — we only wrap pure computation

2. **Create `packages/python-native/pyproject.toml`**
   - Build backend: `maturin`
   - `[tool.maturin] module-name = "odin_sig._native"`
   - This makes the compiled `.so`/`.dylib` importable as `odin_sig._native`

3. **Create `packages/python-native/src/lib.rs`**
   - `#[pymodule]` named `_native`
   - `#[pyclass]` for `LshFamily` (family, bits, signature, bands) — mirrors Python dataclass
   - `#[pyclass]` for `LshConfig` (families, bits, bands) — mirrors Python dataclass
   - 5 `#[pyfunction]`s that convert Python types → Rust types → call `odin_sig::*` → convert back

4. **Test**: `cd packages/python-native && maturin develop` → `python3 -c "from odin_sig._native import simhash_lsh_multi; print('OK')"`

**Commit**: `feat(0DIN-1029): Scaffold PyO3 native extension crate`

### Phase 11b: Implement the 5 function bindings (medium risk)

Each function needs a thin wrapper:

```rust
#[pyfunction]
#[pyo3(signature = (normalized_vector, families=3, bits=256, bands=16))]
fn simhash_lsh_multi(
    normalized_vector: Vec<f32>,
    families: usize,
    bits: usize,
    bands: usize,
) -> Vec<LshFamily> {
    let config = odin_sig::LshConfig { families, bits, bands };
    let results = odin_sig::simhash_lsh_multi(&normalized_vector, &config);
    results.into_iter().map(LshFamily::from).collect()
}
```

Similar thin wrappers for the other 4 functions. Key type conversions:
- `Vec<f32>` ↔ Python `list[float]` (automatic via PyO3)
- `String` ↔ Python `str` (automatic)
- `usize` ↔ Python `int` (automatic)
- `f64` ↔ Python `float` (automatic)
- `Vec<LshFamily>` → Python `list[LshFamily]` (via `#[pyclass]`)

**Commit**: `feat(0DIN-1029): Implement PyO3 bindings for 5 core LSH functions`

### Phase 11c: Wire up transparent fallback in Python SDK (low risk)

1. **Create `packages/python/odin_sig/_accel.py`** — acceleration dispatcher:
   ```python
   """Transparent native acceleration layer."""
   try:
       from odin_sig._native import (
           simhash_lsh_multi as _native_simhash_lsh_multi,
           normalize_vector as _native_normalize_vector,
           hamming_distance_hex as _native_hamming_distance_hex,
           cosine_from_hamming as _native_cosine_from_hamming,
           compute_embedding_sha256 as _native_compute_embedding_sha256,
           LshFamily as NativeLshFamily,
       )
       NATIVE_AVAILABLE = True
   except ImportError:
       NATIVE_AVAILABLE = False
   ```

2. **Modify `packages/python/odin_sig/lsh.py`** — at the bottom, conditionally replace:
   ```python
   # Transparent native acceleration
   from odin_sig._accel import NATIVE_AVAILABLE
   if NATIVE_AVAILABLE:
       from odin_sig._accel import (
           _native_simhash_lsh_multi as simhash_lsh_multi,
           _native_normalize_vector as normalize_vector,
           _native_hamming_distance_hex as hamming_distance_hex,
           _native_cosine_from_hamming as cosine_from_hamming,
       )
   ```
   
   Similarly for `compute_embedding_sha256` in `types.py`.

3. **Modify `packages/python/odin_sig/__init__.py`** — add `NATIVE_AVAILABLE` to exports.

4. **Add `[native]` extra to `packages/python/pyproject.toml`**:
   ```toml
   [project.optional-dependencies]
   native = ["odin-sig-native"]  # or whatever the wheel name is
   ```

**Commit**: `feat(0DIN-1029): Wire transparent Rust fallback in Python SDK`

### Phase 11d: Verify correctness against test vectors (critical)

1. **Build the native extension**: `cd packages/python-native && maturin develop --release`
2. **Run existing test suite**: `cd packages/python && python3 -m pytest tests/test_vectors.py -v`
   - These tests exercise `simhash_lsh_multi`, `hamming_distance_hex`, `cosine_from_hamming`, `normalize_vector`, `compute_embedding_sha256` against canonical test vectors
   - With native acceleration active, they'll test the Rust path
   - **Must produce bit-identical results** to pass

3. **Add a new test** `tests/test_native_fallback.py`:
   - Explicitly test both paths (native and pure-Python) produce identical results
   - Test `NATIVE_AVAILABLE` flag
   - Test that the `LshFamily` returned by native has the same attributes as the Python dataclass

**Commit**: `test(0DIN-1029): Verify native bindings against canonical test vectors`

### Phase 11e: Benchmark and document (low risk)

1. **Run benchmark**: Compare Python-only vs native on the same 3,714-prompt dataset
   - Expected: ~9 sigs/sec → ~5,640 sigs/sec (627× speedup)
   - Measure actual numbers on this machine

2. **Update `demos/RESULTS.md`** with native binding results

3. **Add build instructions** to `packages/python-native/README.md`

4. **Add Makefile targets**: `make native-build`, `make native-test`

**Commit**: `docs(0DIN-1029): Add native binding benchmarks and build instructions`

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyO3 type conversion mismatch | Low | High | Test vectors catch any discrepancy |
| `LshFamily` attribute mismatch | Medium | Medium | Mirror exact Python dataclass fields |
| maturin + hatchling conflict | Low | Medium | Separate crates, separate pyproject.toml |
| Python 3.14 + PyO3 compat | Medium | High | PyO3 0.23 supports 3.13+; verify 3.14 |
| `compute_embedding_sha256` JSON format | Medium | High | Already verified Rust/Python produce same SHA256 in test vectors |

## Key Decision: LshFamily Return Type

The native `simhash_lsh_multi` returns `Vec<LshFamily>` where `LshFamily` is a `#[pyclass]`. But the Python code uses `LSHFamily` (a dataclass from `lsh.py`). Two options:

**Option A (Recommended)**: Return native `LshFamily` as a `#[pyclass]` with identical attributes (`family`, `bits`, `signature`, `bands`). The Python `LSHFamily` dataclass and the native `LshFamily` pyclass are duck-type compatible — both have the same fields. Code that accesses `.signature` or `.bands` works with either.

**Option B**: Convert to Python dicts and construct `LSHFamily` dataclass instances in Python. Adds overhead and complexity.

Going with **Option A** — duck typing is Pythonic, and the test vectors will catch any field mismatch.

## Dependencies

- **PyO3 0.23+** — latest stable, supports Python 3.8-3.13 (need to verify 3.14)
- **maturin 1.x** — build tool for PyO3 projects
- **odin-sig** (path dep) — the existing Rust crate, no-default-features (no ONNX/OpenAI)

## Estimated Effort

| Phase | Complexity | Time |
|-------|-----------|------|
| 11a: Scaffold | Low | ~15 min |
| 11b: Implement bindings | Medium | ~30 min |
| 11c: Wire fallback | Low | ~15 min |
| 11d: Test vectors | Low | ~15 min |
| 11e: Benchmark + docs | Low | ~20 min |
| **Total** | | **~1.5 hours** |
