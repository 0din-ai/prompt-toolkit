# Agent Instructions — 0DIN Prompt Toolkit

This file tells AI coding agents what to do before committing changes in this repo.
**Read it before touching any source file.**

---

## Quick-reference: format / lint commands

| Language | Format (run before commit) | Lint | Test |
|---|---|---|---|
| **Rust** | `cd packages/rust && cargo fmt` | `cd packages/rust && cargo clippy --all-features -- -D warnings` | `cd packages/rust && cargo test --lib --features cm-lsh` |
| **Python** | `cd packages/python && ruff check --fix odin_prompt_toolkit/ tests/` | `cd packages/python && ruff check odin_prompt_toolkit/ tests/` | `cd packages/python && python -m pytest tests/ -v` |
| **TypeScript** | `cd packages/typescript && npm run format` | `cd packages/typescript && npm run lint` | `cd packages/typescript && npm test` |
| **Go** | `cd packages/go && gofmt -w .` | `cd packages/go && go vet ./...` | `cd packages/go && CGO_ENABLED=1 go test ./...` |
| **Docs** | — | `cd docs && npm run build` | — |

`make fmt` runs all formatters. `make lint` runs all linters. `make check` runs both.

---

## Pre-commit checklist (run for every commit)

Before committing, run the commands that correspond to the files you changed:

```
# Rust files (.rs, Cargo.toml, Cargo.lock)
cd packages/rust && cargo fmt
cd packages/rust && cargo clippy --all-features -- -D warnings

# Python files (.py, pyproject.toml)
cd packages/python && ruff check --fix odin_prompt_toolkit/ tests/

# TypeScript files (.ts, package.json, tsconfig.json)
cd packages/typescript && npm run format
cd packages/typescript && npm run lint

# Go files (.go, go.mod, go.sum)
cd packages/go && gofmt -w .
cd packages/go && go vet ./...

# Docs files (docs/**)
cd docs && npm run build
```

If any command fails, fix the issue before committing. Do not commit with a known lint or format error.

---

## Rust

### Format — mandatory, not optional

CI runs `cargo fmt --check`. If you commit unformatted Rust code, the lint job fails.

**Always run `cargo fmt` before committing any `.rs` file:**
```
cd packages/rust && cargo fmt
```

Then verify clean:
```
cd packages/rust && cargo fmt --check
```

`--check` exits non-zero if anything would change. If it exits non-zero after you ran `cargo fmt`, something is wrong with your environment.

### Clippy

```
cd packages/rust && cargo clippy --all-features -- -D warnings
```

Warnings are errors in CI. Fix all diagnostics before committing.

### Tests

```
# Standard unit tests + CM-LSH
cd packages/rust && cargo test --lib --features cm-lsh

# SusFactor unit tests (no model required)
cd packages/rust && cargo test --lib --features susfactor

# Full SusFactor integration (requires SUSFACTOR_MODEL_DIR)
cargo test --features susfactor,susfactor-vertex -p odin-prompt-toolkit
```

### Feature discipline — `susfactor-vertex` must not pull in `ort` or `ndarray`

After changing `Cargo.toml` or any `susfactor` code, verify the feature tree is clean:

```
cargo tree --features susfactor-vertex --no-default-features -p odin-prompt-toolkit \
  | grep -E "\bort\b|\bndarray\b"
```

This must return empty. If it doesn't, you introduced a forbidden dependency — revert and fix before committing.

---

## Python

### Lint and format

```
# Auto-fix what ruff can
cd packages/python && ruff check --fix odin_prompt_toolkit/ tests/

# Verify clean (what CI runs)
cd packages/python && ruff check odin_prompt_toolkit/ tests/
```

`ruff check` must exit 0 before committing.

### Tests

```
cd packages/python && python -m pytest tests/ -v
```

Parity and integration tests skip automatically when `SUSFACTOR_MODEL_DIR` is not set.

---

## TypeScript

### Format and lint

```
cd packages/typescript && npm run format   # prettier
cd packages/typescript && npm run lint     # eslint — must pass for CI
```

### Build check

```
cd packages/typescript && npm run build
```

Run this after any type or API change to catch compile errors before pushing.

### Tests

```
cd packages/typescript && npm test
```

---

## Go

### Format and vet

```
cd packages/go && gofmt -w .
cd packages/go && go vet ./...
```

`gofmt` rewrites files in place. `go vet` must exit 0.

### Tests

```
cd packages/go && CGO_ENABLED=1 go test ./... -count=1
```

Requires `libtokenizers.a` present. Run `bash packages/go/scripts/download-libtokenizers.sh` first if the file is missing.

---

## Docs (Docusaurus)

Before committing any change under `docs/`:

```
cd docs && npm run build
```

This catches broken MDX, bad Mermaid diagrams, and broken links. A build failure here blocks the deploy pipeline.

---

## Branch and PR rules

- **Never commit feature work directly to `main`.** Use a feature branch and open a PR.
- Hotfixes to examples, docs, or comments may go directly to main if the change is trivial and self-contained.
- PR descriptions must reference a Linear issue (see below).

---

## Issue tracking — Linear

This project uses **Linear**, not GitHub Issues.

Before starting any work, search for an existing issue:
```
linear issue query --team 0DIN --search "<topic>"
```

Create an issue only if none exists. Include the Linear issue ID in your branch name and PR description.

---

## Version discipline

Rust, Python, and TypeScript versions must stay in sync. Current series: **`0.7.x`**.

- Patch bumps for language-specific fixes may be done independently.
- Minor and major bumps must touch all three manifests in a single commit:
  - `packages/rust/Cargo.toml`
  - `packages/python/pyproject.toml`
  - `packages/typescript/package.json`

Do not bump one without the others for a coordinated release.

---

## Secrets and credentials

**No hardcoded secrets, project IDs, endpoint URLs, or credentials** anywhere in source files, examples, or docs.

Use placeholder strings:
- `{PROJECT_ID}`
- `{ENDPOINT_ID}`
- `{API_KEY}`

Or reference environment variables: `os.environ["PROJECT_ID"]`, `process.env.PROJECT_ID`, etc.

---

## Spec files

`spec/` is the source of truth for algorithm behavior. Read the relevant spec before implementing any feature:

| File | What it covers |
|---|---|
| `spec/INTEGRATION.md` | Caller-facing API contract — read this first |
| `spec/SUSFACTOR-VERTEX.md` | SusFactor Vertex AI integration spec |
| `spec/test-vectors/` | Golden test vectors across all SDKs |

Implementations must match the spec. If you find a discrepancy, fix the implementation — not the spec — unless the spec has a confirmed error.
