# CI/CD Pipeline Documentation

## Overview

The odin-prompt-toolkit uses GitHub Actions for continuous integration across all three language implementations (Rust, Python, TypeScript).

## Workflows

### Main CI Pipeline (`.github/workflows/ci.yml`)

Runs on:
- Push to `main` branch
- Pull requests targeting `main`

#### Jobs

1. **Rust** (`rust`)
   - Tests with all features enabled
   - Tests with no default features
   - Clippy linting (warnings treated as errors)
   - Documentation generation
   - **Platform**: Ubuntu latest
   - **Rust version**: Stable

2. **Python** (`python`)
   - Tests across Python 3.10, 3.11, 3.12, 3.13
   - Full test suite with pytest
   - Type checking with mypy
   - **Platform**: Ubuntu latest
   - **Matrix strategy**: 4 Python versions

3. **TypeScript** (`typescript`)
   - Tests across Node.js 20 and 22
   - Build verification
   - Full test suite with Jest
   - ESLint checking
   - **Platform**: Ubuntu latest
   - **Matrix strategy**: 2 Node versions

4. **Cross-Validation** (`cross-validate`)
   - Runs after all language tests pass
   - Installs all three SDKs
   - Runs unified validation script
   - Verifies cross-language compatibility
   - **Dependencies**: `rust`, `python`, `typescript` jobs
   - **Total tests validated**: 61 tests

5. **Documentation** (`docs`)
   - Builds Docusaurus documentation site
   - Verifies all documentation builds without errors
   - **Platform**: Ubuntu latest
   - **Node version**: 22

## Local CI Simulation

Run the full CI pipeline locally:

```bash
# Full pipeline
make ci

# Individual components
make test              # All language tests (61 tests)
make cross-validate    # Cross-language validation
make lint              # All linters
make docs              # Build documentation
```

## Pre-commit Hooks

Install pre-commit hooks to catch issues before committing:

```bash
pip install pre-commit
pre-commit install
```

**Hooks run on commit:**
- Rust: `cargo fmt --check`, `cargo clippy`
- Python: `ruff check`, `mypy`
- TypeScript: `eslint`, `tsc --noEmit`

**Manual run:**
```bash
pre-commit run --all-files
```

## CI Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/0din-ai/odin-prompt-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/0din-ai/odin-prompt-toolkit/actions/workflows/ci.yml)
```

## Test Results Summary

| Component | Tests | Platform |
|-----------|-------|----------|
| Rust Core | 43 tests | Ubuntu, Stable Rust |
| Python SDK | 11 tests | Ubuntu, Python 3.10-3.13 |
| TypeScript SDK | 7 tests | Ubuntu, Node 20-22 |
| Cross-validation | 61 total | Ubuntu |
| Documentation | Build check | Ubuntu, Node 22 |

## Troubleshooting

### CI Failing on Rust

- Check Clippy warnings: `cargo clippy --all-features -- -D warnings`
- Verify tests pass locally: `cargo test --all-features`
- Check documentation builds: `cargo doc --all-features --no-deps`

### CI Failing on Python

- Check mypy errors: `cd packages/python && mypy src/odin_prompt_toolkit/`
- Verify tests pass: `cd packages/python && pytest tests/ -v`
- Check for Python version compatibility issues

### CI Failing on TypeScript

- Check lint errors: `cd packages/typescript && npm run lint`
- Verify tests pass: `cd packages/typescript && npm test`
- Check type errors: `cd packages/typescript && npx tsc --noEmit`

### Cross-validation Failing

Run locally to debug:
```bash
python scripts/cross_validate.py
```

The script shows detailed output for each language's test suite.

### Documentation Build Failing

```bash
cd docs
npm ci
npm run build
```

Check for broken links or invalid markdown in `docs/docs/`.

## Maintenance

### Updating CI

When updating the CI pipeline:

1. Test locally first with `make ci`
2. Update `.github/workflows/ci.yml`
3. Create a PR and verify CI passes
4. Monitor first few runs after merge

### Adding New Dependencies

When adding dependencies:

1. Update package files (`Cargo.toml`, `pyproject.toml`, `package.json`)
2. Update installation instructions in README.md
3. Verify CI installs dependencies correctly
4. Update pre-commit hooks if needed

### Version Bumps

Use the version management script:

```bash
./scripts/bump-version.sh 0.2.0
```

This updates all three packages and creates a git tag.

## Performance

Typical CI run times:

- Rust job: ~2-3 minutes
- Python job: ~3-4 minutes (4 Python versions)
- TypeScript job: ~2-3 minutes (2 Node versions)
- Cross-validation: ~4-5 minutes
- Documentation: ~1-2 minutes

**Total pipeline time**: ~10-15 minutes (jobs run in parallel)

## Security

- No secrets required for basic CI
- ONNX models downloaded on-demand (not committed to repo)
- Private repository access required (not public yet)

## Future Enhancements

Potential improvements:

- [ ] Add coverage reporting (codecov/coveralls)
- [ ] Add performance benchmarks
- [ ] Add automated releases on tag push
- [ ] Add security scanning (dependabot, snyk)
- [ ] Add license checking
- [ ] Add changelog generation
