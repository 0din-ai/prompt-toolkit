# Contributing to signature-sdk

## Development Setup

### Prerequisites

- **Rust**: Install via [rustup](https://rustup.rs/)
- **Python**: 3.10+ (3.10, 3.11, 3.12, or 3.13)
- **Node.js**: 20+ (20 or 22)
- **pre-commit**: Install via `pip install pre-commit`

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/0din-ai/signature-sdk.git
cd sig-sdk

# Install dependencies for all languages
make install

# Install pre-commit hooks
pre-commit install
```

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality before commits. The hooks run:

- **Rust**:
  - `cargo fmt --check` - Format checking
  - `cargo clippy --all-features -- -D warnings` - Linting

- **Python**:
  - `ruff check` - Linting
  - `mypy` - Type checking

- **TypeScript**:
  - `npm run lint` - ESLint
  - `tsc --noEmit` - Type checking

### Running Hooks Manually

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hook
pre-commit run rust-fmt --all-files
pre-commit run python-ruff --all-files
pre-commit run typescript-lint --all-files
```

### Skipping Hooks (Emergency Only)

```bash
# Skip pre-commit hooks (use sparingly!)
git commit --no-verify -m "message"
```

## Testing

### Run All Tests

```bash
# All languages
make test

# Individual languages
make test-rust
make test-python
make test-typescript

# Cross-language validation
make cross-validate
```

### Test Vectors

All implementations are validated against canonical test vectors in `spec/test-vectors/`.

```bash
# Generate test vectors (from Rust canonical implementation)
make generate-vectors

# Validate implementations
make validate-vectors  # alias for 'make test'
```

## Code Style

### Rust

- Follow standard Rust conventions
- Use `cargo fmt` for formatting
- Address all `cargo clippy` warnings

### Python

- Follow PEP 8 style guide
- Use `black` for formatting: `make fmt-python`
- Use `ruff` for linting: `make lint-python`
- Use type hints with `mypy` checking

### TypeScript

- Follow project ESLint configuration
- Use Prettier for formatting: `make fmt-typescript`
- Enable strict type checking

## Documentation

- Update relevant README files when adding features
- Add multi-language code examples to Docusaurus docs
- Update `spec/SPEC.md` for algorithm changes
- Update `spec/VERSIONING.md` for signature format changes

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and ensure all tests pass: `make test`
3. Ensure code quality checks pass: `make lint`
4. Update documentation if needed
5. Commit with semantic commit format: `type(scope): description`
   - Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
   - Scope: `rust`, `python`, `typescript`, `docs`, `ci`, etc.
6. Push and create a pull request
7. Wait for CI checks to pass
8. Request review

## Semantic Commit Format

We use semantic commit messages:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `test`: Test changes
- `chore`: Build/tooling changes

**Examples**:
```
feat(rust): add support for custom hash functions
fix(python): correct cosine similarity estimation
docs(guides): add duplicate detection tutorial
test(typescript): add CM-LSH test cases
chore(ci): update GitHub Actions to v4
```

## Release Process

Versions are synchronized across all three language packages.

```bash
# Bump version (updates all three packages)
./scripts/bump-version.sh 0.2.0

# Tag release
git tag v0.2.0
git push origin v0.2.0
```

## Questions?

Open an issue or discussion on GitHub!
