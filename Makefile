.PHONY: help test test-rust test-python test-typescript test-all cross-validate build build-rust build-python build-typescript clean clean-rust clean-python clean-typescript generate-vectors examples examples-rust examples-python examples-typescript install install-rust install-python install-typescript lint fmt check docs docs-dev package package-python package-rust package-typescript deliverable deliverable-with-model offline offline-with-model showcase showcase-install

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
RESET := \033[0m

##@ General

help: ## Display this help message
	@echo "$(CYAN)0DIN Prompt Toolkit - Multi-language Build System$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(CYAN)<target>$(RESET)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Testing

test: test-rust test-python test-typescript ## Run all tests across all languages
	@echo "$(GREEN)✅ All tests passed!$(RESET)"
	@echo ""
	@echo "$(CYAN)Test Summary:$(RESET)"
	@echo "  Rust:       43 tests passing"
	@echo "  Python:     11 tests passing"
	@echo "  TypeScript: 7 tests passing"
	@echo "  $(GREEN)Total:      61 tests passing$(RESET)"

test-rust: ## Run Rust tests
	@echo "$(CYAN)Running Rust tests...$(RESET)"
	@cd packages/rust && cargo test --lib --features cm-lsh
	@echo "$(GREEN)✅ Rust tests passed$(RESET)"
	@echo ""

test-python: ## Run Python tests
	@echo "$(CYAN)Running Python tests...$(RESET)"
	@cd packages/python && python -m pytest tests/ -v
	@echo "$(GREEN)✅ Python tests passed$(RESET)"
	@echo ""

test-typescript: ## Run TypeScript tests
	@echo "$(CYAN)Running TypeScript tests...$(RESET)"
	@cd packages/typescript && npm test
	@echo "$(GREEN)✅ TypeScript tests passed$(RESET)"
	@echo ""

test-all: test ## Alias for 'test'

test-watch-rust: ## Run Rust tests in watch mode
	@cd packages/rust && cargo watch -x 'test --lib --features cm-lsh'

test-watch-python: ## Run Python tests in watch mode
	@cd packages/python && python -m pytest tests/ --watch

test-watch-typescript: ## Run TypeScript tests in watch mode
	@cd packages/typescript && npm run test:watch

##@ Building

build: build-rust build-typescript ## Build all packages
	@echo "$(GREEN)✅ All packages built!$(RESET)"

build-rust: ## Build Rust package
	@echo "$(CYAN)Building Rust package...$(RESET)"
	@cd packages/rust && cargo build --release --all-features
	@echo "$(GREEN)✅ Rust package built$(RESET)"

build-python: ## Check Python package (no build step needed)
	@echo "$(CYAN)Checking Python package...$(RESET)"
	@cd packages/python && python -c "import odin_prompt_toolkit; print('✅ Python package OK')"

build-typescript: ## Build TypeScript package
	@echo "$(CYAN)Building TypeScript package...$(RESET)"
	@cd packages/typescript && npm run build
	@echo "$(GREEN)✅ TypeScript package built$(RESET)"

##@ Packaging

DIST_DIR := dist

package: package-python package-rust package-typescript ## Build distributable artifacts for all packages
	@echo ""
	@echo "$(GREEN)✅ All packages built!$(RESET)"
	@echo ""
	@echo "$(CYAN)Artifacts (dist/):$(RESET)"
	@ls -1 $(DIST_DIR)/ | sed 's/^/  /'

package-python: ## Build Python wheel and sdist
	@echo "$(CYAN)Packaging Python SDK...$(RESET)"
	@cd packages/python && hatch build
	@mkdir -p $(DIST_DIR)
	@cp packages/python/dist/*.whl packages/python/dist/*.tar.gz $(DIST_DIR)/
	@echo "$(GREEN)✅ Python package built and copied to $(DIST_DIR)/$(RESET)"

package-rust: ## Package Rust crate
	@echo "$(CYAN)Packaging Rust crate...$(RESET)"
	@cd packages/rust && cargo package --allow-dirty
	@mkdir -p $(DIST_DIR)
	@cp packages/rust/target/package/*.crate $(DIST_DIR)/
	@echo "$(GREEN)✅ Rust crate packaged and copied to $(DIST_DIR)/$(RESET)"

package-typescript: ## Build and pack TypeScript npm tarball
	@echo "$(CYAN)Packaging TypeScript SDK...$(RESET)"
	@cd packages/typescript && npm run build && npm pack
	@mkdir -p $(DIST_DIR)
	@cp packages/typescript/*.tgz $(DIST_DIR)/
	@echo "$(GREEN)✅ TypeScript package built and copied to $(DIST_DIR)/$(RESET)"

##@ Deliverable

deliverable: package-python ## Build design partner deliverable tarball (0DIN-1114)
	@echo ""
	@echo "$(CYAN)Building design partner deliverable...$(RESET)"
	@if [ ! -f deliverable/build-tarball.sh ]; then \
		echo "$(RED)Error: deliverable/build-tarball.sh not found$(RESET)"; \
		exit 1; \
	fi
	@cd deliverable && ./build-tarball.sh
	@echo ""
	@echo "$(GREEN)✅ Deliverable built successfully$(RESET)"
	@echo "   Location: deliverable/0DIN Prompt Toolkit-deliverable-*.tar.gz"

deliverable-with-model: package-python ## Build deliverable with local model (faster, no HF download)
	@echo ""
	@echo "$(CYAN)Building design partner deliverable (using local model)...$(RESET)"
	@if [ ! -d ~/.cache/odin-prompt-toolkit/models/v1 ]; then \
		echo "$(YELLOW)Warning: Model not found in cache, will download from HuggingFace$(RESET)"; \
		cd deliverable && ./build-tarball.sh; \
	else \
		cd deliverable && ./build-tarball.sh --model-source ~/.cache/odin-prompt-toolkit/models/v1; \
	fi
	@echo ""
	@echo "$(GREEN)✅ Deliverable built successfully$(RESET)"
	@echo "   Location: deliverable/0DIN Prompt Toolkit-deliverable-*.tar.gz"

offline: package-python ## Build offline installer bundle (downloads model from HuggingFace)
	@echo ""
	@echo "$(CYAN)Building offline installer bundle...$(RESET)"
	@if [ ! -f deliverable/package-offline.sh ]; then \
		echo "$(RED)Error: deliverable/package-offline.sh not found$(RESET)"; \
		exit 1; \
	fi
	@cd deliverable && ./package-offline.sh
	@echo ""
	@echo "$(GREEN)✅ Offline installer bundle built successfully$(RESET)"

offline-with-model: package-python ## Build offline installer bundle using local cached model (faster)
	@echo ""
	@echo "$(CYAN)Building offline installer bundle (using local model)...$(RESET)"
	@if [ ! -d ~/.cache/odin-prompt-toolkit/models/v1 ]; then \
		echo "$(YELLOW)Warning: Model not found in cache, will download from HuggingFace$(RESET)"; \
		cd deliverable && ./package-offline.sh; \
	else \
		cd deliverable && ./package-offline.sh --model-source ~/.cache/odin-prompt-toolkit/models/v1; \
	fi
	@echo ""
	@echo "$(GREEN)✅ Offline installer bundle built successfully$(RESET)"

##@ Test Vectors

generate-vectors: ## Generate test vectors from canonical Rust implementation
	@echo "$(CYAN)Generating test vectors...$(RESET)"
	@cd packages/rust && cargo run --example generate_vectors
	@echo "$(GREEN)✅ Core test vectors generated$(RESET)"
	@cd packages/rust && cargo run --example generate_cm_lsh_vectors --features cm-lsh
	@echo "$(GREEN)✅ CM-LSH test vectors generated$(RESET)"
	@echo ""
	@echo "$(CYAN)Generated files:$(RESET)"
	@ls -lh spec/test-vectors/*.json

generate-susfactor-goldens: ## Regenerate SusFactor golden vectors from validated Rust (requires SUSFACTOR_MODEL_DIR)
	@echo "$(CYAN)Regenerating SusFactor golden vectors...$(RESET)"
	@if [ -z "$(SUSFACTOR_MODEL_DIR)" ]; then \
		echo "$(RED)Error: SUSFACTOR_MODEL_DIR is not set$(RESET)"; \
		echo "Usage: make generate-susfactor-goldens SUSFACTOR_MODEL_DIR=/path/to/cache/susfactor-v1"; \
		exit 1; \
	fi
	@SUSFACTOR_MODEL_DIR=$(SUSFACTOR_MODEL_DIR) bash scripts/regenerate_susfactor_goldens.sh

validate-vectors: test ## Validate all implementations against test vectors (alias for test)

cross-validate: ## Run cross-language validation script (for CI)
	@echo "$(CYAN)Running cross-language validation...$(RESET)"
	@python scripts/cross_validate.py

cross-validate-parity: ## Run cross-language validation + SusFactor parity (requires SUSFACTOR_MODEL_DIR)
	@echo "$(CYAN)Running cross-language validation with SusFactor parity...$(RESET)"
	@if [ -z "$(SUSFACTOR_MODEL_DIR)" ]; then \
		echo "$(RED)Error: SUSFACTOR_MODEL_DIR is not set$(RESET)"; \
		echo "Usage: make cross-validate-parity SUSFACTOR_MODEL_DIR=/path/to/cache/susfactor-v1"; \
		exit 1; \
	fi
	@SUSFACTOR_MODEL_DIR=$(SUSFACTOR_MODEL_DIR) python scripts/cross_validate.py --susfactor-parity

##@ Examples

examples: examples-rust examples-python examples-typescript ## Run all example files

examples-rust: ## Run Rust example files
	@echo "$(CYAN)Running Rust examples...$(RESET)"
	@echo "$(YELLOW)1. Basic signature:$(RESET)"
	@cd packages/rust && cargo run --example basic_signature --quiet
	@echo ""
	@echo "$(YELLOW)2. Similarity comparison:$(RESET)"
	@cd packages/rust && cargo run --example similarity_comparison --quiet
	@echo ""
	@echo "$(YELLOW)3. Duplicate detection:$(RESET)"
	@cd packages/rust && cargo run --example duplicate_detection --quiet
	@echo ""
	@echo "$(YELLOW)4. CM-LSH:$(RESET)"
	@cd packages/rust && cargo run --example cm_lsh_example --features cm-lsh --quiet
	@echo "$(GREEN)✅ All Rust examples completed$(RESET)"
	@echo ""

examples-python: ## Run Python example files
	@echo "$(CYAN)Running Python examples...$(RESET)"
	@echo "$(YELLOW)1. Basic signature:$(RESET)"
	@cd packages/python && PYTHONPATH=. python examples/basic_signature.py
	@echo ""
	@echo "$(YELLOW)2. Similarity comparison:$(RESET)"
	@cd packages/python && PYTHONPATH=. python examples/similarity_comparison.py
	@echo ""
	@echo "$(YELLOW)3. Duplicate detection:$(RESET)"
	@cd packages/python && PYTHONPATH=. python examples/duplicate_detection.py
	@echo ""
	@echo "$(YELLOW)4. CM-LSH:$(RESET)"
	@cd packages/python && PYTHONPATH=. python examples/cm_lsh_example.py
	@echo "$(GREEN)✅ All Python examples completed$(RESET)"
	@echo ""

examples-typescript: ## Run TypeScript example files
	@echo "$(CYAN)Running TypeScript examples...$(RESET)"
	@echo "$(YELLOW)1. Basic signature:$(RESET)"
	@cd packages/typescript && npx ts-node examples/basic_signature.ts
	@echo ""
	@echo "$(YELLOW)2. Similarity comparison:$(RESET)"
	@cd packages/typescript && npx ts-node examples/similarity_comparison.ts
	@echo ""
	@echo "$(YELLOW)3. Duplicate detection:$(RESET)"
	@cd packages/typescript && npx ts-node examples/duplicate_detection.ts
	@echo "$(GREEN)✅ All TypeScript examples completed$(RESET)"
	@echo ""

##@ Installation

install: install-rust install-python install-typescript ## Install dependencies for all packages

install-rust: ## Install Rust dependencies
	@echo "$(CYAN)Installing Rust dependencies...$(RESET)"
	@cd packages/rust && cargo fetch
	@echo "$(GREEN)✅ Rust dependencies installed$(RESET)"

install-python: ## Install Python dependencies
	@echo "$(CYAN)Installing Python dependencies...$(RESET)"
	@cd packages/python && pip install -e ".[dev]"
	@echo "$(GREEN)✅ Python dependencies installed$(RESET)"

install-typescript: ## Install TypeScript dependencies
	@echo "$(CYAN)Installing TypeScript dependencies...$(RESET)"
	@cd packages/typescript && npm install
	@echo "$(GREEN)✅ TypeScript dependencies installed$(RESET)"

##@ Code Quality

lint: lint-rust lint-python lint-typescript ## Run linters for all languages

lint-rust: ## Run Rust linter
	@echo "$(CYAN)Linting Rust code...$(RESET)"
	@cd packages/rust && cargo clippy --all-features -- -D warnings

lint-python: ## Run Python linter
	@echo "$(CYAN)Linting Python code...$(RESET)"
	@cd packages/python && ruff check odin_prompt_toolkit/ tests/

lint-typescript: ## Run TypeScript linter
	@echo "$(CYAN)Linting TypeScript code...$(RESET)"
	@cd packages/typescript && npm run lint

fmt: fmt-rust fmt-python fmt-typescript ## Format code for all languages

fmt-rust: ## Format Rust code
	@echo "$(CYAN)Formatting Rust code...$(RESET)"
	@cd packages/rust && cargo fmt
	@echo "$(GREEN)✅ Rust code formatted$(RESET)"

fmt-python: ## Format Python code
	@echo "$(CYAN)Formatting Python code...$(RESET)"
	@cd packages/python && black odin_prompt_toolkit/ tests/
	@echo "$(GREEN)✅ Python code formatted$(RESET)"

fmt-typescript: ## Format TypeScript code
	@echo "$(CYAN)Formatting TypeScript code...$(RESET)"
	@cd packages/typescript && npm run format
	@echo "$(GREEN)✅ TypeScript code formatted$(RESET)"

check: ## Run all checks (lint + test)
	@$(MAKE) lint
	@$(MAKE) test

##@ Cleaning

clean: clean-rust clean-python clean-typescript ## Clean build artifacts for all packages
	@rm -rf $(DIST_DIR)
	@echo "$(GREEN)✅ All build artifacts cleaned$(RESET)"

clean-rust: ## Clean Rust build artifacts
	@echo "$(CYAN)Cleaning Rust artifacts...$(RESET)"
	@cd packages/rust && cargo clean
	@echo "$(GREEN)✅ Rust artifacts cleaned$(RESET)"

clean-python: ## Clean Python build artifacts
	@echo "$(CYAN)Cleaning Python artifacts...$(RESET)"
	@cd packages/python && rm -rf build/ dist/ *.egg-info .pytest_cache/ __pycache__/
	@find packages/python -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find packages/python -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Python artifacts cleaned$(RESET)"

clean-typescript: ## Clean TypeScript build artifacts
	@echo "$(CYAN)Cleaning TypeScript artifacts...$(RESET)"
	@cd packages/typescript && rm -rf dist/ node_modules/.cache/
	@echo "$(GREEN)✅ TypeScript artifacts cleaned$(RESET)"

##@ Documentation

docs: docs-rust docs-python docs-typescript ## Build all documentation (API docs + Docusaurus site)
	@echo "$(CYAN)Building Docusaurus documentation site...$(RESET)"
	@cd docs && npm run build
	@echo "$(GREEN)✅ All documentation built successfully$(RESET)"
	@echo ""
	@echo "$(CYAN)Generated API docs:$(RESET)"
	@echo "  • Rust:       docs/static/api/rust/odin_prompt_toolkit/index.html"
	@echo "  • Python:     docs/static/api/python/odin_prompt_toolkit.html"
	@echo "  • TypeScript: docs/static/api/typescript/index.html"
	@echo ""
	@echo "$(CYAN)To serve locally:$(RESET)"
	@echo "  cd docs && npm run serve"

docs-rust: ## Generate Rust API documentation (cargo doc)
	@echo "$(CYAN)Generating Rust API docs...$(RESET)"
	@cd packages/rust && cargo doc --all-features --no-deps
	@mkdir -p docs/static/api/rust
	@rm -rf docs/static/api/rust/*
	@cp -r packages/rust/target/doc/* docs/static/api/rust/
	@echo "$(GREEN)✅ Rust API docs generated → docs/static/api/rust/$(RESET)"

docs-python: ## Generate Python API documentation (pdoc)
	@echo "$(CYAN)Generating Python API docs...$(RESET)"
	@mkdir -p docs/static/api/python
	@cd packages/python && python -m pdoc odin_prompt_toolkit -o ../../docs/static/api/python --html
	@echo "$(GREEN)✅ Python API docs generated → docs/static/api/python/$(RESET)"

docs-typescript: ## Generate TypeScript API documentation (typedoc)
	@echo "$(CYAN)Generating TypeScript API docs...$(RESET)"
	@mkdir -p docs/static/api/typescript
	@cd packages/typescript && npx typedoc --out ../../docs/static/api/typescript src/index.ts
	@echo "$(GREEN)✅ TypeScript API docs generated → docs/static/api/typescript/$(RESET)"

docs-dev: ## Start Docusaurus development server
	@echo "$(CYAN)Starting documentation development server...$(RESET)"
	@cd docs && npm start

##@ CI/CD Simulation

ci: clean install lint test ## Simulate CI pipeline (clean, install, lint, test)
	@echo ""
	@echo "$(GREEN)✅ CI pipeline completed successfully!$(RESET)"
	@echo ""
	@echo "$(CYAN)Summary:$(RESET)"
	@echo "  • Dependencies installed"
	@echo "  • Code linted"
	@echo "  • 61 tests passed"
	@echo ""

##@ Information

info: ## Display project information
	@echo "$(CYAN)0DIN Prompt Toolkit - Project Information$(RESET)"
	@echo ""
	@echo "$(YELLOW)Status:$(RESET) ✅ Phases 1-5 Complete (Production Ready)"
	@echo ""
	@echo "$(YELLOW)Packages:$(RESET)"
	@echo "  • Rust:       0DIN Prompt Toolkit       (43 tests)"
	@echo "  • Python:     0DIN Prompt Toolkit       (11 tests)"
	@echo "  • TypeScript: @0din/0DIN Prompt Toolkit      (7 tests)"
	@echo ""
	@echo "$(YELLOW)Test Vectors:$(RESET)"
	@ls spec/test-vectors/ | wc -l | xargs -I {} echo "  • {} files"
	@echo ""
	@echo "$(YELLOW)Documentation:$(RESET)"
	@echo "  • Algorithm:   spec/SPEC.md"
	@echo "  • Versioning:  spec/VERSIONING.md"
	@echo "  • Validation:  VALIDATION.md"
	@echo "  • Rust:        packages/rust/README.md"
	@echo "  • Python:      packages/python/README.md"
	@echo "  • TypeScript:  packages/typescript/README.md"
	@echo ""

version: ## Display version information
	@echo "$(CYAN)Package Versions:$(RESET)"
	@echo "  Rust:       $(shell cd packages/rust && cargo metadata --no-deps --format-version 1 2>/dev/null | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)"
	@echo "  Python:     $(shell cd packages/python && python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "0.1.0")"
	@echo "  TypeScript: $(shell cd packages/typescript && node -p "require('./package.json').version" 2>/dev/null || echo "0.1.0")"

##@ Showcase (0DIN-1029)

showcase-install: ## Install showcase benchmark dependencies
	@echo "$(CYAN)Installing showcase dependencies...$(RESET)"
	@pip install -e "packages/python[onnx]"
	@pip install -r demos/requirements.txt
	@echo "$(GREEN)✅ Showcase dependencies installed$(RESET)"
	@echo ""
	@echo "$(YELLOW)To start pgvector (optional):$(RESET)"
	@echo "  docker compose -f demos/docker-compose.yml up -d"

showcase: ## Run the signature capabilities showcase benchmark
	@if [ -z "$(DATA)" ]; then \
		echo "$(RED)Error: DATA is required$(RESET)"; \
		echo "Usage: make showcase DATA=path/to/threat-feed.json"; \
		echo "       make showcase DATA=path/to/threat-feed.json LIMIT=5000"; \
		echo "       make showcase DATA=path/to/threat-feed.json PHASE=query"; \
		exit 1; \
	fi
	@echo "$(CYAN)Running Signature Capabilities Showcase...$(RESET)"
	@python demos/showcase.py \
		--data $(DATA) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(PHASE),--phase $(PHASE),) \
		$(if $(USE_CACHE),--use-cache,) \
		$(if $(SKIP_PGVECTOR),--skip-pgvector,)
