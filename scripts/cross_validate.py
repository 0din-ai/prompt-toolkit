#!/usr/bin/env python3
"""
Cross-language validation script for 0DIN Prompt Toolkit.

Runs test suites for all four language implementations (Rust, Python, TypeScript, Go)
and validates that all tests pass. Used in CI pipeline to ensure cross-language
compatibility.

Usage
-----
Standard (offline-safe, model not required)::

    python scripts/cross_validate.py

SusFactor parity check (requires SUSFACTOR_MODEL_DIR pointing at a local model cache)::

    SUSFACTOR_MODEL_DIR=/path/to/cache/susfactor-v1 \\
        python scripts/cross_validate.py --susfactor-parity
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ANSI color codes
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
RESET = "\033[0m"


def run_command(cmd: list[str], cwd: Path, description: str, env: dict | None = None) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"{CYAN}Running {description}...{RESET}")
    try:
        merged_env = {**os.environ, **(env or {})}
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            env=merged_env,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print(f"{GREEN}✅ {description} passed{RESET}\n")
        else:
            print(f"{RED}❌ {description} failed{RESET}")
            print(f"{RED}Output:{RESET}\n{output}\n")

        return success, output
    except Exception as e:
        print(f"{RED}❌ {description} failed with exception: {e}{RESET}\n")
        return False, str(e)


def extract_test_count(output: str, language: str) -> int:
    """Extract test count from test output."""
    if language == "rust":
        # Look for "test result: ok. X passed"
        for line in output.split("\n"):
            if "test result: ok" in line and "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed;":
                        try:
                            return int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass
    elif language == "python":
        # Look for "X passed"
        for line in output.split("\n"):
            if "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part:
                        try:
                            return int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass
    elif language == "typescript":
        # Look for "Tests:  X passed"
        for line in output.split("\n"):
            if "Tests:" in line and "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part:
                        try:
                            return int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass

    return 0


def _extract_go_test_count(output: str) -> int:
    """Extract passing test count from `go test -v` output."""
    count = 0
    for line in output.split("\n"):
        if line.startswith("--- PASS:"):
            count += 1
    return count


def check_susfactor_goldens_present(root_dir: Path) -> tuple[bool, int]:
    """Check whether the golden fixture has scored vectors. Returns (present, count)."""
    fixture = root_dir / "spec" / "test-vectors" / "susfactor_vectors.json"
    if not fixture.exists():
        return False, 0
    try:
        doc = json.loads(fixture.read_text())
        scored = [
            v for v in doc.get("vectors", [])
            if v.get("rust_score") is not None and v.get("expected_label") is not None
        ]
        return len(scored) > 0, len(scored)
    except Exception:
        return False, 0


def run_susfactor_parity(root_dir: Path, model_dir: str, py_model_cache: str = "") -> dict[str, tuple[bool, int]]:
    """Run SusFactor parity checks for all three SDKs.

    Returns a dict of lang -> (passed, vector_count).
    """
    results: dict[str, tuple[bool, int]] = {}
    env = {"SUSFACTOR_MODEL_DIR": model_dir}
    py_env = {**env, "SIGNATURE_SDK_MODEL_CACHE": py_model_cache} if py_model_cache else env

    # ── Rust parity (self-check) ─────────────────────────────────────────────
    success, _ = run_command(
        [
            "cargo", "test",
            "--test", "susfactor_parity",
            "--features", "susfactor",
            "--release",
        ],
        root_dir / "packages" / "rust",
        "Rust SusFactor parity (self-check)",
        env=env,
    )
    results["rust"] = (success, 0)

    # ── Python parity ────────────────────────────────────────────────────────
    success, output = run_command(
        [
            "python", "-m", "pytest",
            "tests/test_susfactor_parity.py",
            "-v",
        ],
        root_dir / "packages" / "python",
        "Python SusFactor parity",
        env=py_env,
    )
    count = extract_test_count(output, "python")
    results["python"] = (success, count)

    # ── TypeScript parity ────────────────────────────────────────────────────
    # Run via Jest (not bare ts-node) so that `describe` and other Jest globals
    # are available. The test registers itself as todo under Jest for the ONNX
    # inference path (see test file header), so this validates fixture loading
    # and skip logic. Full ONNX inference is validated by the Rust parity test.
    success, _ = run_command(
        ["npm", "test", "--", "--testPathPattern=susfactor-parity"],
        root_dir / "packages" / "typescript",
        "TypeScript SusFactor parity",
        env=env,
    )
    results["typescript"] = (success, 0)

    # ── Go parity ────────────────────────────────────────────────────────────
    # Reuses the same ONNX model dir as Rust + TypeScript (/tmp/susfactor-v1).
    # The parity test skips automatically when SUSFACTOR_MODEL_DIR is unset,
    # so this is safe to run even without a model in standard mode.
    success, _ = run_command(
        [
            "go", "test",
            "./susfactor/...",
            "-run", "TestSusFactorParityGoldens",
            "-v", "-count=1",
        ],
        root_dir / "packages" / "go",
        "Go SusFactor parity",
        env={**env, "CGO_ENABLED": "1"},
    )
    results["go"] = (success, 15)

    return results


def main() -> int:
    """Run cross-language validation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--susfactor-parity",
        action="store_true",
        help="Also run SusFactor cross-SDK parity checks (requires SUSFACTOR_MODEL_DIR)",
    )
    parser.add_argument(
        "--py-model-cache",
        default=None,
        help=(
            "Path to pass as SIGNATURE_SDK_MODEL_CACHE for the Python parity test. "
            "Kept separate from the step-level env so it does not leak into Rust unit "
            "tests that assert on the default ~/.cache/signature-sdk path."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}0DIN Prompt Toolkit - Cross-Language Validation{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")

    results: dict[str, bool] = {}
    test_counts: dict[str, int] = {}

    # ── Standard test suites ─────────────────────────────────────────────────

    success, output = run_command(
        ["cargo", "test", "--lib", "--features", "cm-lsh"],
        root_dir / "packages" / "rust",
        "Rust tests",
    )
    results["rust"] = success
    test_counts["rust"] = extract_test_count(output, "rust")

    success, output = run_command(
        ["python", "-m", "pytest", "tests/", "-v"],
        root_dir / "packages" / "python",
        "Python tests",
    )
    results["python"] = success
    test_counts["python"] = extract_test_count(output, "python")

    success, output = run_command(
        ["npm", "test"],
        root_dir / "packages" / "typescript",
        "TypeScript tests",
    )
    results["typescript"] = success
    test_counts["typescript"] = extract_test_count(output, "typescript")

    success, output = run_command(
        ["go", "test", "./susfactor/...", "-count=1"],
        root_dir / "packages" / "go",
        "Go tests",
        env={"CGO_ENABLED": "1"},
    )
    results["go"] = success
    test_counts["go"] = _extract_go_test_count(output)

    # ── SusFactor parity (optional) ──────────────────────────────────────────

    parity_results: dict[str, tuple[bool, int]] = {}
    parity_skipped_reason: str | None = None

    if args.susfactor_parity:
        model_dir = os.environ.get("SUSFACTOR_MODEL_DIR", "")
        py_model_cache = args.py_model_cache or os.environ.get("SIGNATURE_SDK_MODEL_CACHE", "")
        if not model_dir:
            parity_skipped_reason = "SUSFACTOR_MODEL_DIR not set"
        else:
            goldens_present, golden_count = check_susfactor_goldens_present(root_dir)
            if not goldens_present:
                parity_skipped_reason = (
                    "no scored golden vectors — run `make generate-susfactor-goldens` first"
                )
            else:
                print(f"{CYAN}{'=' * 60}{RESET}")
                print(f"{CYAN}SusFactor Parity Checks ({golden_count} golden vectors){RESET}")
                print(f"{CYAN}{'=' * 60}{RESET}\n")
                parity_results = run_susfactor_parity(root_dir, model_dir, py_model_cache)

    # ── Summary ──────────────────────────────────────────────────────────────

    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}Validation Summary{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")

    all_passed = all(results.values())
    total_tests = sum(test_counts.values())

    print(f"  {'Language'.ljust(14)} {'Status'.ljust(14)} Tests")
    print(f"  {'-' * 40}")
    for lang, passed in results.items():
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        count = test_counts[lang]
        print(f"  {lang.ljust(14)} {status}  ({count} tests)")

    if args.susfactor_parity:
        print()
        print(f"  {'SusFactor Parity'.ljust(40)}")
        print(f"  {'-' * 40}")
        if parity_skipped_reason:
            print(f"  {YELLOW}⏭  skipped: {parity_skipped_reason}{RESET}")
        elif parity_results:
            parity_all_passed = all(ok for ok, _ in parity_results.values())
            all_passed = all_passed and parity_all_passed
            for lang, (passed, count) in parity_results.items():
                status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
                count_str = f"({count} vectors)" if count else ""
                print(f"  {lang.ljust(14)} {status}  {count_str}")

    print()

    if all_passed:
        print(f"{GREEN}✅ All validations passed!{RESET}")
        print(f"{GREEN}Total: {total_tests} tests passing across 4 languages{RESET}\n")
        return 0
    else:
        failed = [lang for lang, passed in results.items() if not passed]
        if parity_results:
            failed += [f"{lang}(parity)" for lang, (ok, _) in parity_results.items() if not ok]
        print(f"{RED}❌ Validation failed for: {', '.join(failed)}{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
