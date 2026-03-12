#!/usr/bin/env python3
"""
Cross-language validation script for 0DIN Prompt Toolkit.

Runs test suites for all three language implementations (Rust, Python, TypeScript)
and validates that all tests pass. Used in CI pipeline to ensure cross-language
compatibility.
"""

import subprocess
import sys
from pathlib import Path

# ANSI color codes
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[0;33m"
RESET = "\033[0m"


def run_command(cmd: list[str], cwd: Path, description: str) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"{CYAN}Running {description}...{RESET}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
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
                # Format: "test result: ok. 43 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out"
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
                            # Format is usually "11 passed in 0.23s"
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


def main() -> int:
    """Run cross-language validation."""
    # Find project root
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}0DIN Prompt Toolkit - Cross-Language Validation{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")

    results = {}
    test_counts = {}

    # Run Rust tests
    success, output = run_command(
        ["cargo", "test", "--lib", "--features", "cm-lsh"],
        root_dir / "packages" / "rust",
        "Rust tests",
    )
    results["rust"] = success
    test_counts["rust"] = extract_test_count(output, "rust")

    # Run Python tests
    success, output = run_command(
        ["python", "-m", "pytest", "tests/", "-v"],
        root_dir / "packages" / "python",
        "Python tests",
    )
    results["python"] = success
    test_counts["python"] = extract_test_count(output, "python")

    # Run TypeScript tests
    success, output = run_command(
        ["npm", "test"],
        root_dir / "packages" / "typescript",
        "TypeScript tests",
    )
    results["typescript"] = success
    test_counts["typescript"] = extract_test_count(output, "typescript")

    # Print summary
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}Validation Summary{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}\n")

    all_passed = all(results.values())
    total_tests = sum(test_counts.values())

    for lang, passed in results.items():
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        count = test_counts[lang]
        print(f"  {lang.ljust(12)} {status}  ({count} tests)")

    print()

    if all_passed:
        print(f"{GREEN}✅ All validations passed!{RESET}")
        print(f"{GREEN}Total: {total_tests} tests passing across 3 languages{RESET}\n")
        return 0
    else:
        failed = [lang for lang, passed in results.items() if not passed]
        print(f"{RED}❌ Validation failed for: {', '.join(failed)}{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
