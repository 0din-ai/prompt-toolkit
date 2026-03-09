#!/usr/bin/env python3
"""
Post-installation verification script for signature-sdk deliverable.

This script runs a quick smoke test to verify:
1. signature_sdk package is importable
2. Native acceleration is available (if installed)
3. Basic signature generation works
4. Output matches expected format

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import sys
from typing import Optional

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def print_success(msg: str) -> None:
    """Print success message in green."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_error(msg: str) -> None:
    """Print error message in red."""
    print(f"{RED}✗{RESET} {msg}")


def print_warning(msg: str) -> None:
    """Print warning message in yellow."""
    print(f"{YELLOW}⚠{RESET} {msg}")


def verify_import() -> bool:
    """Verify that signature_sdk can be imported."""
    try:
        import signature_sdk

        print_success(
            f"signature_sdk imported successfully (v{signature_sdk.__version__})"
        )
        return True
    except ImportError as e:
        print_error(f"Failed to import signature_sdk: {e}")
        return False


def verify_native_acceleration() -> Optional[bool]:
    """Check if native acceleration is available."""
    try:
        from signature_sdk import NATIVE_AVAILABLE

        if NATIVE_AVAILABLE:
            print_success("Native Rust acceleration is ENABLED (653× faster)")
            return True
        else:
            print_warning("Native acceleration not available (using pure Python)")
            print_warning("  Install native wheels for better performance:")
            print_warning("  pip install signature-sdk-native")
            return False
    except ImportError:
        print_warning("Could not check native acceleration status")
        return None


def verify_signature_generation() -> bool:
    """Verify basic signature generation works."""
    try:
        from signature_sdk import simhash_lsh_multi, normalize_vector

        # Test vector: simple 3D vector
        test_vector = [1.0, 2.0, 3.0]
        normalized = normalize_vector(test_vector)

        # Generate signature with default config (3 families × 256 bits × 16 bands)
        families_result = simhash_lsh_multi(normalized, families=3, bits=256, bands=16)

        # Verify we got 3 families
        if len(families_result) != 3:
            print_error(f"Expected 3 families, got {len(families_result)}")
            return False

        # Verify each family has correct structure
        for i, family in enumerate(families_result):
            if len(family.bands) != 16:
                print_error(f"Family {i}: expected 16 bands, got {len(family.bands)}")
                return False

            # Verify signature is 64 hex chars (256 bits)
            if len(family.signature) != 64:
                print_error(
                    f"Family {i}: expected 64 hex chars, got {len(family.signature)}"
                )
                return False

        print_success("Signature generation works correctly")
        print(f"  Generated signature for test vector: {test_vector}")
        print(
            f"  First family signature (256 bits): {families_result[0].signature[:16]}..."
        )

        return True

    except Exception as e:
        print_error(f"Signature generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_signature_format() -> bool:
    """Verify signature string formatting and parsing."""
    try:
        from signature_sdk import (
            signature_string,
            parse_signature_string,
            SignatureVersion,
        )

        # Create a test signature string
        test_sig = "a" * 64
        version = SignatureVersion.V1
        sig_str = signature_string(version, test_sig)

        # Verify format
        if not sig_str.startswith("0din-v1:"):
            print_error(f"Invalid signature format: {sig_str}")
            return False

        # Verify parsing roundtrip
        parsed = parse_signature_string(sig_str)

        if parsed.version != version:
            print_error(f"Version mismatch: expected {version}, got {parsed.version}")
            return False

        if parsed.signature != test_sig:
            print_error(f"Signature mismatch after parsing")
            return False

        print_success("Signature format validation passed")
        return True

    except Exception as e:
        print_error(f"Signature format validation failed: {e}")
        return False


def verify_hamming_distance() -> bool:
    """Verify Hamming distance calculation."""
    try:
        from signature_sdk import hamming_distance_hex

        # Test known cases
        test_cases = [
            ("ff00", "00ff", 16),  # Completely different
            ("ffff", "ffff", 0),  # Identical
            ("f0f0", "0f0f", 16),  # Alternating
        ]

        for sig_a, sig_b, expected_distance in test_cases:
            distance = hamming_distance_hex(sig_a, sig_b)
            if distance != expected_distance:
                print_error(
                    f"Hamming distance mismatch: hamming_distance_hex('{sig_a}', '{sig_b}') "
                    f"= {distance}, expected {expected_distance}"
                )
                return False

        print_success("Hamming distance calculation works correctly")
        return True

    except Exception as e:
        print_error(f"Hamming distance calculation failed: {e}")
        return False


def main() -> int:
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("signature-sdk Installation Verification")
    print("=" * 60 + "\n")

    checks = [
        ("Package Import", verify_import),
        ("Signature Generation", verify_signature_generation),
        ("Signature Format", verify_signature_format),
        ("Hamming Distance", verify_hamming_distance),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            results.append(False)

    # Check native acceleration separately (not a failure if unavailable)
    print(f"\n[Native Acceleration]")
    verify_native_acceleration()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)

    if all(results):
        print_success(f"All {total} checks passed!")
        print(
            "\nInstallation verified successfully. You're ready to use signature-sdk!"
        )
        return 0
    else:
        print_error(f"{passed}/{total} checks passed")
        print("\nSome checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
