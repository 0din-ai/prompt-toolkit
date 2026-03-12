//! Generate test vectors from the canonical Rust implementation.
//!
//! This program generates JSON test vectors that all three language implementations
//! (Rust, Python, TypeScript) must pass for cross-language validation.
//!
//! Run with: cargo run --example generate_vectors

use odin_prompt_toolkit::{
    compute_embedding_sha256, cosine_from_hamming, hamming_distance_hex, normalize_vector,
    simhash_lsh_multi, LshConfig,
};
use serde_json::json;
use std::fs;

// SplitMix64 hash function for test vector generation
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// Get deterministic +1/-1 sign from (family, bit, dim)
fn sign_for(family: usize, bit: usize, dim: usize) -> i8 {
    let seed = ((family as u64) << 48) ^ ((bit as u64) << 24) ^ (dim as u64);
    let h = splitmix64(seed);
    if (h & 1) == 1 {
        1
    } else {
        -1
    }
}

fn main() {
    println!("Generating test vectors from canonical Rust implementation...\n");

    // 1. SplitMix64 test vectors
    generate_splitmix64_vectors();

    // 2. sign_for test vectors
    generate_sign_for_vectors();

    // 3. SimHash LSH test vectors
    generate_simhash_vectors();

    // 4. Hamming distance test vectors
    generate_hamming_vectors();

    // 5. Cosine from Hamming test vectors
    generate_cosine_vectors();

    // 6. SHA256 test vectors
    generate_sha256_vectors();

    // 7. Signature string format test vectors
    generate_signature_format_vectors();

    println!("\n✅ All test vectors generated successfully!");
    println!("📁 Output directory: ../../spec/test-vectors/");
}

fn generate_splitmix64_vectors() {
    println!("Generating SplitMix64 test vectors...");

    let test_cases = vec![
        0u64,
        1u64,
        42u64,
        12345u64,
        0x9E3779B97F4A7C15u64, // Golden ratio constant
        u64::MAX,              // Maximum value
        u64::MAX / 2,          // Mid-range
    ];

    let vectors: Vec<_> = test_cases
        .iter()
        .map(|&input| {
            let output = splitmix64(input);
            json!({
                "input": input,
                "output": format!("{:016X}", output)
            })
        })
        .collect();

    let output = json!({
        "description": "SplitMix64 PRNG test vectors",
        "algorithm": "splitmix64(x) with constants 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB",
        "vectors": vectors
    });

    fs::write(
        "../../spec/test-vectors/splitmix64.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write splitmix64.json");

    println!("  ✓ Generated {} test cases", test_cases.len());
}

fn generate_sign_for_vectors() {
    println!("Generating sign_for test vectors...");

    let mut vectors = vec![];

    // Test various combinations of (family, bit, dim)
    for family in 0..3 {
        for bit in [0, 1, 127, 255] {
            for dim in [0, 1, 10, 100, 383, 1535] {
                vectors.push(json!({
                    "family": family,
                    "bit": bit,
                    "dim": dim,
                    "sign": sign_for(family, bit, dim)
                }));
            }
        }
    }

    let output = json!({
        "description": "sign_for(family, bit, dim) test vectors",
        "algorithm": "seed = (family << 48) XOR (bit << 24) XOR dim; sign = splitmix64(seed) & 1 ? +1 : -1",
        "vectors": vectors
    });

    fs::write(
        "../../spec/test-vectors/sign_for.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write sign_for.json");

    println!("  ✓ Generated {} test cases", vectors.len());
}

fn generate_simhash_vectors() {
    println!("Generating SimHash LSH test vectors...");

    let test_cases = vec![
        (
            "unit_4d",
            vec![0.5, 0.5, 0.5, 0.5],
            "4-dimensional unit vector",
        ),
        (
            "unit_10d",
            vec![
                0.31622776, 0.31622776, 0.31622776, 0.31622776, 0.31622776, 0.31622776, 0.31622776,
                0.31622776, 0.31622776, 0.31622776,
            ],
            "10-dimensional unit vector",
        ),
        (
            "zeros_and_ones",
            normalize_vector(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            "Alternating 1s and 0s (normalized)",
        ),
        (
            "negative_values",
            normalize_vector(&[1.0, -1.0, 1.0, -1.0]),
            "Positive and negative values",
        ),
        (
            "small_384d",
            (0..384)
                .map(|i| (i as f32 / 384.0) * 2.0 - 1.0)
                .collect::<Vec<_>>(),
            "384-dimensional vector (typical for V1)",
        ),
    ];

    let config = LshConfig {
        families: 3,
        bits: 256,
        bands: 16,
    };

    let mut vectors = vec![];

    for (name, input_vector, description) in test_cases {
        // Normalize if not already
        let normalized = if input_vector.iter().map(|x| x * x).sum::<f32>().sqrt() > 0.999
            && input_vector.iter().map(|x| x * x).sum::<f32>().sqrt() < 1.001
        {
            input_vector
        } else {
            normalize_vector(&input_vector)
        };

        let families = simhash_lsh_multi(&normalized, &config);

        let families_json: Vec<_> = families
            .iter()
            .map(|f| {
                json!({
                    "family": f.family,
                    "bits": f.bits,
                    "signature": f.signature,
                    "bands": f.bands
                })
            })
            .collect();

        vectors.push(json!({
            "name": name,
            "description": description,
            "input": normalized,
            "config": {
                "families": config.families,
                "bits": config.bits,
                "bands": config.bands
            },
            "expected": families_json
        }));
    }

    let output = json!({
        "description": "SimHash LSH test vectors",
        "algorithm": "Random hyperplane LSH with deterministic SplitMix64-based hyperplanes",
        "vectors": vectors
    });

    fs::write(
        "../../spec/test-vectors/simhash.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write simhash.json");

    println!("  ✓ Generated {} test cases", vectors.len());
}

fn generate_hamming_vectors() {
    println!("Generating Hamming distance test vectors...");

    let test_cases = vec![
        ("0000", "ffff", 16, "All bits different"),
        ("0000", "0000", 0, "Identical"),
        ("0f0f", "f0f0", 16, "Alternating nibbles"),
        ("abcd", "abcd", 0, "Identical complex"),
        (
            "a",
            "b",
            1,
            "Single char difference (a=1010, b=1011, 1 bit diff)",
        ),
        ("abc", "def", 7, "Different chars (3 nibbles, 7 bits diff)"),
        ("00", "01", 1, "1 bit different"),
        ("ff", "7f", 1, "1 bit different (high)"),
        (
            "0123456789abcdef",
            "fedcba9876543210",
            64,
            "Reversed hex (16 hex chars = 64 bits total)",
        ),
        (
            "aaaaaaaaaaaaaaaa",
            "5555555555555555",
            64,
            "All 64 bits different (a=1010, 5=0101)",
        ),
    ];

    let vectors: Vec<_> = test_cases
        .iter()
        .map(|(a, b, expected, description)| {
            let actual = hamming_distance_hex(a, b);
            assert_eq!(
                actual, *expected,
                "Hamming distance mismatch for {} vs {}: expected {}, got {}",
                a, b, expected, actual
            );
            json!({
                "a": a,
                "b": b,
                "distance": expected,
                "description": description
            })
        })
        .collect();

    let output = json!({
        "description": "Hamming distance test vectors for hex-encoded signatures",
        "algorithm": "XOR nibbles, popcount each, sum",
        "vectors": vectors
    });

    fs::write(
        "../../spec/test-vectors/hamming.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write hamming.json");

    println!("  ✓ Generated {} test cases", test_cases.len());
}

fn generate_cosine_vectors() {
    println!("Generating cosine from Hamming test vectors...");

    let test_cases = vec![
        (0, 256, "Identical signatures (cosine = 1.0)"),
        (128, 256, "Half bits different (cosine = 0.0)"),
        (256, 256, "Opposite signatures (cosine = -1.0)"),
        (64, 256, "Quarter bits different (cosine ≈ 0.707)"),
        (192, 256, "Three quarters different (cosine ≈ -0.707)"),
        (32, 256, "12.5% different"),
        (16, 256, "6.25% different"),
        (1, 256, "Single bit different"),
    ];

    let vectors: Vec<_> = test_cases
        .iter()
        .map(|(distance, total_bits, description)| {
            let cosine = cosine_from_hamming(*distance, *total_bits);
            json!({
                "distance": distance,
                "total_bits": total_bits,
                "cosine_similarity": cosine,
                "description": description
            })
        })
        .collect();

    let output = json!({
        "description": "Cosine similarity estimation from Hamming distance",
        "algorithm": "cos(PI * distance / total_bits)",
        "vectors": vectors
    });

    fs::write(
        "../../spec/test-vectors/cosine.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write cosine.json");

    println!("  ✓ Generated {} test cases", test_cases.len());
}

fn generate_sha256_vectors() {
    println!("Generating SHA256 canonical format test vectors...");

    let test_cases = vec![
        (vec![0.1, 0.2, 0.3], "Simple 3-element vector"),
        (
            vec![0.123456789, 0.987654321, -0.111111111],
            "High precision (6-decimal quantization)",
        ),
        (
            vec![0.123456001, 0.999999999],
            "Sub-6-decimal jitter (should match next)",
        ),
        (
            vec![0.123456002, 0.999999998],
            "Sub-6-decimal jitter variant",
        ),
        (vec![1.0, 2.0, 3.0], "Whole numbers (must include .0)"),
        (vec![0.0, -0.0, 0.5], "Zero and negative zero"),
        (
            (0..10).map(|i| i as f32 / 10.0).collect(),
            "Ten evenly-spaced values",
        ),
    ];

    let mut vectors = vec![];

    for (input, description) in test_cases {
        let hash = compute_embedding_sha256(&input);

        // Compute expected JSON format
        let quantized: Vec<f64> = input
            .iter()
            .map(|&x| {
                let x64 = x as f64;
                (x64 * 1_000_000.0).round() / 1_000_000.0
            })
            .collect();

        let json_parts: Vec<String> = quantized
            .iter()
            .map(|&x| {
                let s = format!("{}", x);
                if x.fract() == 0.0 && !s.contains('.') {
                    format!("{}.0", s)
                } else {
                    s
                }
            })
            .collect();
        let expected_json = format!("[{}]", json_parts.join(", "));

        vectors.push(json!({
            "input": input,
            "expected_json": expected_json,
            "expected_sha256": hash,
            "description": description
        }));
    }

    let output = json!({
        "description": "Canonical SHA256 hash of normalized embeddings",
        "algorithm": "Quantize to 6 decimals, format as JSON array with space after comma, SHA256 hash",
        "note": "Whole numbers must include .0 (e.g., [1.0, 2.0])",
        "vectors": vectors
    });

    fs::write(
        "../../spec/test-vectors/sha256.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write sha256.json");

    println!("  ✓ Generated {} test cases", vectors.len());
}

fn generate_signature_format_vectors() {
    println!("Generating signature format test vectors...");

    let test_cases = vec![
        (
            "0din-v0:deadbeef12345678",
            Some(("v0", "deadbeef12345678")),
            "Valid V0 signature",
        ),
        (
            "0din-v1:cafebabe87654321",
            Some(("v1", "cafebabe87654321")),
            "Valid V1 signature",
        ),
        (
            "0din-v0:a3f9c2e1b8d4f7a2c5e8b1d3f6a9c2e5b8d1f4a7c2e5b8d1f4a7c2e5b8d1f4a7c2",
            Some((
                "v0",
                "a3f9c2e1b8d4f7a2c5e8b1d3f6a9c2e5b8d1f4a7c2e5b8d1f4a7c2e5b8d1f4a7c2",
            )),
            "Valid V0 signature (full 256-bit)",
        ),
        ("invalid:foo", None, "Invalid prefix (should be 0din-)"),
        ("0din-v99:foo", None, "Unsupported version"),
        ("0din-v0", None, "Missing signature component"),
        (
            "0din-v0:foo:bar",
            None,
            "Too many colons (V0 should have one colon)",
        ),
    ];

    let mut vectors = vec![];
    let num_cases = test_cases.len();

    for (input, expected, description) in test_cases {
        if let Some((version, signature)) = expected {
            vectors.push(json!({
                "input": input,
                "valid": true,
                "expected_version": version,
                "expected_signature": signature,
                "description": description
            }));
        } else {
            vectors.push(json!({
                "input": input,
                "valid": false,
                "error": true,
                "description": description
            }));
        }
    }

    let output = json!({
        "description": "Signature string format parsing test vectors",
        "format": "0din-v{N}:<hex_signature>",
        "vectors": vectors
    });

    fs::write(
        "../../spec/test-vectors/signature_format.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write signature_format.json");

    println!("  ✓ Generated {} test cases", num_cases);
}
