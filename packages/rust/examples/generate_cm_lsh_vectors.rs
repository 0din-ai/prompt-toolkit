//! Generate CM-LSH test vectors from the canonical Rust implementation.
//!
//! This program generates JSON test vectors for Confidence Matrix LSH that all
//! three language implementations (Rust, Python, TypeScript) must pass.
//!
//! Run with: cargo run --example generate_cm_lsh_vectors --features cm-lsh

#[cfg(feature = "cm-lsh")]
use signature_sdk::create_default_cm_lsh;
use serde_json::json;
use std::fs;

#[cfg(feature = "cm-lsh")]
fn main() {
    println!("Generating CM-LSH test vectors from canonical Rust implementation...\n");

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
            "small_384d",
            (0..384)
                .map(|i| (i as f32 / 384.0) * 2.0 - 1.0)
                .collect::<Vec<_>>(),
            "384-dimensional vector (typical for V1)",
        ),
        (
            "alternating",
            (0..100)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect::<Vec<_>>(),
            "100-dimensional alternating +1/-1",
        ),
        (
            "random_seed_42",
            generate_pseudo_random(256, 42),
            "256-dimensional pseudo-random (seed=42)",
        ),
    ];

    let mut vectors = vec![];

    for (name, embedding, description) in test_cases {
        // Create default CM-LSH hasher
        let cm_lsh = create_default_cm_lsh(embedding.len(), 0);

        // Generate hash
        let hash = cm_lsh.hash(&embedding);

        // Test similarity with itself (should be ~1.0)
        let self_sim = cm_lsh.sim(&hash, &hash);

        // Test similarity with slightly perturbed version
        let perturbed: Vec<f32> = embedding.iter().map(|x| x * 0.95).collect();
        let perturbed_hash = cm_lsh.hash(&perturbed);
        let perturbed_sim = cm_lsh.sim(&hash, &perturbed_hash);

        vectors.push(json!({
            "name": name,
            "description": description,
            "input": embedding,
            "expected": {
                "hash_a": hash.hash_a,
                "hash_b": hash.hash_b,
                "bits": hash.bits,
                "bands": hash.bands,
                "lsh_ts_compat": hash.lsh_ts_compat(),
                "self_similarity": self_sim,
                "perturbed_similarity": perturbed_sim
            }
        }));
    }

    // Test similarity computation between different vectors
    let similarity_tests = vec![
        (
            "identical",
            vec![0.5; 384],
            vec![0.5; 384],
            "Identical vectors should have similarity ~1.0",
        ),
        (
            "similar",
            vec![0.5; 384],
            (0..384).map(|i| 0.5 + (i as f32 * 0.001)).collect(),
            "Similar vectors should have high similarity",
        ),
        (
            "different",
            vec![0.5; 384],
            vec![-0.5; 384],
            "Opposite vectors should have low similarity",
        ),
    ];

    let mut sim_vectors = vec![];
    let cm_lsh = create_default_cm_lsh(384, 0);

    for (name, e1, e2, description) in similarity_tests {
        let h1 = cm_lsh.hash(&e1);
        let h2 = cm_lsh.hash(&e2);
        let similarity = cm_lsh.sim(&h1, &h2);

        sim_vectors.push(json!({
            "name": name,
            "description": description,
            "embedding1": e1,
            "embedding2": e2,
            "similarity": similarity
        }));
    }

    let output = json!({
        "description": "Confidence Matrix LSH (CM-LSH) test vectors",
        "algorithm": "Hybrid LSH-TS (256 bits) + ITQ (256 bits) with confidence matrix",
        "note": "CM-LSH always produces 512-bit signatures. First 256 bits are LSH-TS compatible.",
        "hash_vectors": vectors,
        "similarity_vectors": sim_vectors
    });

    fs::write(
        "../../spec/test-vectors/cm_lsh.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .expect("Failed to write cm_lsh.json");

    println!("✅ Generated {} hash test cases", vectors.len());
    println!("✅ Generated {} similarity test cases", sim_vectors.len());
    println!("📁 Output: ../../spec/test-vectors/cm_lsh.json");
}

#[cfg(not(feature = "cm-lsh"))]
fn main() {
    eprintln!("Error: This example requires the 'cm-lsh' feature");
    eprintln!("Run with: cargo run --example generate_cm_lsh_vectors --features cm-lsh");
    std::process::exit(1);
}

/// Generate pseudo-random values for testing (deterministic)
fn generate_pseudo_random(count: usize, seed: u64) -> Vec<f32> {
    let mut values = Vec::with_capacity(count);
    let mut state = seed;

    for _ in 0..count {
        state = splitmix64(state);
        // Convert to f32 in range [-1, 1]
        let norm = (state as f64) / (u64::MAX as f64);
        values.push((norm * 2.0 - 1.0) as f32);
    }

    values
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
