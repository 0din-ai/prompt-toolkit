//! Similarity comparison example.
//!
//! This example demonstrates:
//! - Comparing multiple vectors using LSH signatures
//! - Computing Hamming distance between signatures
//! - Estimating cosine similarity from Hamming distance
//!
//! Run with: cargo run --example similarity_comparison

use odin_sig::{
    cosine_from_hamming, hamming_distance_hex, normalize_vector, simhash_lsh_multi, LshConfig,
};

fn main() {
    println!("=== LSH Similarity Comparison ===\n");

    // Three example vectors (unnormalized)
    let vector_a = vec![1.0, 1.0, 1.0, 1.0]; // Original
    let vector_b = vec![1.0, 0.9, 1.1, 1.0]; // Similar (small perturbation)
    let vector_c = vec![-1.0, -1.0, -1.0, -1.0]; // Opposite direction

    println!("Input vectors:");
    println!("  A: {:?}", vector_a);
    println!("  B: {:?} (similar to A)", vector_b);
    println!("  C: {:?} (opposite to A)\n", vector_c);

    // Normalize all vectors
    let norm_a = normalize_vector(&vector_a);
    let norm_b = normalize_vector(&vector_b);
    let norm_c = normalize_vector(&vector_c);

    // Generate signatures
    let config = LshConfig::default();
    let sig_a = simhash_lsh_multi(&norm_a, &config);
    let sig_b = simhash_lsh_multi(&norm_b, &config);
    let sig_c = simhash_lsh_multi(&norm_c, &config);

    println!("Signatures (first family only):");
    println!("  A: {}", &sig_a[0].signature[..16]);
    println!("  B: {}", &sig_b[0].signature[..16]);
    println!("  C: {}", &sig_c[0].signature[..16]);
    println!("     (showing first 16 hex chars of 64)\n");

    // Compare all pairs
    println!("Pairwise comparisons:\n");

    compare_signatures("A vs B", &sig_a[0].signature, &sig_b[0].signature);
    compare_signatures("A vs C", &sig_a[0].signature, &sig_c[0].signature);
    compare_signatures("B vs C", &sig_b[0].signature, &sig_c[0].signature);

    println!("\n✓ Comparison complete!");
    println!("\nInterpretation:");
    println!("  - Similarity > 0.9: Very similar");
    println!("  - Similarity 0.7-0.9: Moderately similar");
    println!("  - Similarity < 0.5: Dissimilar");
}

fn compare_signatures(label: &str, sig1: &str, sig2: &str) {
    let hamming = hamming_distance_hex(sig1, sig2);
    let similarity = cosine_from_hamming(hamming, 256);

    println!("{}:", label);
    println!("  Hamming distance: {}/256 bits differ", hamming);
    println!("  Estimated cosine similarity: {:.4}", similarity);
    println!();
}
