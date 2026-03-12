//! Duplicate detection using LSH bands.
//!
//! This example demonstrates:
//! - Batch signature generation for multiple vectors
//! - Using LSH bands for efficient candidate generation
//! - Finding near-duplicates in a collection
//!
//! Run with: cargo run --example duplicate_detection

use odin_prompt_toolkit::{
    cosine_from_hamming, hamming_distance_hex, normalize_vector, simhash_lsh_multi, LshConfig,
};
use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Duplicate Detection with LSH ===\n");

    // Example vectors representing different documents
    // Vectors 0, 1, 2 are similar (duplicates)
    // Vectors 3, 4 are different
    let vectors = vec![
        vec![1.0, 1.0, 1.0, 1.0],     // Doc 0
        vec![1.0, 0.95, 1.05, 1.0],   // Doc 1 (near-duplicate of 0)
        vec![0.98, 1.02, 1.0, 1.01],  // Doc 2 (near-duplicate of 0, 1)
        vec![0.0, 1.0, 0.0, 1.0],     // Doc 3 (different)
        vec![-1.0, -1.0, -1.0, -1.0], // Doc 4 (opposite, different)
    ];

    println!("Processing {} documents...\n", vectors.len());

    // Normalize and generate signatures
    let config = LshConfig::default();
    let signatures: Vec<_> = vectors
        .iter()
        .map(|v| {
            let normalized = normalize_vector(v);
            simhash_lsh_multi(&normalized, &config)
        })
        .collect();

    // Build band index for candidate generation
    // Map: (band_index, band_value) -> [doc_ids]
    let mut band_index: HashMap<(usize, String), Vec<usize>> = HashMap::new();

    for (doc_id, sig) in signatures.iter().enumerate() {
        // Use first family only for this example
        let family = &sig[0];

        for (band_idx, band_value) in family.bands.iter().enumerate() {
            band_index
                .entry((band_idx, band_value.clone()))
                .or_default()
                .push(doc_id);
        }
    }

    // Find candidate pairs (documents that share at least one band)
    let mut candidates: HashSet<(usize, usize)> = HashSet::new();

    for docs in band_index.values() {
        if docs.len() > 1 {
            // Multiple documents match this band
            for i in 0..docs.len() {
                for j in i + 1..docs.len() {
                    let pair = if docs[i] < docs[j] {
                        (docs[i], docs[j])
                    } else {
                        (docs[j], docs[i])
                    };
                    candidates.insert(pair);
                }
            }
        }
    }

    println!(
        "Found {} candidate pairs from band matching\n",
        candidates.len()
    );

    // Verify candidates with full Hamming distance
    let threshold = 0.85; // Cosine similarity threshold for duplicates
    let mut duplicates = Vec::new();

    for (id1, id2) in candidates {
        let sig1 = &signatures[id1][0].signature;
        let sig2 = &signatures[id2][0].signature;

        let hamming = hamming_distance_hex(sig1, sig2);
        let similarity = cosine_from_hamming(hamming, 256);

        if similarity >= threshold {
            duplicates.push((id1, id2, similarity));
        }
    }

    // Sort by similarity (descending)
    duplicates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("Detected duplicates (similarity >= {}):\n", threshold);
    for (id1, id2, sim) in duplicates {
        println!("  Doc {} <-> Doc {}: {:.4}", id1, id2, sim);
    }

    println!("\n✓ Duplicate detection complete!");
    println!("\nKey insight:");
    println!("  - Band matching reduces comparisons from O(n²) to O(n)");
    println!("  - Only candidate pairs need full Hamming distance computation");
    println!("  - Tune bands/bits ratio for precision/recall tradeoff");
}
