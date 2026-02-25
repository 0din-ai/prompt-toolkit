//! Basic LSH signature generation example.
//!
//! This example demonstrates:
//! - Generating an LSH signature from a normalized vector
//! - Default configuration (3 families, 256 bits, 16 bands)
//! - Formatting and parsing signature strings
//!
//! Run with: cargo run --example basic_signature

use odin_sig::{simhash_lsh_multi, LshConfig};

fn main() {
    println!("=== Basic LSH Signature Generation ===\n");

    // Example normalized vector (4 dimensions for clarity)
    // In practice, this would come from an embedding model (384 or 1536 dims)
    let normalized_vector = vec![0.5, 0.5, 0.5, 0.5];

    println!("Input vector: {:?}", normalized_vector);
    println!("Vector dimensions: {}\n", normalized_vector.len());

    // Generate LSH signatures with default configuration
    let config = LshConfig::default();
    println!("Configuration:");
    println!("  Families: {}", config.families);
    println!("  Bits per signature: {}", config.bits);
    println!("  Bands: {}\n", config.bands);

    let families = simhash_lsh_multi(&normalized_vector, &config);

    // Display results for each family
    for family in &families {
        println!("Family {}:", family.family);
        println!("  Signature (hex): {}", family.signature);
        println!(
            "  Signature length: {} hex chars = {} bits",
            family.signature.len(),
            family.bits
        );
        println!("  Number of bands: {}", family.bands.len());
        println!(
            "  Band 0: {} (first {} hex chars)",
            family.bands[0],
            family.bands[0].len()
        );
        println!();
    }

    // Format as 0din signature string (V1 format)
    let primary_sig = &families[0].signature;
    let signature_string = format!("0din-v1:{}", primary_sig);

    println!("Formatted signature string:");
    println!("  {}", signature_string);
    println!();

    // In a real application, you would:
    // 1. Store this signature in a database with the original text
    // 2. Use bands for efficient similarity search (LSH indexing)
    // 3. Compare signatures using hamming distance

    println!("✓ Signature generation complete!");
}
