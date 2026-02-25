//! Confidence Matrix LSH (CM-LSH) example.
//!
//! This example demonstrates:
//! - Enhanced LSH with confidence matrix
//! - Dual hash structure (512-bit signature + 512-bit confidence)
//! - Backward compatibility with standard LSH (first 256 bits)
//! - Calibrated similarity estimation
//!
//! Run with: cargo run --example cm_lsh_example --features cm-lsh

#[cfg(feature = "cm-lsh")]
use odin_sig::{create_default_cm_lsh, normalize_vector};

#[cfg(not(feature = "cm-lsh"))]
fn main() {
    eprintln!("This example requires the 'cm-lsh' feature.");
    eprintln!("Run with: cargo run --example cm_lsh_example --features cm-lsh");
    std::process::exit(1);
}

#[cfg(feature = "cm-lsh")]
fn main() {
    println!("=== Confidence Matrix LSH (CM-LSH) ===\n");

    // Example vectors
    let vector_a = vec![1.0, 1.0, 1.0, 1.0];
    let vector_b = vec![1.0, 0.9, 1.1, 1.0]; // Similar to A
    let vector_c = vec![-1.0, -1.0, -1.0, -1.0]; // Opposite to A

    println!("Input vectors:");
    println!("  A: {:?}", vector_a);
    println!("  B: {:?} (similar to A)", vector_b);
    println!("  C: {:?} (opposite to A)\n", vector_c);

    // Normalize vectors
    let norm_a = normalize_vector(&vector_a);
    let norm_b = normalize_vector(&vector_b);
    let norm_c = normalize_vector(&vector_c);

    // Create CM-LSH hasher with default configuration
    // This uses identity ITQ (no learned rotation) for simplicity
    // Family 0 for deterministic results
    let hasher = create_default_cm_lsh(norm_a.len(), 0);

    println!("CM-LSH Configuration:");
    println!("  Total bits: 512 (256 LSH-TS + 256 ITQ)");
    println!("  First 256 bits: LSH-TS compatible");
    println!("  Confidence matrix: Alpha-weighted agreement\n");

    // Generate dual hashes
    let hash_a = hasher.hash(&norm_a);
    let hash_b = hasher.hash(&norm_b);
    let hash_c = hasher.hash(&norm_c);

    println!("Dual hashes (showing first 32 hex chars of 128):");
    println!(
        "  A: hash={} conf={}",
        &hash_a.hash_a[..32],
        &hash_a.hash_b[..32]
    );
    println!(
        "  B: hash={} conf={}",
        &hash_b.hash_a[..32],
        &hash_b.hash_b[..32]
    );
    println!(
        "  C: hash={} conf={}",
        &hash_c.hash_a[..32],
        &hash_c.hash_b[..32]
    );
    println!();

    // Demonstrate LSH-TS backward compatibility
    println!("LSH-TS compatibility (first 256 bits):");
    println!("  A: {}", &hash_a.lsh_ts_compat()[..16]);
    println!("  B: {}", &hash_b.lsh_ts_compat()[..16]);
    println!("  C: {}", &hash_c.lsh_ts_compat()[..16]);
    println!("     (showing first 16 hex chars of 64)\n");

    // Compute calibrated similarities
    println!("Calibrated similarities:\n");

    let sim_ab = hasher.sim(&hash_a, &hash_b);
    let sim_ac = hasher.sim(&hash_a, &hash_c);
    let sim_bc = hasher.sim(&hash_b, &hash_c);

    println!("  A vs B: {:.4}", sim_ab);
    println!("  A vs C: {:.4}", sim_ac);
    println!("  B vs C: {:.4}", sim_bc);

    println!("\n✓ CM-LSH example complete!");
    println!("\nKey advantages of CM-LSH:");
    println!("  - Confidence matrix weights reliable bits higher");
    println!("  - Isotonic calibration improves similarity estimates");
    println!("  - Dual hash (LSH + ITQ) for better quantization");
    println!("  - Backward compatible with standard LSH");
}
