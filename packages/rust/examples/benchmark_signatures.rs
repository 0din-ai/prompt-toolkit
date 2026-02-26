//! Signature generation benchmark
//!
//! Measures the throughput of LSH signature generation in the Rust SDK.
//! This demonstrates the performance difference between pure-Python and
//! native implementations for the signature capabilities showcase (0DIN-1029).
//!
//! Usage:
//!   cargo run --release --example benchmark_signatures -- --count 10000
//!
//! Example output:
//!   Generating 10,000 signatures (384-dim random vectors)...
//!   Time: 1.234s
//!   Throughput: 8,100 signatures/sec
//!   Per-signature: 0.123ms

use odin_sig::{simhash_lsh_multi, LshConfig};
use rand::Rng;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let count = if args.len() > 2 && args[1] == "--count" {
        args[2].parse::<usize>().unwrap_or(10_000)
    } else {
        10_000
    };

    println!(
        "Generating {} signatures (384-dim random vectors)...",
        count
    );

    // Generate random normalized vectors (simulating embeddings)
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..384).map(|_| rng.gen_range(-1.0..1.0)).collect();
            // Normalize
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        })
        .collect();

    // Benchmark signature generation
    let config = LshConfig::default();
    let start = Instant::now();

    for vec in &vectors {
        let _ = simhash_lsh_multi(vec, &config);
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let throughput = count as f64 / elapsed_secs;
    let per_sig_ms = elapsed_secs * 1000.0 / count as f64;

    println!("Time: {:.3}s", elapsed_secs);
    println!("Throughput: {:.0} signatures/sec", throughput);
    println!("Per-signature: {:.3}ms", per_sig_ms);
    println!();
    println!(
        "Note: Python SDK achieves ~9 sigs/sec on the same hardware ({:.0}× slower)",
        throughput / 9.0
    );
}
