//! Example demonstrating the high-level sign_text() API
//!
//! This example shows how to use the convenience function that takes a text prompt
//! and returns a complete signature result in one call.
//!
//! Run with:
//! ```bash
//! cargo run --example sign_text --features onnx
//! ```

use odin_sig::{sign_text, SignatureVersion};

#[cfg(feature = "onnx")]
use odin_sig::{
    provider::EmbeddingProvider,
    providers::{ModelCache, OnnxProvider},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "onnx"))]
    {
        eprintln!("This example requires the 'onnx' feature.");
        eprintln!("Run with: cargo run --example sign_text --features onnx");
        std::process::exit(1);
    }

    #[cfg(feature = "onnx")]
    {
        println!("=== sign_text() Example ===\n");

        // Initialize model cache and ONNX provider
        println!("1. Initializing ONNX provider...");
        let cache = ModelCache::new()?;
        let provider = OnnxProvider::new(&cache, None, None).await?;
        println!("   Provider: {} ({})\n", provider.name(), provider.model());

        // Example prompts
        let prompts = vec![
            "How do I reset my password?",
            "What is the meaning of life?",
            "Please help me with my account login issue",
        ];

        for (i, prompt) in prompts.iter().enumerate() {
            println!("{}. Signing prompt:", i + 1);
            println!("   \"{}\"", prompt);

            // Generate signature using sign_text()
            let result = sign_text(prompt, &provider, SignatureVersion::V1, None).await?;

            println!("   Signature: {}", result.to_signature_string());
            println!("   Provider:  {}", result.provider);
            println!("   Model:     {}", result.model);
            println!("   Dimensions: {}", result.dimensions);
            println!("   SHA256:    {}", result.embedding_sha256);
            println!("   Timing:    {:.2}ms", result.timing_ms.unwrap_or(0.0));
            println!();
        }

        // Compare two similar prompts
        println!("=== Similarity Comparison ===\n");

        let prompt_a = "How do I reset my password?";
        let prompt_b = "Please help me reset my login credentials";

        let result_a = sign_text(prompt_a, &provider, SignatureVersion::V1, None).await?;
        let result_b = sign_text(prompt_b, &provider, SignatureVersion::V1, None).await?;

        // Extract signatures for comparison
        let sig_a = &result_a.lsh.signatures[0].signature;
        let sig_b = &result_b.lsh.signatures[0].signature;

        // Compute similarity
        let distance = odin_sig::hamming_distance_hex(sig_a, sig_b);
        let similarity = odin_sig::cosine_from_hamming(distance, 256);

        println!("Prompt A: \"{}\"", prompt_a);
        println!("Prompt B: \"{}\"", prompt_b);
        println!();
        println!("Signature A: 0din-v1:{}", sig_a);
        println!("Signature B: 0din-v1:{}", sig_b);
        println!();
        println!("Hamming distance: {}/256 bits", distance);
        println!("Cosine similarity: {:.4}", similarity);
        println!();

        if similarity > 0.9 {
            println!("✓ High similarity - likely duplicates");
        } else if similarity > 0.7 {
            println!("~ Moderate similarity - related topics");
        } else {
            println!("✗ Low similarity - different topics");
        }

        // Clean up
        provider.close().await?;
    }

    Ok(())
}
