//! # signature-sdk
//!
//! Multi-language SDK for LSH (Locality-Sensitive Hashing) signature generation
//! for AI prompt similarity detection.
//!
//! This crate provides the canonical Rust implementation of the signature-sdk algorithm,
//! which is also available in Python and TypeScript.
//!
//! ## Quick Start
//!
//! ### High-Level API (Recommended)
//!
//! The easiest way to generate signatures is using `sign_text()`:
//!
//! ```rust,no_run
//! # #[cfg(feature = "onnx")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use odin_prompt_toolkit::{sign_text, SignatureVersion};
//! use odin_prompt_toolkit::providers::{ModelCache, OnnxProvider};
//!
//! // Initialize ONNX provider (local, no API key needed)
//! let cache = ModelCache::new()?;
//! let provider = OnnxProvider::new(&cache, None, None, 0, 0).await?;
//!
//! // Generate signature from text
//! let result = sign_text(
//!     "How do I reset my password?",
//!     SignatureVersion::V1,
//!     Some(&provider),
//!     None,
//! ).await?;
//!
//! println!("Signature: {}", result.to_signature_string());
//! // Output: "0din-v1:8d000000ac854dae..."
//! # Ok(())
//! # }
//! ```
//!
//! ### Low-Level API (Manual Pipeline)
//!
//! For more control, use the composable primitives:
//!
//! ```rust,no_run
//! # #[cfg(feature = "onnx")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use odin_prompt_toolkit::{simhash_lsh_multi, LshConfig};
//! use odin_prompt_toolkit::provider::EmbeddingProvider;
//! use odin_prompt_toolkit::providers::{ModelCache, OnnxProvider};
//!
//! let cache = ModelCache::new()?;
//! let provider = OnnxProvider::new(&cache, None, None, 0, 0).await?;
//!
//! // 1. Generate embedding
//! let embedding = provider.generate_embedding("Hello, world!").await?;
//!
//! // 2. Compute LSH signatures (embedding is already normalized)
//! let config = LshConfig::default();
//! let families = simhash_lsh_multi(&embedding.normalized_embedding, &config);
//!
//! println!("Signature: {}", families[0].signature);
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `openai` | Yes | OpenAI API embedding provider (V0 signatures, 1536 dims) |
//! | `onnx` | Yes | Local ONNX embedding provider (V1 signatures, 1024 dims) |
//! | `cm-lsh` | No | Confidence Matrix LSH (experimental, higher accuracy) |
//!
//! ## Signature Versions
//!
//! - **V0**: OpenAI text-embedding-3-large (1536 dimensions, API-based)
//! - **V1**: 0din-jailbreak-embeddings-small ONNX (1024 dimensions, local)
//! - **Latest**: Resolves to V1
//!
//! V0 and V1 signatures are **not comparable** due to different embedding spaces.
//!
//! ## Algorithm
//!
//! SimHash via Random Hyperplane LSH (Charikar 2002):
//! - Deterministic hyperplanes via SplitMix64 PRNG
//! - Default: 3 families × 256 bits × 16 bands
//! - Hex-encoded signatures (64 hex chars = 256 bits)
//! - Hamming distance → cosine similarity via `cos(π × d/n)`
//!
//! See the [specification](https://github.com/0din-ai/signature-sdk/blob/main/spec/SPEC.md)
//! for complete algorithm details.

pub mod error;
pub mod hasher;
pub mod hashers;
pub mod lsh;
pub mod provider;
pub mod providers;
pub mod sign;
pub mod types;

pub use error::{Result, SigError};
pub use hasher::Hasher;
pub use hashers::get_hasher;
pub use lsh::{
    compute_embedding_sha256, cosine_from_hamming, hamming_distance_hex, normalize_vector,
    simhash_lsh_multi,
};
pub use sign::sign_text;
pub use types::{
    parse_signature_string, signature_string, ComparisonResult, EmbeddingResult, HashAlgorithm,
    LshConfig, LshFamily, LshOutput, ParsedSignature, PromptInfo, QualityStats, SignatureResult,
    SignatureVersion,
};

// Re-export providers based on features
#[cfg(feature = "openai")]
pub use providers::OpenAIProvider;

#[cfg(feature = "onnx")]
pub use providers::{ModelCache, OnnxProvider};

// Re-export CM-LSH components when feature is enabled
#[cfg(feature = "cm-lsh")]
pub use hashers::{
    create_default_cm_lsh, gen_hyperplanes, Calibrator, DualHash, HybridCMLSH, HybridParams,
    ITQParams,
};

// Threat feed integration
#[cfg(feature = "threatfeed")]
pub mod threatfeed;

// SusFactor jailbreak classifier
#[cfg(feature = "susfactor")]
pub mod susfactor;

#[cfg(feature = "susfactor")]
pub use susfactor::{SusFactorClassifier, SusFactorResult};
