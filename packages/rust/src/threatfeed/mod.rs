//! Threat feed integration for fetching and caching known threat signatures.
//!
//! This module provides the ability to fetch detection signatures from the 0din
//! portal's threat feed API, cache them locally with a band index, and perform
//! fast similarity lookup against the cache.
//!
//! # Example
//!
//! ```rust,no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use odin_prompt_toolkit::threatfeed::{ThreatFeedClient, ThreatFeedCache};
//! use odin_prompt_toolkit::SignatureVersion;
//!
//! // Sync signatures from the portal
//! let client = ThreatFeedClient::new("your-api-token", None, None);
//! let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
//! cache.sync(&client, true).await?;
//!
//! // Query for similar signatures
//! let matches = cache.query(
//!     "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
//!     0.85,
//!     10,
//! );
//! # Ok(())
//! # }
//! ```

pub mod cache;
pub mod client;
pub mod compare;
pub mod types;

pub use cache::ThreatFeedCache;
pub use client::ThreatFeedClient;
pub use compare::compare_to_threatfeed;
pub use types::{CachedSignature, SyncResult, ThreatFeedEntry, ThreatMatch};
