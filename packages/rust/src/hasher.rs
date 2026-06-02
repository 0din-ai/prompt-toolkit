//! Trait for hash algorithm implementations.

use crate::types::{LshConfig, LshFamily};
use crate::SigError;

/// Trait for hash algorithm implementations
///
/// Each hasher takes a normalized embedding vector and LSH configuration,
/// and produces LSH signatures suitable for similarity matching.
pub trait Hasher: Send + Sync {
    /// Algorithm name (e.g., "lsh")
    fn name(&self) -> &str;

    /// Compute LSH signatures from a normalized embedding vector
    ///
    /// # Arguments
    /// * `embedding` - Normalized embedding vector (unit length)
    /// * `config` - LSH configuration (families, bits, bands)
    ///
    /// # Returns
    /// Vector of `LshFamily` results, one per family
    fn compute(&self, embedding: &[f32], config: &LshConfig) -> Result<Vec<LshFamily>, SigError>;
}
