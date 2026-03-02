//! Hash algorithm implementations.
//!
//! This module contains different LSH algorithm implementations that can be
//! selected via the `HashAlgorithm` enum.

mod lsh;

#[cfg(feature = "cm-lsh")]
mod cm_lsh;

pub use lsh::SimHashLsh;

#[cfg(feature = "cm-lsh")]
pub use cm_lsh::{
    create_default_cm_lsh, gen_hyperplanes, Calibrator, DualHash, HybridCMLSH, HybridParams,
    ITQParams,
};

use crate::hasher::Hasher;
use crate::types::HashAlgorithm;
use std::sync::Arc;

/// Get a hasher implementation for the specified algorithm.
///
/// # Arguments
/// * `algorithm` - The hash algorithm to use
///
/// # Returns
/// An `Arc<dyn Hasher>` that can be used to compute signatures
///
/// # Example
/// ```
/// use odin_sig::types::HashAlgorithm;
/// use odin_sig::hashers::get_hasher;
///
/// let hasher = get_hasher(HashAlgorithm::Lsh);
/// assert_eq!(hasher.name(), "lsh");
/// ```
pub fn get_hasher(algorithm: HashAlgorithm) -> Arc<dyn Hasher> {
    match algorithm {
        HashAlgorithm::Lsh => Arc::new(SimHashLsh),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_hasher_lsh() {
        let hasher = get_hasher(HashAlgorithm::Lsh);
        assert_eq!(hasher.name(), "lsh");
    }

    #[test]
    fn test_get_hasher_default() {
        let hasher = get_hasher(HashAlgorithm::default());
        assert_eq!(hasher.name(), "lsh");
    }
}
