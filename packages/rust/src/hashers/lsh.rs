//! SimHash LSH implementation (current default algorithm).

use crate::hasher::Hasher;
use crate::lsh::simhash_lsh_multi;
use crate::types::{LshConfig, LshFamily};
use crate::SigError;

/// SimHash LSH implementation
///
/// This is the current default algorithm using deterministic random hyperplanes
/// with SplitMix64 for reproducible signature generation.
pub struct SimHashLsh;

impl Hasher for SimHashLsh {
    fn name(&self) -> &str {
        "lsh"
    }

    fn compute(&self, embedding: &[f32], config: &LshConfig) -> Result<Vec<LshFamily>, SigError> {
        Ok(simhash_lsh_multi(embedding, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_lsh_name() {
        let hasher = SimHashLsh;
        assert_eq!(hasher.name(), "lsh");
    }

    #[test]
    fn test_simhash_lsh_compute() {
        let hasher = SimHashLsh;
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let config = LshConfig::default();

        let result = hasher.compute(&embedding, &config).unwrap();
        assert_eq!(result.len(), config.families);
        assert_eq!(result[0].bits, config.bits);
        assert_eq!(result[0].bands.len(), config.bands);
    }

    #[test]
    fn test_simhash_lsh_deterministic() {
        let hasher = SimHashLsh;
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        let config = LshConfig::default();

        let result1 = hasher.compute(&embedding, &config).unwrap();
        let result2 = hasher.compute(&embedding, &config).unwrap();

        assert_eq!(result1[0].signature, result2[0].signature);
    }
}
