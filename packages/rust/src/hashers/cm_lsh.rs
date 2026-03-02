//! Confidence Matrix LSH (CM-LSH) implementation.
//!
//! This module provides an enhanced LSH implementation that combines:
//! - Standard LSH-TS (256-bit random hyperplane hash)
//! - ITQ (Iterative Quantization) for improved quantization
//! - Confidence matrix to weight reliable bits higher
//! - Isotonic calibration for accurate similarity estimates
//!
//! The result is a DualHash with:
//! - hash_a: 512-bit signature (first 256 bits = LSH-TS compatible)
//! - hash_b: 512-bit confidence matrix
//! - Calibrated similarity function
//!
//! Ported from Python implementation in research/src/signature_cli/core/cm_lsh.py

use crate::error::SigError;
use crate::hasher::Hasher;
use crate::types::LshFamily;

/// Result of CM-LSH hashing with confidence matrix.
#[derive(Debug, Clone)]
pub struct DualHash {
    /// 512-bit signature (128 hex chars, first 64 = LSH-TS compat)
    pub hash_a: String,
    /// 512-bit confidence matrix (128 hex chars)
    pub hash_b: String,
    /// Band slices for LSH indexing
    pub bands: Vec<String>,
    /// Number of bits
    pub bits: usize,
}

impl DualHash {
    /// Get LSH-TS compatible 256-bit signature (first 64 hex chars).
    pub fn lsh_ts_compat(&self) -> String {
        self.hash_a[..64].to_string()
    }
}

/// Parameters for Iterative Quantization transformation.
#[derive(Debug, Clone)]
pub struct ITQParams {
    /// Mean vector for centering (dims,)
    pub mean: Vec<f32>,
    /// PCA projection matrix (256, dims)
    pub pca: Vec<Vec<f32>>,
    /// ITQ rotation matrix (256, 256)
    pub rotation: Vec<Vec<f32>>,
}

/// Combined parameters for hybrid LSH+ITQ.
#[derive(Debug, Clone)]
pub struct HybridParams {
    /// LSH-TS hyperplanes (256, dims)
    pub lsh_ts_hyperplanes: Vec<Vec<f32>>,
    /// ITQ parameters
    pub itq: ITQParams,
}

/// Isotonic regression calibrator for similarity scores.
///
/// Maps raw similarity scores to calibrated cosine similarity estimates
/// using piecewise linear interpolation.
#[derive(Debug, Clone)]
pub struct Calibrator {
    /// X thresholds for piecewise linear function
    pub x_thresh: Vec<f32>,
    /// Y thresholds for piecewise linear function
    pub y_thresh: Vec<f32>,
    /// Minimum x value for clipping
    pub x_min: f32,
    /// Maximum x value for clipping
    pub x_max: f32,
}

impl Calibrator {
    /// Create a new calibrator.
    pub fn new(x_thresh: Vec<f32>, y_thresh: Vec<f32>, x_min: f32, x_max: f32) -> Self {
        Self {
            x_thresh,
            y_thresh,
            x_min,
            x_max,
        }
    }

    /// Predict calibrated similarity from raw score.
    pub fn predict(&self, x: f32) -> f32 {
        let x_clipped = x.clamp(self.x_min, self.x_max);
        interp(&self.x_thresh, &self.y_thresh, x_clipped)
    }

    /// Create a linear (identity) calibrator for default usage.
    pub fn linear() -> Self {
        Self {
            x_thresh: vec![0.0, 1.0],
            y_thresh: vec![0.0, 1.0],
            x_min: 0.0,
            x_max: 1.0,
        }
    }
}

/// Hybrid Confidence Matrix LSH implementation.
///
/// Combines LSH-TS and ITQ for improved similarity search:
/// - LSH-TS: 256-bit random hyperplane hash (deterministic, backward compatible)
/// - ITQ: 256-bit quantized projection (learned, rotation optimized)
/// - Confidence: weights reliable bits higher in similarity computation
/// - Calibration: maps raw scores to accurate cosine similarities
pub struct HybridCMLSH {
    params: HybridParams,
    calibrator: Calibrator,
    alpha: f32,
    family: usize,
}

impl HybridCMLSH {
    /// Create a new CM-LSH hasher.
    pub fn new(params: HybridParams, calibrator: Calibrator, alpha: f32, family: usize) -> Self {
        Self {
            params,
            calibrator,
            alpha,
            family,
        }
    }

    /// Generate CM-LSH hash from embedding.
    pub fn hash(&self, embedding: &[f32]) -> DualHash {
        // Normalize embedding
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let emb: Vec<f32> = if norm > 1e-8 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        };

        self.gen_hash(&emb)
    }

    /// Compute calibrated similarity between two hashes.
    pub fn sim(&self, h1: &DualHash, h2: &DualHash) -> f32 {
        let raw_sim = self.raw_sim(h1, h2);
        self.calibrator.predict(raw_sim)
    }

    /// Compare two embeddings via CM-LSH.
    pub fn cmp(&self, e1: &[f32], e2: &[f32]) -> f32 {
        let h1 = self.hash(e1);
        let h2 = self.hash(e2);
        self.sim(&h1, &h2)
    }

    /// Check if two hashes represent duplicates.
    pub fn is_dup(&self, h1: &DualHash, h2: &DualHash, threshold: f32) -> bool {
        self.sim(h1, h2) >= threshold
    }

    /// Verify that LSH-TS portion matches standalone LSH-TS hash.
    pub fn verify_lsh_ts(&self, embedding: &[f32]) -> bool {
        let h = self.hash(embedding);

        // Normalize embedding
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let emb: Vec<f32> = if norm > 1e-8 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        };

        // Generate standalone LSH-TS hash
        let lsh_ts = lsh_ts_hash(&emb, self.family, 256);

        h.lsh_ts_compat() == lsh_ts
    }

    /// Generate hash from normalized embedding.
    fn gen_hash(&self, emb: &[f32]) -> DualHash {
        // 1. LSH-TS projection (256 bits)
        let p1 = matmul_vec(&self.params.lsh_ts_hyperplanes, emb);

        // 2. ITQ projection (256 bits)
        let centered: Vec<f32> = emb
            .iter()
            .zip(self.params.itq.mean.iter())
            .map(|(x, m)| x - m)
            .collect();

        let pca_proj = matmul_vec(&self.params.itq.pca, &centered);
        let p2 = matmul_vec(&self.params.itq.rotation, &pca_proj);

        // 3. Combine projections (512 bits total)
        let mut proj = p1;
        proj.extend(p2);

        // 4. Sign bits (hash_a)
        let signs: Vec<bool> = proj.iter().map(|&x| x > 0.0).collect();

        // 5. Confidence bits (hash_b)
        // Use 45th percentile as threshold
        let abs_proj: Vec<f32> = proj.iter().map(|x| x.abs()).collect();
        let conf_thresh = percentile(&abs_proj, 45.0);
        let conf_bits: Vec<bool> = abs_proj.iter().map(|&x| x > conf_thresh).collect();

        // 6. Pack into hex strings
        let hash_a = pack_bits(&signs);
        let hash_b = pack_bits(&conf_bits);

        // 7. Split into bands (64 bands for LSH indexing)
        let band_len = hash_a.len() / 64;
        let bands: Vec<String> = (0..64)
            .map(|i| {
                let start = i * band_len;
                let end = (start + band_len).min(hash_a.len());
                hash_a[start..end].to_string()
            })
            .collect();

        DualHash {
            hash_a,
            hash_b,
            bands,
            bits: 512,
        }
    }

    /// Compute raw similarity before calibration.
    fn raw_sim(&self, h1: &DualHash, h2: &DualHash) -> f32 {
        let a1 = unpack_bits(&h1.hash_a);
        let a2 = unpack_bits(&h2.hash_a);
        let b1 = unpack_bits(&h1.hash_b);
        let b2 = unpack_bits(&h2.hash_b);

        // Compute agreement and confidence overlap
        let agree: Vec<bool> = a1.iter().zip(a2.iter()).map(|(x, y)| x == y).collect();
        let both_conf: Vec<bool> = b1.iter().zip(b2.iter()).map(|(x, y)| x & y).collect();

        // Weighted similarity: alpha * (confident agreement) + (1-alpha) * (overall agreement)
        let conf_count = both_conf.iter().filter(|&&x| x).count();
        if conf_count > 0 {
            let conf_agree_count: usize = agree
                .iter()
                .zip(both_conf.iter())
                .filter(|(_, &conf)| conf)
                .filter(|(&agr, _)| agr)
                .count();

            let conf_agree_rate = conf_agree_count as f32 / conf_count as f32;
            let overall_agree_rate =
                agree.iter().filter(|&&x| x).count() as f32 / agree.len() as f32;

            self.alpha * conf_agree_rate + (1.0 - self.alpha) * overall_agree_rate
        } else {
            agree.iter().filter(|&&x| x).count() as f32 / agree.len() as f32
        }
    }
}

impl Hasher for HybridCMLSH {
    fn name(&self) -> &str {
        "cm-lsh"
    }

    fn compute(
        &self,
        embedding: &[f32],
        _config: &crate::types::LshConfig,
    ) -> Result<Vec<LshFamily>, SigError> {
        let dual_hash = self.hash(embedding);

        // Return a single family with the 512-bit signature
        Ok(vec![LshFamily {
            family: self.family,
            bits: 512,
            signature: dual_hash.hash_a,
            bands: dual_hash.bands,
        }])
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Generate deterministic random hyperplanes for LSH.
pub fn gen_hyperplanes(family: usize, bits: usize, dims: usize) -> Vec<Vec<f32>> {
    let mut hp = vec![vec![0.0f32; dims]; bits];
    for (b, row) in hp.iter_mut().enumerate().take(bits) {
        for (d, val) in row.iter_mut().enumerate().take(dims) {
            *val = sign_for(family, b, d);
        }
    }
    hp
}

/// Create a default CM-LSH instance without training.
///
/// This creates a minimal CM-LSH configuration using:
/// - Random hyperplanes for LSH-TS (deterministic based on family)
/// - Identity transformations for ITQ (no learned rotation)
/// - Linear calibrator (no adjustment)
///
/// Note: Always produces 512-bit output (256 from LSH-TS + 256 from ITQ).
/// For dimensions < 256, ITQ output is padded with zeros.
pub fn create_default_cm_lsh(dimensions: usize, family: usize) -> HybridCMLSH {
    // Generate LSH-TS hyperplanes (always 256 bits)
    let lsh_hp = gen_hyperplanes(family, 256, dimensions);

    // Create identity ITQ parameters (always 256 bits output)
    let itq_dims = dimensions.min(256);
    let mean = vec![0.0f32; dimensions];

    // PCA: identity for first itq_dims dimensions
    let mut pca = vec![vec![0.0f32; dimensions]; 256];
    for (i, row) in pca.iter_mut().enumerate().take(itq_dims) {
        if i < dimensions {
            row[i] = 1.0;
        }
    }

    // Rotation: identity
    let mut rotation = vec![vec![0.0f32; 256]; 256];
    for (i, row) in rotation.iter_mut().enumerate().take(256) {
        row[i] = 1.0;
    }

    let itq = ITQParams {
        mean,
        pca,
        rotation,
    };

    let params = HybridParams {
        lsh_ts_hyperplanes: lsh_hp,
        itq,
    };

    let calibrator = Calibrator::linear();

    HybridCMLSH::new(params, calibrator, 0.65, family)
}

/// Generate standalone LSH-TS hash (for verification).
fn lsh_ts_hash(emb: &[f32], family: usize, bits: usize) -> String {
    let mut bool_bits = Vec::with_capacity(bits);
    for b in 0..bits {
        let dot: f64 = emb
            .iter()
            .enumerate()
            .map(|(d, &val)| val as f64 * sign_for(family, b, d) as f64)
            .sum();
        bool_bits.push(dot > 0.0);
    }
    pack_bits(&bool_bits)
}

/// Pack boolean array into hex string.
fn pack_bits(bits: &[bool]) -> String {
    let mut hex_chars = String::with_capacity(bits.len() / 4);
    for chunk in bits.chunks(4) {
        let n = (if chunk[0] { 8 } else { 0 })
            + (if chunk.get(1).copied().unwrap_or(false) {
                4
            } else {
                0
            })
            + (if chunk.get(2).copied().unwrap_or(false) {
                2
            } else {
                0
            })
            + (if chunk.get(3).copied().unwrap_or(false) {
                1
            } else {
                0
            });
        hex_chars.push(char::from_digit(n, 16).unwrap());
    }
    hex_chars
}

/// Unpack hex string into boolean array.
fn unpack_bits(hex_str: &str) -> Vec<bool> {
    let mut bits = Vec::new();
    for c in hex_str.chars() {
        let n = c.to_digit(16).unwrap();
        bits.push(n & 8 != 0);
        bits.push(n & 4 != 0);
        bits.push(n & 2 != 0);
        bits.push(n & 1 != 0);
    }
    bits
}

/// Matrix-vector multiplication: (rows, cols) × (cols,) -> (rows,)
fn matmul_vec(mat: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
    mat.iter()
        .map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Linear interpolation (like numpy.interp)
fn interp(x: &[f32], y: &[f32], xp: f32) -> f32 {
    if xp <= x[0] {
        return y[0];
    }
    if xp >= x[x.len() - 1] {
        return y[y.len() - 1];
    }

    for i in 0..x.len() - 1 {
        if xp >= x[i] && xp <= x[i + 1] {
            let t = (xp - x[i]) / (x[i + 1] - x[i]);
            return y[i] + t * (y[i + 1] - y[i]);
        }
    }
    y[y.len() - 1]
}

/// Compute percentile of a slice
fn percentile(data: &[f32], p: f32) -> f32 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() as f32 - 1.0)) as usize;
    sorted[idx]
}

/// SplitMix64 hash function for deterministic random generation
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Get deterministic +1/-1 sign from (family, bit, dim)
fn sign_for(family: usize, bit: usize, dim: usize) -> f32 {
    let seed = ((family as u64) << 48) ^ ((bit as u64) << 24) ^ (dim as u64);
    let h = splitmix64(seed);
    if (h & 1) == 1 {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_bits() {
        let bits = vec![true, false, true, false, true, true, false, false];
        let hex = pack_bits(&bits);
        assert_eq!(hex, "ac");

        let unpacked = unpack_bits(&hex);
        assert_eq!(unpacked, bits);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_interp() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 10.0, 20.0];

        assert_eq!(interp(&x, &y, 0.0), 0.0);
        assert_eq!(interp(&x, &y, 0.5), 5.0);
        assert_eq!(interp(&x, &y, 1.0), 10.0);
        assert_eq!(interp(&x, &y, 1.5), 15.0);
        assert_eq!(interp(&x, &y, 2.0), 20.0);
    }

    #[test]
    fn test_matmul_vec() {
        let mat = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vec = vec![5.0, 6.0];
        let result = matmul_vec(&mat, &vec);
        assert_eq!(result, vec![17.0, 39.0]); // [1*5+2*6, 3*5+4*6]
    }

    #[test]
    fn test_default_cm_lsh() {
        let cm_lsh = create_default_cm_lsh(384, 0);
        let embedding = vec![0.5; 384];
        let hash = cm_lsh.hash(&embedding);

        assert_eq!(hash.bits, 512);
        assert_eq!(hash.hash_a.len(), 128); // 512 bits / 4 bits per hex char
        assert_eq!(hash.hash_b.len(), 128);
        assert_eq!(hash.bands.len(), 64);
    }

    #[test]
    fn test_lsh_ts_compatibility() {
        let cm_lsh = create_default_cm_lsh(384, 0);

        // Use a non-trivial embedding to test
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();

        // Verify first 256 bits match standalone LSH-TS
        let h = cm_lsh.hash(&embedding);
        let standalone = lsh_ts_hash(&embedding, 0, 256);

        let lsh_ts_compat = h.lsh_ts_compat();

        // Debug: print if they don't match
        if lsh_ts_compat != standalone {
            println!("CM-LSH LSH-TS compat: {}", &lsh_ts_compat[..20]);
            println!("Standalone LSH-TS:    {}", &standalone[..20]);
        }

        assert_eq!(
            lsh_ts_compat, standalone,
            "CM-LSH first 256 bits should match standalone LSH-TS"
        );
    }

    #[test]
    fn test_similarity() {
        let cm_lsh = create_default_cm_lsh(384, 0);

        let e1 = vec![0.5; 384];
        let e2 = vec![0.5; 384];
        let e3 = vec![0.1; 384];

        let h1 = cm_lsh.hash(&e1);
        let h2 = cm_lsh.hash(&e2);
        let h3 = cm_lsh.hash(&e3);

        // Identical embeddings should have high similarity
        let sim_identical = cm_lsh.sim(&h1, &h2);
        assert!(sim_identical > 0.9, "sim_identical = {}", sim_identical);

        // Different embeddings should have lower similarity
        let sim_different = cm_lsh.sim(&h1, &h3);
        assert!(
            sim_different < sim_identical,
            "sim_different = {}, sim_identical = {}",
            sim_different,
            sim_identical
        );
    }
}
