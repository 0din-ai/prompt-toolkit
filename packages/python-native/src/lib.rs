//! Native Python bindings for odin-sig LSH functions
//!
//! This crate provides PyO3 bindings for the performance-critical LSH functions
//! from the odin-sig Rust library. When installed, the Python SDK automatically
//! uses these native implementations for ~627× speedup over pure Python.

use pyo3::prelude::*;

/// LSH family result containing signature and bands
///
/// This mirrors the Python LSHFamily dataclass from odin_sig.lsh
#[pyclass]
#[derive(Clone)]
pub struct LshFamily {
    #[pyo3(get)]
    pub family: usize,
    #[pyo3(get)]
    pub bits: usize,
    #[pyo3(get)]
    pub signature: String,
    #[pyo3(get)]
    pub bands: Vec<String>,
}

#[pymethods]
impl LshFamily {
    #[new]
    fn new(family: usize, bits: usize, signature: String, bands: Vec<String>) -> Self {
        LshFamily {
            family,
            bits,
            signature,
            bands,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LshFamily(family={}, bits={}, signature='{}...', bands={})",
            self.family,
            self.bits,
            &self.signature.chars().take(8).collect::<String>(),
            self.bands.len()
        )
    }
}

impl From<odin_sig::LshFamily> for LshFamily {
    fn from(f: odin_sig::LshFamily) -> Self {
        LshFamily {
            family: f.family,
            bits: f.bits,
            signature: f.signature,
            bands: f.bands,
        }
    }
}

/// LSH configuration
///
/// This mirrors the Python LshConfig dataclass from odin_sig.types
#[pyclass]
#[derive(Clone)]
pub struct LshConfig {
    #[pyo3(get)]
    pub families: usize,
    #[pyo3(get)]
    pub bits: usize,
    #[pyo3(get)]
    pub bands: usize,
}

#[pymethods]
impl LshConfig {
    #[new]
    #[pyo3(signature = (families=3, bits=256, bands=16))]
    fn new(families: usize, bits: usize, bands: usize) -> Self {
        LshConfig {
            families,
            bits,
            bands,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LshConfig(families={}, bits={}, bands={})",
            self.families, self.bits, self.bands
        )
    }
}

impl From<LshConfig> for odin_sig::LshConfig {
    fn from(c: LshConfig) -> Self {
        odin_sig::LshConfig {
            families: c.families,
            bits: c.bits,
            bands: c.bands,
        }
    }
}

/// Generate SimHash LSH signatures for a normalized embedding vector
///
/// Args:
///     normalized_vector: L2-normalized embedding vector (list of floats)
///     families: Number of independent hash families (default: 3)
///     bits: Number of bits per signature (default: 256)
///     bands: Number of bands to split signature into (default: 16)
///
/// Returns:
///     List of LshFamily objects, one per family
#[pyfunction]
#[pyo3(signature = (normalized_vector, families=3, bits=256, bands=16))]
fn simhash_lsh_multi(
    normalized_vector: Vec<f32>,
    families: usize,
    bits: usize,
    bands: usize,
) -> Vec<LshFamily> {
    let config = odin_sig::LshConfig {
        families,
        bits,
        bands,
    };
    let results = odin_sig::simhash_lsh_multi(&normalized_vector, &config);
    results.into_iter().map(LshFamily::from).collect()
}

/// L2-normalize a vector
///
/// Args:
///     vector: Input vector (list of floats)
///
/// Returns:
///     L2-normalized vector. If the input has zero magnitude, returns the original.
#[pyfunction]
fn normalize_vector(vector: Vec<f32>) -> Vec<f32> {
    odin_sig::normalize_vector(&vector)
}

/// Compute Hamming distance between two hex-encoded signatures
///
/// Args:
///     a: First hex signature string
///     b: Second hex signature string
///
/// Returns:
///     Number of differing bits
#[pyfunction]
fn hamming_distance_hex(a: &str, b: &str) -> usize {
    odin_sig::hamming_distance_hex(a, b)
}

/// Estimate cosine similarity from Hamming distance
///
/// Uses the formula: cos(π × distance / total_bits)
///
/// Args:
///     distance_bits: Hamming distance in bits
///     total_bits: Total number of bits in the signatures
///
/// Returns:
///     Estimated cosine similarity in [-1, 1]
#[pyfunction]
fn cosine_from_hamming(distance_bits: usize, total_bits: usize) -> f64 {
    odin_sig::cosine_from_hamming(distance_bits, total_bits)
}

/// Compute canonical SHA-256 hash of a normalized embedding
///
/// The embedding is quantized to 6 decimal places and serialized as JSON
/// with specific formatting (spaces after commas, .0 for whole numbers)
/// to ensure deterministic hashing across implementations.
///
/// Args:
///     normalized_embedding: L2-normalized embedding vector
///
/// Returns:
///     Hex-encoded SHA-256 hash
#[pyfunction]
fn compute_embedding_sha256(normalized_embedding: Vec<f32>) -> String {
    odin_sig::compute_embedding_sha256(&normalized_embedding)
}

/// Native Python extension module for odin-sig
///
/// This module is imported as `odin_sig_native` and provides accelerated
/// implementations of the core LSH functions.
#[pymodule]
fn odin_sig_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LshFamily>()?;
    m.add_class::<LshConfig>()?;
    m.add_function(wrap_pyfunction!(simhash_lsh_multi, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_vector, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_hex, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_from_hamming, m)?)?;
    m.add_function(wrap_pyfunction!(compute_embedding_sha256, m)?)?;
    Ok(())
}
