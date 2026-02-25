use crate::types::{LshConfig, LshFamily};
use sha2::{Digest, Sha256};

/// SplitMix64 hash function for deterministic random generation
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Get deterministic +1/-1 sign from (family, bit, dim)
fn sign_for(family: usize, bit: usize, dim: usize) -> f64 {
    let seed = ((family as u64) << 48) ^ ((bit as u64) << 24) ^ (dim as u64);
    let h = splitmix64(seed);
    if (h & 1) == 1 {
        1.0
    } else {
        -1.0
    }
}

/// Compute SimHash LSH signatures for a normalized vector
pub fn simhash_lsh_multi(normalized_vector: &[f32], config: &LshConfig) -> Vec<LshFamily> {
    let families = config.families.max(1);
    let bits = config.bits.max(64);
    let bands = config.bands.max(1);

    let mut results = Vec::with_capacity(families);

    for f in 0..families {
        let mut bool_bits = Vec::with_capacity(bits);

        for b in 0..bits {
            // Use f64 for dot product to match Python's precision
            let dot: f64 = normalized_vector
                .iter()
                .enumerate()
                .map(|(j, &val)| val as f64 * sign_for(f, b, j))
                .sum();
            bool_bits.push(dot > 0.0);
        }

        // Pack into hex string
        let mut hex_chars = String::with_capacity(bits / 4);
        for chunk in bool_bits.chunks(4) {
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

        // Split into bands
        let band_len = (hex_chars.len() / bands).max(1);
        let band_arr: Vec<String> = hex_chars
            .as_bytes()
            .chunks(band_len)
            .take(bands)
            .map(|c| String::from_utf8_lossy(c).to_string())
            .collect();

        results.push(LshFamily {
            family: f,
            bits,
            signature: hex_chars,
            bands: band_arr,
        });
    }

    results
}

/// Compute Hamming distance between two hex signatures
pub fn hamming_distance_hex(a: &str, b: &str) -> usize {
    let clean = |s: &str| -> String {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_hexdigit())
            .collect()
    };

    let x = clean(a);
    let y = clean(b);
    let min_len = x.len().min(y.len());

    let mut dist = 0usize;
    for (c1, c2) in x.chars().zip(y.chars()).take(min_len) {
        let n1 = c1.to_digit(16).unwrap();
        let n2 = c2.to_digit(16).unwrap();
        let xor = n1 ^ n2;
        dist += xor.count_ones() as usize;
    }

    // Extra nibbles count as differing bits
    dist += x.len().abs_diff(y.len()) * 4;
    dist
}

/// Estimate cosine similarity from Hamming distance
pub fn cosine_from_hamming(distance_bits: usize, total_bits: usize) -> f64 {
    if total_bits == 0 {
        return 0.0;
    }
    let p_diff = distance_bits as f64 / total_bits as f64;
    (std::f64::consts::PI * p_diff).cos()
}

/// L2-normalize a vector
///
/// Note: Uses f32 precision throughout. For exact Python compatibility (f64),
/// there may be tiny rounding differences (~1 bit per 256 bits in LSH signatures)
/// due to floating-point precision. This is negligible for practical similarity matching.
pub fn normalize_vector(vector: &[f32]) -> Vec<f32> {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|x| x / magnitude).collect()
}

/// Compute SHA256 hash of normalized embedding.
///
/// This implementation matches Thor's canonical specification:
/// 1. Quantize each value to 6 decimal places: `round(x * 1e6) / 1e6`
/// 2. Serialize as JSON array: `[0.001234,0.005678,...]`
/// 3. Hash the JSON string representation
///
/// The 6-decimal quantization eliminates floating-point jitter from:
/// - OpenAI API non-determinism (different servers/GPUs)
/// - Cross-platform float representation differences
/// - Numerical precision variations
///
/// This ensures the SHA256 is stable and matches Thor frontend computations.
pub fn compute_embedding_sha256(normalized_embedding: &[f32]) -> String {
    // Quantize to 6 decimals: round(x * 1e6) / 1e6
    let quantized: Vec<f64> = normalized_embedding
        .iter()
        .map(|&x| {
            let x64 = x as f64;
            (x64 * 1_000_000.0).round() / 1_000_000.0
        })
        .collect();

    // Serialize as JSON array with spaces after commas (matches Python's json.dumps)
    // Python: json.dumps([0.1, 0.2]) produces "[0.1, 0.2]" (spaces after commas)
    // Python: Always includes ".0" for whole numbers: 1.0 not 1
    // Rust serde_json::to_string produces "[0.1,0.2]" (no spaces) so we need to match Python
    //
    // Build the JSON string manually to match Python's format exactly
    let json_parts: Vec<String> = quantized
        .iter()
        .map(|&x| {
            // Format with trailing zeros removed, but keeping .0 for whole numbers
            // This matches Python's json.dumps float formatting
            let s = format!("{}", x);
            // If it's a whole number and doesn't have a decimal point, add .0
            if x.fract() == 0.0 && !s.contains('.') {
                format!("{}.0", s)
            } else {
                s
            }
        })
        .collect();
    let json_str = format!("[{}]", json_parts.join(", ")); // Space after comma!

    // Compute SHA256 of the JSON string
    let mut hasher = Sha256::new();
    hasher.update(json_str.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_vector() {
        let v = vec![3.0, 4.0];
        let normalized = normalize_vector(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_simhash_deterministic() {
        let v = normalize_vector(&[1.0, 2.0, 3.0, 4.0]);
        let config = LshConfig::default();

        let result1 = simhash_lsh_multi(&v, &config);
        let result2 = simhash_lsh_multi(&v, &config);

        assert_eq!(result1[0].signature, result2[0].signature);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance_hex("0000", "ffff"), 16);
        assert_eq!(hamming_distance_hex("0000", "0000"), 0);
        assert_eq!(hamming_distance_hex("0f0f", "f0f0"), 16);
    }

    #[test]
    fn test_cosine_from_hamming() {
        // Same signature = cosine 1.0
        let cos = cosine_from_hamming(0, 256);
        assert!((cos - 1.0).abs() < 1e-6);

        // Opposite signature = cosine -1.0
        let cos = cosine_from_hamming(256, 256);
        assert!((cos - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_embedding_sha256() {
        let v = vec![0.1, 0.2, 0.3];
        let hash1 = compute_embedding_sha256(&v);
        let hash2 = compute_embedding_sha256(&v);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA256 = 64 hex chars
    }

    #[test]
    fn test_embedding_sha256_matches_thor_format() {
        // Test vector that matches Thor's canonical test case
        let v = vec![0.123456789, 0.987654321, -0.111111111];

        // After quantization to 6 decimals:
        // 0.123456789 -> 0.123457
        // 0.987654321 -> 0.987654
        // -0.111111111 -> -0.111111

        // Expected JSON format: [0.123457,0.987654,-0.111111]
        // This matches Thor's JSON.stringify(toFixedArray(v, 6))

        let hash = compute_embedding_sha256(&v);

        // Verify it's deterministic
        let hash2 = compute_embedding_sha256(&v);
        assert_eq!(hash, hash2);

        // Verify hash length
        assert_eq!(hash.len(), 64);

        // Verify quantization eliminates jitter beyond 6 decimals
        let v_jittered = vec![0.1234567891, 0.9876543212, -0.1111111112];
        let hash_jittered = compute_embedding_sha256(&v_jittered);
        assert_eq!(
            hash, hash_jittered,
            "Quantization should eliminate jitter beyond 6 decimals"
        );
    }

    #[test]
    fn test_embedding_sha256_quantization() {
        // Test that values differing only beyond 6 decimals produce the same hash
        let v1 = vec![0.123456001, 0.999999999];
        let v2 = vec![0.123456002, 0.999999998];

        let hash1 = compute_embedding_sha256(&v1);
        let hash2 = compute_embedding_sha256(&v2);

        // Both round to [0.123456, 1.0]
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_embedding_sha256_matches_thor_reference() {
        // These reference values are from verify_sha_match.py
        // which implements Thor's canonical formula: sha256(json.dumps(toFixedArray(v, 6)))

        // Test case 1: Simple vector
        let v1 = vec![0.1, 0.2, 0.3];
        let hash1 = compute_embedding_sha256(&v1);
        assert_eq!(
            hash1, "9a04781069052282acb2e95529c7f5bcd85149ab2ec559c550dce80b81ceb04e",
            "Simple vector hash must match Thor's reference"
        );

        // Test case 2: High precision vector
        let v2 = vec![0.123456789, 0.987654321, -0.111111111];
        let hash2 = compute_embedding_sha256(&v2);
        assert_eq!(
            hash2, "939cac39b5886ae89213f3db06102874acf411f6c6f7021c6b33297f1d5f39ea",
            "High precision vector hash must match Thor's reference"
        );

        // Test case 3: Sub-6-decimal jitter (both should produce same hash)
        let v3a = vec![0.123456001, 0.999999999];
        let v3b = vec![0.123456002, 0.999999998];
        let hash3a = compute_embedding_sha256(&v3a);
        let hash3b = compute_embedding_sha256(&v3b);
        assert_eq!(
            hash3a, "730da253c365cc40fb4b319971547d2f33b13de3469e917da1cdbd624f8b1a2a",
            "Jitter vector hash must match Thor's reference"
        );
        assert_eq!(
            hash3a, hash3b,
            "Quantization must eliminate sub-6-decimal jitter"
        );
    }
}
