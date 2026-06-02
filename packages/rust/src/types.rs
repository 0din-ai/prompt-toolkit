use serde::{Deserialize, Serialize};

/// Hash algorithm selection (internal/undocumented)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HashAlgorithm {
    /// SimHash LSH (current default)
    #[default]
    Lsh,
}

impl std::fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashAlgorithm::Lsh => write!(f, "lsh"),
        }
    }
}

impl std::str::FromStr for HashAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lsh" | "simhash" => Ok(HashAlgorithm::Lsh),
            _ => Err(format!("Unknown algorithm: {}", s)),
        }
    }
}

/// Signature version identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignatureVersion {
    /// v0: text-embedding-3-large + LSH (256-bit, 3 families, 1536 dims)
    V0,
    /// v1: 0din-jailbreak-embeddings-small ONNX + LSH (256-bit, 3 families, 1024 dims)
    V1,
    /// Latest version (resolves to V1)
    #[serde(rename = "latest")]
    Latest,
}

impl std::fmt::Display for SignatureVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignatureVersion::V0 => write!(f, "v0"),
            SignatureVersion::V1 => write!(f, "v1"),
            SignatureVersion::Latest => write!(f, "latest"),
        }
    }
}

impl std::str::FromStr for SignatureVersion {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "v0" => Ok(SignatureVersion::V0),
            "v1" => Ok(SignatureVersion::V1),
            "latest" => Ok(SignatureVersion::Latest),
            _ => Err(format!("Unknown signature version: {}", s)),
        }
    }
}

impl SignatureVersion {
    /// Resolve Latest to the actual current version.
    ///
    /// # Breaking Change
    ///
    /// As of the V1 implementation, `Latest` resolves to `V1` (was `V0` previously).
    /// This means applications using `Latest` will now receive V1 signatures by default.
    ///
    /// # Example
    ///
    /// ```
    /// use odin_prompt_toolkit::types::SignatureVersion;
    ///
    /// assert_eq!(SignatureVersion::V0.resolve(), SignatureVersion::V0);
    /// assert_eq!(SignatureVersion::V1.resolve(), SignatureVersion::V1);
    /// assert_eq!(SignatureVersion::Latest.resolve(), SignatureVersion::V1);
    /// ```
    pub fn resolve(&self) -> Self {
        match self {
            SignatureVersion::Latest => SignatureVersion::V1,
            other => *other,
        }
    }

    /// Get algorithm for this version
    pub fn to_algorithm(&self) -> HashAlgorithm {
        match self.resolve() {
            SignatureVersion::V0 => HashAlgorithm::Lsh,
            SignatureVersion::V1 => HashAlgorithm::Lsh,
            SignatureVersion::Latest => unreachable!("Latest should be resolved"),
        }
    }

    /// Get version from algorithm (defaults to latest version with that algorithm)
    pub fn from_algorithm(algorithm: HashAlgorithm) -> Self {
        match algorithm {
            HashAlgorithm::Lsh => SignatureVersion::V1, // Changed: defaults to V1 now
        }
    }

    /// Get the embedding dimensions for this version.
    ///
    /// Returns the number of dimensions in the embedding vector for each version:
    /// - V0: 1536 dimensions (OpenAI text-embedding-3-large)
    /// - V1: 1024 dimensions (0din-jailbreak-embeddings-small ONNX)
    ///
    /// # Panics
    ///
    /// Panics if called on `Latest` without first calling [`resolve()`](Self::resolve).
    ///
    /// # Example
    ///
    /// ```
    /// use odin_prompt_toolkit::types::SignatureVersion;
    ///
    /// assert_eq!(SignatureVersion::V0.embedding_dimensions(), 1536);
    /// assert_eq!(SignatureVersion::V1.embedding_dimensions(), 1024);
    /// assert_eq!(SignatureVersion::Latest.resolve().embedding_dimensions(), 1024);
    /// ```
    pub fn embedding_dimensions(&self) -> usize {
        match self.resolve() {
            SignatureVersion::V0 => 1536, // OpenAI text-embedding-3-large
            SignatureVersion::V1 => 1024, // 0din-jailbreak-embeddings-small
            SignatureVersion::Latest => unreachable!("Latest should be resolved"),
        }
    }

    /// Default version for serde deserialization of older payloads.
    ///
    /// Used by `#[serde(default = "SignatureVersion::default_version")]`
    /// on `ComparisonResult.version` to maintain backward compatibility.
    pub fn default_version() -> Self {
        SignatureVersion::V1
    }
}

/// LSH configuration parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LshConfig {
    pub families: usize,
    pub bits: usize,
    pub bands: usize,
}

impl Default for LshConfig {
    fn default() -> Self {
        Self {
            families: 3,
            bits: 256,
            bands: 16,
        }
    }
}

/// Result of LSH hashing for one family
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshFamily {
    pub family: usize,
    pub bits: usize,
    pub signature: String,
    pub bands: Vec<String>, // contiguous slices
}

/// Embedding result from a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub embedding: Vec<f32>,
    pub normalized_embedding: Vec<f32>,
    pub normalized_embedding_sha256: String,
    pub model: String,
    pub dimensions: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_hash_algorithm_default() {
        assert_eq!(HashAlgorithm::default(), HashAlgorithm::Lsh);
    }

    #[test]
    fn test_hash_algorithm_from_str() {
        assert_eq!(HashAlgorithm::from_str("lsh").unwrap(), HashAlgorithm::Lsh);
        assert_eq!(
            HashAlgorithm::from_str("simhash").unwrap(),
            HashAlgorithm::Lsh
        );
        assert!(HashAlgorithm::from_str("unknown").is_err());
    }

    #[test]
    fn test_hash_algorithm_display() {
        assert_eq!(HashAlgorithm::Lsh.to_string(), "lsh");
    }

    #[test]
    fn test_hash_algorithm_serde() {
        let json = serde_json::to_string(&HashAlgorithm::Lsh).unwrap();
        assert_eq!(json, "\"lsh\"");

        let parsed: HashAlgorithm = serde_json::from_str("\"lsh\"").unwrap();
        assert_eq!(parsed, HashAlgorithm::Lsh);
    }

    #[test]
    fn test_signature_version_resolve() {
        assert_eq!(SignatureVersion::V0.resolve(), SignatureVersion::V0);
        assert_eq!(SignatureVersion::V1.resolve(), SignatureVersion::V1);
        assert_eq!(SignatureVersion::Latest.resolve(), SignatureVersion::V1); // Changed: now resolves to V1
    }

    #[test]
    fn test_signature_version_to_algorithm() {
        assert_eq!(SignatureVersion::V0.to_algorithm(), HashAlgorithm::Lsh);
        assert_eq!(SignatureVersion::V1.to_algorithm(), HashAlgorithm::Lsh);
        assert_eq!(SignatureVersion::Latest.to_algorithm(), HashAlgorithm::Lsh);
    }

    #[test]
    fn test_signature_version_from_algorithm() {
        assert_eq!(
            SignatureVersion::from_algorithm(HashAlgorithm::Lsh),
            SignatureVersion::V1 // Changed: now defaults to V1
        );
    }

    #[test]
    fn test_signature_version_embedding_dimensions() {
        assert_eq!(SignatureVersion::V0.embedding_dimensions(), 1536);
        assert_eq!(SignatureVersion::V1.embedding_dimensions(), 1024);
        assert_eq!(SignatureVersion::Latest.embedding_dimensions(), 1024); // Resolves to V1
    }

    #[test]
    fn test_signature_string_v0() {
        let result = SignatureResult {
            signature: String::new(),
            version: SignatureVersion::V0,
            prompt_preview: "test".to_string(),
            prompt_length: 4,
            provider: "openai".to_string(),
            model: "test".to_string(),
            dimensions: 1536,
            embedding_sha256: "abc".to_string(),
            lsh: LshOutput {
                config: LshConfig::default(),
                signatures: vec![LshFamily {
                    family: 0,
                    bits: 256,
                    signature: "deadbeef".to_string(),
                    bands: vec![],
                }],
            },
            timing_ms: None,
        };

        assert_eq!(result.to_signature_string(), "0din-v0:deadbeef");
    }

    #[test]
    fn test_signature_string_v1() {
        let result = SignatureResult {
            signature: String::new(),
            version: SignatureVersion::V1,
            prompt_preview: "test".to_string(),
            prompt_length: 4,
            provider: "onnx".to_string(),
            model: "intfloat/multilingual-e5-large".to_string(),
            dimensions: 1024,
            embedding_sha256: "abc".to_string(),
            lsh: LshOutput {
                config: LshConfig::default(),
                signatures: vec![LshFamily {
                    family: 0,
                    bits: 256,
                    signature: "cafebabe".to_string(),
                    bands: vec![],
                }],
            },
            timing_ms: None,
        };

        assert_eq!(result.to_signature_string(), "0din-v1:cafebabe");
    }

    #[test]
    fn test_parse_signature_string_v0() {
        let parsed = parse_signature_string("0din-v0:deadbeef").unwrap();
        assert_eq!(parsed.version(), SignatureVersion::V0);
        assert_eq!(parsed.signature(), "deadbeef");

        match parsed {
            ParsedSignature::V0 { signature } => {
                assert_eq!(signature, "deadbeef");
            }
            _ => panic!("Expected V0 signature"),
        }
    }

    #[test]
    fn test_parse_signature_string_v1() {
        let parsed = parse_signature_string("0din-v1:cafebabe").unwrap();
        assert_eq!(parsed.version(), SignatureVersion::V1);
        assert_eq!(parsed.signature(), "cafebabe");

        match parsed {
            ParsedSignature::V1 { signature } => {
                assert_eq!(signature, "cafebabe");
            }
            _ => panic!("Expected V1 signature"),
        }
    }

    #[test]
    fn test_parse_signature_string_invalid_prefix() {
        let result = parse_signature_string("invalid:deadbeef");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must start with '0din-'"));
    }

    #[test]
    fn test_parse_signature_string_unsupported_version() {
        let result = parse_signature_string("0din-v99:deadbeef");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Unsupported signature version"));
    }

    #[test]
    fn test_parse_signature_string_invalid_v0_format() {
        let result = parse_signature_string("0din-v0:dead:beef");
        assert!(result.is_err());
    }
}

/// Complete signature result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureResult {
    /// Primary signature in 0din format (e.g., "0din-v1:\<sig\>:\<mask\>")
    pub signature: String,

    pub version: SignatureVersion,
    pub prompt_preview: String,
    pub prompt_length: usize,
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
    pub embedding_sha256: String,
    pub lsh: LshOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_ms: Option<f64>,
}

impl SignatureResult {
    /// Generate signature string in 0din format
    ///
    /// Format:
    /// - v0: `0din-v0:<signature>`
    /// - v1: `0din-v1:<signature>`
    pub fn to_signature_string(&self) -> String {
        let resolved_version = self.version.resolve();
        let primary_sig = &self.lsh.signatures[0];

        match resolved_version {
            SignatureVersion::V0 => {
                format!("0din-v0:{}", primary_sig.signature)
            }
            SignatureVersion::V1 => {
                format!("0din-v1:{}", primary_sig.signature)
            }
            SignatureVersion::Latest => {
                unreachable!("Latest should be resolved before string generation")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshOutput {
    pub config: LshConfig,
    pub signatures: Vec<LshFamily>,
}

/// Comparison result between two signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub prompt_a: PromptInfo,
    pub prompt_b: PromptInfo,
    pub hamming_distance: usize,
    pub cosine_similarity: f64,
    pub lsh_config: LshConfig,
    /// Resolved signature version used for both embeddings.
    ///
    /// Always a concrete version (`V0` or `V1`) — never `Latest`.
    ///
    /// Defaults to `V1` when deserializing older payloads that lack this field,
    /// preserving backward compatibility with pre-existing serialized results.
    #[serde(default = "SignatureVersion::default_version")]
    pub version: SignatureVersion,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality_stats: Option<QualityStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptInfo {
    pub preview: String,
    pub length: usize,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    pub absolute_error: f64,
    pub signed_error: f64,
    pub squared_error: f64,
    pub quality_rating: String,
}

/// Parsed signature string
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedSignature {
    /// v0 signature (OpenAI text-embedding-3-large)
    V0 { signature: String },
    /// v1 signature (multilingual-e5-small ONNX)
    V1 { signature: String },
}

impl ParsedSignature {
    /// Get the version of this parsed signature
    pub fn version(&self) -> SignatureVersion {
        match self {
            ParsedSignature::V0 { .. } => SignatureVersion::V0,
            ParsedSignature::V1 { .. } => SignatureVersion::V1,
        }
    }

    /// Get the primary signature hash
    pub fn signature(&self) -> &str {
        match self {
            ParsedSignature::V0 { signature } => signature,
            ParsedSignature::V1 { signature } => signature,
        }
    }
}

/// Format a signature string in 0din format
///
/// Creates a signature string from a version and hex signature.
///
/// # Arguments
///
/// * `version` - The signature version (V0, V1, or Latest)
/// * `signature` - The hex-encoded signature string
///
/// # Returns
///
/// Formatted signature string: `0din-v{N}:<signature>`
///
/// # Example
///
/// ```
/// use odin_prompt_toolkit::{signature_string, SignatureVersion};
///
/// let sig = signature_string(SignatureVersion::V1, "abcd1234");
/// assert_eq!(sig, "0din-v1:abcd1234");
/// ```
pub fn signature_string(version: SignatureVersion, signature: &str) -> String {
    let version = version.resolve();
    format!("0din-{}:{}", version, signature)
}

/// Parse a signature string in 0din format
///
/// Format:
/// - v0: `0din-v0:<signature>`
/// - v1: `0din-v1:<signature>`
///
/// # Errors
///
/// Returns an error if:
/// - The string doesn't start with `0din-`
/// - The version is not v0 or v1
/// - The format is invalid for the specified version
pub fn parse_signature_string(s: &str) -> Result<ParsedSignature, String> {
    if !s.starts_with("0din-") {
        return Err("Invalid signature format: must start with '0din-'".to_string());
    }

    let parts: Vec<&str> = s.splitn(3, ':').collect();

    if parts.len() < 2 {
        return Err("Invalid signature format: missing components".to_string());
    }

    match parts[0] {
        "0din-v0" => {
            if parts.len() != 2 {
                return Err("Invalid v0 signature format: expected 0din-v0:<signature>".to_string());
            }
            Ok(ParsedSignature::V0 {
                signature: parts[1].to_string(),
            })
        }
        "0din-v1" => {
            if parts.len() != 2 {
                return Err("Invalid v1 signature format: expected 0din-v1:<signature>".to_string());
            }
            Ok(ParsedSignature::V1 {
                signature: parts[1].to_string(),
            })
        }
        _ => Err(format!("Unsupported signature version: {}", parts[0])),
    }
}
