//! High-level comparison API for threat feed matching.

use crate::error::Result;
use crate::types::SignatureResult;

use super::cache::ThreatFeedCache;
use super::types::ThreatMatch;

/// Compare a signature result against the threat feed cache.
///
/// Extracts the primary signature (family 0) from the result and queries
/// the cache for similar known threat signatures.
///
/// # Arguments
///
/// * `result` - Signature result from `sign_text()`
/// * `cache` - Pre-loaded threat feed cache
/// * `threshold` - Minimum cosine similarity threshold (default: 0.85)
/// * `max_results` - Maximum number of results to return (default: 10)
///
/// # Returns
///
/// Vector of matches sorted by cosine similarity descending.
///
/// # Example
///
/// ```rust,no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use odin_prompt_toolkit::{sign_text, SignatureVersion};
/// use odin_prompt_toolkit::threatfeed::{ThreatFeedCache, compare_to_threatfeed};
///
/// // Sign some text
/// let result = sign_text("suspicious prompt", SignatureVersion::V1, None, None).await?;
///
/// // Load the threat feed cache
/// let mut cache = ThreatFeedCache::new(SignatureVersion::V1, None, None);
/// cache.load()?;
///
/// // Compare against known threats
/// let matches = compare_to_threatfeed(&result, &cache, 0.85, 10)?;
/// for m in &matches {
///     println!("Match: {} (similarity: {:.3})", m.title, m.cosine_similarity);
/// }
/// # Ok(())
/// # }
/// ```
pub fn compare_to_threatfeed(
    result: &SignatureResult,
    cache: &ThreatFeedCache,
    threshold: f64,
    max_results: usize,
) -> Result<Vec<ThreatMatch>> {
    let primary_sig = &result.lsh.signatures[0].signature;
    Ok(cache.query(primary_sig, threshold, max_results))
}
