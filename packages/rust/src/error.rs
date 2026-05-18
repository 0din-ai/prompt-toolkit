use thiserror::Error;

#[derive(Error, Debug)]
pub enum SigError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Threat feed API error: {0}")]
    ThreatFeedApi(String),

    #[error("Threat feed cache error: {0}")]
    ThreatFeedCache(String),
}

pub type Result<T> = std::result::Result<T, SigError>;
