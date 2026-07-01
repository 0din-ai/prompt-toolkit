//! Vertex AI authentication wrapper.
//!
//! Re-exports the `gcp_auth` `TokenProvider` trait under the `VertexAuth` alias
//! so callers can reference it without importing `gcp_auth` directly.

pub use gcp_auth::TokenProvider as VertexAuth;
