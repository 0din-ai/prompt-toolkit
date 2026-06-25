//! Extract SusFactor golden vectors from the validated Rust implementation.
//!
//! Reads the corpus from `spec/test-vectors/susfactor_vectors.json`, runs each
//! prompt through the real ONNX model, and writes `rust_score` (and
//! `expected_label` for near-boundary entries where it was null) back to the
//! same file.
//!
//! # Usage
//!
//! ```sh
//! SUSFACTOR_MODEL_DIR=/path/to/cache/susfactor-v1 \
//!   cargo run --example extract_susfactor_goldens --features susfactor --release
//! ```
//!
//! The model directory must contain:
//!   onnx/model.onnx
//!   onnx/model.onnx_data   (external weights)
//!   tokenizer.json
//!
//! # What this writes
//!
//! - `rust_score`: f64 from the f32 model output (full precision preserved).
//! - `expected_label`: filled in for vectors where it was null (derived from
//!   rust_score vs. the fixture threshold).
//! - `provenance` block: ORT version, model dir, timestamp of this run.
//!
//! Commit the result. Regeneration is a deliberate, reviewed act — only when
//! the model is retrained/re-exported, or re-validated in production.

use std::path::Path;

use odin_prompt_toolkit::providers::ModelCache;
use odin_prompt_toolkit::susfactor::SusFactorClassifier;
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Locate the model ────────────────────────────────────────────────────
    let model_dir = std::env::var("SUSFACTOR_MODEL_DIR").unwrap_or_else(|_| {
        eprintln!(
            "error: SUSFACTOR_MODEL_DIR is not set.\n\
             Point it at a directory containing:\n\
             \tonnx/model.onnx\n\
             \tonnx/model.onnx_data\n\
             \ttokenizer.json"
        );
        std::process::exit(1);
    });

    let onnx_path = Path::new(&model_dir).join("onnx").join("model.onnx");
    if !onnx_path.exists() {
        eprintln!(
            "error: model.onnx not found at {}\n\
             Download 0dinai/susfactor-e5-large-onnx from HuggingFace first.",
            onnx_path.display()
        );
        std::process::exit(1);
    }

    // ── Locate the fixture file ──────────────────────────────────────────────
    // examples/ lives at packages/rust/examples/, so ../../.. is the repo root.
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("spec")
        .join("test-vectors")
        .join("susfactor_vectors.json");

    let fixture_path = fixture_path
        .canonicalize()
        .unwrap_or_else(|_| fixture_path.clone());

    let raw = std::fs::read_to_string(&fixture_path)?;
    let mut doc: Value = serde_json::from_str(&raw)?;

    // ── Load classifier ──────────────────────────────────────────────────────
    println!("Loading SusFactor from: {model_dir}");
    let cache = ModelCache::new()?;
    let clf = SusFactorClassifier::new(
        &cache,
        None,                    // model name: default
        Some(model_dir.clone()), // source: local dir
        None,                    // threshold: default 0.5
    )
    .await?;

    println!("Model loaded. Running {} prompts...\n", {
        doc["vectors"].as_array().map(|v| v.len()).unwrap_or(0)
    });

    let threshold = doc["threshold"].as_f64().unwrap_or(0.5) as f32;

    // ── Score each vector ────────────────────────────────────────────────────
    let vectors = doc["vectors"]
        .as_array_mut()
        .expect("vectors must be an array");

    for entry in vectors.iter_mut() {
        let name = entry["name"].as_str().unwrap_or("?").to_string();
        let prompt = entry["prompt"].as_str().unwrap_or("").to_string();

        let result = clf.classify(&prompt).await?;

        // rust_score records chunk[0].score for both single- and multi-chunk
        // prompts. For single-chunk prompts this is the only score. For
        // multi-chunk prompts callers compare chunk[0].score against rust_score
        // and check is_suspicious for the overall label.
        let chunk0 = &result.chunks[0];

        // Store score at full f64 precision (the model outputs f32; we widen
        // to f64 for JSON so downstream parsers don't lose mantissa bits).
        entry["rust_score"] = json!(chunk0.score as f64);

        // Fill expected_label for near-boundary entries (where it was null).
        // Use is_suspicious (any-chunk) as the canonical label gate.
        if entry["expected_label"].is_null() {
            let label = if result.is_suspicious {
                "suspicious"
            } else {
                "safe"
            };
            entry["expected_label"] = json!(label);
            let n = result.chunks.len();
            println!(
                "  {name}: score={:.6}  label={label}  chunks={n}  (was null — now filled)",
                chunk0.score
            );
        } else {
            let expected = entry["expected_label"].as_str().unwrap_or("?");
            let actual_is_suspicious = result.is_suspicious;
            let expected_is_suspicious = expected == "suspicious";
            let ok = if actual_is_suspicious == expected_is_suspicious {
                "✅"
            } else {
                "❌ MISMATCH"
            };
            let n = result.chunks.len();
            println!(
                "  {name}: score={:.6}  {ok}  chunks={n}  (expected={expected})",
                chunk0.score
            );
        }
    }

    // ── Update provenance ────────────────────────────────────────────────────
    // Use Unix epoch seconds — no chrono dep needed.
    let epoch_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    doc["provenance"] = json!({
        "generated_by": "packages/rust/examples/extract_susfactor_goldens.rs",
        "model_dir": model_dir,
        "ort_optimization_level": "Level3",
        "epoch_secs": epoch_secs,
        "note": "Regenerate only when the model is retrained/re-exported or re-validated in production. Run: make generate-susfactor-goldens"
    });

    // ── Write back ───────────────────────────────────────────────────────────
    let out = serde_json::to_string_pretty(&doc)?;
    std::fs::write(&fixture_path, out)?;
    println!("\n✅ Scores written to {}", fixture_path.display());
    println!("   Review the diff, then commit if expected_label values look correct.");

    Ok(())
}
