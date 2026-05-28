//! Example: Sign an ML model directory using OpenSSF model signing.
//!
//! Usage:
//!   cargo run --example sign_model -- ./path/to/model/

use gpu_hashlib::signing::{ArtifactType, ManifestBuilder, verify_artifact};
use gpu_hashlib::{list_backends, HashAlgorithm};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line
    let args: Vec<String> = std::env::args().collect();
    let model_dir = if args.len() > 1 {
        Path::new(&args[1])
    } else {
        eprintln!("Usage: sign_model <model_directory>");
        eprintln!("\nCreating a temporary example model...");
        return run_demo();
    };

    if !model_dir.is_dir() {
        eprintln!("Error: {:?} is not a directory", model_dir);
        std::process::exit(1);
    }

    // Show available backends
    println!("=== Available Hashing Backends ===\n");
    for info in list_backends() {
        println!(
            "  {} ({}) — {} device(s), available: {}",
            info.name, info.vendor, info.device_count, info.available
        );
    }
    println!();

    // Build the signing manifest
    println!("=== Computing Digests for {:?} ===\n", model_dir);

    let manifest = ManifestBuilder::new(
        model_dir
            .file_name()
            .unwrap()
            .to_str()
            .unwrap_or("model"),
        ArtifactType::MlModel,
    )
    .algorithm(HashAlgorithm::Sha256)
    .metadata("tool", "gpu-hashlib")
    .metadata("tool_version", env!("CARGO_PKG_VERSION"))
    .add_directory(model_dir)?
    .build()?;

    // Print manifest
    println!("=== Signing Manifest ===\n");
    println!("{}", manifest.to_json()?);

    // Print DSSE info
    let pae = manifest.dsse_pae()?;
    println!("\n=== DSSE PAE ({} bytes) ===", pae.len());
    println!("Payload type: {}", gpu_hashlib::signing::SigningManifest::payload_type());

    // Verify
    println!("\n=== Verification ===\n");
    let result = verify_artifact(&manifest, model_dir)?;
    println!("Passed: {}", result.passed);
    for fr in &result.file_results {
        println!("  {:?} — {:?}", fr.path, fr.status);
    }

    Ok(())
}

fn run_demo() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;

    // Create fake model files
    std::fs::write(dir.path().join("model.onnx"), b"fake onnx model weights")?;
    std::fs::write(
        dir.path().join("config.json"),
        b"{\"framework\": \"pytorch\", \"version\": \"2.0\"}",
    )?;
    std::fs::write(dir.path().join("tokenizer.json"), b"{\"vocab_size\": 50000}")?;

    println!("=== Demo: Signing temporary model ===\n");

    let manifest = ManifestBuilder::new("demo-model-v1.0", ArtifactType::MlModel)
        .algorithm(HashAlgorithm::Sha256)
        .metadata("framework", "pytorch")
        .metadata("demo", "true")
        .add_directory(dir.path())?
        .build()?;

    println!("{}\n", manifest.to_json()?);

    // Verify — should pass
    let result = verify_artifact(&manifest, dir.path())?;
    println!("Verification passed: {}\n", result.passed);

    // Tamper and re-verify — should fail
    std::fs::write(dir.path().join("model.onnx"), b"TAMPERED!")?;
    let result = verify_artifact(&manifest, dir.path())?;
    println!("After tampering: passed = {} (expected false)", result.passed);

    Ok(())
}
