//! Artifact Verification
//!
//! Verifies that model files match their signed digests.
//! This is the verification counterpart to the signing/manifest module.

use super::manifest::SigningManifest;
use crate::backend::HashAlgorithm;
use crate::error::{HashError, HashResult};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Status of a single file verification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationStatus {
    /// File matches its expected digest.
    Ok,
    /// File digest does not match.
    DigestMismatch {
        expected: String,
        actual: String,
    },
    /// File is missing from disk.
    FileMissing,
    /// File is present on disk but not in the manifest.
    ExtraFile,
    /// Verification error.
    Error(String),
}

/// Result of verifying a single file or the entire artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Overall pass/fail.
    pub passed: bool,
    /// Per-file results.
    pub file_results: Vec<FileVerificationResult>,
    /// Combined digest verification (if present in manifest).
    pub combined_digest_ok: Option<bool>,
    /// Timestamp of verification.
    pub verified_at: chrono::DateTime<chrono::Utc>,
    /// Backend used for re-hashing.
    pub hashing_backend: String,
}

/// Verification result for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileVerificationResult {
    /// File path.
    pub path: std::path::PathBuf,
    /// Verification status.
    pub status: VerificationStatus,
}

/// Verify an artifact against a signing manifest.
///
/// Re-computes hashes of all files listed in the manifest and compares
/// them against the expected digests.
///
/// # Arguments
///
/// * `manifest` - The signed manifest to verify against
/// * `artifact_root` - Root directory containing the artifact files
pub fn verify_artifact(
    manifest: &SigningManifest,
    artifact_root: &Path,
) -> HashResult<VerificationResult> {
    let mut file_results = Vec::new();
    let mut all_ok = true;

    // Collect all file data for batch hashing
    let mut paths_to_hash = Vec::new();
    let mut expected_digests = Vec::new();
    let mut file_data = Vec::new();

    for entry in &manifest.entries {
        let file_path = artifact_root.join(&entry.path);

        if !file_path.exists() {
            file_results.push(FileVerificationResult {
                path: entry.path.clone(),
                status: VerificationStatus::FileMissing,
            });
            all_ok = false;
            continue;
        }

        match std::fs::read(&file_path) {
            Ok(data) => {
                paths_to_hash.push(entry.path.clone());
                expected_digests.push(entry.digest.clone());
                file_data.push(data);
            }
            Err(e) => {
                file_results.push(FileVerificationResult {
                    path: entry.path.clone(),
                    status: VerificationStatus::Error(format!("Read error: {e}")),
                });
                all_ok = false;
            }
        }
    }

    // Batch hash all readable files
    if !file_data.is_empty() {
        let hashes = crate::hash_batch(&file_data, manifest.algorithm)?;

        for ((path, expected), hash) in paths_to_hash
            .iter()
            .zip(expected_digests.iter())
            .zip(hashes.iter())
        {
            let actual_hex = hash.to_hex();
            let status = if &actual_hex == expected {
                VerificationStatus::Ok
            } else {
                all_ok = false;
                VerificationStatus::DigestMismatch {
                    expected: expected.clone(),
                    actual: actual_hex,
                }
            };

            file_results.push(FileVerificationResult {
                path: path.clone(),
                status,
            });
        }
    }

    // Verify combined digest if present
    let combined_ok = if let Some(ref combined) = manifest.combined_digest {
        let computed_input: Vec<u8> = manifest
            .entries
            .iter()
            .flat_map(|e| e.digest.as_bytes().to_vec())
            .collect();
        let computed = crate::hash(&computed_input, manifest.algorithm)?;
        let expected_str = format!(
            "{}:{}",
            manifest.algorithm.openssf_id(),
            computed.to_hex()
        );
        let ok = &expected_str == combined;
        if !ok {
            all_ok = false;
        }
        Some(ok)
    } else {
        None
    };

    Ok(VerificationResult {
        passed: all_ok,
        file_results,
        combined_digest_ok: combined_ok,
        verified_at: chrono::Utc::now(),
        hashing_backend: "auto".to_string(),
    })
}

/// Quick verification: just check if the combined digest matches.
pub fn verify_combined_digest(
    manifest: &SigningManifest,
    artifact_root: &Path,
) -> HashResult<bool> {
    let result = verify_artifact(manifest, artifact_root)?;
    Ok(result.passed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signing::manifest::{ArtifactType, ManifestBuilder};

    #[test]
    fn test_verify_artifact_ok() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.bin"), b"weights").unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();

        let manifest = ManifestBuilder::new("test", ArtifactType::MlModel)
            .add_directory(dir.path())
            .unwrap()
            .build()
            .unwrap();

        let result = verify_artifact(&manifest, dir.path()).unwrap();
        assert!(result.passed);
        assert_eq!(result.file_results.len(), 2);
    }

    #[test]
    fn test_verify_artifact_tampered() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.bin"), b"weights").unwrap();

        let manifest = ManifestBuilder::new("test", ArtifactType::MlModel)
            .add_directory(dir.path())
            .unwrap()
            .build()
            .unwrap();

        // Tamper with the file
        std::fs::write(dir.path().join("model.bin"), b"TAMPERED").unwrap();

        let result = verify_artifact(&manifest, dir.path()).unwrap();
        assert!(!result.passed);
    }

    #[test]
    fn test_verify_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.bin"), b"weights").unwrap();

        let manifest = ManifestBuilder::new("test", ArtifactType::MlModel)
            .add_directory(dir.path())
            .unwrap()
            .build()
            .unwrap();

        // Delete the file
        std::fs::remove_file(dir.path().join("model.bin")).unwrap();

        let result = verify_artifact(&manifest, dir.path()).unwrap();
        assert!(!result.passed);
        assert!(result.file_results.iter().any(|r| r.status == VerificationStatus::FileMissing));
    }
}
