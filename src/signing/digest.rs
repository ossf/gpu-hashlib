//! Artifact digest computation.
//!
//! Computes GPU-accelerated hashes of model files, datasets, and other
//! artifacts for use in OpenSSF model signing manifests.

use crate::backend::HashAlgorithm;
use crate::error::{HashError, HashResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Digest of a single artifact (file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactDigest {
    /// File path (relative to artifact root).
    pub path: PathBuf,
    /// Hash algorithm used.
    pub algorithm: HashAlgorithm,
    /// Hex-encoded digest.
    pub digest_hex: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

impl ArtifactDigest {
    /// The digest in the format expected by OpenSSF: "sha256:abcdef..."
    pub fn openssf_digest_string(&self) -> String {
        format!("{}:{}", self.algorithm.openssf_id(), self.digest_hex)
    }
}

/// Bundle of digests for a multi-file artifact (e.g., a model directory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigestBundle {
    /// Artifact name / identifier.
    pub name: String,
    /// Individual file digests.
    pub digests: Vec<ArtifactDigest>,
    /// Combined digest of all files (Merkle-like: hash of concatenated hashes).
    pub combined_digest: Option<ArtifactDigest>,
    /// Timestamp of digest computation.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Compute the digest of a single file.
pub fn compute_artifact_digest(
    path: &Path,
    algorithm: HashAlgorithm,
) -> HashResult<ArtifactDigest> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| HashError::Io(format!("Cannot stat {:?}: {}", path, e)))?;

    let hash_output = crate::hash_file(path, algorithm)?;

    Ok(ArtifactDigest {
        path: path.to_path_buf(),
        algorithm,
        digest_hex: hash_output.to_hex(),
        size_bytes: metadata.len(),
    })
}

/// Compute digests for all files in a directory (recursive).
pub fn compute_directory_digests(
    dir: &Path,
    algorithm: HashAlgorithm,
) -> HashResult<DigestBundle> {
    let mut file_paths = Vec::new();
    collect_files(dir, &mut file_paths)?;
    file_paths.sort(); // Deterministic ordering

    let mut digests = Vec::with_capacity(file_paths.len());

    // Read all files
    let mut file_data = Vec::with_capacity(file_paths.len());
    for path in &file_paths {
        let data = std::fs::read(path)
            .map_err(|e| HashError::Io(format!("Failed to read {:?}: {}", path, e)))?;
        file_data.push(data);
    }

    // Batch hash all files (GPU-accelerated)
    let hash_outputs = crate::hash_batch(&file_data, algorithm)?;

    for (path, (data, hash)) in file_paths
        .iter()
        .zip(file_data.iter().zip(hash_outputs.iter()))
    {
        let relative = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .to_path_buf();

        digests.push(ArtifactDigest {
            path: relative,
            algorithm,
            digest_hex: hash.to_hex(),
            size_bytes: data.len() as u64,
        });
    }

    // Compute combined digest (hash of all hex digests concatenated)
    let combined_input: Vec<u8> = digests
        .iter()
        .flat_map(|d| d.digest_hex.as_bytes().to_vec())
        .collect();
    let combined_hash = crate::hash(&combined_input, algorithm)?;

    let combined = ArtifactDigest {
        path: dir.to_path_buf(),
        algorithm,
        digest_hex: combined_hash.to_hex(),
        size_bytes: digests.iter().map(|d| d.size_bytes).sum(),
    };

    Ok(DigestBundle {
        name: dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "artifact".to_string()),
        digests,
        combined_digest: Some(combined),
        timestamp: chrono::Utc::now(),
    })
}

/// Recursively collect all files in a directory.
fn collect_files(dir: &Path, files: &mut Vec<PathBuf>) -> HashResult<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| HashError::Io(format!("Cannot read directory {:?}: {}", dir, e)))?;

    for entry in entries {
        let entry = entry.map_err(|e| HashError::Io(e.to_string()))?;
        let path = entry.path();

        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            collect_files(&path, files)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_compute_file_digest() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"model weights data").unwrap();

        let digest = compute_artifact_digest(&file_path, HashAlgorithm::Sha256).unwrap();
        assert_eq!(digest.size_bytes, 18);
        assert_eq!(digest.digest_hex.len(), 64);
        assert!(digest.openssf_digest_string().starts_with("sha256:"));
    }

    #[test]
    fn test_compute_directory_digests() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.bin"), b"weights").unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();

        let bundle =
            compute_directory_digests(dir.path(), HashAlgorithm::Sha256).unwrap();

        assert_eq!(bundle.digests.len(), 2);
        assert!(bundle.combined_digest.is_some());
    }
}
