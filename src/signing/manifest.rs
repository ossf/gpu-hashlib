//! Signing Manifest for OpenSSF Model Signing
//!
//! Generates and parses signing manifests compatible with the OpenSSF
//! Model Signing specification (DSSE envelope format).
//!
//! Reference: https://github.com/sigstore/model-transparency

use crate::backend::HashAlgorithm;
use crate::error::{HashError, HashResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Type of artifact being signed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    /// ML model (ONNX, PyTorch, TensorFlow, etc.)
    MlModel,
    /// Training dataset
    Dataset,
    /// Model configuration
    Config,
    /// Container image
    ContainerImage,
    /// Generic artifact
    Generic,
    /// Custom type
    Custom(String),
}

impl std::fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactType::MlModel => write!(f, "ml_model"),
            ArtifactType::Dataset => write!(f, "dataset"),
            ArtifactType::Config => write!(f, "config"),
            ArtifactType::ContainerImage => write!(f, "container_image"),
            ArtifactType::Generic => write!(f, "generic"),
            ArtifactType::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// A single entry in the signing manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    /// Relative file path.
    pub path: PathBuf,
    /// Hash algorithm.
    pub algorithm: HashAlgorithm,
    /// Hex-encoded digest.
    pub digest: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

impl ManifestEntry {
    /// OpenSSF-style digest string: "sha256:abcdef..."
    pub fn digest_string(&self) -> String {
        format!("{}:{}", self.algorithm.openssf_id(), self.digest)
    }
}

/// Signing manifest — the document that gets signed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigningManifest {
    /// Manifest format version.
    pub version: String,

    /// Artifact name / identifier.
    pub artifact_name: String,

    /// Artifact type.
    pub artifact_type: ArtifactType,

    /// Individual file entries.
    pub entries: Vec<ManifestEntry>,

    /// Combined digest of all files (hash of concatenated per-file digests).
    pub combined_digest: Option<String>,

    /// Hash algorithm used for all entries.
    pub algorithm: HashAlgorithm,

    /// GPU backend used for hashing.
    pub hashing_backend: String,

    /// Timestamp of manifest creation.
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Optional metadata.
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl SigningManifest {
    /// Create a manifest from a digest bundle.
    pub fn from_digest_bundle(
        bundle: &super::digest::DigestBundle,
        artifact_type: ArtifactType,
    ) -> Self {
        let entries: Vec<ManifestEntry> = bundle
            .digests
            .iter()
            .map(|d| ManifestEntry {
                path: d.path.clone(),
                algorithm: d.algorithm,
                digest: d.digest_hex.clone(),
                size_bytes: d.size_bytes,
            })
            .collect();

        let combined = bundle
            .combined_digest
            .as_ref()
            .map(|d| d.openssf_digest_string());

        let algorithm = bundle
            .digests
            .first()
            .map(|d| d.algorithm)
            .unwrap_or(HashAlgorithm::Sha256);

        Self {
            version: "0.1.0".to_string(),
            artifact_name: bundle.name.clone(),
            artifact_type,
            entries,
            combined_digest: combined,
            algorithm,
            hashing_backend: "auto".to_string(),
            created_at: bundle.timestamp,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> HashResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| HashError::Signing(format!("Failed to serialize manifest: {e}")))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> HashResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| HashError::Signing(format!("Failed to parse manifest: {e}")))
    }

    /// The payload bytes that should be signed (canonical JSON).
    pub fn signable_payload(&self) -> HashResult<Vec<u8>> {
        let json = self.to_json()?;
        Ok(json.into_bytes())
    }

    /// DSSE envelope payload type.
    pub fn payload_type() -> &'static str {
        "application/vnd.openssf.model-signing.manifest+json;version=0.1"
    }

    /// Build a DSSE pre-authentication encoding (PAE).
    ///
    /// PAE(type, payload) = "DSSEv1" + SP + len(type) + SP + type + SP + len(payload) + SP + payload
    pub fn dsse_pae(&self) -> HashResult<Vec<u8>> {
        let payload = self.signable_payload()?;
        let payload_type = Self::payload_type();

        let mut pae = Vec::new();
        pae.extend_from_slice(b"DSSEv1 ");
        pae.extend_from_slice(payload_type.len().to_string().as_bytes());
        pae.push(b' ');
        pae.extend_from_slice(payload_type.as_bytes());
        pae.push(b' ');
        pae.extend_from_slice(payload.len().to_string().as_bytes());
        pae.push(b' ');
        pae.extend_from_slice(&payload);

        Ok(pae)
    }

    /// Add custom metadata.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Builder for constructing signing manifests step by step.
pub struct ManifestBuilder {
    artifact_name: String,
    artifact_type: ArtifactType,
    algorithm: HashAlgorithm,
    entries: Vec<ManifestEntry>,
    metadata: std::collections::HashMap<String, String>,
}

impl ManifestBuilder {
    /// Create a new manifest builder.
    pub fn new(name: &str, artifact_type: ArtifactType) -> Self {
        Self {
            artifact_name: name.to_string(),
            artifact_type,
            algorithm: HashAlgorithm::Sha256,
            entries: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the hash algorithm.
    pub fn algorithm(mut self, algorithm: HashAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Add a file to the manifest by path (hashes it).
    pub fn add_file(mut self, path: &std::path::Path) -> HashResult<Self> {
        let digest = super::digest::compute_artifact_digest(path, self.algorithm)?;
        self.entries.push(ManifestEntry {
            path: digest.path,
            algorithm: digest.algorithm,
            digest: digest.digest_hex,
            size_bytes: digest.size_bytes,
        });
        Ok(self)
    }

    /// Add a directory recursively.
    pub fn add_directory(mut self, dir: &std::path::Path) -> HashResult<Self> {
        let bundle =
            super::digest::compute_directory_digests(dir, self.algorithm)?;
        for d in &bundle.digests {
            self.entries.push(ManifestEntry {
                path: d.path.clone(),
                algorithm: d.algorithm,
                digest: d.digest_hex.clone(),
                size_bytes: d.size_bytes,
            });
        }
        Ok(self)
    }

    /// Add metadata.
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Build the manifest.
    pub fn build(self) -> HashResult<SigningManifest> {
        // Compute combined digest
        let combined_input: Vec<u8> = self
            .entries
            .iter()
            .flat_map(|e| e.digest.as_bytes().to_vec())
            .collect();
        let combined = crate::hash(&combined_input, self.algorithm)?;

        Ok(SigningManifest {
            version: "0.1.0".to_string(),
            artifact_name: self.artifact_name,
            artifact_type: self.artifact_type,
            entries: self.entries,
            combined_digest: Some(format!(
                "{}:{}",
                self.algorithm.openssf_id(),
                combined.to_hex()
            )),
            algorithm: self.algorithm,
            hashing_backend: "auto".to_string(),
            created_at: chrono::Utc::now(),
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_builder() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.bin"), b"weights").unwrap();

        let manifest = ManifestBuilder::new("test-model", ArtifactType::MlModel)
            .algorithm(HashAlgorithm::Sha256)
            .metadata("framework", "pytorch")
            .add_directory(dir.path())
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(manifest.artifact_name, "test-model");
        assert_eq!(manifest.entries.len(), 1);
        assert!(manifest.combined_digest.is_some());

        // Round-trip JSON
        let json = manifest.to_json().unwrap();
        let parsed = SigningManifest::from_json(&json).unwrap();
        assert_eq!(parsed.artifact_name, "test-model");
    }

    #[test]
    fn test_dsse_pae() {
        let manifest = SigningManifest {
            version: "0.1.0".to_string(),
            artifact_name: "test".to_string(),
            artifact_type: ArtifactType::Generic,
            entries: vec![],
            combined_digest: None,
            algorithm: HashAlgorithm::Sha256,
            hashing_backend: "cpu".to_string(),
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
        };

        let pae = manifest.dsse_pae().unwrap();
        assert!(pae.starts_with(b"DSSEv1 "));
    }
}
