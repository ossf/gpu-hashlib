//! # gpu-hashlib — Generic Multi-Vendor GPU-Accelerated Hashing
//!
//! A Rust library providing GPU-accelerated cryptographic hashing with a unified
//! API across Intel Xe (SYCL/oneAPI) and NVIDIA (CUDA) GPUs. Designed for
//! OpenSSF model signing and general artifact verification.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    Public API (lib.rs)                    │
//! │  hash() · hash_batch() · verify() · measure() · sign()  │
//! ├──────────────────────────────────────────────────────────┤
//! │               GpuBackend Trait (backend/mod.rs)           │
//! ├──────────────┬───────────────┬───────────────────────────┤
//! │ Intel SYCL   │ NVIDIA CUDA   │  CPU Fallback (ring)      │
//! │ (feature:    │ (feature:     │  (always available)        │
//! │  "intel")    │  "nvidia")    │                            │
//! ├──────────────┴───────────────┴───────────────────────────┤
//! │         Signing (OpenSSF / sigstore / C2PA)              │
//! ├──────────────────────────────────────────────────────────┤
//! │         Measurement & Benchmarking Utilities             │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Feature Flags
//!
//! | Feature            | Description                                      |
//! |--------------------|--------------------------------------------------|
//! | `intel`            | Intel Xe GPU backend via SYCL/oneAPI              |
//! | `nvidia`           | NVIDIA GPU backend via CUDA                       |
//! | `gpu-all`          | Enable all GPU backends                           |
//! | `openssf-signing`  | OpenSSF model signing integration                 |
//! | `measure`          | Measurement and benchmarking utilities             |
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use gpu_hashlib::{HashAlgorithm, hash, hash_batch, verify};
//!
//! // Hash with automatic backend selection (GPU if available, else CPU)
//! let digest = hash(b"model weights data", HashAlgorithm::Sha256)?;
//!
//! // Verify a hash
//! assert!(verify(b"model weights data", &digest, HashAlgorithm::Sha256)?);
//!
//! // Batch hash (leverages GPU parallelism)
//! let chunks: Vec<Vec<u8>> = load_model_chunks("model.onnx");
//! let digests = hash_batch(&chunks, HashAlgorithm::Sha256)?;
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod backend;
pub mod error;
pub mod measure;
pub mod signing;

// Re-exports for convenience
pub use backend::{BackendInfo, GpuBackend, HashAlgorithm, HashOutput};
pub use error::{HashError, HashResult};

use backend::auto::AutoBackend;

// ─── Top-Level Convenience API ──────────────────────────────────────────────

/// Hash data using the best available backend (GPU → CPU fallback).
///
/// Automatically selects Intel GPU, NVIDIA GPU, or CPU in priority order.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_hashlib::{hash, HashAlgorithm};
///
/// let digest = hash(b"Hello, World!", HashAlgorithm::Sha256)?;
/// println!("{}", hex::encode(&digest.as_bytes()));
/// ```
pub fn hash(data: &[u8], algorithm: HashAlgorithm) -> HashResult<HashOutput> {
    let backend = AutoBackend::new(algorithm)?;
    backend.hash(data)
}

/// Hash multiple messages in parallel using the best available backend.
///
/// This is significantly faster than hashing one-by-one on GPUs.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_hashlib::{hash_batch, HashAlgorithm};
///
/// let messages = vec![b"chunk1".to_vec(), b"chunk2".to_vec()];
/// let digests = hash_batch(&messages, HashAlgorithm::Sha256)?;
/// ```
pub fn hash_batch(messages: &[Vec<u8>], algorithm: HashAlgorithm) -> HashResult<Vec<HashOutput>> {
    let backend = AutoBackend::new(algorithm)?;
    backend.hash_batch(messages)
}

/// Verify that data matches an expected hash.
///
/// # Example
///
/// ```rust,ignore
/// use gpu_hashlib::{hash, verify, HashAlgorithm};
///
/// let digest = hash(b"data", HashAlgorithm::Sha256)?;
/// assert!(verify(b"data", &digest.as_bytes(), HashAlgorithm::Sha256)?);
/// ```
pub fn verify(data: &[u8], expected: &[u8], algorithm: HashAlgorithm) -> HashResult<bool> {
    let digest = hash(data, algorithm)?;
    Ok(digest.as_bytes() == expected)
}

/// Hash a file from disk using the best available backend.
///
/// For large files, the data is read into memory and hashed.
/// Future versions may support streaming/chunked GPU hashing.
pub fn hash_file(
    path: &std::path::Path,
    algorithm: HashAlgorithm,
) -> HashResult<HashOutput> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| HashError::Io(format!("Failed to open {}: {}", path.display(), e)))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| HashError::Io(format!("Failed to read {}: {}", path.display(), e)))?;
    hash(&data, algorithm)
}

/// Hash multiple files in parallel.
pub fn hash_files(
    paths: &[&std::path::Path],
    algorithm: HashAlgorithm,
) -> HashResult<Vec<HashOutput>> {
    use std::io::Read;
    let mut messages = Vec::with_capacity(paths.len());
    for path in paths {
        let mut file = std::fs::File::open(path)
            .map_err(|e| HashError::Io(format!("Failed to open {:?}: {}", path, e)))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| HashError::Io(format!("Failed to read {:?}: {}", path, e)))?;
        messages.push(data);
    }
    hash_batch(&messages, algorithm)
}

/// List all available backends and their devices.
pub fn list_backends() -> Vec<BackendInfo> {
    let mut backends = vec![BackendInfo {
        name: "cpu".to_string(),
        vendor: "ring/sha2".to_string(),
        available: true,
        device_count: 1,
        devices: vec!["CPU (software fallback)".to_string()],
    }];

    #[cfg(feature = "intel")]
    {
        if let Ok(info) = backend::intel::IntelBackend::probe() {
            backends.push(info);
        }
    }

    #[cfg(feature = "nvidia")]
    {
        if let Ok(info) = backend::nvidia::NvidiaBackend::probe() {
            backends.push(info);
        }
    }

    backends
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_sha256() {
        let digest = hash(b"Hello, World!", HashAlgorithm::Sha256).unwrap();
        assert_eq!(digest.as_bytes().len(), 32);
    }

    #[test]
    fn test_hash_sha384() {
        let digest = hash(b"Hello, World!", HashAlgorithm::Sha384).unwrap();
        assert_eq!(digest.as_bytes().len(), 48);
    }

    #[test]
    fn test_hash_sha512() {
        let digest = hash(b"Hello, World!", HashAlgorithm::Sha512).unwrap();
        assert_eq!(digest.as_bytes().len(), 64);
    }

    #[test]
    fn test_verify() {
        let data = b"test data";
        let digest = hash(data, HashAlgorithm::Sha256).unwrap();
        assert!(verify(data, digest.as_bytes(), HashAlgorithm::Sha256).unwrap());
        assert!(!verify(b"wrong data", digest.as_bytes(), HashAlgorithm::Sha256).unwrap());
    }

    #[test]
    fn test_batch_hash() {
        let messages = vec![
            b"msg1".to_vec(),
            b"msg2".to_vec(),
            b"msg3".to_vec(),
        ];
        let digests = hash_batch(&messages, HashAlgorithm::Sha256).unwrap();
        assert_eq!(digests.len(), 3);
        for d in &digests {
            assert_eq!(d.as_bytes().len(), 32);
        }
    }

    #[test]
    fn test_known_vector_empty() {
        let digest = hash(b"", HashAlgorithm::Sha256).unwrap();
        assert_eq!(
            digest.to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_list_backends() {
        let backends = list_backends();
        assert!(!backends.is_empty());
        assert!(backends.iter().any(|b| b.name == "cpu"));
    }
}
