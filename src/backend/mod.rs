//! # GPU Backend Abstraction Layer
//!
//! This module defines the `GpuBackend` trait — the generic interface that
//! all vendor-specific backends (Intel, NVIDIA, CPU) implement.
//!
//! ## Design Principles
//!
//! 1. **Trait-based dispatch**: All backends implement `GpuBackend`.
//! 2. **Automatic selection**: `AutoBackend` picks the best available GPU.
//! 3. **CPU fallback**: Always works, even without any GPU.
//! 4. **Zero-cost when unused**: Vendor code is behind feature flags.

pub mod auto;
pub mod cpu;

#[cfg(feature = "intel")]
#[cfg_attr(docsrs, doc(cfg(feature = "intel")))]
pub mod intel;

#[cfg(feature = "nvidia")]
#[cfg_attr(docsrs, doc(cfg(feature = "nvidia")))]
pub mod nvidia;

use crate::error::HashResult;
use serde::{Deserialize, Serialize};

// ─── Shared Types ───────────────────────────────────────────────────────────

/// Supported hash algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// SHA-256 (32-byte output)
    Sha256,
    /// SHA-384 (48-byte output)
    Sha384,
    /// SHA-512 (64-byte output)
    Sha512,
}

impl HashAlgorithm {
    /// Output size in bytes.
    pub fn output_size(&self) -> usize {
        match self {
            HashAlgorithm::Sha256 => 32,
            HashAlgorithm::Sha384 => 48,
            HashAlgorithm::Sha512 => 64,
        }
    }

    /// Block size in bytes.
    pub fn block_size(&self) -> usize {
        match self {
            HashAlgorithm::Sha256 => 64,
            HashAlgorithm::Sha384 | HashAlgorithm::Sha512 => 128,
        }
    }

    /// OID string for this algorithm (used in signing).
    pub fn oid(&self) -> &'static str {
        match self {
            HashAlgorithm::Sha256 => "2.16.840.1.101.3.4.2.1",
            HashAlgorithm::Sha384 => "2.16.840.1.101.3.4.2.2",
            HashAlgorithm::Sha512 => "2.16.840.1.101.3.4.2.3",
        }
    }

    /// OpenSSF model signing algorithm identifier.
    pub fn openssf_id(&self) -> &'static str {
        match self {
            HashAlgorithm::Sha256 => "sha256",
            HashAlgorithm::Sha384 => "sha384",
            HashAlgorithm::Sha512 => "sha512",
        }
    }
}

impl std::str::FromStr for HashAlgorithm {
    type Err = crate::error::HashError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sha256" | "sha-256" => Ok(HashAlgorithm::Sha256),
            "sha384" | "sha-384" => Ok(HashAlgorithm::Sha384),
            "sha512" | "sha-512" => Ok(HashAlgorithm::Sha512),
            _ => Err(crate::error::HashError::InvalidAlgorithm(format!(
                "Unknown algorithm: {s}. Supported: sha256, sha384, sha512"
            ))),
        }
    }
}

impl std::fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashAlgorithm::Sha256 => write!(f, "SHA-256"),
            HashAlgorithm::Sha384 => write!(f, "SHA-384"),
            HashAlgorithm::Sha512 => write!(f, "SHA-512"),
        }
    }
}

/// Output from a hash operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashOutput {
    /// Raw hash bytes.
    bytes: Vec<u8>,
    /// Algorithm used.
    algorithm: HashAlgorithm,
    /// Which backend produced this hash.
    backend_name: String,
}

impl HashOutput {
    /// Create a new hash output.
    pub fn new(bytes: Vec<u8>, algorithm: HashAlgorithm, backend: &str) -> Self {
        Self {
            bytes,
            algorithm,
            backend_name: backend.to_string(),
        }
    }

    /// Get the hash bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Hex-encoded hash.
    pub fn to_hex(&self) -> String {
        hex::encode(&self.bytes)
    }

    /// Algorithm used.
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// Which backend produced this.
    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    /// Consume into raw bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

impl AsRef<[u8]> for HashOutput {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl std::fmt::Display for HashOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl PartialEq for HashOutput {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.algorithm == other.algorithm
    }
}

/// Information about a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// Backend name (e.g., "intel", "nvidia", "cpu").
    pub name: String,
    /// Vendor name.
    pub vendor: String,
    /// Whether this backend is available on the current system.
    pub available: bool,
    /// Number of devices.
    pub device_count: usize,
    /// Device names.
    pub devices: Vec<String>,
}

// ─── The Core Trait ─────────────────────────────────────────────────────────

/// Generic GPU hashing backend.
///
/// Every vendor backend (Intel SYCL, NVIDIA CUDA, CPU fallback) implements
/// this trait. The `AutoBackend` picks the best one at runtime.
///
/// # Implementors
///
/// - `CpuBackend` — always available, uses `ring`/`sha2`
/// - `IntelBackend` — requires `intel` feature and Intel GPU
/// - `NvidiaBackend` — requires `nvidia` feature and NVIDIA GPU
pub trait GpuBackend: Send + Sync {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// The hash algorithm this instance is configured for.
    fn algorithm(&self) -> HashAlgorithm;

    /// Hash a single message.
    fn hash(&self, data: &[u8]) -> HashResult<HashOutput>;

    /// Hash multiple messages in parallel.
    ///
    /// Default implementation falls back to sequential hashing.
    fn hash_batch(&self, messages: &[Vec<u8>]) -> HashResult<Vec<HashOutput>> {
        messages.iter().map(|m| self.hash(m)).collect()
    }

    /// Hash contiguous fixed-size messages (most efficient for GPU).
    ///
    /// `data` must be `message_size * num_messages` bytes long.
    fn hash_batch_fixed(
        &self,
        data: &[u8],
        message_size: usize,
    ) -> HashResult<Vec<HashOutput>> {
        if message_size == 0 {
            return Err(crate::error::HashError::InvalidInput(
                "message_size cannot be zero".into(),
            ));
        }
        if data.len() % message_size != 0 {
            return Err(crate::error::HashError::InvalidInput(format!(
                "Data length {} not divisible by message_size {}",
                data.len(),
                message_size
            )));
        }
        let messages: Vec<Vec<u8>> = data.chunks(message_size).map(|c| c.to_vec()).collect();
        self.hash_batch(&messages)
    }

    /// Whether this backend is using a real GPU (not CPU fallback).
    fn is_gpu_accelerated(&self) -> bool;

    /// Backend info for diagnostics.
    fn info(&self) -> BackendInfo;
}
