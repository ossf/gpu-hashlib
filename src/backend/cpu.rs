//! CPU fallback backend using the `ring` crate.
//! 

use super::{BackendInfo, GpuBackend, HashAlgorithm, HashOutput};
use crate::error::{HashResult};
use ring::digest;

/// CPU-based hashing backend using the `ring` crate.
pub struct CpuBackend {
    algorithm: HashAlgorithm,
}

impl CpuBackend {
    /// Create a new CPU backend for the given algorithm.
    pub fn new(algorithm: HashAlgorithm) -> Self {
        Self { algorithm }
    }

    fn ring_algorithm(&self) -> &'static digest::Algorithm {
        match self.algorithm {
            HashAlgorithm::Sha256 => &digest::SHA256,
            HashAlgorithm::Sha384 => &digest::SHA384,
            HashAlgorithm::Sha512 => &digest::SHA512,
        }
    }
}

impl GpuBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    fn hash(&self, data: &[u8]) -> HashResult<HashOutput> {
        let result = digest::digest(self.ring_algorithm(), data);
        Ok(HashOutput::new(
            result.as_ref().to_vec(),
            self.algorithm,
            "cpu",
        ))
    }

    fn hash_batch(&self, messages: &[Vec<u8>]) -> HashResult<Vec<HashOutput>> {
        let alg = self.ring_algorithm();
        Ok(messages
            .iter()
            .map(|m| {
                let result = digest::digest(alg, m);
                HashOutput::new(result.as_ref().to_vec(), self.algorithm, "cpu")
            })
            .collect())
    }

    fn is_gpu_accelerated(&self) -> bool {
        false
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "cpu".to_string(),
            vendor: "ring".to_string(),
            available: true,
            device_count: 1,
            devices: vec!["CPU (ring crate)".to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_sha256_known_vector() {
        let backend = CpuBackend::new(HashAlgorithm::Sha256);
        let result = backend.hash(b"").unwrap();
        assert_eq!(
            result.to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_cpu_batch() {
        let backend = CpuBackend::new(HashAlgorithm::Sha256);
        let messages = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let results = backend.hash_batch(&messages).unwrap();
        assert_eq!(results.len(), 3);
        // All should be different
        assert_ne!(results[0].to_hex(), results[1].to_hex());
    }

    #[test]
    fn test_cpu_not_gpu_accelerated() {
        let backend = CpuBackend::new(HashAlgorithm::Sha256);
        assert!(!backend.is_gpu_accelerated());
    }
}
