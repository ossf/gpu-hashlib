//! # Measurement & Verification Utilities
//!
//! Functions for measuring hash performance, verifying correctness across
//! backends, and producing machine-readable measurement reports.

use crate::backend::{BackendInfo, GpuBackend, HashAlgorithm, HashOutput};
use crate::error::{HashError, HashResult};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Result of a single hash measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashMeasurement {
    /// Algorithm used.
    pub algorithm: HashAlgorithm,
    /// Backend name.
    pub backend: String,
    /// Input size in bytes.
    pub input_bytes: usize,
    /// Number of messages (1 for single hash).
    pub message_count: usize,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Throughput in bytes/second.
    pub throughput_bytes_per_sec: f64,
    /// Throughput in messages/second.
    pub throughput_msgs_per_sec: f64,
    /// The hash output (hex).
    pub hash_hex: String,
}

/// Summary comparing multiple backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Algorithm tested.
    pub algorithm: HashAlgorithm,
    /// Input description.
    pub input_description: String,
    /// Per-backend measurements.
    pub measurements: Vec<HashMeasurement>,
    /// Whether all backends produced identical hashes.
    pub hashes_match: bool,
    /// Speedup of fastest GPU over CPU (if applicable).
    pub gpu_speedup: Option<f64>,
}

/// Measure a single hash operation.
pub fn measure_hash(
    backend: &dyn GpuBackend,
    data: &[u8],
) -> HashResult<HashMeasurement> {
    let start = Instant::now();
    let output = backend.hash(data)?;
    let duration = start.elapsed();

    let throughput = if duration.as_secs_f64() > 0.0 {
        data.len() as f64 / duration.as_secs_f64()
    } else {
        f64::INFINITY
    };

    Ok(HashMeasurement {
        algorithm: backend.algorithm(),
        backend: backend.name().to_string(),
        input_bytes: data.len(),
        message_count: 1,
        duration,
        throughput_bytes_per_sec: throughput,
        throughput_msgs_per_sec: 1.0 / duration.as_secs_f64(),
        hash_hex: output.to_hex(),
    })
}

/// Measure a batch hash operation.
pub fn measure_batch(
    backend: &dyn GpuBackend,
    messages: &[Vec<u8>],
) -> HashResult<HashMeasurement> {
    let total_bytes: usize = messages.iter().map(|m| m.len()).sum();

    let start = Instant::now();
    let outputs = backend.hash_batch(messages)?;
    let duration = start.elapsed();

    let throughput = if duration.as_secs_f64() > 0.0 {
        total_bytes as f64 / duration.as_secs_f64()
    } else {
        f64::INFINITY
    };

    let first_hex = outputs
        .first()
        .map(|o| o.to_hex())
        .unwrap_or_default();

    Ok(HashMeasurement {
        algorithm: backend.algorithm(),
        backend: backend.name().to_string(),
        input_bytes: total_bytes,
        message_count: messages.len(),
        duration,
        throughput_bytes_per_sec: throughput,
        throughput_msgs_per_sec: messages.len() as f64 / duration.as_secs_f64(),
        hash_hex: first_hex,
    })
}

/// Measure with multiple iterations and return the median.
pub fn measure_hash_median(
    backend: &dyn GpuBackend,
    data: &[u8],
    iterations: usize,
) -> HashResult<HashMeasurement> {
    let mut measurements = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        measurements.push(measure_hash(backend, data)?);
    }

    measurements.sort_by(|a, b| a.duration.partial_cmp(&b.duration).unwrap());

    Ok(measurements.swap_remove(measurements.len() / 2))
}

/// Verify that a backend produces correct SHA output by comparing to CPU.
pub fn verify_backend_correctness(
    backend: &dyn GpuBackend,
    test_vectors: &[&[u8]],
) -> HashResult<bool> {
    let cpu = crate::backend::cpu::CpuBackend::new(backend.algorithm());

    for data in test_vectors {
        let cpu_hash = cpu.hash(data)?;
        let gpu_hash = backend.hash(data)?;

        if cpu_hash.as_bytes() != gpu_hash.as_bytes() {
            log::error!(
                "Hash mismatch for {} bytes: CPU={} GPU={}",
                data.len(),
                cpu_hash.to_hex(),
                gpu_hash.to_hex()
            );
            return Ok(false);
        }
    }

    Ok(true)
}

/// Compare all available backends on the same input.
pub fn compare_backends(
    data: &[u8],
    algorithm: HashAlgorithm,
) -> HashResult<ComparisonReport> {
    let mut measurements = Vec::new();

    // Always measure CPU
    let cpu = crate::backend::cpu::CpuBackend::new(algorithm);
    measurements.push(measure_hash(&cpu, data)?);

    // Measure Intel if available
    #[cfg(feature = "intel")]
    {
        if let Ok(intel) = crate::backend::intel::IntelBackend::new(algorithm) {
            measurements.push(measure_hash(&intel, data)?);
        }
    }

    // Measure NVIDIA if available
    #[cfg(feature = "nvidia")]
    {
        if let Ok(nvidia) = crate::backend::nvidia::NvidiaBackend::new(algorithm) {
            measurements.push(measure_hash(&nvidia, data)?);
        }
    }

    // Check all hashes match
    let hashes_match = measurements
        .windows(2)
        .all(|w| w[0].hash_hex == w[1].hash_hex);

    // Compute GPU speedup over CPU
    let cpu_duration = measurements
        .iter()
        .find(|m| m.backend == "cpu")
        .map(|m| m.duration);

    let fastest_gpu = measurements
        .iter()
        .filter(|m| m.backend != "cpu")
        .min_by(|a, b| a.duration.partial_cmp(&b.duration).unwrap());

    let gpu_speedup = match (cpu_duration, fastest_gpu) {
        (Some(cpu), Some(gpu)) if gpu.duration.as_secs_f64() > 0.0 => {
            Some(cpu.as_secs_f64() / gpu.duration.as_secs_f64())
        }
        _ => None,
    };

    Ok(ComparisonReport {
        algorithm,
        input_description: format!("{} bytes", data.len()),
        measurements,
        hashes_match,
        gpu_speedup,
    })
}

/// Generate a JSON measurement report.
#[cfg(feature = "measure")]
pub fn report_json(report: &ComparisonReport) -> HashResult<String> {
    serde_json::to_string_pretty(report)
        .map_err(|e| HashError::Other(format!("JSON serialization failed: {e}")))
}

/// Standard SHA test vectors for correctness verification.
pub fn standard_test_vectors() -> Vec<Vec<u8>> {
    vec![
        b"".to_vec(),
        b"abc".to_vec(),
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".to_vec(),
        vec![0x61; 1_000_000], // 1M 'a' characters
        vec![0xAB; 4096],      // 4KB pattern
        vec![0x00; 65536],     // 64KB zeros
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    #[test]
    fn test_measure_hash() {
        let backend = CpuBackend::new(HashAlgorithm::Sha256);
        let m = measure_hash(&backend, b"Hello").unwrap();
        assert_eq!(m.input_bytes, 5);
        assert_eq!(m.message_count, 1);
        assert!(!m.hash_hex.is_empty());
    }

    #[test]
    fn test_verify_cpu_correctness() {
        let backend = CpuBackend::new(HashAlgorithm::Sha256);
        let vectors: Vec<&[u8]> = vec![b"", b"abc", b"test"];
        assert!(verify_backend_correctness(&backend, &vectors).unwrap());
    }

    #[test]
    fn test_compare_backends() {
        let report = compare_backends(b"test data", HashAlgorithm::Sha256).unwrap();
        assert!(report.hashes_match);
        assert!(!report.measurements.is_empty());
    }
}
