//! Unified error types for all GPU backends.

use std::fmt;

/// Result type for hash operations.
pub type HashResult<T> = Result<T, HashError>;

/// Errors that can occur during hashing operations.
#[derive(Debug)]
pub enum HashError {
    /// No compatible GPU device found (falls back to CPU).
    NoDeviceFound(String),

    /// Invalid hash algorithm specified.
    InvalidAlgorithm(String),

    /// GPU memory allocation failed.
    MemoryAllocation(String),

    /// GPU kernel execution failed.
    KernelExecution(String),

    /// Invalid input data.
    InvalidInput(String),

    /// Backend not initialized.
    NotInitialized(String),

    /// I/O error (file operations).
    Io(String),

    /// Signing/verification error.
    Signing(String),

    /// Backend-specific error.
    Backend {
        backend: String,
        message: String,
    },

    /// Generic error.
    Other(String),
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashError::NoDeviceFound(msg) => write!(f, "No GPU device found: {msg}"),
            HashError::InvalidAlgorithm(msg) => write!(f, "Invalid algorithm: {msg}"),
            HashError::MemoryAllocation(msg) => write!(f, "GPU memory allocation failed: {msg}"),
            HashError::KernelExecution(msg) => write!(f, "Kernel execution failed: {msg}"),
            HashError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            HashError::NotInitialized(msg) => write!(f, "Not initialized: {msg}"),
            HashError::Io(msg) => write!(f, "I/O error: {msg}"),
            HashError::Signing(msg) => write!(f, "Signing error: {msg}"),
            HashError::Backend { backend, message } => {
                write!(f, "[{backend}] {message}")
            }
            HashError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for HashError {}

impl From<std::io::Error> for HashError {
    fn from(e: std::io::Error) -> Self {
        HashError::Io(e.to_string())
    }
}

impl From<String> for HashError {
    fn from(s: String) -> Self {
        HashError::Other(s)
    }
}
