//! # OpenSSF Model Signing Integration
//!
//! This module provides hash-based signing and verification for ML model
//! artifacts, compatible with the [OpenSSF Model Signing] specification.
//!
//! [OpenSSF Model Signing]: https://github.com/sigstore/model-transparency
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────┐
//! │           Model / Artifact Files          │
//! ├──────────────────────────────────────────┤
//! │      GPU-Accelerated Hash Digests        │  ← this library
//! ├──────────────────────────────────────────┤
//! │     Signing Manifest (DSSE envelope)      │
//! ├──────────────────────────────────────────┤
//! │   Sigstore / OIDC / Certificate Chain     │  ← optional sigstore dep
//! └──────────────────────────────────────────┘
//! ```
//!
//! ## Workflow
//!
//! 1. **Digest**: Compute hashes of model files using GPU acceleration
//! 2. **Manifest**: Build a signing manifest with file digests
//! 3. **Sign**: Sign the manifest (via sigstore or external PKI)
//! 4. **Verify**: Re-hash files and compare against signed manifest

pub mod digest;
pub mod manifest;
pub mod verify;

pub use digest::{ArtifactDigest, DigestBundle, compute_artifact_digest};
pub use manifest::{SigningManifest, ManifestEntry, ArtifactType};
pub use verify::{VerificationResult, VerificationStatus, verify_artifact};
