# **gpu-hashlib**

Generic multi-vendor GPU-accelerated hashing library for ML model signing and artifact verification.

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]()


## 
**Motivation**

ML model integrity is critical for supply chain security. Hashing large model artifacts (multi-GB weights, checkpoints) is CPU-bound and slow. GPU-accelerated hashing provides significant speedups for signing and verification workflows, enabling practical adoption of model signing in CI/CD pipelines.


## 
**Objective**

Provide a unified Rust API for GPU-accelerated cryptographic hashing across multiple GPU vendors (Intel, NVIDIA), with automatic fallback to CPU, designed for OpenSSF model signing and general artifact verification.


## 
**Scope**

**In scope:**
- GPU-accelerated SHA-256/384/512 hashing
- Intel Xe GPU backend via SYCL/oneAPI
- NVIDIA GPU backend via CUDA
- CPU fallback via ring/sha2
- OpenSSF model signing manifest generation and verification
- DSSE pre-authentication encoding

**Out of scope:**
- Key management and actual cryptographic signing (delegated to sigstore)
- Non-SHA hash algorithms
- GPU backends for other vendors (AMD, Apple Silicon)


## 
**Prior Work**

*   Intel atlas-c2pa-lib — SYCL SHA kernels for C2PA content authenticity


## 
**Active Projects**

### Architecture

```
                    +-----------------------+
                    |    Public API         |
                    |    (lib.rs)           |
                    |                       |
                    |  hash()  hash_batch() |
                    |  verify()  sign()     |
                    +-----------+-----------+
                                |
                    +-----------+-----------+
                    |    GpuBackend Trait    |
                    |    (backend/mod.rs)    |
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                 |                 |
    +---------+-------+ +------+--------+ +------+--------+
    |   Intel SYCL    | |  NVIDIA CUDA  | | CPU Fallback  |
    |   feature:      | |  feature:     | | (ring/sha2)   |
    |   "intel"       | |  "nvidia"     | | always on     |
    +-----------------+ +---------------+ +---------------+

    +-----------------------------------------------------+
    |          OpenSSF Model Signing  (signing/)           |
    +-----------------------------------------------------+
    |       Measurement & Benchmarking  (measure/)        |
    +-----------------------------------------------------+
```

### Features

| Feature | Description | Status |
|---|---|---|
| `intel` | Intel Xe GPU via SYCL/oneAPI | ✅ Full implementation |
| `nvidia` | NVIDIA GPU via CUDA | 🔲 Barebones (trait wired, kernels stubbed) |
| `openssf-signing` | OpenSSF model signing integration | ✅ Implemented |
| `measure` | Benchmarking & comparison utilities | ✅ Implemented |
| CPU fallback | ring-based software hashing | ✅ Always available |

### Project Structure

```
gpu-hashlib/
├── Cargo.toml
├── build.rs                          # Compiles SYCL/CUDA kernels
├── src/
│   ├── lib.rs                        # Public API
│   ├── error.rs                      # Unified error types
│   ├── backend/
│   │   ├── mod.rs                    # GpuBackend trait + shared types
│   │   ├── auto.rs                   # Automatic backend selection
│   │   ├── cpu.rs                    # CPU fallback (ring)
│   │   ├── intel/
│   │   │   ├── mod.rs                # Intel backend impl
│   │   │   ├── ffi.rs                # FFI to libgpu_hash_intel.so
│   │   │   ├── stub.c               # Stub when SYCL unavailable
│   │   │   └── sycl/
│   │   │       ├── gpu_hash.h        # C API header
│   │   │       └── gpu_hash.cpp      # SYCL SHA kernels
│   │   └── nvidia/
│   │       ├── mod.rs                # NVIDIA backend impl (barebones)
│   │       ├── ffi.rs                # FFI to libgpu_hash_nvidia.so
│   │       └── cuda/
│   │           ├── gpu_hash_cuda.h   # C API header
│   │           └── gpu_hash_cuda.cu  # CUDA stubs (TODO)
│   ├── signing/
│   │   ├── mod.rs                    # OpenSSF signing entry point
│   │   ├── digest.rs                 # Artifact digest computation
│   │   ├── manifest.rs               # DSSE manifest generation
│   │   └── verify.rs                 # Artifact verification
│   └── measure/
│       └── mod.rs                    # Benchmarking & comparison
├── benches/
│   └── gpu_hash_benchmark.rs
├── examples/
│   └── sign_model.rs
└── tests/
```

# 
**Get Involved**

*   Official communications occur on the [ADD LINK TO YOUR WG MAILING LIST] (ex: https://lists.openssf.org/g/openssf-tac/topics).  \
[Manage your subscriptions to Open SSF mailing lists](https://lists.openssf.org/g/main/subgroups).
*   [Add Slack information if availabable]

## 


### 
**Quick Start**

```toml
[dependencies]
gpu-hashlib = "0.1"

# With Intel GPU support
gpu-hashlib = { version = "0.1", features = ["intel"] }

# With all GPU backends
gpu-hashlib = { version = "0.1", features = ["gpu-all"] }
```

#### Basic Hashing

```rust
use gpu_hashlib::{hash, hash_batch, verify, HashAlgorithm};

// Automatic backend selection (GPU → CPU fallback)
let digest = hash(b"model weights", HashAlgorithm::Sha256)?;
println!("{}", digest.to_hex());

// Verify
assert!(verify(b"model weights", digest.as_bytes(), HashAlgorithm::Sha256)?);

// Batch hash (GPU-parallel)
let chunks = vec![b"chunk1".to_vec(), b"chunk2".to_vec()];
let digests = hash_batch(&chunks, HashAlgorithm::Sha256)?;
```

#### Building

**CPU only (no GPU features):**

```bash
cargo build
cargo test
```

**With Intel GPU:**

```bash
source /opt/intel/oneapi/setvars.sh
cargo build --features intel
cargo test --features intel
```

**With NVIDIA GPU (when implemented):**

```bash
cargo build --features nvidia
```

**All backends:**

```bash
source /opt/intel/oneapi/setvars.sh
cargo build --features gpu-all
```

*   Issues: File issues in this repository

## 
**Meeting times**

[TODO: Update with your WG meeting details]
*   Every other Tuesday @ 10:00am PST (Link to calendar invite)
*   [Meeting Minutes](https://docs.google.com/document/d/1uXQI1vI5_HyOvxHMexrnTY_ruBrynbPl5yOd1UM4g3A/edit#heading=h.yworp6sxzb6g)

# 
**Governance**

[TODO: Update this link to your specific CHARTER.md file]
The [CHARTER.md](CHARTER.md) outlines the scope and governance of our group activities.


[OPTIONAL]
*   Lead name 
*   Co-Lead name

#
**Intellectual Property**

In accordance with the [OpenSSF Charter (PDF)](https://charter.openssf.org/), work produced by this group is licensed as follows:

1. Software source code
* Apache License, Version 2.0, available at https://www.apache.org/licenses/LICENSE-2.0;
2. Data
* Any of the Community Data License Agreements, available at https://www.cdla.io;
3. Specifications
* Community Specification License, Version 1.0, available at https://github.com/CommunitySpecification/1.0
4. All other Documentation
* Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/

**Antitrust Policy Notice**

Linux Foundation meetings involve participation by industry competitors, and it is the intention of the Linux Foundation to conduct all of its activities in accordance with applicable antitrust and competition laws. It is therefore extremely important that attendees adhere to meeting agendas, and be aware of, and not participate in, any activities that are prohibited under applicable US state, federal or foreign antitrust and competition laws.

Examples of types of actions that are prohibited at Linux Foundation meetings and in connection with Linux Foundation activities are described in the Linux Foundation Antitrust Policy available at http://www.linuxfoundation.org/antitrust-policy. If you have questions about these matters, please contact your company counsel, or if you are a member of the Linux Foundation, feel free to contact Andrew Updegrove of the firm of Gesmer Updegrove LLP, which provides legal counsel to the Linux Foundation.
