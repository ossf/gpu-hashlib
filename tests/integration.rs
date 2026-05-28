//! Integration tests for gpu-hashlib

use gpu_hashlib::{hash, hash_batch, hash_file, verify, list_backends, HashAlgorithm};
use std::io::Write;

#[test]
fn test_sha256_known_vectors() {
    // NIST test vector: empty string
    let d = hash(b"", HashAlgorithm::Sha256).unwrap();
    assert_eq!(d.to_hex(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    // "abc"
    let d = hash(b"abc", HashAlgorithm::Sha256).unwrap();
    assert_eq!(d.to_hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

#[test]
fn test_sha512_known_vector() {
    let d = hash(b"abc", HashAlgorithm::Sha512).unwrap();
    assert_eq!(
        d.to_hex(),
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
    );
}

#[test]
fn test_sha384_known_vector() {
    let d = hash(b"abc", HashAlgorithm::Sha384).unwrap();
    assert_eq!(
        d.to_hex(),
        "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7"
    );
}

#[test]
fn test_verify_roundtrip() {
    let data = b"The quick brown fox jumps over the lazy dog";
    let digest = hash(data, HashAlgorithm::Sha256).unwrap();
    assert!(verify(data, digest.as_bytes(), HashAlgorithm::Sha256).unwrap());
    assert!(!verify(b"wrong", digest.as_bytes(), HashAlgorithm::Sha256).unwrap());
}

#[test]
fn test_batch_consistency() {
    let messages: Vec<Vec<u8>> = (0..100).map(|i| format!("message-{i}").into_bytes()).collect();
    let batch = hash_batch(&messages, HashAlgorithm::Sha256).unwrap();

    // Each batch result should match individual hashing
    for (i, msg) in messages.iter().enumerate() {
        let single = hash(msg, HashAlgorithm::Sha256).unwrap();
        assert_eq!(batch[i].as_bytes(), single.as_bytes(), "Mismatch at index {i}");
    }
}

#[test]
fn test_hash_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.bin");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"file contents").unwrap();
    drop(f);

    let digest = hash_file(&path, HashAlgorithm::Sha256).unwrap();
    let expected = hash(b"file contents", HashAlgorithm::Sha256).unwrap();
    assert_eq!(digest.as_bytes(), expected.as_bytes());
}

#[test]
fn test_list_backends_has_cpu() {
    let backends = list_backends();
    assert!(backends.iter().any(|b| b.name == "cpu" && b.available));
}

#[test]
fn test_signing_workflow() {
    use gpu_hashlib::signing::{ArtifactType, ManifestBuilder, verify_artifact};

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("weights.bin"), b"model-weights-data").unwrap();
    std::fs::write(dir.path().join("config.json"), b"{\"layers\": 12}").unwrap();

    let manifest = ManifestBuilder::new("test-model", ArtifactType::MlModel)
        .algorithm(HashAlgorithm::Sha256)
        .metadata("version", "1.0")
        .add_directory(dir.path())
        .unwrap()
        .build()
        .unwrap();

    // Verify should pass
    let result = verify_artifact(&manifest, dir.path()).unwrap();
    assert!(result.passed);

    // Tamper → should fail
    std::fs::write(dir.path().join("weights.bin"), b"TAMPERED").unwrap();
    let result = verify_artifact(&manifest, dir.path()).unwrap();
    assert!(!result.passed);
}
