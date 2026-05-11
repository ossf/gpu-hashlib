use gpu_hashlib::backend::auto::AutoBackend;
use gpu_hashlib::backend::cpu::CpuBackend;
use gpu_hashlib::backend::GpuBackend;
use gpu_hashlib::HashAlgorithm;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_single_cpu_vs_gpu(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096, 65536, 1048576];
    let mut group = c.benchmark_group("single_hash");

    for &size in &sizes {
        let data = vec![0xABu8; size];

        group.bench_with_input(BenchmarkId::new("cpu", size), &data, |b, d| {
            let backend = CpuBackend::new(HashAlgorithm::Sha256);
            b.iter(|| backend.hash(black_box(d)))
        });

        group.bench_with_input(BenchmarkId::new("auto", size), &data, |b, d| {
            let backend = AutoBackend::new(HashAlgorithm::Sha256).unwrap();
            b.iter(|| backend.hash(black_box(d)))
        });
    }

    group.finish();
}

fn bench_batch_cpu_vs_gpu(c: &mut Criterion) {
    let batch_sizes = [10, 50, 100, 500, 1000, 5000];
    let msg_size = 4096;
    let mut group = c.benchmark_group("batch_hash");

    for &count in &batch_sizes {
        let messages: Vec<Vec<u8>> = (0..count).map(|_| vec![0xCDu8; msg_size]).collect();

        group.bench_with_input(BenchmarkId::new("cpu", count), &messages, |b, m| {
            let backend = CpuBackend::new(HashAlgorithm::Sha256);
            b.iter(|| backend.hash_batch(black_box(m)))
        });

        group.bench_with_input(BenchmarkId::new("auto", count), &messages, |b, m| {
            let backend = AutoBackend::new(HashAlgorithm::Sha256).unwrap();
            b.iter(|| backend.hash_batch(black_box(m)))
        });
    }

    group.finish();
}

fn bench_algorithms(c: &mut Criterion) {
    let data = vec![0xEFu8; 65536];
    let mut group = c.benchmark_group("algorithms");

    for alg in &[HashAlgorithm::Sha256, HashAlgorithm::Sha384, HashAlgorithm::Sha512] {
        group.bench_with_input(BenchmarkId::new("auto", format!("{}", alg)), &data, |b, d| {
            let backend = AutoBackend::new(*alg).unwrap();
            b.iter(|| backend.hash(black_box(d)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_single_cpu_vs_gpu, bench_batch_cpu_vs_gpu, bench_algorithms);
criterion_main!(benches);
