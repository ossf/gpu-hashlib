.PHONY: build build-intel build-nvidia build-all test test-intel lint fmt clean

# ─── Default (CPU only) ──────────────────────────────────
build:
	cargo build

test:
	cargo test

# ─── Intel GPU ────────────────────────────────────────────
build-intel:
	cargo build --features intel

test-intel:
	cargo test --features intel

bench-intel:
	cargo bench --features intel

# ─── NVIDIA GPU ───────────────────────────────────────────
build-nvidia:
	cargo build --features nvidia

test-nvidia:
	cargo test --features nvidia

# ─── All GPU backends ─────────────────────────────────────
build-all:
	cargo build --features gpu-all

test-all:
	cargo test --features gpu-all

bench-all:
	cargo bench --features gpu-all

# ─── Quality ──────────────────────────────────────────────
lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

fmt-cpp:
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format -i src/backend/intel/sycl/gpu_hash.cpp \
		               src/backend/intel/sycl/gpu_hash.h \
		               src/backend/nvidia/cuda/gpu_hash_cuda.cu \
		               src/backend/nvidia/cuda/gpu_hash_cuda.h; \
	else echo "clang-format not found"; fi

fmt-all: fmt fmt-cpp

# ─── Examples ─────────────────────────────────────────────
example-sign:
	cargo run --example sign_model

# ─── Clean ────────────────────────────────────────────────
clean:
	cargo clean
