FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

RUN apt update && apt install -y curl

FROM rust:latest AS go-tools
RUN cargo build --features intel
RUN cargo test --features intel

WORKDIR /home
