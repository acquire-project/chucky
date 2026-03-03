# syntax=docker/dockerfile:1

# CUDA 12.8 required for SM 100 (Blackwell) target
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        libzstd-dev \
        libomp-dev \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/nvcomp.tar.xz \
        "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz" \
    && mkdir -p /opt/nvcomp \
    && tar -xJf /tmp/nvcomp.tar.xz -C /opt/nvcomp --strip-components=1 \
    && rm /tmp/nvcomp.tar.xz

ENV CMAKE_PREFIX_PATH="/opt/nvcomp"

WORKDIR /src
COPY . .

RUN cmake --preset docker -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build
