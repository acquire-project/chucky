# syntax=docker/dockerfile:1

# CUDA 12.8 required for SM 100 (Blackwell) target
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        libblosc-dev \
        libsnappy-dev \
        zlib1g-dev \
        libzstd-dev \
        liblz4-dev \
        libomp-dev \
        libssl-dev \
        git \
        wget \
        unzip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build AWS C libraries from source (not packaged in Ubuntu 24.04).
# Explicit &&-chain so any failure aborts the layer.
RUN git clone --depth 1 --branch v0.12.6 https://github.com/awslabs/aws-c-common.git      /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.9.13  https://github.com/awslabs/aws-c-cal.git          /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.2.10  https://github.com/awslabs/aws-checksums.git      /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v1.7.0   https://github.com/aws/s2n-tls.git               /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.26.3  https://github.com/awslabs/aws-c-io.git           /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.3.2   https://github.com/awslabs/aws-c-compression.git  /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.10.12 https://github.com/awslabs/aws-c-http.git         /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.2.4   https://github.com/awslabs/aws-c-sdkutils.git     /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.10.1  https://github.com/awslabs/aws-c-auth.git         /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b \
 && git clone --depth 1 --branch v0.11.5  https://github.com/awslabs/aws-c-s3.git           /tmp/b && cmake -S /tmp/b -B /tmp/b/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/aws -DCMAKE_PREFIX_PATH=/opt/aws -DBUILD_TESTING=OFF && cmake --build /tmp/b/build --target install && rm -rf /tmp/b

# AWS CLI for S3 integration tests (standalone, no Python needed)
RUN wget -qO /tmp/awscliv2.zip \
        "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

RUN wget -qO /tmp/nvcomp.tar.xz \
        "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz" \
    && mkdir -p /opt/nvcomp \
    && tar -xJf /tmp/nvcomp.tar.xz -C /opt/nvcomp --strip-components=1 \
    && rm /tmp/nvcomp.tar.xz

ENV CMAKE_PREFIX_PATH="/opt/nvcomp:/opt/aws"

WORKDIR /src
COPY . .

RUN cmake --preset docker -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build
