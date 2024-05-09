
# ███████ ███████ ███    ███ ██████  ███████  ██████  
# ██      ██      ████  ████ ██   ██ ██      ██       
# █████   █████   ██ ████ ██ ██████  █████   ██   ███ 
# ██      ██      ██  ██  ██ ██      ██      ██    ██ 
# ██      ██      ██      ██ ██      ███████  ██████  
FROM ubuntu:22.04 as ffmpeg

# RUN sed -i 's/htt[p|ps]:\/\/archive.ubuntu.com\/ubuntu\//mirror:\/\/mirrors.ubuntu.com\/mirrors.txt/g' /etc/apt/sources.list
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y build-essential wget curl clang pkg-config yasm  \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /ffmpeg/

RUN wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n7.0.tar.gz

RUN tar -xvzf n7.0.tar.gz 

WORKDIR /ffmpeg/FFmpeg-n7.0

RUN ./configure

RUN make -j10 all

RUN make install

################################################################################

FROM ghcr.io/webassembly/wasi-sdk as WASI_SDK

################################################################################

# ██    ██  ██████  ██       ██████        ██████  ███████ 
#  ██  ██  ██    ██ ██      ██    ██       ██   ██ ██      
#   ████   ██    ██ ██      ██    ██ █████ ██████  ███████ 
#    ██    ██    ██ ██      ██    ██       ██   ██      ██ 
#    ██     ██████  ███████  ██████        ██   ██ ███████ 

FROM rust:1.78 as yolo_rs_builder

WORKDIR /app/

# TODO DELETE - FOR DEBUGGING PURPOSES ONLY
RUN apt-get update && apt install fzf

# ONLY IF I NEED TO INSTALL WASM-SDK
# ENV WASI_VERSION=20
# ENV WASI_VERSION_FULL=${WASI_VERSION}.0
# RUN wget https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${WASI_VERSION}/wasi-sdk-${WASI_VERSION_FULL}-linux.tar.gz
# tar xvf wasi-sdk-${WASI_VERSION_FULL}-linux.tar.gz

# Install FFMPEG from earlier build.
COPY --from=ffmpeg /usr/local/lib /usr/local/lib
COPY --from=ffmpeg /usr/local/include /usr/local/include
COPY --from=ffmpeg /usr/local/share/ffmpeg /usr/local/share/ffmpeg

# Copy for wasi-emulated-process-clocks
RUN mkdir -p /opt/wasi-sdk/share/wasi-sysroot/include
RUN mkdir -p /opt/wasi-sdk/share/wasi-sysroot/lib
COPY --from=WASI_SDK /wasi-sysroot/lib /opt/wasi-sdk/share/wasi-sysroot/lib
COPY --from=WASI_SDK /wasi-sysroot/include /opt/wasi-sdk/share/wasi-sysroot/include
# Copy for clang_rt.builtins-wasm32
RUN mkdir -p /opt/wasi-sdk/lib/clang
RUN mkdir -p /opt/wasi-sdk/lib/llvm-17  
COPY --from=WASI_SDK /lib/clang /opt/wasi-sdk/lib/clang
COPY --from=WASI_SDK /lib/llvm-17 /opt/wasi-sdk/lib/llvm-17

# RUN apt-get upgrade -y
# RUN apt-get update

RUN rustup target add wasm32-wasi
COPY yolo-rs-wasm/ yolo-rs-wasm
COPY yolo-rs-video-plugin/ yolo-rs-video-plugin

# RUN cd yolo-rs-wasm/ && cargo build --target wasm32-wasi --release --examples
# RUN cd yolo-rs-video-plugin/ && cargo build --release

################################################################################

# Output WasmEdge 

# FROM wasmedge/wasmedge:ubuntu-build-gcc-plugins-deps

# SHELL [ "/bin/bash", "-c" ]
# ENV SHELL=/bin/bash

# ENV PATH="/root/.cargo/bin:${PATH}"

# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s  -- -y
# RUN ls
# RUN echo "$SHELL" 

# RUN source "$HOME/.cargo/env" 
# RUN cat "$HOME/.cargo/env" 

# FROM wasmedge/wasmedge:ubuntu-build-gcc-plugins-deps
