
# ███████ ███████ ███    ███ ██████  ███████  ██████  
# ██      ██      ████  ████ ██   ██ ██      ██       
# █████   █████   ██ ████ ██ ██████  █████   ██   ███ 
# ██      ██      ██  ██  ██ ██      ██      ██    ██ 
# ██      ██      ██      ██ ██      ███████  ██████  
FROM ubuntu:22.04 as ffmpeg

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

# ██     ██  █████  ███████ ██     ███████ ██████  ██   ██ 
# ██     ██ ██   ██ ██      ██     ██      ██   ██ ██  ██  
# ██  █  ██ ███████ ███████ ██     ███████ ██   ██ █████   
# ██ ███ ██ ██   ██      ██ ██          ██ ██   ██ ██  ██  
#  ███ ███  ██   ██ ███████ ██     ███████ ██████  ██   ██ 

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
RUN apt-get update && apt install -y fzf clang

# Install FFMPEG from earlier build
COPY --from=ffmpeg /usr/local/lib /usr/local/lib
COPY --from=ffmpeg /usr/local/include /usr/local/include
COPY --from=ffmpeg /usr/local/share/ffmpeg /usr/local/share/ffmpeg

# Copy for wasi-emulated-process-clocks
RUN mkdir -p /opt/wasi-sdk/share/wasi-sysroot/include
RUN mkdir -p /opt/wasi-sdk/share/wasi-sysroot/lib
COPY --from=WASI_SDK /wasi-sysroot/lib /opt/wasi-sdk/share/wasi-sysroot/lib
COPY --from=WASI_SDK /wasi-sysroot/include /opt/wasi-sdk/share/wasi-sysroot/include

# Copy for clang_rt.builtins-wasm32
RUN mkdir -p /lib/clang
RUN mkdir -p /lib/llvm-17  
COPY --from=WASI_SDK /lib/clang /lib/clang
COPY --from=WASI_SDK /lib/llvm-17 /lib/llvm-17

# Add wasm32-wasi target to toolchain
RUN rustup target add wasm32-wasi

# Copy video plugin wasm directory
COPY yolo-rs-wasm/ yolo-rs-wasm
COPY yolo-rs-video-plugin/ yolo-rs-video-plugin

# Build Plugin and build WASM
RUN cd yolo-rs-wasm/ && cargo build --target wasm32-wasi --release --examples
RUN cd yolo-rs-video-plugin/ && cargo build --release

################################################################################