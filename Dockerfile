FROM wasmedge/wasmedge:ubuntu-build-gcc-plugins-deps

WORKDIR /app/

ENV PATH="/root/.cargo/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y \
    build-essential \
    curl \
    clang \ 
    pkg-config \ 
    libavcodec-dev \ 
    libavformat-dev \ 
    libavfilter-dev \ 
    libavdevice-dev

RUN apt-get update

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s  -- -y

RUN rustup target add wasm32-wasi
COPY yolo-rs-wasm/ yolo-rs-wasm
COPY yolo-rs-video-plugin/ yolo-rs-video-plugin

RUN cd yolo-rs-video-plugin/ && cargo build --release
RUN cd yolo-rs-wasm/ && cargo build --target wasm32-wasi --release --examples

