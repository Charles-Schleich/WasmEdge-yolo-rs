FROM wasmedge/wasmedge:ubuntu-build-gcc-plugins-deps

WORKDIR /app/

COPY yolo-rs-wasm/ yolo-rs-wasm
COPY yolo-rs-video-plugin/ yolo-rs-video-plugin