<div align="center">
  <h1><code>YOLO-rs</code></h1>
  </p>
</div>

# A Rust library for YOLO tasks for WasmEdge WASI-NN

> [!NOTE]  
> This library is a work in progress and is likely to change


#### FFmpeg requirements
This library relies on ffmpeg for video processing, 

```bash
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    clang \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    pkg-config
```

### Build + Run Examples 

Build examples using the script (run from project root `/`)
`./scripts/build-and-optimize-examples.sh`

Run Examples (run from project root `/`)
`./scripts/run-example.sh`

```bash
wasmedge --dir .:. ./target/wasm32-wasi/release/examples/image-inference.wasm \
    --model-path ./yolov8n.torchscript \
    --image-path examples/dog.png \
    --class-names-path examples/class_names
```

#### Video-inference example
```bash
WASMEDGE_PLUGIN_PATH=./target/x86_64-unknown-linux-gnu/release/libyolo_rs_video_plugin.so \
  wasmedge --dir .:. \
  target/wasm32-wasi/release/examples/video-inference.wasm \
    --model-path ./yolov8n.torchscript \
    --video-path examples/dog.png \
    --class-names-path examples/class_names
```


WASMEDGE_PLUGIN_PATH=./target/x86_64-unknown-linux-gnu/release/libyolo_rs_video_plugin.so   \
wasmedge --dir .:. \
  target/wasm32-wasi/release/examples/video-inference.wasm \
  --model-path ./yolo-rs-wasm/yolov8n.torchscript \
  --video-path ./yolo-rs-wasm/720p30.mp4 \
  --class-names-path ./yolo-rs-wasm/examples/class_names


WASMEDGE_PLUGIN_PATH=./target/x86_64-unknown-linux-gnu/release/libyolo_rs_video_plugin.so   \
wasmedge --dir .:. \
  target/wasm32-wasi/release/examples/video-inference.wasm \
  --model-path ./yolo-rs-wasm/yolov8n.torchscript \
  --video-path ./yolo-rs-wasm/small_bunny_1080p_60fps.mp4 \
  --class-names-path ./yolo-rs-wasm/examples/class_names
  