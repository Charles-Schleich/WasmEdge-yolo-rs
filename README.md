<div align="center">
  <h1><code>WasmEdge-Yolo-rs</code></h1>
  </p>
</div>

# A Rust library for Yolo tasks using WasmEdge + WASI-NN + FFMPEG 

> [!NOTE]  
> This library is a work in progress and is likely to change

### Requirements

##### WasmEdge
`wasmedge` built with the WASI-NN Plugin  
https://wasmedge.org/docs/start/install  

##### FFmpeg libraries
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


## Run Examples 
### Image Inference  
#### Build  
From directory `./yolo-rs-wasm`  
`cargo build --release`  
  
#### Run  
From project root `./`  
```bash
wasmedge --dir .:. ./target/wasm32-wasi/release/examples/image-inference.wasm \
    --model-path ./yolo-rs-wasm/example_inputs/yolov8n.torchscript \
    --image-path ./yolo-rs-wasm/example_inputs/busy_street.png \
    --class-names-path ./yolo-rs-wasm/example_inputs/class_names
```

### Video-inference example
#### Build  
From directory `./yolo-rs-wasm`  
`cargo build --release`  

From directory `./yolo-rs-video-plugin`  
`cargo build --release`  

#### Run  
From project root `./`
```bash
WASMEDGE_PLUGIN_PATH=./target/x86_64-unknown-linux-gnu/release/libyolo_rs_video_plugin.so   \
wasmedge --dir .:. \
  target/wasm32-wasi/release/examples/video-inference.wasm \
  --model-path ./yolo-rs-wasm/example_inputs/yolov8n.torchscript \
  --input-video-path ./yolo-rs-wasm/example_inputs/times_square.mp4 \
  --output-video-path ./yolo-rs-wasm/example_inputs/times_square_output.mp4 \
  --class-names-path ./yolo-rs-wasm/example_inputs/class_names
```