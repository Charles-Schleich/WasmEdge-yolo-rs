<div align="center">
  <h1><code>WasmEdge-Yolo-rs Video Plugin</code></h1>
  </p>
</div>

# A Rust Plugin for WasmEdge to support Video Processing of Yolo tasks

> [!NOTE]  
> This library is a work in progress and is likely to change

### Requirements

`wasmedge` built with the WASI-NN Plugin
https://wasmedge.org/docs/start/install 

### Optimizing the WASM Binary in the examples  
Rust outputs unoptimized wasm by default, wasmedge has an Ahead of Time compiler that can improve the performance significantly !

`wasmedge compile ./target/wasm32-wasi/release/examples/image-inference.wasm ./image-inference-optimized.wasm`

If you are running into performance issues with your WASM binary, consider this a first port of call for improving your runtime execution. 

You can read more on the AOT compiler here 
https://wasmedge.org/docs/start/build-and-run/aot/

## Examples 

### Video-inference example
#### Build  
From directory `./yolo-rs-wasm`  
`cargo build --release --example video-inference`  

From directory `./yolo-rs-video-plugin`  
`cargo build --release`  

#### Run  
From project root `./`
```bash
WASMEDGE_PLUGIN_PATH=./target/x86_64-unknown-linux-gnu/release/libyolo_rs_video_plugin.so   \
wasmedge --dir .:. \
  ./video-inference-optimized.wasm \
  --model-path ./yolo-rs-wasm/example_inputs/yolov8n.torchscript \
  --input-video-path ./yolo-rs-wasm/example_inputs/times_square.mp4 \
  --output-video-path ./yolo-rs-wasm/example_inputs/times_square_output.mp4 \
  --class-names-path ./yolo-rs-wasm/example_inputs/class_names
```
