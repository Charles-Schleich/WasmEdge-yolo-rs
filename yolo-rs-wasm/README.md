<div align="center">
  <h1><code>WasmEdge-Yolo-rs WASM Library</code></h1>
  </p>
</div>

# A Rust library for YOLO tasks using WasmEdge WASI-NN

> [!NOTE]  
> This library is a work in progress and is likely to change

### Requirements

`wasmedge` built with the WASI-NN Plugin  
https://wasmedge.org/docs/start/install  

## Examples 

#### Image Inference  
###### Build  
From directory `./yolo-rs-wasm`  
`cargo build --release`  
  
From project root `./`  
```bash
wasmedge --dir .:. ./target/wasm32-wasi/release/examples/image-inference.wasm \
    --model-path ./yolo-rs-wasm/example_inputs/yolov8n.torchscript \
    --image-path ./yolo-rs-wasm/example_inputs/busy_street.png \
    --class-names-path ./yolo-rs-wasm/example_inputs/class_names
```
