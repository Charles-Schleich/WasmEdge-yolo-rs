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

### Optimizing the WASM Binary in the examples  
Rust outputs unoptimized wasm by default, wasmedge has an Ahead of Time compiler that can improve the performance significantly !

`wasmedge compile ./target/wasm32-wasi/release/examples/image-inference.wasm ./image-inference-optimized.wasm`

If you are running into performance issues with your WASM binary, consider this a first port of call for improving your runtime execution. 

You can read more on the AOT compiler here 
https://wasmedge.org/docs/start/build-and-run/aot/

## Examples 

### Image Inference  
##### Build  
From directory `./yolo-rs-wasm`  
`cargo build --release --example image-inference`  


##### Run  
From project root `./`  
```bash
wasmedge --dir .:. ./image-inference-optimized.wasm \
    --model-path ./yolo-rs-wasm/example_inputs/yolov8n.torchscript \
    --image-path ./yolo-rs-wasm/example_inputs/busy_street.png \
    --class-names-path ./yolo-rs-wasm/example_inputs/class_names
```
