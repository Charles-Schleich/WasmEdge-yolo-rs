## WasmEdge Yolo-rs Image + Video Processing 

> [!NOTE]  
> This library is a work in progress and is likely to change

This is a small plugin developed for the WasmEdge Runtime to support video processing of [yolo-rs](https://github.com/Charles-Schleich/yolo-rs)
Using FFMPEG for native video processing. 

Build `host_library` and `wasm_app` in separate terminals

#### To build:
Terminal 1:  
`cd yolo-rs-video-plugin && cargo build --release`
Terminal 2:  
`cd yolo-rs-wasm && cargo build --release`  

<!-- 
Quick build
cd wasm_app/ && cargo build --release && cd ../ && cd host_library/ && cargo build --release && cd .. 
-->

#### To run:
From project root  
`WASMEDGE_PLUGIN_PATH=/home/charles/we/yolo_ffmpeg_plugin/target   wasmedge  ./target/wasm32-wasi/release/wasm_app.wasm` 


TODO:
Mention that wasmedge is required.

