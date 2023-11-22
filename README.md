<div align="center">
  <h1><code>WasmEdge-Yolo-rs</code></h1>
  </p>
</div>

## A Rust library for Yolo tasks using WasmEdge + WASI-NN + FFMPEG 

> [!NOTE]  
> This library is a work in progress and is likely to change

This repo is split into 2 parts: 
- `yolo-rs-wasm` : A library to process images and videos in WasmEdge, using trained YoloV8 weights
- `yolo-rs-video-plugin` : A WasmEdge Plugin to load videos, split out frames, pass frames `yolo-rs-wasm`, and re-assemble frames with detection information drawn onto the frame

### Requirements

##### WasmEdge
`wasmedge` built with the WASI-NN Plugin  
https://wasmedge.org/docs/start/install  
https://wasmedge.org/docs/start/install/#wasi-nn-plug-in-with-pytorch-backend  


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


### Optimizing the WASM Binary in the examples  
Rust outputs unoptimized wasm by default, wasmedge has an Ahead of Time compiler that can improve the performance significantly !

From project root `./`  
`wasmedge compile ./target/wasm32-wasi/release/examples/image-inference.wasm ./image-inference-optimized.wasm`

If you are running into performance issues with your WASM binary, consider this a first port of call for improving your runtime execution. 

You can read more on the AOT compiler here 
https://wasmedge.org/docs/start/build-and-run/aot/


## Run Examples 
### Image Inference  example
#### Build  
From directory `./yolo-rs-wasm`  
`cargo build --release --example image-inference`  

#### Run  
From project root `./`  
```bash
wasmedge --dir .:. ./image-inference-optimized.wasm \
    --model-path ./yolo-rs-wasm/example_inputs/yolov8n.torchscript \
    --image-path ./yolo-rs-wasm/example_inputs/busy_street.png \
    --class-names-path ./yolo-rs-wasm/example_inputs/class_names
```

Note: We are running the optimized WASM binary in the command above, if you wish to run the unoptimized binary for comparison please use the path `./target/wasm32-wasi/release/examples/image-inference.wasm`


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


#### Limitations / TODO's
- This library and plugin are intended to be used with the WasmEdge runtime only.
- The WasmEdge Runtime requires the Wasi-NN plugin to be built (See Requirements above)
- In its current form the video plugin does not copy over the Audio Stream from the original video 
- In its current form the video plugin re-encodes each frame as I-Frames, as at the time of writing, the FFMPEG encoder was not setting DTS (Decoding Time Stamp) and (PTS) Presentation Time Stamp, and no short term alternative currently (This is a WIP to fix)
This results in much larger output video, as the plugin effectively encodes 
To slightly compensate, the bitrate is reduced significantly to save on file size. Quality is barely effected from uncompressed video
- By default, Hardcoded Attempts to Decode / Encode H264 video, this is a trivial change to fix to support other codecs
 
###### References 
- Video Compression Picture Types (I vs B vs P Frames) : https://en.wikipedia.org/wiki/Video_compression_picture_types
