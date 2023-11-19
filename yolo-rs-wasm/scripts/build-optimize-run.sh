cargo build --example image-inference --release --target wasm32-wasi

wasmedge compile \
    ./target/wasm32-wasi/release/examples/image-inference.wasm \
    ./target/wasm32-wasi/release/examples/image-inference.wasm

wasmedge --dir .:. ./target/wasm32-wasi/release/examples/image-inference.wasm \
    --model-path ./yolov8n.torchscript \
    --image-path examples/dog.png \
    --class-names-path examples/class_names