cargo build --example image-inference --release --target wasm32-wasi

wasmedge compile ./target/wasm32-wasi/release/examples/image-inference.wasm \
    ./target/wasm32-wasi/release/examples/image-inference.wasm
