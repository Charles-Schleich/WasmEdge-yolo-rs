set -e 

cargo build --examples --release --target wasm32-wasi

echo "Optimizing image inference..."
wasmedge compile ./target/wasm32-wasi/release/examples/image-inference.wasm \
    ./target/wasm32-wasi/release/examples/image-inference.wasm
echo "... Done"

echo "Optimizing image inference..."
wasmedge compile ./target/wasm32-wasi/release/examples/image-inference.wasm \
    ./target/wasm32-wasi/release/examples/image-inference.wasm
echo "... Done"
