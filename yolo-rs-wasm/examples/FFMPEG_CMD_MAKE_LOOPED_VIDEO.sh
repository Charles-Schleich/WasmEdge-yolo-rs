ffmpeg -loop 1 \
    -framerate 1/10 \
    -i image.png \
    -i audio.mp3 \
    -c:v libx264 \
    -tune stillimage \
    -c:a aac \
    -r 10 \
    -b:a 192k \
    -pix_fmt yuv420p \
    -t 10 out.mp4

~/code/wasm_edge/FFmpeg_FRESH_INSTALL/ffmpeg/ffmpeg -loop 1 \
    -framerate 1/10 \
    -i dog.png \
    -c:v libx264 \
    -c:a aac \
    -r 10 \
    -b:a 192k \
    -pix_fmt yuv420p \
    -t 10 out.mp4

