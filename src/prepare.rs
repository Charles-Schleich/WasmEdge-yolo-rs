use image::{GenericImage, RgbImage};

// Function to normalize and resize image for YOLO
const SIZE: usize = 640;
const SIZE_U32: u32 = 640;
type Channel = Vec<Vec<f32>>;

pub(crate) fn pre_process_image(image: RgbImage) -> [Channel; 3] {
    // let mut file_img = File::open(path).unwrap();
    // let mut img_buf = Vec::new();
    // file_img.read_to_end(&mut img_buf).unwrap();
    // let img = image::load_from_memory(&img_buf).unwrap().to_rgb8();

    let (height, width);
    if image.width() > image.height() {
        // height is the shorter length
        height = SIZE_U32 * image.height() / image.width();
        width = SIZE_U32;
    } else {
        // width is the shorter length
        width = SIZE_U32 * image.width() / image.height();
        height = SIZE_U32;
    }

    let resized: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> = image::imageops::resize(
        &image,
        width,
        height,
        ::image::imageops::FilterType::Triangle,
    );

    // We need the image to fit the 640 x 640 size,
    // and we want to keep the aspect ratio of the original image
    // So we fill the remaining pixels with black,
    let mut resized_640x640 = RgbImage::new(SIZE_U32, SIZE_U32);
    resized_640x640.copy_from(&resized, 0, 0).unwrap();

    // Split intoChannels
    let mut red: Channel = vec![vec![0.0; SIZE]; SIZE];
    let mut blue: Channel = vec![vec![0.0; SIZE]; SIZE];
    let mut green: Channel = vec![vec![0.0; SIZE]; SIZE];

    for (_, pixel) in resized_640x640.enumerate_rows() {
        for (x, y, rgb) in pixel {
            let x = x as usize;
            let y = y as usize;

            red[y][x] = rgb.0[0] as f32 / 255.0;
            green[y][x] = rgb.0[1] as f32 / 255.0;
            blue[y][x] = rgb.0[2] as f32 / 255.0;
        }
    }

    let final_tensor: [Vec<Vec<f32>>; 3];
    final_tensor = [red, green, blue];

    final_tensor
}
