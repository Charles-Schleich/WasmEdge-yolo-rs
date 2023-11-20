use std::fs;

use clap::Parser;
use image::ImageFormat;
use yolo_rs::{process::draw_bounding_boxes_on_mut_image, ConfThresh, IOUThresh, Yolo, YoloBuilder};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path to model file, with encoding
    #[arg(short, long)]
    model_path: String,

    /// classnames path
    #[arg(short, long)]
    class_names_path: String,

    /// image path
    #[arg(short, long)]
    image_path: String,
}

pub fn main() {
    let args = Args::parse();
    println!("model_bin_name {}", args.model_path);
    println!("image_name {}", args.image_path);

    // Create YOLO instance
    let yolo: Yolo = YoloBuilder::new()
        .classes_file(args.class_names_path)
        .unwrap()
        // .execution_target(wasi_nn::ExecutionTarget::GPU)
        .build_from_files([args.model_path])
        .unwrap();

    let conf_thresh = ConfThresh(0.5);
    let iou_thresh = IOUThresh(0.5);

    // Load in the image
    let image_bytes = fs::read(args.image_path).unwrap();
    let rgb_image = image::load_from_memory(&image_bytes).unwrap().to_rgb8();

    // Run inference
    // for i in 0..100 {
    // let vec_result = yolo.infer(&conf_thresh, &iou_thresh, &rgb_image).unwrap();
    // println!("{:?}", vec_result);
    // }

    let vec_result = yolo.infer(&conf_thresh, &iou_thresh, &rgb_image).unwrap();
    println!("Detection Results {:?}",vec_result); 
    let output_image = draw_bounding_boxes_on_mut_image(rgb_image, &vec_result, &yolo.font());

    output_image
        .save_with_format("output.png", ImageFormat::Png)
        .unwrap();
}
