use std::fs;

use clap::Parser;
use image::ImageFormat;
use yolo_rs::{process::draw_bounding_boxes_to_image, ConfThresh, IOUThresh, Yolo, YoloBuilder};

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
        .build_from_files([args.model_path])
        .unwrap();

    yolo.test_features();

}
