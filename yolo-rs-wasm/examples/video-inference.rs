use std::fs;

use clap::Parser;
use yolo_rs::{ConfThresh, IOUThresh, Yolo, YoloBuilder};

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
    video_path: String,
}

pub fn main() {
    let args = Args::parse();
    println!("model_bin_name {}", args.model_path);
    println!("video_path {}", args.video_path);

    // Create YOLO instance
    let yolo: Yolo = YoloBuilder::new()
        .classes_file(args.class_names_path)
        .unwrap()
        .build_from_files([args.model_path])
        .unwrap();

    let conf_thresh = ConfThresh(0.5);
    let iou_thresh = IOUThresh(0.5);

    let video_results = yolo.infer_video(args.video_path, &conf_thresh, &iou_thresh).unwrap();
    
}
