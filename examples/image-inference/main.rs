use clap::Parser;
use std::env;
use yolo_rs::{Yolo, YoloBuilder};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// model path
    #[arg(short, long)]
    model_path: String,

    /// image path
    #[arg(short, long)]
    image_path: String,
}

pub fn main() {
    let args = Args::parse();

    // let args: Vec<String> = env::args().collect();
    // let model_bin_name: &str = &args[1];
    // let image_name: &str = &args[2];

    println!("model_bin_name {}", args.model_path);
    println!("image_name {}", args.image_path);

    let yolo: Yolo = YoloBuilder::new()
        .build_from_files([args.model_path])
        .unwrap();

    let x = yolo.infer_file("./dog.png");

    // for result in results {
    //     if result.probability > 0.5 {
    //         println!("{:?}", result);
    //     }
    // }
}
