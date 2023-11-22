use clap::Parser;
use log::LevelFilter;
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode};
use yolo_rs::{utils, ConfThresh, DrawBoundingBoxes, IOUThresh, Yolo, YoloBuilder};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path to model file, with encoding
    #[arg(short, long)]
    model_path: String,

    /// classnames path
    #[arg(short, long)]
    class_names_path: String,

    /// video_path
    #[arg(short, long)]
    input_video_path: String,

    /// video_path
    #[arg(short, long)]
    output_video_path: String,
}

pub fn main() {
    let args = Args::parse();
    println!("model_bin_name {}", args.model_path);
    println!("input_video_path {}", args.input_video_path);
    println!("output_video_path {}", args.output_video_path);

    CombinedLogger::init(vec![TermLogger::new(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Always,
    )])
    .unwrap();

    // Create YOLO instance
    let yolo: Yolo = YoloBuilder::new()
        .classes_file(args.class_names_path)
        .unwrap()
        .build_from_files([args.model_path])
        .unwrap();

    let conf_thresh = ConfThresh(0.5);
    let iou_thresh = IOUThresh(0.5);

    let video_results = yolo
        .infer_video(
            args.input_video_path,
            args.output_video_path,
            &conf_thresh,
            &iou_thresh,
            DrawBoundingBoxes::TrueWithFont(
                utils::load_font("./yolo-rs-wasm/assets/ClearSans-Medium.ttf").unwrap(),
            ),
        )
        .unwrap();

    for (idx, result) in video_results.iter().enumerate() {
        println!("Inference Results for Frame {}, {:?}", idx, result);
    }
}
