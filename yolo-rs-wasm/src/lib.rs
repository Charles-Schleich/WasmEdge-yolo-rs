use image::{GenericImage, ImageBuffer, Rgb, RgbImage};
use imageproc::rect::Rect;
use log::{debug, info, LevelFilter};
use prepare::ResizeScale;
use prgrs::Prgrs;
use process::{
    apply_confidence_and_scale, non_maximum_supression, process_output_buffer_to_tensor,
};
use rusttype::Font;
use std::{fs, io, path::Path};
use wasi_nn::{ExecutionTarget, Graph, GraphEncoding};

use crate::video_proc::yolo_rs_video_plugin;
mod prepare;
pub mod process;
mod video_proc;

pub struct Yolo {
    // inference_type: YoloType
    font: Font<'static>,
    graph: Graph,
    classes: Vec<String>,
}
pub enum YoloType {
    Pose,
    Segment,
    Detection,
}
#[derive(thiserror::Error, Debug)]
pub enum RuntimeError {
    #[error("image error")]
    ImageError(#[from] image::ImageError),

    #[error("file error")]
    FileError(#[from] io::Error),

    #[error("graph error")]
    GraphError(#[from] wasi_nn::Error),

    #[error("result processing error")]
    PostProcessingError(#[from] PostProcessingError),
}

#[derive(thiserror::Error, Debug)]
pub enum PostProcessingError {
    #[error("Cannot broadcast to Array dimension")]
    BroadcastArrayDims,
}

const INPUT_WIDTH: usize = 640;
const INPUT_HEIGHT: usize = 640;
const OUTPUT_CLASSES: usize = 80;
const OUTPUT_OBJECTS: usize = 8400;

pub struct ConfThresh(pub f32);

impl ConfThresh {
    pub fn new(conf: f32) -> Self {
        Self(conf)
    }
}

pub struct IOUThresh(pub f32);

impl IOUThresh {
    pub fn new(conf: f32) -> Self {
        Self(conf)
    }
}

impl Yolo {
    /// Creates a new instance of YOLO, including graph and classes
    pub fn new(graph: Graph, classes: Vec<String>, font: Font<'static>) -> Self {
        Yolo {
            graph,
            classes,
            font,
        }
    }

    // Convienence function to run load file and poarse as image
    fn load_image_from_file<P: AsRef<Path>>(image_path: P) -> Result<RgbImage, RuntimeError> {
        let path = image_path.as_ref();
        let image_bytes = fs::read(path)?;
        Ok(image::load_from_memory(&image_bytes)?.to_rgb8())
    }

    pub fn font(self) -> Font<'static> {
        self.font
    }

    pub fn infer_file<P: AsRef<Path>>(
        self,
        image_path: P,
        conf_thresh: &ConfThresh,
        iou_thresh: &IOUThresh,
    ) -> Result<Vec<InferenceResult>, RuntimeError> {
        let image_buffer = Yolo::load_image_from_file(image_path)?;
        self.infer(conf_thresh, iou_thresh, &image_buffer)
    }

    #[cfg(not(feature = "pure-rust"))]
    pub fn test_features(&self) {
        println!("Test Default")
    }

    #[cfg(feature = "pure-rust")]
    pub fn test_features(&self) {
        println!("Test pure rust")
    }

    pub fn infer_video<P: AsRef<Path>>(
        self,
        video_path: P,
        conf_thresh: &ConfThresh,
        iou_thresh: &IOUThresh,
    ) -> Result<Vec<InferenceResult>, RuntimeError> {
        self.process_video(video_path, conf_thresh, iou_thresh);
        Ok(Vec::new())
    }

    fn process_video<P: AsRef<Path>>(
        self,
        video_path: P,
        conf_thresh: &ConfThresh,
        iou_thresh: &IOUThresh,
    ) -> Result<(), ()> {
        debug!("Start Proc Video");

        // TODO: Remove Unwrap
        let mut filename = video_path.as_ref().to_str().unwrap().to_string();

        yolo_rs_video_plugin::init_plugin_logging_with_log_level(LevelFilter::Info);

        let (mut width, mut height, mut frame_count): (i32, i32, i32) = (0, 0, 10);
        let width_ptr = std::ptr::addr_of_mut!(width);
        let height_ptr = std::ptr::addr_of_mut!(height);
        let frame_count_ptr = std::ptr::addr_of_mut!(frame_count);

        debug!("Call load_video_to_host_memory() ");
        let result = unsafe {
            yolo_rs_video_plugin::load_video_to_host_memory(
                filename.as_mut_ptr() as usize as i32,
                filename.len() as i32,
                filename.capacity() as i32,
                width_ptr,
                height_ptr,
                frame_count_ptr,
            )
        };

        let image_buf_size: usize = (width * height * 3) as usize;
        debug!("WIDTH {}", width);
        debug!("HEIGHT {}", height);
        debug!("Number of Frames {}", frame_count);

        info!("Begin Processing {} frames ", frame_count);
        
        for idx in 0..frame_count {
            // for idx in Prgrs::new(0..frame_count, frame_count as usize) {

            println!("Process Frame {idx}");
            debug!("------ Run for frame {}", idx);
            let mut image_buf: Vec<u8> = vec![0; image_buf_size];

            let buf_ptr_raw = image_buf.as_mut_ptr() as usize as i32;
            let buf_len = image_buf.len() as i32;
            let buf_capacity = image_buf.capacity() as i32;
            debug!("WASM image_buf_ptr {:?}", buf_ptr_raw);
            debug!("WASM image_buf_len {:?}", buf_len);
            debug!("WASM image_buf_capacity {:?}", buf_capacity);

            {
                unsafe {
                    yolo_rs_video_plugin::get_frame(idx, buf_ptr_raw, buf_len, buf_capacity)
                };

                let mut image_buf: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                    ImageBuffer::from_vec(width as u32, height as u32, image_buf).unwrap();
               println!("Before inference {idx}");
               println!("{} {}", image_buf.width(), image_buf.height());

                // TODO: Remove Unwrap
                let vec_results: Vec<InferenceResult> = self.infer(
                    conf_thresh,
                    iou_thresh,
                    &image_buf
                ).unwrap();

               println!("After Inference {idx}");
               let image_after_drawing = process::draw_bounding_boxes_to_image(
                    image_buf,
                    vec_results,
                    &self.font
                );

                unsafe { yolo_rs_video_plugin::write_frame(idx, buf_ptr_raw, buf_len) };
            }
        }

        info!("Finished Writing {:?} Frames To Plugin", frame_count);

        let mut out: Vec<&str> = filename.split(".").collect::<Vec<&str>>();
        out.insert(0, "./");
        out.insert(out.len() - 1, "_out.");
        let mut output_filename = out.join("");

        info!("Begin Encode Video {:?}", output_filename);
        let output_code = unsafe {
            yolo_rs_video_plugin::assemble_output_frames_to_video(
                output_filename.as_mut_ptr() as usize as i32,
                output_filename.len() as i32,
                output_filename.capacity() as i32,
            )
        };

        info!("Finished Encoding Video : {}", output_filename);

        Ok(())
    }

    pub fn infer(
        &self,
        conf_thresh: &ConfThresh,
        iou_thresh: &IOUThresh,
        image_buffer: &RgbImage,
    ) -> Result<Vec<InferenceResult>, RuntimeError> {
        let (bytes, resize_scale): ([Vec<Vec<f32>>; 3], ResizeScale) =
            prepare::pre_process_image(image_buffer);

        let tensor_data = bytes
            .into_iter()
            .flatten()
            .collect::<Vec<Vec<f32>>>()
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>();

        let mut context = self.graph.init_execution_context().unwrap();
        context
            .set_input(
                0,
                wasi_nn::TensorType::F32,
                // Input
                &[1, 3, self::INPUT_WIDTH, INPUT_HEIGHT],
                &tensor_data,
            )
            .unwrap();

        let mut output_buffer = vec![0f32; 1 * OUTPUT_OBJECTS * OUTPUT_CLASSES];

        // Execute the inference.
        context.compute().unwrap();
        context.get_output(0, &mut output_buffer)?;

        // Process inference results into Vector of Results
        let output_tensor = process_output_buffer_to_tensor(&output_buffer);

        let vec_results =
            apply_confidence_and_scale(output_tensor, conf_thresh, &self.classes, resize_scale);
        if vec_results.len()==0{
            return OK(vec_results);
         }
        let vec_results = non_maximum_supression(iou_thresh, vec_results)?;

        Ok(vec_results)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error("Classes must be added to the YoloBuilder before building")]
    MissingClasses,

    #[error("error creating Graph")]
    GraphError(#[from] wasi_nn::Error),

    #[error("error reading file containing classes")]
    FileError(#[from] std::io::Error),
}

pub struct YoloBuilder {
    // inference_type: YoloType
    graph_encoding: GraphEncoding,
    execution_target: ExecutionTarget,
    classes: Option<Vec<String>>,
}

impl YoloBuilder {
    #[inline(always)]
    pub fn new() -> Self {
        YoloBuilder {
            graph_encoding: GraphEncoding::Pytorch,
            execution_target: ExecutionTarget::CPU,
            classes: None,
        }
    }

    #[inline(always)]
    pub fn execution_target(mut self, execution_target: ExecutionTarget) -> Self {
        self.execution_target = execution_target;
        self
    }

    #[inline(always)]
    pub fn graph_encoding(mut self, graph_encoding: GraphEncoding) -> Self {
        self.graph_encoding = graph_encoding;
        self
    }

    #[inline(always)]
    pub fn classes(mut self, classes: Vec<String>) -> Self {
        self.classes = Some(classes);
        self
    }

    /// Function that takes a path to a classes text file,
    /// each class is separated by a newline
    #[inline(always)]
    pub fn classes_file<P>(mut self, classes_path: P) -> Result<Self, BuildError>
    where
        P: AsRef<std::path::Path>,
    {
        let path = classes_path.as_ref();
        let classes = fs::read_to_string(path)?
            .lines()
            .map(String::from)
            .collect::<Vec<String>>();

        self.classes = Some(classes);
        Ok(self)
    }

    // TODO remove unwrap
    fn load_font() -> Font<'static> {
        let font_data: &[u8] = include_bytes!("../assets/ClearSans-Medium.ttf");
        Font::try_from_bytes(font_data).unwrap()
    }

    #[inline(always)]
    pub fn build_from_bytes<B>(self, bytes_array: impl AsRef<[B]>) -> Result<Yolo, BuildError>
    where
        B: AsRef<[u8]>,
    {
        match self.classes {
            Some(classes) => {
                let graph = wasi_nn::GraphBuilder::new(self.graph_encoding, self.execution_target)
                    .build_from_bytes(bytes_array)?;

                Ok(Yolo::new(graph, classes, YoloBuilder::load_font()))
            }
            None => Err(BuildError::MissingClasses),
        }
    }

    #[inline(always)]
    pub fn build_from_files<P>(self, files: impl AsRef<[P]>) -> Result<Yolo, BuildError>
    where
        P: AsRef<std::path::Path>,
    {
        match self.classes {
            Some(classes) => {
                let graph = wasi_nn::GraphBuilder::new(self.graph_encoding, self.execution_target)
                    .build_from_files(files)?;
                Ok(Yolo::new(graph, classes, YoloBuilder::load_font()))
            }
            None => Err(BuildError::MissingClasses),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceResult {
    b_box: Rect,
    class: String,
    confidence: f32,
}
