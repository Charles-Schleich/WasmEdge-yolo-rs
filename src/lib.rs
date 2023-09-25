use std::{fs, io, path::Path};

use image::GenericImage;
use wasi_nn::{ExecutionTarget, Graph, GraphEncoding};
mod prepare;

pub struct Yolo {
    // inference_type: YoloType
    graph: Graph,
    classes: Vec<String>,
}

pub struct InferenceResult;

#[derive(thiserror::Error, Debug)]
pub enum RuntimeError {
    #[error("image error")]
    ImageError(#[from] image::ImageError),

    #[error("file error")]
    FileError(#[from] io::Error),

    #[error("graph error")]
    GraphError(#[from] wasi_nn::Error),
}

const INPUT_WIDTH: usize = 640;
const INPUT_HEIGHT: usize = 640;
const OUTPUT_CLASSES: usize = 80;
const OUTPUT_OBJECTS: usize = 8400;

impl Yolo {
    /// Creates a new instance of YOLO, including graph and classes
    pub fn new(graph: Graph, classes: Vec<String>) -> Self {
        Yolo { graph, classes }
    }

    pub fn infer_file<P: AsRef<Path>>(
        self,
        image_path: P,
    ) -> Result<InferenceResult, RuntimeError>
// where
        // P: AsRef<std::path::PathBuf>,
    {
        let path = image_path.as_ref();
        let image_bytes = fs::read(path)?;
        let image_buffer = image::load_from_memory(&image_bytes).unwrap().to_rgb8();

        let bytes: [Vec<Vec<f32>>; 3] = prepare::pre_process_image(image_buffer);
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

        Ok(InferenceResult)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error("Classes must be added to the YoloBuilder before building")]
    MissingClasses,

    #[error("error creating Graph")]
    GraphError(wasi_nn::Error),
}

impl From<wasi_nn::Error> for BuildError {
    fn from(value: wasi_nn::Error) -> Self {
        BuildError::GraphError(value)
    }
}

pub enum YoloType {
    Pose,
    Segment,
    Detection,
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

    #[inline(always)]
    pub fn build_from_bytes<B>(self, bytes_array: impl AsRef<[B]>) -> Result<Yolo, BuildError>
    where
        B: AsRef<[u8]>,
    {
        match self.classes {
            Some(classes) => {
                let graph = wasi_nn::GraphBuilder::new(self.graph_encoding, self.execution_target)
                    .build_from_bytes(bytes_array)?;

                Ok(Yolo::new(graph, classes))
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
                Ok(Yolo::new(graph, classes))
            }
            None => Err(BuildError::MissingClasses),
        }
    }
}
