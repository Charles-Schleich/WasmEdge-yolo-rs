use wasi_nn::{Error, ExecutionTarget, Graph, GraphEncoding};

pub struct Yolo {
    // inference_type: YoloType
    graph: Graph,
    classes: Vec<String>,
}

pub struct InferenceResult;

impl Yolo {
    pub fn new(graph: Graph, classes: Vec<String>) -> Self {
        Yolo { graph, classes }
    }

    pub fn infer() -> InferenceResult {
        InferenceResult
    }
}

#[derive(Debug)]
pub enum BuildError {
    MissingClasses,
    GraphError(Error),
}

impl From<Error> for BuildError {
    fn from(value: Error) -> Self {
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
