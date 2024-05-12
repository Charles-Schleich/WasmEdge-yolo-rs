use std::sync::{Arc, Mutex};

mod decode_video;
mod encode_video;
mod time;

use ffmpeg_next::{
    dictionary,
    format::Pixel,
    frame,
    picture::{self},
    Codec, Rational,
};

use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode};
use wasmedge_sdk::{
    error::HostFuncError,
    host_function,
    plugin::{ffi, PluginDescriptor, PluginModuleBuilder, PluginVersion},
    Caller, Memory, NeverType, WasmValue,
};

use std::fmt::Debug;

use log::{debug, error, LevelFilter};

#[derive(Debug, Copy, Clone)]
pub struct Width(pub u32);
#[derive(Debug, Copy, Clone)]
pub struct Height(pub u32);
#[derive(Debug, Copy, Clone)]
pub struct AspectRatio(pub Rational);
#[derive(Debug, Copy, Clone)]
pub struct FrameRate(pub Option<Rational>);

#[derive(Debug, Copy, Clone)]
pub struct BitRate(pub usize);
#[derive(Debug, Copy, Clone)]
pub struct MaxBitRate(pub usize);

#[derive(Clone)]
pub struct VideoInfo {
    pub codec: Codec,
    pub format: Pixel,
    pub width: Width,
    pub height: Height,
    pub aspect_ratio: AspectRatio,
    pub frame_rate: FrameRate,
    pub input_stream_meta_data: dictionary::Owned<'static>, // TODO: Remove static, Add lifetime to VideoInfo
    pub itcx_number_streams: u32,
    pub bitrate: BitRate,
    pub max_bitrate: MaxBitRate,
}

pub enum VideoProcessingPluginError {}

impl Debug for VideoInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoInfo")
            .field("codec", &self.codec.name())
            .field("format", &self.format)
            .field("width", &self.width.0)
            .field("height", &self.height.0)
            .field("aspect_ratio", &self.aspect_ratio.0)
            .field("frame_rate", &self.frame_rate.0)
            .field("input_stream_meta_data", &self.input_stream_meta_data)
            .field("itcx_number_streams", &self.itcx_number_streams)
            .finish()
    }
}

impl VideoInfo {
    pub fn new(
        codec: Codec,
        format: Pixel,
        width: Width,
        height: Height,
        aspect_ratio: AspectRatio,
        frame_rate: FrameRate,
        input_stream_meta_data: dictionary::Owned<'static>, // TODO change this
        itcx_number_streams: u32,
        bitrate: BitRate,
        max_bitrate: MaxBitRate,
    ) -> Self {
        VideoInfo {
            codec,
            format,
            width,
            height,
            aspect_ratio,
            frame_rate,
            input_stream_meta_data,
            itcx_number_streams,
            bitrate,
            max_bitrate,
        }
    }

    pub fn width(&self) -> u32 {
        self.width.0
    }

    pub fn height(&self) -> u32 {
        self.height.0
    }
}

#[host_function]
fn init_plugin_logging(
    caller: Caller,
    args: Vec<WasmValue>,
    _data: &mut Arc<Mutex<FramesMap>>,
) -> Result<Vec<WasmValue>, HostFuncError> {
    let log_level_ptr = args[0].to_i32() as *mut i32;

    let mut main_memory = caller.memory(0).ok_or(HostFuncError::User(1))?;

    let log_level_main_memory = main_memory.try_get_ptr::<u32>(log_level_ptr as u32, 1)?;

    let log_level = match unsafe { *log_level_main_memory } {
        0 => LevelFilter::Off,
        1 => LevelFilter::Error,
        2 => LevelFilter::Warn,
        3 => LevelFilter::Info,
        4 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    if let Err(err) = CombinedLogger::init(vec![TermLogger::new(
        log_level,
        Config::default(),
        TerminalMode::Mixed,
        ColorChoice::Always,
    )]) {
        eprintln!("Could not Initialize Plugin Logging {}", err);
    };

    return Ok(vec![WasmValue::from_i32(0)]);
}

trait TryGetPointer {
    fn try_get_ptr<T>(&mut self, offset: u32, len: u32) -> Result<*mut T, HostFuncError>;
}

impl TryGetPointer for Memory {
    fn try_get_ptr<T>(&mut self, offset: u32, len: u32) -> Result<*mut T, HostFuncError> {
        match self.data_pointer_mut(offset, len) {
            Ok(x) => Ok(x as *mut T),
            Err(err) => {
                error!("Error Getting Value from Pointer {}", err);
                Err(HostFuncError::User(1))
            }
        }
    }
}

#[host_function]
fn load_video_to_host_memory(
    caller: Caller,
    args: Vec<WasmValue>,
    data: &mut Arc<Mutex<FramesMap>>,
) -> Result<Vec<WasmValue>, HostFuncError> {
    debug!("Load_video");

    let data_guard = match data.lock() {
        Ok(x) => x,
        Err(err) => {
            error!("Mutex Carrying plugin Data Poisoned {err}");
            return Err(HostFuncError::Runtime(1));
        }
    };

    let mut main_memory = caller.memory(0).ok_or(HostFuncError::User(1))?;

    let filename_ptr = args[0].to_i32();
    let filename_len = args[1].to_i32();
    let filaname_capacity = args[2].to_i32();

    let width_ptr = args[3].to_i32() as *mut i32;
    let height_ptr = args[4].to_i32() as *mut i32;
    let frames_ptr = args[5].to_i32() as *mut i32;

    let width_ptr_main_memory = main_memory.try_get_ptr::<u32>(width_ptr as u32, 1)?;
    let height_ptr_main_memory = main_memory.try_get_ptr::<u32>(height_ptr as u32, 1)?;
    let frames_ptr_main_memory = main_memory.try_get_ptr::<u32>(frames_ptr as u32, 1)?;

    let filename_ptr_main_memory =
        main_memory.try_get_ptr::<u8>(filename_ptr as u32, filename_len as u32)?;

    let filename: String = unsafe {
        String::from_raw_parts(
            filename_ptr_main_memory,
            filename_len as usize,
            filaname_capacity as usize,
        )
    };

    debug!("Call FFMPEG dump Frames");

    let res = match decode_video::dump_frames(&filename) {
        Ok((frames, video_info)) => {
            debug!("Input Frame Count {}", frames.len());
            if frames.is_empty() {
                unsafe {
                    *width_ptr_main_memory = frames[0].input_frame.width();
                    *height_ptr_main_memory = frames[0].input_frame.height();
                }
            } else {
                error!("Video file {} contained No Frames", filename);
                return Err(HostFuncError::User(1));
            }

            let mut vid_gaurd = data_guard;
            vid_gaurd.video_info = Some(video_info);
            vid_gaurd.frames = frames;
            unsafe {
                *frames_ptr_main_memory = vid_gaurd.frames.len() as u32;
            }
            Ok(vec![WasmValue::from_i32(0)])
        }
        Err(err) => {
            error!("Error Loading Frames {:?}", err);
            Err(HostFuncError::User(1))
        }
    };

    // Need to forget x otherwise we get a double free
    std::mem::forget(filename);
    res
}

#[host_function]
fn get_frame(
    caller: Caller,
    args: Vec<WasmValue>,
    data: &mut Arc<Mutex<FramesMap>>,
) -> Result<Vec<WasmValue>, HostFuncError> {
    debug!("get_frame");

    let data_guard = match data.lock() {
        Ok(x) => x,
        Err(err) => {
            error!("Mutex Carrying plugin Data Poisoned {err}");
            return Err(HostFuncError::Runtime(1));
        }
    };

    let mut main_memory = caller.memory(0).ok_or(HostFuncError::User(1))?;

    let idx: i32 = args[0].to_i32();
    let image_buf_ptr = args[1].to_i32();
    let image_buf_len = args[2].to_i32() as usize;
    let image_buf_capacity = args[3].to_i32() as usize;

    debug!("LIB image_buf_ptr {:?}", image_buf_ptr);
    debug!("LIB image_buf_len {:?}", image_buf_len);
    debug!("LIB image_buf_capacity {:?}", image_buf_capacity);

    let image_ptr_wasm_memory = main_memory
        .data_pointer_mut(image_buf_ptr as u32, image_buf_len as u32)
        .expect("Could not get Data pointer");

    let mut vec =
        unsafe { Vec::from_raw_parts(image_ptr_wasm_memory, image_buf_len, image_buf_capacity) };

    if let Some(frame) = data_guard.frames.get(idx as usize) {
        debug!("LIB data {:?}", frame.input_frame.data(0).len());
        vec.copy_from_slice(frame.input_frame.data(0));
    } else {
        error!("Return error if frame does not exist");
    };

    // Need to forget x otherwise we get a double free
    std::mem::forget(vec);
    Ok(vec![WasmValue::from_i32(0)])
}

#[host_function]
fn write_frame(
    caller: Caller,
    args: Vec<WasmValue>,
    data: &mut Arc<Mutex<FramesMap>>,
) -> Result<Vec<WasmValue>, HostFuncError> {
    debug!("write_frame");

    let mut data_guard = match data.lock() {
        Ok(x) => x,
        Err(err) => {
            error!("Mutex Carrying plugin Data Poisoned {err}");
            return Err(HostFuncError::Runtime(1));
        }
    };

    let mut main_memory = caller.memory(0).ok_or(HostFuncError::User(1))?;

    let video_info = data_guard
        .video_info
        .as_ref()
        .expect("Could not get Video Info data ");

    let idx = args[0].to_i32() as usize;
    let image_buf_ptr = args[1].to_i32();
    let image_buf_len = args[2].to_i32() as usize;

    let image_ptr_wasm_memory = main_memory
        .data_pointer_mut(image_buf_ptr as u32, image_buf_len as u32)
        .expect("Could not get Data pointer");

    let vec = unsafe {
        Vec::from_raw_parts(
            image_ptr_wasm_memory,
            image_buf_len,
            (video_info.width() * video_info.height() * 3) as usize,
        )
    };

    debug!(
        "BUFFER SIZE {}",
        video_info.width() * video_info.height() * 3
    );

    let mut video_frame = frame::Video::new(
        ffmpeg_next::format::Pixel::RGB24,
        video_info.width.0,
        video_info.height.0,
    );

    {
        let data = video_frame.data_mut(0);
        data.copy_from_slice(&vec);
    }

    debug!("Writing Frame {idx}");

    if let Some(frame_map) = data_guard.frames.get_mut(idx) {
        frame_map.output_frame = Some(video_frame);
    } else {
        // Need to forget x otherwise we get a double free
        std::mem::forget(vec);
        return Ok(vec![WasmValue::from_i32(1)]);
    };

    // Need to forget x otherwise we get a double free
    std::mem::forget(vec);
    Ok(vec![WasmValue::from_i32(0)])
}

#[host_function]
fn assemble_output_frames_to_video(
    caller: Caller,
    args: Vec<WasmValue>,
    data: &mut Arc<Mutex<FramesMap>>,
) -> Result<Vec<WasmValue>, HostFuncError> {
    debug!("assemble_video");
    let mut data_guard = match data.lock() {
        Ok(x) => x,
        Err(err) => {
            error!("Mutex Carrying plugin Data Poisoned {err}");
            return Err(HostFuncError::Runtime(1));
        }
    };

    let mut main_memory = caller.memory(0).ok_or(HostFuncError::Runtime(1))?;

    let filename_ptr = args[0].to_i32();
    let filename_len = args[1].to_i32();
    let filaname_capacity = args[2].to_i32();

    // TODO proper Handling of errors
    let filename_ptr_main_memory = main_memory.try_get_ptr::<u8>(filename_ptr as u32, 1)?;

    let video_struct = &mut (*data_guard);
    let frames = &mut video_struct.frames;
    let video_info = match &video_struct.video_info {
        Some(video_info) => video_info,
        None => {
            error!("No Video Information when attempting to Assemble output assemble_output_frames_to_video");
            return Err(HostFuncError::User(1));
        }
    };

    let output_file: String = unsafe {
        String::from_raw_parts(
            filename_ptr_main_memory,
            filename_len as usize,
            filaname_capacity as usize,
        )
    };

    // Check Frames have all been Written
    // Save Indexes of frames that have not been written
    let (mut frames, missing_frames) = frames.iter_mut().enumerate().fold(
        (Vec::new(), Vec::new()),
        |(mut iter_frames, mut iter_missing), (idx, frame_map)| {
            match frame_map.output_frame.as_mut() {
                Some(fr) => {
                    // TODO REMOVE CLONE
                    iter_frames.push((fr.clone(), frame_map.frame_type, frame_map.timestamp))
                }
                None => iter_missing.push(idx),
            };
            (iter_frames, iter_missing)
        },
    );

    if missing_frames.is_empty() {
        error!("Error Missing Frames {:?} ", missing_frames);
        return Err(HostFuncError::User(1));
    }

    let mut video_encoder = encode_video::VideoEncoder::new(video_info, &output_file)
        .map_err(|_| HostFuncError::User(1))?;

    if let Err(err) = video_encoder.receive_and_process_decoded_frames(&mut frames) {
        error!("Encode stream Error {:?}", err);
    };

    // Need to forget x otherwise we get a double free
    std::mem::forget(output_file);
    Ok(vec![WasmValue::from_i32(0)])
}

#[derive(Clone)]
struct FramesMap {
    frames: Frames,
    video_info: Option<VideoInfo>,
}

#[derive(Clone)]
pub struct FrameMap {
    input_frame: frame::Video,
    // Input Frame Type
    frame_type: picture::Type,
    // Input Frame Timestamp
    timestamp: Option<i64>,
    // Option as we are not sure if it has been processed yet or not
    output_frame: Option<frame::Video>,
}

type Frames = Vec<FrameMap>;
type ShareFrames = Arc<Mutex<FramesMap>>;

/// Defines Plugin module instance
unsafe extern "C" fn create_test_module(
    _arg1: *const ffi::WasmEdge_ModuleDescriptor,
) -> *mut ffi::WasmEdge_ModuleInstanceContext {
    let module_name: &str = "yolo-rs-video";

    let video_frames = FramesMap {
        frames: Vec::new(),
        video_info: None,
    };

    let video_frames_arc = Box::new(Arc::new(Mutex::new(video_frames)));

    type Width = i32;
    type Height = i32;
    type Frames = i32;

    let plugin_module = PluginModuleBuilder::<NeverType>::new()
        .with_func::<i32, i32, ()>("init_plugin_logging", init_plugin_logging, None)
        .expect("failed to create init_plugin_logging host function")
        .with_func::<(i32, i32, i32, Width, Height, Frames), i32, ShareFrames>(
            "load_video_to_host_memory",
            load_video_to_host_memory,
            Some(video_frames_arc.clone()),
        )
        .expect("failed to create load_video_to_host_memory host function")
        .with_func::<(i32, i32, i32, i32), i32, ShareFrames>(
            "get_frame",
            get_frame,
            Some(video_frames_arc.clone()),
        )
        .expect("failed to create get_frame host function")
        .with_func::<(i32, i32, i32), i32, ShareFrames>(
            "write_frame",
            write_frame,
            Some(video_frames_arc.clone()),
        )
        .expect("failed to create write_frame host function")
        .with_func::<(i32, i32, i32), i32, ShareFrames>(
            "assemble_output_frames_to_video",
            assemble_output_frames_to_video,
            Some(video_frames_arc.clone()),
        )
        .expect("failed to create assemble_output_frames_to_video host function")
        .build(module_name)
        .expect("failed to create plugin module");

    let boxed_module = Box::new(plugin_module);
    let module = Box::leak(boxed_module);

    module.as_raw_ptr() as *mut _
}

/// Defines PluginDescriptor
#[export_name = "WasmEdge_Plugin_GetDescriptor"]
pub extern "C" fn plugin_hook() -> *const ffi::WasmEdge_PluginDescriptor {
    const NAME: &str = "yolo-rs-video-plugin";
    const DESC: &str = "This is a yolo video processing plugin utilizing FFMPEG";
    let version = PluginVersion::new(0, 0, 0, 0);
    let plugin_descriptor = PluginDescriptor::new(NAME, DESC, version)
        .expect("Failed to create plugin descriptor")
        .add_module_descriptor(NAME, DESC, Some(create_test_module))
        .expect("Failed to add module descriptor");

    let boxed_plugin = Box::new(plugin_descriptor);
    let plugin = Box::leak(boxed_plugin);

    plugin.as_raw_ptr()
}
