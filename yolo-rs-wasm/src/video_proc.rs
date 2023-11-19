extern crate simplelog;

pub mod yolo_rs_video_plugin {
    use log::LevelFilter;

    pub fn init_plugin_logging_with_log_level(level_filter: LevelFilter) {
        let level_filter_i32 = level_filter as i32;
        let level_filter_ptr = std::ptr::addr_of!(level_filter_i32);
        unsafe { init_plugin_logging(level_filter_ptr) };
    }

    #[link(wasm_import_module = "yolo-rs-video")]
    extern "C" {
        pub fn init_plugin_logging(level: *const i32) -> i32;

        pub fn load_video_to_host_memory(
            str_ptr: i32,
            str_len: i32,
            str_capacity: i32,
            width_ptr: *mut i32,
            height_ptr: *mut i32,
            frame_count: *mut i32,
        ) -> i32;

        pub fn get_frame(
            frame_index: i32,
            image_buf_ptr: i32,
            image_buf_len: i32,
            image_buf_capacity: i32,
        ) -> i32;

        pub fn write_frame(frame_index: i32, image_buf_ptr: i32, image_buf_len: i32) -> i32;

        pub fn assemble_output_frames_to_video(
            str_ptr: i32,
            str_len: i32,
            str_capacity: i32,
        ) -> i32;

    }
}
