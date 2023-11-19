use ffmpeg::{
    codec, dictionary, encoder,
    format::{input, Pixel},
    frame,
    media::Type,
    software::scaling::{context::Context, flag::Flags},
    util::frame::video::Video,
};

use ffmpeg::Error as FFmpegError;

use log::debug;

use crate::{
    AspectRatio, BitRate, FrameMap, FrameRate, Frames, Height, MaxBitRate, VideoInfo, Width,
};
#[derive(Debug)]
pub enum VideoDecoderError {
    FFMpegError(FFmpegError),
    CodecError(String),
}

impl From<FFmpegError> for VideoDecoderError {
    fn from(value: FFmpegError) -> Self {
        VideoDecoderError::FFMpegError(value)
    }
}

pub fn dump_frames(filename: &String) -> Result<(Frames, VideoInfo), VideoDecoderError> {
    ffmpeg::init()?;

    let mut frame_index = 0;
    let mut frames = Vec::new();
    let codec;
    let input = input(filename);
    let (width, height, aspect_ratio, frame_rate, format);
    let input_stream_meta_data: dictionary::Owned;

    let itcx_number_streams;

    let (bitrate, max_bitrate);

    match input {
        Ok(mut ictx) => {
            let input = ictx
                .streams()
                .best(Type::Video)
                .ok_or(ffmpeg::Error::StreamNotFound)?;
            itcx_number_streams = ictx.nb_streams();

            let video_stream_index: usize = input.index();

            input_stream_meta_data = ictx.metadata().to_owned();

            let mut decoder = input.decoder()?.video()?;

            codec = encoder::find(codec::Id::H264).ok_or(VideoDecoderError::CodecError(
                "Could not Find Codec h264".into(),
            ))?;

            debug!("Decoder Codec");
            debug!("  BitRate {:?}", decoder.bit_rate());
            debug!("  MaxBitRate {:?}", decoder.max_bit_rate());
            debug!("  TimeBase {:?}", decoder.time_base());
            debug!("  Codec");
            debug!("      Name  {:?}", codec.name());
            debug!("      Descr {:?}", codec.description());

            // I am wrapping these in Structs so its less likely that I make Type Errors
            bitrate = BitRate(decoder.bit_rate());
            max_bitrate = MaxBitRate(decoder.bit_rate());
            width = Width(decoder.width());
            height = Height(decoder.height());
            aspect_ratio = AspectRatio(decoder.aspect_ratio());
            frame_rate = FrameRate(decoder.frame_rate());
            format = decoder.format();

            // Scaler to convert YUV420 encoded frame -> RGB Raw frame
            let mut scaler = Context::get(
                decoder.format(),
                decoder.width(),
                decoder.height(),
                Pixel::RGB24,
                decoder.width(),
                decoder.height(),
                Flags::BILINEAR,
            )?;

            // Closure to process out frames
            let mut receive_and_process_decoded_frames =
                |decoder: &mut ffmpeg::decoder::Video| -> Result<(), ffmpeg::Error> {
                    let mut decoded_frame = frame::Video::empty();
                    while decoder.receive_frame(&mut decoded_frame).is_ok() {
                        let mut rgb_frame = Video::empty();
                        scaler.run(&decoded_frame, &mut rgb_frame)?;
                        debug!(
                            "R_Frame {frame_index} : {:?} {:?} {:?} {:?} ",
                            decoded_frame.kind(),
                            decoded_frame.timestamp(),
                            decoded_frame.duration(),
                            decoded_frame.display_number()
                        );

                        let frame_map = FrameMap {
                            input_frame: rgb_frame,
                            frame_type: decoded_frame.kind(),
                            timestamp: decoded_frame.timestamp(),
                            output_frame: None,
                        };

                        frames.push(frame_map);
                        frame_index += 1;
                    }
                    Ok(())
                };

            // Iterator over Input Context Packets
            for (idx, res) in ictx.packets().enumerate() {
                let (stream, packet) = res?;
                if stream.index() == video_stream_index {
                    debug!("PKT {idx} PTS{:?}   DTS:{:?}", packet.pts(), packet.dts());
                    decoder.send_packet(&packet)?;
                    receive_and_process_decoded_frames(&mut decoder)?;
                }
            }
            decoder.send_eof()?;
            receive_and_process_decoded_frames(&mut decoder)?;
        }
        Err(err) => return Err(VideoDecoderError::from(err)),
    };

    let video_info = VideoInfo {
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
    };

    Ok((frames, video_info))
}
