use std::time::Duration;

use ffmpeg_next::{
    codec::{self, Parameters},
    encoder::Video as AVEncoder,
    format::{self, context::output::Output, Pixel},
    frame, picture,
    software::scaling::{Context as Scaler, Flags},
    util::frame::video::Video as AVFrame,
    Dictionary, Error as FFmpegError, Packet, Rational,
};

use log::{debug, error, info, warn};

use std::collections::BTreeMap;

use crate::{time::Time, VideoInfo};

#[derive(Debug)]
pub enum VideoEncoderError {
    FFMpegError(FFmpegError),
    CodecError(String),
}

impl From<FFmpegError> for VideoEncoderError {
    fn from(value: FFmpegError) -> Self {
        VideoEncoderError::FFMpegError(value)
    }
}

pub(crate) struct VideoEncoder {
    // Encoder
    encoder: AVEncoder,
    // Output Context
    octx: Output,
    // Output Time Base
    _packet_order_map: BTreeMap<i64, Packet>, // ost_time_bases: Vec<Rational>,
    // Frame scaler / Converter between formats
    scaler: Scaler,
    // FrameRate
    frame_rate: i32,
}

impl VideoEncoder {
    pub fn new(v_info: &VideoInfo, output_file: &String) -> Result<Self, VideoEncoderError> {
        info!(" 1 ");
        let mut octx = format::output(output_file)?;
        info!(" 2 ");
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);
        info!(" 3 "); //
        let codec: ffmpeg_next::Codec = ffmpeg_next::encoder::find(codec::Id::H264).ok_or(
            VideoEncoderError::CodecError("Could not Find Codec h264".into()),
        )?;

        println!("Encoder Codec");
        println!("  Codec");
        println!("      Name  {:?}", codec.name());
        println!("      Descr {:?}", codec.description());
        println!("      Descr {:?}", codec.is_video());
        println!("      Descr {:?}", codec.is_audio());

        info!(" 4 "); //
        let mut ost: ffmpeg_next::StreamMut<'_> = octx.add_stream(codec)?;
        println!("{:?}", ost.metadata());

        info!(" 5 "); //
        let mut encoder = ffmpeg_next::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;
        // let mut encoder = ffmpeg_next::codec::encoder::new().video()?;
        //
        ost.set_parameters(&encoder);
        encoder.set_height(v_info.height.0);
        encoder.set_width(v_info.width.0);
        encoder.set_format(v_info.format);
        encoder.set_frame_rate(v_info.frame_rate.0);
        encoder.set_time_base(ffmpeg_next::rescale::TIME_BASE);

        // Keeping the Bit Rate VERY high to not lose information
        let bitrate_uncompressed = (3 * 8 * v_info.height.0 * v_info.width.0) as usize;
        encoder.set_bit_rate(bitrate_uncompressed / 2);
        info!(" 7 ");
        let mut dict = Dictionary::new();
        dict.set("preset", "slow");
        // dict.set("preset", "medium");
        info!(" 8 ");
        let mut encoder: AVEncoder = encoder.open_as_with(codec, dict)?;
        info!(" 9 ");
        // let params: Parameters = ffmpeg_next::codec::parameters::Parameters::new();
        // ost.set_parameters(params);

        if global_header {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }
        info!(" 10 ");

        octx.set_metadata(v_info.input_stream_meta_data.clone());
        format::context::output::dump(&octx, 0, Some(output_file));
        octx.write_header()?;
        info!(" 11 ");

        // Write Every Frame out to encoder packet
        let scaler = Scaler::get(
            Pixel::RGB24,
            v_info.width.0,
            v_info.height.0,
            Pixel::YUV420P,
            v_info.width.0,
            v_info.height.0,
            Flags::empty(),
        )?;
        info!(" 12 ");

        encoder.set_threading(codec::threading::Config {
            kind: codec::threading::Type::None,
            count: 0,
        });

        debug!("==================================");
        debug!("Encoder Settings");
        debug!("e.format() {:?}", encoder.format());
        debug!("e.medium() {:?}", encoder.medium());
        debug!("e.time_base() {:?}", encoder.time_base());
        debug!("e.threading() {:?}", encoder.threading());
        let codec = encoder.codec().ok_or(VideoEncoderError::CodecError(
            "Could not get Codec from Encoder".into(),
        ))?;
        debug!("e.codec().name {:?}", codec.name());
        debug!("e.codec().capabilities {:?}", codec.capabilities());
        debug!("e.codec().description {:?}", codec.description());
        debug!("e.codec().id {:?}", codec.id());
        debug!("==================================");

        // TODO: Should i rather fail here ?
        let frame_rate = match v_info.frame_rate.0 {
            Some(fr) => fr,
            None => {
                warn!("No Frame rate from Decoder Found, Defaulting to 30FPS for encoder");
                Rational::new(30, 1)
            }
        };

        Ok(VideoEncoder {
            encoder,
            octx,
            _packet_order_map: BTreeMap::new(),
            scaler,
            frame_rate: frame_rate.0,
        })
    }

    pub fn receive_and_process_decoded_frames(
        &mut self,
        frames: &mut [(frame::Video, picture::Type, Option<i64>)],
    ) -> Result<(), VideoEncoderError> {
        let duration: Time = Duration::from_nanos(1_000_000_000 / self.frame_rate as u64).into();

        let mut position = Time::zero();

        for (out_frame_rgb, _frame_type, _) in frames.iter_mut() {
            let frame_timestamp_rescale = position
                .aligned_with_rational(self.encoder.time_base())
                .into_value();

            out_frame_rgb.set_pts(frame_timestamp_rescale);

            let mut frame_yuv420 = self.scale(out_frame_rgb)?;

            // TODO Fix Encoding here
            // This will result in HUGE, essentially uncompressed video files,
            // If the bit rate is set to (Single Frame Byte amount) * number of Frames
            frame_yuv420.set_kind(picture::Type::I);

            debug!("F Send {:?}", frame_yuv420.pts(),);
            self.encoder.send_frame(&frame_yuv420)?;

            if let Some(mut packet) = self.encoder_receive_packet()? {
                // Leaving this here should i want to try reorder the packets again in the future
                // self.packet_order_map.insert(packet.pts().unwrap(), packet);
                self.write_encoded_packets(&mut packet, 0);
            }

            let aligned_position = position.aligned_with(&duration);
            position = aligned_position.add();
        }

        // Leaving this here should i want to try reorder the packets again in the futue
        // while let Some((k, mut packet)) = self.packet_order_map.pop_first() {
        //     debug!("Writing Packet {:?}", k);
        //     self.write_encoded_packets(&mut packet, 0);
        // }

        self.finish()?;

        Ok(())
    }

    fn scale(&mut self, frame: &mut AVFrame) -> Result<AVFrame, FFmpegError> {
        let mut frame_scaled = AVFrame::empty();
        self.scaler.run(frame, &mut frame_scaled)?;

        // Copy over PTS from old frame.
        frame_scaled.set_pts(frame.pts());

        Ok(frame_scaled)
    }

    fn flush(&mut self) -> Result<(), FFmpegError> {
        // Maximum number of invocations to `encoder_receive_packet`
        // to drain the items still on the queue before giving up.
        const MAX_DRAIN_ITERATIONS: u32 = 100;

        // Notify the encoder that the last frame has been sent.
        self.encoder.send_eof()?;

        // We need to drain the items still in the encoders queue.
        for _ in 0..MAX_DRAIN_ITERATIONS {
            let mut packet = Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(_) => self.write_encoded_packets(&mut packet, 0),
                Err(_) => break,
            };
        }

        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), FFmpegError> {
        self.flush()?;
        self.octx.write_trailer()?;
        Ok(())
    }

    fn encoder_receive_packet(&mut self) -> Result<Option<Packet>, FFmpegError> {
        let mut packet = Packet::empty();
        let encode_result = self.encoder.receive_packet(&mut packet);
        match encode_result {
            Ok(()) => Ok(Some(packet)),
            Err(err) => Err(err), // TODO process properly
        }
    }

    fn write_encoded_packets(&mut self, packet: &mut Packet, ost_index: usize) {
        packet.set_stream(ost_index);
        packet.set_position(-1);
        debug!(
            "P Write S {:?} {:?} {:?}",
            packet.pts(),
            packet.dts(),
            packet.duration()
        );

        packet.rescale_ts(
            self.encoder.time_base(),
            // TODO: Will Defaulting to TIME_BASE cause a potential source of errors here ?
            self.octx
                .stream(0)
                .expect("Could not Find Stream at index")
                .time_base(),
        );

        debug!("P Write F {:?} {:?}", packet.pts(), packet.dts());

        let write_interleaved = packet.write_interleaved(&mut self.octx);
        if let Err(err) = write_interleaved {
            error!("write_interleaved {:?}", err);
        };
    }
}
