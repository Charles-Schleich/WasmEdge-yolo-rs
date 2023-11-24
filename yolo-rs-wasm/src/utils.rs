use std::path::Path;

use crate::InferenceResult;
use image::{Rgb, RgbImage};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use rusttype::{Font, Scale};

/// Convieience Function to draw bounding boxes to image
pub fn draw_bounding_boxes_on_mut_image(
    mut rgb_image: RgbImage,
    vec_results: &Vec<InferenceResult>,
    font: &Font<'static>,
) -> RgbImage {
    let color = Rgb([0u8, 0u8, 255u8]);

    for result in vec_results {
        let conf = result.confidence;

        let rect: Rect = result.b_box;

        draw_hollow_rect_mut(&mut rgb_image, rect, color);

        let class_conf = format!("{} {}", result.class, conf);

        // TODO: calc font size based on image dims
        // let height = 12.4;
        // let scale = Scale {
        //     x: height * 2.0,
        //     y: height,
        // };

        let scale = Scale::uniform(25.0);
        draw_text_mut(
            &mut rgb_image,
            color,
            rect.left() + 5,
            rect.top() - 30,
            scale,
            font,
            &class_conf,
        );
    }
    rgb_image
}

#[derive(thiserror::Error, Debug)]
pub enum FontLoadError {
    #[error("error parsing bytes as font")]
    InvalidFontData,

    #[error("error reading file containing classes")]
    FileError(#[from] std::io::Error),
}

/// Tries to load a font from a path, Fails if it cannot
/// - succeeds if font path points to valid .ttf format file
/// - fails otherwise
pub fn load_font<P>(font_path: P) -> Result<Font<'static>, FontLoadError>
where
    P: AsRef<Path>,
{
    let bytes = std::fs::read(font_path)?;
    let font: Font = Font::try_from_vec(bytes).ok_or(FontLoadError::InvalidFontData)?;
    Ok(font)
}
