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

        let rect: Rect = result.b_box.into();

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
            &font,
            &class_conf,
        );
    }
    rgb_image
}
