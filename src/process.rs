use crate::prepare::ResizeScale;
use crate::{ConfThresh, IOUThresh, InferenceResult};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

/// Function to process output tensor from YOLOv8 Detection Model
/// TODO: more efficient parsing: remove transpose convert from buffer directly to
pub fn process_output_buffer_to_tensor(buffer: &[f32]) -> Vec<Vec<f32>> {
    // Output buffer is in format
    // 8400 x 84 as a single Vec of f32
    // i.e. [x1,x2,x3,..,x8400, y1,y2,y3,...,y84000,]
    let mut columns = Vec::new();
    for col_slice in buffer.chunks_exact(8400) {
        let col_vec = col_slice.to_vec();
        columns.push(col_vec);
    }

    // transpose 84 rows x 8400 columns as a single Vec of f32
    let rows = transpose(columns);
    rows
}

// Row Format is
// [x,y,w,h,p1,p2,p3...p80]
// where:
// x,y are the pixel locations of the top left corner of the bounding box,
// w,h are the width and height of bounding box,
// p1,p2..p80, are the class probabilities.
pub(crate) fn apply_confidence_and_scale(
    rows: Vec<Vec<f32>>,
    conf_thresh: ConfThresh,
    classes: Vec<String>,
    scale: ResizeScale,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        // Get maximum likeliehood for each detection
        // Iterator of only class probabilities
        let mut prob_iter = row.clone().into_iter().skip(4);
        let max = prob_iter.clone().reduce(|a, b| a.max(b)).unwrap();

        if max < conf_thresh.0 {
            continue;
        }

        let index = prob_iter.position(|element| element == max).unwrap();
        let class = classes.get(index).unwrap().to_string();

        // The output of the x and y cooridnates are at the CENTER of the bounding box
        // which means if we want to get them to the top left hand corners,
        // we must shift x by width * 0.5, and y by height * 0.5
        let raw_w = row[2];
        let raw_h = row[3];
        let x = ((row[0] - 0.5 * raw_w) * scale.0).round() as u32;
        let y = ((row[1] - 0.5 * raw_h) * scale.0).round() as u32;
        let w = (raw_w * scale.0).round() as u32;
        let h = (raw_h * scale.0).round() as u32;

        results.push(InferenceResult {
            x,
            y,
            width: w,
            height: h,
            confidence: max,
            class,
        });
    }
    results
}

fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    if v.is_empty() {
        return v;
    }

    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

pub(crate) fn apply_conf_and_nms(
    conf_thresh: ConfThresh,
    iou_thresh: IOUThresh,
    results: Vec<InferenceResult>,
) -> Vec<InferenceResult> {
    let high_conf = results
        .into_iter()
        .filter(|x| x.confidence > conf_thresh.0)
        .collect::<Vec<InferenceResult>>();

    high_conf
    // TODO apply NMS
}

/// Convieience Function to draw bounding boxes to image
pub fn draw_bounding_boxes_to_image(
    mut rgb_image: RgbImage,
    vec_results: Vec<InferenceResult>,
) -> RgbImage {
    let color = Rgb([0u8, 0u8, 255u8]);

    for result in vec_results {
        let conf = result.confidence;
        let class = result.class.clone();

        let rect: Rect = result.into();
        draw_hollow_rect_mut(&mut rgb_image, rect, color)
    }
    rgb_image
}
