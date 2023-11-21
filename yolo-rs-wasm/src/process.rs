use std::collections::HashSet;

use crate::prepare::ResizeScale;
use crate::{ConfThresh, IOUThresh, InferenceResult, PostProcessingError, RuntimeError};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use itertools::Itertools;
use ndarray::{s, Array2, ArrayView, Axis, Dim, Zip};
use rusttype::{Font, Scale};

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
    conf_thresh: &ConfThresh,
    classes: &Vec<String>,
    scale: ResizeScale,
) -> Vec<InferenceResult> {
    let mut results = Vec::new();
    for (_, row) in rows.iter().enumerate() {
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
        // TODO remove allocation of raw_w
        let raw_w = row[2];
        let raw_h = row[3];
        let x = ((row[0] - 0.5 * raw_w) * scale.0).round() as u32;
        let y = ((row[1] - 0.5 * raw_h) * scale.0).round() as u32;
        let w = (raw_w * scale.0).round() as u32;
        let h = (raw_h * scale.0).round() as u32;

        results.push(InferenceResult {
            b_box: Rect::at(x as i32, y as i32).of_size(w, h),
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

// Non-vectorized Non Maximum supression
pub(crate) fn non_maximum_supression(
    iou_thresh: &IOUThresh,
    mut results: Vec<InferenceResult>,
) -> Result<Vec<InferenceResult>, RuntimeError> {
    results.sort_by(|x, y| y.confidence.total_cmp(&x.confidence));

    // TODO make computation more efficient
    //  Potentially grouping by class type reduces the set of bounding boxes that need to be calucated over
    //  i.e group all class1, all class2 ...
    //

    // TODO integration creation of ND array into iterator
    let b_boxes: Vec<[f64; 4]> = results
        .iter()
        .map(|r| {
            let b = &r.b_box;
            let coords: [f64; 4] = [b.left(), b.top(), b.right(), b.bottom()].map(f64::from);
            coords
        })
        .collect();

    let nd_bboxes = bboxes_to_ndarray(b_boxes);
    if nd_bboxes.is_empty() {
        return Ok(Vec::new());
    }

    let iou_matrix = vectorized_iou(nd_bboxes.clone(), nd_bboxes)?;

    // TODO look at using Two Array pointers to walk list of values
    // discarding values that we do not need
    // let results_combinations = results.iter().enumerate().zip(results.iter().enumerate());
    // TODO REMOVE CLONE and re-write comparison function
    let results_combinations = results
        .clone()
        .into_iter()
        .enumerate()
        .combinations(2)
        .collect::<Vec<Vec<(usize, InferenceResult)>>>()
        .into_iter()
        .map(|v: Vec<(usize, InferenceResult)>| (v[0].clone(), v[1].clone()))
        .collect::<Vec<((usize, InferenceResult), (usize, InferenceResult))>>();

    let mut keep = (0..results.len() as i32).collect::<HashSet<i32>>();

    // We are checking to see if we want to keep inference result - 1
    for ((idx_1, r_1), (idx_2, r_2)) in results_combinations {
        // When to skip comparison

        if idx_1 == idx_2 {
            continue;
        }

        if r_1.class != r_2.class {
            continue;
        }

        // TODO: potentially unsafe indexing
        let iou = iou_matrix[[idx_1, idx_2]];
        if iou > iou_thresh.0 as f64 {
            keep.remove(&(idx_2 as i32));
        }
    }

    let mut keepers = Vec::new();
    for elem in keep {
        keepers.push(results[elem as usize].clone());
    }

    // TODO See if i can make use of references and not allocate new arrays
    Ok(keepers)
}

// Calculate intersection over union for rectangle
pub fn iou(box1: Rect, box2: Rect) -> f32 {
    let area = |r: Rect| r.width() * r.height();

    let area1 = area(box1);
    let area2 = area(box2);

    let area_boxes = area1 + area2;

    match box1.intersect(box2) {
        Some(intersection) => area(intersection) as f32 / (area_boxes - area(intersection)) as f32,
        None => 1.,
    }
}

/// Converts a bounding box to an ArrayBase
/// Map [x1,y1,x2,y2] -> to ArrayBase<OwnedRepr<A>, D>
pub fn bboxes_to_ndarray(arr_b_boxes: Vec<[f64; 4]>) -> Array2<f64> {
    let mut data = Vec::new();
    let ncols = arr_b_boxes.first().map_or(0, |row| row.len());
    let mut nrows = 0;

    for i in 0..arr_b_boxes.len() {
        data.extend_from_slice(&arr_b_boxes[i]);
        nrows += 1;
    }

    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

pub fn vectorized_iou(
    boxes_a: Array2<f64>,
    boxes_b: Array2<f64>,
) -> Result<Array2<f64>, RuntimeError> {
    // TODO See if i can more make use of references and not allocate new arrays

    let box_area =
        |bbox: ArrayView<f64, Dim<[usize; 1]>>| (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    let (num_boxes, _elems_per_box) = boxes_a.dim();

    let area_a = boxes_a.map_axis(Axis(1), |row| box_area(row));
    let area_b = boxes_b.map_axis(Axis(1), |row| box_area(row));

    let boxes_a_new_axis = boxes_a.clone().insert_axis(Axis(1));
    // let boxes_b_new_axis = boxes_b.clone();
    let a_top_left = boxes_a_new_axis.slice(s![.., .., ..2]);

    let a_top_left_bc = a_top_left.broadcast((num_boxes, num_boxes, 2)).ok_or(
        RuntimeError::PostProcessingError(PostProcessingError::BroadcastArrayDims),
    )?;
    let b_top_left = boxes_b.slice(s![.., ..2]);

    // Elementwise maximum
    let top_left = Zip::from(&a_top_left_bc)
        .and_broadcast(b_top_left)
        .map_collect(|x, &y| x.max(y));

    let a_bot_right = boxes_a_new_axis.slice(s![.., .., 2..]);
    let b_bot_right = boxes_b.slice(s![.., 2..]);
    let a_bot_right_bc = a_bot_right.broadcast((num_boxes, num_boxes, 2)).unwrap();

    // Elementwise minumum
    let bottom_right = Zip::from(a_bot_right_bc)
        .and_broadcast(&b_bot_right)
        .map_collect(|x, &y| x.min(y));

    // Difference between right and bottom left
    let bot_right_top_left = bottom_right - top_left;
    let area_inter = bot_right_top_left.map_axis(Axis(2), |x| x.product());
    let iou = area_inter.clone() / (area_a.insert_axis(Axis(1)) + area_b - area_inter);
    Ok(iou)
}

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

#[cfg(test)]
mod tests {
    use crate::process::{bboxes_to_ndarray, iou, vectorized_iou};
    use imageproc::rect::Rect;
    use ndarray::array;

    #[test]
    fn test_iou() {
        // top left (1,1)  bot right (3,3)
        let box1 = Rect::at(1, 1).of_size(2, 2);
        // top left (2,2)  bot right (3,3)
        let box2 = Rect::at(2, 2).of_size(1, 1);

        let iou_out = iou(box1, box2);

        assert_eq!(iou_out, 0.25);

        // top left (1,1)  bot right (4,4)
        let box1 = Rect::at(1, 1).of_size(3, 3);
        // top left (2,2)  bot right (5,5)
        let box2 = Rect::at(2, 2).of_size(3, 3);
        let iou_out = iou(box1, box2);

        assert_eq!(iou_out, 0.2857143);
    }

    // Testing with 2 bounding boxes
    // format [x1,y1,x2,y2]
    #[test]
    fn test_vectorized_iou_2_boxes() {
        let box1: [f64; 4] = [1., 1., 3., 3.];
        let box2: [f64; 4] = [2., 2., 3., 3.];
        let all_boxes = vec![box1, box2];
        let all_boxes_arr = bboxes_to_ndarray(all_boxes);
        let expected_iou = array!([1., 0.25], [0.25, 1.]);
        let actual_iou = vectorized_iou(all_boxes_arr.clone(), all_boxes_arr).unwrap();

        assert_eq!(expected_iou, actual_iou);
    }

    // Testing with 3 bounding boxes
    // format [x1,y1,x2,y2]
    #[test]
    fn test_vectorized_iou() {
        let box1: [f64; 4] = [1., 1., 3., 3.];
        let box2 = [2., 2., 3., 3.];
        let box3 = [3., 3., 4., 4.];
        let all_boxes = vec![box1, box2, box3];

        let all_boxes_arr = bboxes_to_ndarray(all_boxes);
        let expected_iou = array!([1., 0.25, 0.], [0.25, 1.0, 0.0], [0., 0., 1.0]);
        let actual_iou = vectorized_iou(all_boxes_arr.clone(), all_boxes_arr).unwrap();

        assert_eq!(expected_iou, actual_iou);
    }
    #[test]
    fn test_nms() {
        // write tests for non-maximum supression
        todo!();
    }
}
