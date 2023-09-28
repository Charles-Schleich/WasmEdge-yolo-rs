use crate::prepare::ResizeScale;
use crate::{ConfThresh, IOUThresh, InferenceResult};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ndarray::{array, s, Array, Array2, ArrayView, Axis, Dim, Zip};

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
    iou_thresh: IOUThresh,
    results: Vec<InferenceResult>,
) -> Vec<InferenceResult> {
    //  process elements into ndarray
    let mut nd_arr: ndarray::ArrayBase<ndarray::OwnedRepr<u32>, ndarray::Dim<[usize; 2]>> =
        Array::zeros((0, 4));

    // for r in results.iter() {
    //     let bbox = &r.b_box;
    //     nd_arr
    //         .push_row(array![bbox.left(), bbox.top(), bbox.left() + bbox.(w), bbox.y + bbox.h].view())
    //         .unwrap();
    // }

    // println!("{}", nd_arr);
    // [1., 2., 3., 4.]
    // todo!();
    // results.iter().group_by(|x|x)
    // TODO apply NMS
    results
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

// Map [x1,y1,x2,y2] -> to ArrayBase<OwnedRepr<A>, D>
pub fn map_bounding_boxes_to_ndarray(arr_b_boxes: Vec<[f64; 4]>) -> Array<f64, Dim<[usize; 2]>> {
    let mut data = Vec::new();
    let ncols = arr_b_boxes.first().map_or(0, |row| row.len());
    let mut nrows = 0;

    for i in 0..arr_b_boxes.len() {
        data.extend_from_slice(&arr_b_boxes[i]);
        nrows += 1;
    }

    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

pub fn vectorized_iou(boxes_a: Array<f64, Dim<[usize; 2]>>, boxes_b: Array<f64, Dim<[usize; 2]>>) {
    let box_area =
        |bbox: ArrayView<f64, Dim<[usize; 1]>>| (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);

    let area_a = boxes_a.map_axis(Axis(1), |row| box_area(row));
    let area_b = boxes_b.map_axis(Axis(1), |row| box_area(row));
    // boxes_a.slice(info)
    // TODO USE insert_axis
    // https://github.com/rust-ndarray/ndarray/pull/354
    // https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.insert_axis_inplace
    
    let a_top_left = boxes_a.slice(s![.., ..2]);
    let b_top_left = boxes_b.slice(s![.., ..2]);

    // Elementwise maximum
    let top_left = Zip::from(a_top_left)
        .and(&b_top_left)
        .map_collect(|x, &y| x.max(y));

    let a_bottom_right = boxes_a.slice(s![.., 2..]);
    let b_bottom_right = boxes_b.slice(s![.., 2..]);
    // Elementwise minumum

    let bottom_right = Zip::from(a_bottom_right)
        .and(&b_bottom_right)
        .map_collect(|x, &y| x.min(y));

    eprintln!("TL BR");
    eprintln!("{}", top_left);
    eprintln!("{}", bottom_right);
    // top_left
    // bottom_right
    //  .for_each(|x, &y, | {
    //     x.max(y)
    // });
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

        let rect: Rect = result.b_box.into();
        draw_hollow_rect_mut(&mut rgb_image, rect, color);

        let color = Rgb([255u8, 0u8, 0u8]);
        let rect: Rect = Rect::at(10, 10).of_size(10, 20);
        draw_hollow_rect_mut(&mut rgb_image, rect, color)
    }
    rgb_image
}

#[cfg(test)]
mod tests {
    use imageproc::rect::Rect;
    use ndarray::{ArrayView, Axis, Dim};

    use crate::process::{iou, map_bounding_boxes_to_ndarray, vectorized_iou};

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

    #[test]
    fn test_vectorized_iou() {
        let box1: [f64; 4] = [1., 1., 3., 3.];
        let box2 = [2., 2., 3., 3.];
        // let box3 = [3., 3., 4., 4.];
        let all_boxes = vec![box1, box2];

        let all_boxes_arr = map_bounding_boxes_to_ndarray(all_boxes);

        // let box_area =
        //     |bbox: ArrayView<f64, Dim<[usize; 1]>>| (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
        // let box_area = |bbox: [f64; 4]| (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);

        // let output = arr.map_axis(Axis(1), |row| box_area(row));
        // boxes_a: Array<f64, Dim<[usize; 2]>>
        // boxes_b: Array<f64, Dim<[usize; 2]>>

        vectorized_iou(all_boxes_arr.clone(), all_boxes_arr);
        // println!("{}", output);
        // assert_eq!(box_area(box1), 4.0);
        assert_eq!(1.0, 4.0);
    }
}
