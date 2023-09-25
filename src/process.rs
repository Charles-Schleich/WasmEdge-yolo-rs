// Function to process output tensor from YOLO
fn post_process_results(buffer: &[f32]) -> Vec<InferenceResult> {
    // Output buffer is in columar format
    // 84 rows x 8400 columns as a single Vec of f32
    let mut columns = Vec::new();
    for col_slice in buffer.chunks_exact(8400) {
        let col_vec = col_slice.to_vec();
        columns.push(col_vec);
    }

    let rows = transpose(columns);

    // Row Format is
    // [x,y,w,h,p1,p2,p3...p80]
    // where:
    // x,y are the pixel locations of the top left corner of the bounding box,
    // w,h are the width and height of bounding box,
    // p1,p2..p80, are the class probabilities.
    let mut results = Vec::new();

    for row in rows {
        let x = row[0].round() as u32;
        let y = row[1].round() as u32;
        let width = row[2].round() as u32;
        let height = row[3].round() as u32;

        // Get maximum likeliehood for each detection
        // Iterator of only class probabilities
        let mut prob_iter = row.clone().into_iter().skip(4);
        let max = prob_iter.clone().reduce(|a, b| a.max(b)).unwrap();
        let index = prob_iter.position(|element| element == max).unwrap();
        let class = YOLO_CLASSES.get(index).unwrap().to_string();

        results.push(InferenceResult {
            x,
            y,
            width,
            height,
            probability: max,
            class,
        });
    }
    results
}
