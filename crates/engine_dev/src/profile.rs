use engine_core::BrickCosts;
use serde_json::{Value, json};

use crate::protocol::{Detail, ProfileRequest};
use crate::results::{ProfileResult, ProfiledBrick};

pub fn profiled(frames: &[BrickCosts], request: &ProfileRequest) -> (ProfileResult, Value) {
    let count = frames.len().max(1) as f64;
    let mut bricks: Vec<ProfiledBrick> = Vec::new();
    for costs in frames {
        for row in &costs.rows {
            let index = match bricks.iter().position(|brick| brick.brick == row.brick) {
                Some(index) => index,
                None => {
                    bricks.push(ProfiledBrick {
                        brick: row.brick,
                        cpu_ms: 0.0,
                        cpu_ms_max: 0.0,
                        gpu_ms: None,
                        gpu_ms_max: None,
                        heap_live_bytes: Some(0),
                        heap_peak_bytes: Some(0),
                        allocations_per_frame: Some(0.0),
                        gpu_bytes: Some(0),
                        created_after_warmup: Some(0),
                        duplicate_upload_bytes: Some(0),
                    });
                    bricks.len() - 1
                }
            };
            let brick = &mut bricks[index];
            brick.cpu_ms += row.cpu_ms / count;
            brick.cpu_ms_max = brick.cpu_ms_max.max(row.cpu_ms);
            if let Some(gpu) = row.gpu_ms {
                brick.gpu_ms = Some(brick.gpu_ms.unwrap_or(0.0) + gpu / count);
                brick.gpu_ms_max = Some(brick.gpu_ms_max.unwrap_or(0.0).max(gpu));
            }
            brick.heap_live_bytes = Some(row.heap_live_bytes);
            brick.heap_peak_bytes = brick
                .heap_peak_bytes
                .map(|peak| peak.max(row.heap_peak_bytes));
            brick.allocations_per_frame = brick
                .allocations_per_frame
                .map(|mean| mean + row.allocations as f64 / count);
            brick.gpu_bytes = Some(row.gpu_bytes);
            brick.created_after_warmup = brick
                .created_after_warmup
                .map(|sum| sum + row.created_after_warmup);
            brick.duplicate_upload_bytes = brick
                .duplicate_upload_bytes
                .map(|sum| sum + row.duplicate_upload_bytes);
        }
    }
    let cost = |brick: &ProfiledBrick| brick.cpu_ms + brick.gpu_ms.unwrap_or(0.0);
    bricks.sort_by(|a, b| cost(b).total_cmp(&cost(a)));
    let cpu_ms = bricks.iter().map(|brick| brick.cpu_ms).sum();
    let gpu_ms = bricks
        .iter()
        .any(|brick| brick.gpu_ms.is_some())
        .then(|| bricks.iter().filter_map(|brick| brick.gpu_ms).sum());
    let contents = json!({
        "request": serde_json::to_value(request).expect("requests serialize"),
        "bricks": serde_json::to_value(&bricks).expect("rows serialize"),
        "frames": serde_json::to_value(frames).expect("costs serialize"),
    });
    let total = bricks.len();
    let shown = match request.detail {
        Detail::Full => bricks,
        Detail::Short => bricks
            .into_iter()
            .take(request.top)
            .map(|brick| ProfiledBrick {
                heap_live_bytes: None,
                heap_peak_bytes: None,
                allocations_per_frame: None,
                gpu_bytes: None,
                created_after_warmup: None,
                duplicate_upload_bytes: None,
                ..brick
            })
            .collect(),
    };
    (
        ProfileResult {
            frames: frames.len() as u32,
            interval_ms: request.interval_ms,
            cpu_ms,
            gpu_ms,
            bricks: shown,
            total,
            temporary: true,
            artifact: None,
        },
        contents,
    )
}
