use engine_profile::Row;
use serde_json::{Value, json};

use crate::protocol::{Detail, ProfileRequest};
use crate::results::{ProfileResult, ProfiledRow};

pub fn profiled(frames: &[Vec<Row>], request: &ProfileRequest) -> (ProfileResult, Value) {
    let count = frames.len().max(1) as f64;
    let mut rows: Vec<ProfiledRow> = Vec::new();
    for frame in frames {
        for row in frame {
            let index = match rows.iter().position(|profiled| profiled.name == row.name) {
                Some(index) => index,
                None => {
                    rows.push(ProfiledRow {
                        name: row.name,
                        cpu_ms: 0.0,
                        cpu_ms_max: 0.0,
                        runs_per_frame: 0.0,
                        allocations_per_frame: Some(0.0),
                    });
                    rows.len() - 1
                }
            };
            let profiled = &mut rows[index];
            profiled.cpu_ms += row.ms / count;
            profiled.cpu_ms_max = profiled.cpu_ms_max.max(row.worst_ms);
            profiled.runs_per_frame += row.runs as f64 / count;
            profiled.allocations_per_frame = profiled
                .allocations_per_frame
                .map(|mean| mean + row.allocations as f64 / count);
        }
    }
    rows.sort_by(|a, b| b.cpu_ms.total_cmp(&a.cpu_ms));
    let contents = json!({
        "request": serde_json::to_value(request).expect("request_serializes"),
        "rows": serde_json::to_value(&rows).expect("row_serializes"),
    });
    let total = rows.len();
    let shown = match request.detail {
        Detail::Full => rows,
        Detail::Short => rows
            .into_iter()
            .take(request.top)
            .map(|row| ProfiledRow {
                allocations_per_frame: None,
                ..row
            })
            .collect(),
    };
    (
        ProfileResult {
            frames: frames.len() as u32,
            interval_ms: request.interval_ms,
            rows: shown,
            total,
            temporary: true,
            artifact: None,
        },
        contents,
    )
}
