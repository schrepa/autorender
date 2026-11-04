use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_standard_smokes::requests::{accepted, control, grid, number, seen, status, tick};
use serde_json::{Value, json};

use super::pixels::{Pixels, differing, pixels};

const ALPHAS: [f64; 3] = [0.0, 0.37, 0.81];

struct Case {
    candidates: f64,
    visible: f64,
    pixels: Option<Pixels>,
}

fn meshes(rendering: &Value) -> &Value {
    &rendering["meshes_stats"]
}

fn partition(rendering: &Value) -> Result<(), SmokeFailure> {
    let field = |name: &str| number(&meshes(rendering)[name]);
    let parts = field("invalid_transforms")
        + field("missing_assets")
        + field("culled_instances")
        + field("visible_instances");
    smoke_eq!(
        json!(field("render_candidates")),
        json!(parts),
        "counters_partition_candidates"
    );
    smoke_eq!(
        json!([field("dropped_instances"), field("instances")]),
        json!([0.0, field("visible_instances")]),
        "submitted_equals_visible_below_budget"
    );
    Ok(())
}

fn read_back(session: &mut dyn SmokeSession, alpha: f64) -> Result<Value, SmokeFailure> {
    let mut reply = Value::Null;
    for _ in 0..50 {
        control(session, "present", json!({ "alpha": alpha }), "presents")?;
        reply = accepted(
            session,
            "capture",
            json!({ "alpha": alpha }),
            "read_back_capture",
        )?;
        if number(&meshes(&reply["result"]["rendering"])["read_back_age"]) <= 1.0 {
            break;
        }
    }
    let rendering = reply["result"]["rendering"].clone();
    let age = meshes(&rendering)["read_back_age"].clone();
    let wanted = json!({ "at_most": 1 });
    smoke_eq!(
        seen(number(&age) <= 1.0, age, &wanted),
        wanted,
        "preview_count_arrives"
    );
    partition(&rendering)?;
    Ok(rendering)
}

fn compared(session: &mut dyn SmokeSession, alpha: f64) -> Result<Case, SmokeFailure> {
    let first = accepted(
        session,
        "capture",
        json!({ "alpha": alpha }),
        "preview_capture",
    )?;
    let reference = first["result"]["rendering"].clone();
    partition(&reference)?;
    let reference_pixels = pixels(&first)?;
    let read = read_back(session, alpha)?;
    let case = Case {
        candidates: number(&meshes(&read)["render_candidates"]),
        visible: number(&meshes(&read)["visible_instances"]),
        pixels: reference_pixels,
    };
    smoke_eq!(
        meshes(&read)["render_candidates"].clone(),
        meshes(&reference)["render_candidates"].clone(),
        "gpu_candidates_match"
    );
    Ok(case)
}

fn cases(session: &mut dyn SmokeSession) -> Result<Vec<Case>, SmokeFailure> {
    let partial = json!({ "position": [0, 30, 0], "target": [60, 20, 40], "vertical_fov_radians": 0.9, "near": 0.1, "far": 400 });
    let mut cases = Vec::new();
    for view in [None, Some(partial)] {
        if let Some(pose) = view {
            control(session, "camera", pose, "camera_moves")?;
        }
        for alpha in ALPHAS {
            cases.push(compared(session, alpha)?);
        }
    }
    let views: Vec<Value> = cases
        .iter()
        .map(|case| json!([case.candidates, case.visible]))
        .collect();
    let wanted = json!("a view culls");
    smoke_eq!(
        seen(
            cases.iter().any(|case| case.visible < case.candidates),
            json!(views),
            &wanted,
        ),
        wanted,
        "gpu_culling_exercised"
    );
    let wanted = json!("every view draws");
    smoke_eq!(
        seen(
            cases.iter().all(|case| case.visible > 0.0),
            json!(views),
            &wanted,
        ),
        wanted,
        "every_view_draws"
    );
    Ok(cases)
}

fn presented(session: &mut dyn SmokeSession, moving: bool) -> Result<(), SmokeFailure> {
    session.without(&[]);
    session.source(&grid(24, moving).to_string());
    control(session, "pause", json!({}), "pauses")?;
    control(session, "render", json!({}), "renders")?;
    control(session, "step", json!({ "ticks": 3 }), "steps")?;
    let overview = status(session, "status_answers")?;
    let cases = cases(session)?;
    if moving && let (Some(first), Some(last)) = (&cases[0].pixels, &cases[2].pixels) {
        let changed = differing(first, last, 8)?;
        let wanted = json!("> 0");
        smoke_eq!(
            seen(changed > 0, json!(changed), &wanted),
            wanted,
            "fractional_motion"
        );
    }
    let after = status(session, "status_after_previews")?;
    smoke_eq!(
        json!(tick(&after)),
        json!(tick(&overview)),
        "previews_keep_ticks"
    );
    let body = json!({ "component": "PresentationState" });
    accepted(session, "inspect", body, "inspects_presentation")?;
    control(session, "resume", json!({}), "resumes")?;
    let shown = (0..3).filter(|_| session.frame()).count();
    smoke_eq!(json!(shown), json!(3), "ordinary_frames_draw");
    Ok(())
}

pub(crate) fn gpu_presentation(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    presented(session, true)?;
    presented(session, false)
}
