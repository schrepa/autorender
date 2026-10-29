use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use serde_json::{Value, json};

use super::requests::{
    Pixels, accepted, control, differing, frames, grid, number, pixels, spawn, status, tick,
};

const ALPHAS: [f64; 3] = [0.0, 0.37, 0.81];

struct Case {
    candidates: f64,
    reference_visible: f64,
    read_back_visible: f64,
    pixels: Option<Pixels>,
}

fn partition(rendering: &Value) -> SmokeResult {
    let field = |name: &str| number(&rendering[name]);
    let parts = field("invalid_transforms")
        + field("missing_assets")
        + field("culled_instances")
        + field("visible_instances");
    engine_core::smoke_eq!(
        json!(field("render_candidates")),
        json!(parts),
        "counters_partition_candidates"
    );
    engine_core::smoke_eq!(
        json!([field("dropped_instances"), field("instances")]),
        json!([0.0, field("visible_instances")]),
        "submitted_equals_visible_below_budget"
    );
    Ok(())
}

fn read_back(session: &mut dyn SmokeSession, alpha: f64) -> Result<(Value, Option<Pixels>), SmokeFailure> {
    let mut reply = Value::Null;
    for _ in 0..50 {
        control(session, "present", json!({ "alpha": alpha }), "presents")?;
        reply = accepted(session, "capture", json!({ "alpha": alpha }), "read_back_capture")?;
        if number(&reply["result"]["rendering"]["read_back_age"]) <= 1.0 {
            break;
        }
    }
    let rendering = reply["result"]["rendering"].clone();
    engine_core::smoke_eq!(
        rendering["read_back_age"].clone(),
        json!(1),
        "preview_count_arrives"
    );
    let pending = status(session, "status_after_preview")?;
    engine_core::smoke_eq!(
        pending["observation"]["presentation"]["gpu_pending"].clone(),
        json!(true),
        "gpu_previews_leave_cpu_poses_pending"
    );
    partition(&rendering)?;
    Ok((rendering, pixels(&reply)?))
}

fn equivalent(reference: Option<&Pixels>, read_back: Option<&Pixels>) -> SmokeResult {
    let (Some(reference), Some(read_back)) = (reference, read_back) else {
        return Ok(());
    };
    let total = reference.rgba.len() / 4;
    let changed = differing(reference, read_back, 8)?;
    engine_core::smoke_check!(
        changed * 1000 <= total,
        "gpu_cpu_pixel_equivalence",
        json!({ "changed": changed, "total": total }),
        json!({ "changed": total / 1000 })
    );
    Ok(())
}

fn compared(session: &mut dyn SmokeSession, alpha: f64) -> Result<Case, SmokeFailure> {
    let first = accepted(session, "capture", json!({ "alpha": alpha }), "preview_capture")?;
    let reference = first["result"]["rendering"].clone();
    partition(&reference)?;
    let reference_pixels = pixels(&first)?;
    let (read, read_pixels) = read_back(session, alpha)?;
    equivalent(reference_pixels.as_ref(), read_pixels.as_ref())?;
    let candidates = number(&reference["render_candidates"]);
    let case = Case {
        candidates,
        reference_visible: number(&reference["visible_instances"]),
        read_back_visible: number(&read["visible_instances"]),
        pixels: reference_pixels,
    };
    let within = (candidates * 0.001).max(2.0);
    engine_core::smoke_check!(
        (case.read_back_visible - case.reference_visible).abs() <= within,
        "gpu_visible_count",
        json!({ "read_back": case.read_back_visible, "reference": case.reference_visible }),
        json!({ "within": within })
    );
    engine_core::smoke_eq!(
        read["render_candidates"].clone(),
        reference["render_candidates"].clone(),
        "gpu_candidates_match"
    );
    Ok(case)
}

fn settled(session: &mut dyn SmokeSession) -> SmokeResult {
    let mut shadows = Value::Null;
    for _ in 0..120 {
        shadows = frames(session, 2, "settles")?["result"]["rendering"]["shadows"].clone();
        if number(&shadows["tiles_missing"]) == 0.0 && number(&shadows["tiles_drawn"]) == 0.0 {
            break;
        }
    }
    engine_core::smoke_check!(
        number(&shadows["tiles_missing"]) == 0.0 && number(&shadows["tiles_drawn"]) == 0.0,
        "shadows_settle",
        shadows.clone(),
        json!({ "tiles_missing": 0, "tiles_drawn": 0 })
    );
    Ok(())
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
        .map(|case| json!([case.candidates, case.reference_visible, case.read_back_visible]))
        .collect();
    engine_core::smoke_check!(
        cases.iter().any(|case| case.reference_visible < case.candidates),
        "gpu_culling_exercised",
        json!(views),
        json!("a view culls")
    );
    engine_core::smoke_check!(
        cases.iter().all(|case| case.reference_visible > 0.0),
        "every_view_draws",
        json!(views),
        json!("every view draws")
    );
    Ok(cases)
}

fn presented(session: &mut dyn SmokeSession, moving: bool) -> SmokeResult {
    session.restart(&[]);
    spawn(session, &grid(24, moving))?;
    control(session, "pause", json!({}), "pauses")?;
    if !moving {
        settled(session)?;
    }
    control(session, "render", json!({}), "renders")?;
    control(session, "step", json!({ "ticks": 3 }), "steps")?;
    let overview = status(session, "status_answers")?;
    let cases = cases(session)?;
    if moving && let (Some(first), Some(last)) = (&cases[0].pixels, &cases[2].pixels) {
        let changed = differing(first, last, 8)?;
        engine_core::smoke_check!(changed > 0, "fractional_motion", json!(changed), json!("> 0"));
    }
    let after = status(session, "status_after_previews")?;
    engine_core::smoke_eq!(json!(tick(&after)), json!(tick(&overview)), "previews_keep_ticks");
    materialized(session)
}

fn materialized(session: &mut dyn SmokeSession) -> SmokeResult {
    let body = json!({ "component": "PresentationState" });
    let state = accepted(session, "inspect", body, "inspects_presentation")?;
    engine_core::smoke_eq!(
        json!([
            state["observation"]["presentation"]["gpu_pending"],
            state["result"]["value"]["gpu_pending"],
            state["result"]["value"]["mode"],
        ]),
        json!([true, false, "interpolated"]),
        "inspection_materializes"
    );
    control(session, "resume", json!({}), "resumes")?;
    let shown = (0..3).filter(|_| session.frame()).count();
    engine_core::smoke_eq!(json!(shown), json!(3), "ordinary_frames_draw");
    let resumed = control(session, "pause", json!({}), "pauses_again")?;
    let presentation = &resumed["observation"]["presentation"];
    engine_core::smoke_eq!(
        json!([presentation["mode"], presentation["gpu_pending"]]),
        json!(["interpolated", true]),
        "ordinary_frames_gpu"
    );
    Ok(())
}

pub(crate) fn gpu_presentation(session: &mut dyn SmokeSession) -> SmokeResult {
    presented(session, true)?;
    presented(session, false)
}
