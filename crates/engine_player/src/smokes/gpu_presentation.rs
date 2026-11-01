use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_standard_smokes::requests::{accepted, control, frames, grid, number, seen, status, tick};
use serde_json::{Value, json};

use super::pixels::{Pixels, differing, pixels};

const ALPHAS: [f64; 3] = [0.0, 0.37, 0.81];
const QUIET_FRAMES: u32 = 13;
const SETTLE_LIMIT: u32 = 600;

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

fn read_back(
    session: &mut dyn SmokeSession,
    alpha: f64,
) -> Result<(Value, Option<Pixels>), SmokeFailure> {
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
    let pending = status(session, "status_after_preview")?;
    smoke_eq!(
        pending["observation"]["presentation"]["gpu_pending"].clone(),
        json!(true),
        "gpu_previews_leave_cpu_poses_pending"
    );
    partition(&rendering)?;
    Ok((rendering, pixels(&reply)?))
}

fn equivalent(
    reference: Option<&Pixels>,
    read_back: Option<&Pixels>,
) -> Result<(), SmokeFailure> {
    let (Some(reference), Some(read_back)) = (reference, read_back) else {
        return Ok(());
    };
    let total = reference.rgba.len() / 4;
    let changed = differing(reference, read_back, 8)?;
    let wanted = json!({ "changed": total / 1000 });
    smoke_eq!(
        seen(
            changed * 1000 <= total,
            json!({ "changed": changed, "total": total }),
            &wanted,
        ),
        wanted,
        "gpu_cpu_pixel_equivalence"
    );
    Ok(())
}

fn quiet(
    session: &mut dyn SmokeSession,
    name: &str,
    args: &Value,
    check: &'static str,
) -> Result<(), SmokeFailure> {
    let previewing = name == "present";
    let (mut calm, mut shadows) = (0, Value::Null);
    for _ in 0..SETTLE_LIMIT {
        let reply = control(session, name, args.clone(), check)?;
        shadows = reply["result"]["rendering"]["shadow_stats"].clone();
        let idle = |field: &str| number(&shadows[field]) == 0.0;
        let still = number(&shadows["tile_cut"]["cut"]) == 0.0
            && idle("tiles_drawn")
            && idle("sky_drawn")
            && (!previewing || number(&shadows["moving_cut"]["kept"]) == 0.0);
        calm = if still { calm + 1 } else { 0 };
        if calm >= QUIET_FRAMES {
            return Ok(());
        }
    }
    let wanted = json!({ "quiet_frames": QUIET_FRAMES });
    smoke_eq!(
        seen(
            calm >= QUIET_FRAMES,
            json!({ "control": name, "shadows": shadows }),
            &wanted,
        ),
        wanted,
        "frames_settle"
    );
    Ok(())
}

fn current(session: &mut dyn SmokeSession) -> Result<Option<Pixels>, SmokeFailure> {
    quiet(session, "render", &json!({}), "exact_frames_render")?;
    let reply = accepted(session, "capture", json!({}), "exact_capture")?;
    pixels(&reply)
}

fn compared(
    session: &mut dyn SmokeSession,
    alpha: f64,
    exact: Option<&Pixels>,
) -> Result<Case, SmokeFailure> {
    if exact.is_some() {
        quiet(
            session,
            "present",
            &json!({ "alpha": alpha }),
            "previews_settle",
        )?;
    }
    let first = accepted(
        session,
        "capture",
        json!({ "alpha": alpha }),
        "preview_capture",
    )?;
    let reference = first["result"]["rendering"].clone();
    partition(&reference)?;
    let reference_pixels = pixels(&first)?;
    let (read, read_pixels) = read_back(session, alpha)?;
    equivalent(exact, read_pixels.as_ref())?;
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

fn settled(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let mut shadows = Value::Null;
    for _ in 0..120 {
        shadows = frames(session, 2, "settles")?["result"]["rendering"]["shadow_stats"].clone();
        if number(&shadows["tile_cut"]["cut"]) == 0.0 && number(&shadows["tiles_drawn"]) == 0.0 {
            break;
        }
    }
    let wanted = json!({ "tile_cut": { "cut": 0 }, "tiles_drawn": 0 });
    smoke_eq!(
        seen(
            number(&shadows["tile_cut"]["cut"]) == 0.0 && number(&shadows["tiles_drawn"]) == 0.0,
            shadows.clone(),
            &wanted,
        ),
        wanted,
        "shadows_settle"
    );
    Ok(())
}

fn cases(session: &mut dyn SmokeSession, moving: bool) -> Result<Vec<Case>, SmokeFailure> {
    let partial = json!({ "position": [0, 30, 0], "target": [60, 20, 40], "vertical_fov_radians": 0.9, "near": 0.1, "far": 400 });
    let mut cases = Vec::new();
    for view in [None, Some(partial)] {
        if let Some(pose) = view {
            control(session, "camera", pose, "camera_moves")?;
        }
        let exact = if moving { None } else { current(session)? };
        for alpha in ALPHAS {
            cases.push(compared(session, alpha, exact.as_ref())?);
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
    if !moving {
        settled(session)?;
    }
    control(session, "render", json!({}), "renders")?;
    control(session, "step", json!({ "ticks": 3 }), "steps")?;
    let overview = status(session, "status_answers")?;
    let cases = cases(session, moving)?;
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
    materialized(session)
}

fn materialized(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let body = json!({ "component": "PresentationState" });
    let state = accepted(session, "inspect", body, "inspects_presentation")?;
    smoke_eq!(
        json!([
            state["observation"]["presentation"]["gpu_pending"],
            state["result"]["value"]["gpu_pending"],
            state["result"]["value"]["mode"],
        ]),
        json!([true, true, "interpolated"]),
        "inspection_keeps_gpu_poses"
    );
    control(session, "resume", json!({}), "resumes")?;
    let shown = (0..3).filter(|_| session.frame()).count();
    smoke_eq!(json!(shown), json!(3), "ordinary_frames_draw");
    let resumed = control(session, "pause", json!({}), "pauses_again")?;
    let presentation = &resumed["observation"]["presentation"];
    smoke_eq!(
        json!([presentation["mode"], presentation["gpu_pending"]]),
        json!(["interpolated", true]),
        "ordinary_frames_gpu"
    );
    Ok(())
}

pub(crate) fn gpu_presentation(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    presented(session, true)?;
    presented(session, false)
}
