use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::rendered_when;
use engine_standard_smokes::requests::{accepted, control, grid, number, seen};
use serde_json::{Value, json};

const SIDE: usize = 24;

struct Counted {
    candidates: f64,
    visible: f64,
}

fn meshes(reply: &Value) -> &Value {
    &reply["result"]["rendering"]["bricks"]["meshes_stats"]
}

fn partition(reply: &Value) -> Result<(), SmokeFailure> {
    let field = |name: &str| number(&meshes(reply)[name]);
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

fn counted(session: &mut dyn SmokeSession) -> Result<Counted, SmokeFailure> {
    let preview = accepted(session, "capture", json!({ "alpha": 0 }), "preview_capture")?;
    partition(&preview)?;
    let read = rendered_when(
        session,
        |session| {
            control(session, "present", json!({ "alpha": 0 }), "presents")?;
            accepted(
                session,
                "capture",
                json!({ "alpha": 0 }),
                "read_back_capture",
            )
        },
        |reply| number(&meshes(reply)["read_back_age"]) <= 1.0,
    )?;
    partition(&read)?;
    smoke_eq!(
        meshes(&read)["render_candidates"].clone(),
        meshes(&preview)["render_candidates"].clone(),
        "gpu_candidates_match"
    );
    Ok(Counted {
        candidates: number(&meshes(&read)["render_candidates"]),
        visible: number(&meshes(&read)["visible_instances"]),
    })
}

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid(SIDE).to_string());
    control(session, "pause", json!({}), "pauses")?;
    control(session, "step", json!({ "ticks": 3 }), "steps")?;
    let whole = counted(session)?;
    let partial = json!({ "position": [0, 30, 0], "target": [60, 20, 40], "vertical_fov_radians": 0.9, "near": 0.1, "far": 400 });
    control(session, "camera", partial, "camera_moves")?;
    let cases = [whole, counted(session)?];
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
    Ok(())
}
