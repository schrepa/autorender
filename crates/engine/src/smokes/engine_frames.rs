use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{BrickList, BrickRegistry, brick_name, smoke_check, smoke_eq};
use engine_render::render_bricks::render_bricks;
use serde_json::{Value, json};

use super::requests::{
    Pixels, accepted, control, differing, frame, frames, names, pixels, spawn, status,
};

const FRAMES: u32 = 30;

struct Rendered {
    bricks: Vec<&'static str>,
    passes: Vec<&'static str>,
}

fn rendered(session: &mut dyn SmokeSession) -> Rendered {
    let mut found = Rendered {
        bricks: Vec::new(),
        passes: Vec::new(),
    };
    let world = session.app().world();
    for (brick, render) in world
        .get_resource::<BrickRegistry>()
        .into_iter()
        .flat_map(render_bricks)
    {
        found.bricks.push(brick);
        found.passes.extend(
            render
                .parts
                .iter()
                .flat_map(|part| part.decl.passes)
                .map(|pass| pass.name),
        );
    }
    found
}

fn lit() -> [&'static str; 2] {
    [
        brick_name::<engine_lighting::LightingModelBrick>(),
        brick_name::<engine_meshes::MeshesBrick>(),
    ]
}

fn core_only(base: &BrickList, found: &Rendered) -> BrickList {
    base.clone().without(found.bricks.iter().copied())
}

fn lit_meshes(base: &BrickList, found: &Rendered) -> BrickList {
    let kept = lit();
    base.clone().without(
        found
            .bricks
            .iter()
            .copied()
            .filter(|brick| !kept.contains(brick)),
    )
}

fn scene() -> Value {
    json!({
        "id": "engine-frames", "revision": 1,
        "camera": { "position": [0, 4, 9], "target": [0, 1, 0], "fov_degrees": 60, "near": 0.1, "far": 1000 },
        "materials": [
            { "id": "stone", "color": [0.7, 0.6, 0.5, 1] },
            { "id": "paint", "color": [0.25, 0.6, 0.85, 1] },
        ],
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": [
            { "id": "floor", "mesh": "box", "material": "stone", "transform": { "translation": [0, -0.5, 0], "scale": [12, 1, 12] } },
            { "id": "block", "mesh": "box", "material": "paint", "transform": { "translation": [0, 1, 0], "scale": [2, 2, 2] } },
            { "id": "lamp", "light": { "color": [1, 0.8, 0.6], "intensity": 10, "range": 10 }, "transform": { "translation": [0, 3, 2] } },
            { "id": "ground", "terrain": { "origin": [-128, -20, -128], "tiles": [2, 2], "spacing": 1, "layers": [{ "type": "noise", "seed": 3, "height": 8, "wavelength": 120 }] } },
        ],
    })
}

fn framed(session: &mut dyn SmokeSession, bricks: BrickList) -> Result<Value, SmokeFailure> {
    session.restart(bricks);
    spawn(session, &scene())?;
    control(session, "pause", json!({}), "pauses")?;
    control(session, "render", json!({}), "renders_without_errors")?;
    control(session, "step", json!({ "ticks": 3 }), "steps")?;
    control(session, "resume", json!({}), "resumes")?;
    let before = status(session, "status_answers")?;
    let shown = (0..FRAMES).filter(|_| session.frame()).count();
    smoke_eq!(json!(shown), json!(FRAMES), "frames_presented");
    let after = status(session, "status_answers_after_frames")?;
    let played = frame(&after).saturating_sub(frame(&before));
    smoke_check!(
        played >= u64::from(FRAMES),
        "frames_play",
        json!(played),
        json!(FRAMES)
    );
    control(session, "pause", json!({}), "pauses_again")?;
    let reply = frames(session, FRAMES, "frames_run_without_errors")?;
    Ok(reply["result"]["rendering"].clone())
}

fn captured(
    session: &mut dyn SmokeSession,
    check: &'static str,
) -> Result<Option<Pixels>, SmokeFailure> {
    let reply = accepted(session, "capture", json!({}), check)?;
    pixels(&reply)
}

pub(crate) fn core_only_frames(session: &mut dyn SmokeSession) -> SmokeResult {
    let base = session.bricks();
    let found = rendered(session);
    smoke_check!(
        !found.bricks.is_empty(),
        "render_bricks_listed",
        json!(found.bricks),
        json!("the render bricks of the list")
    );
    let rendering = framed(session, core_only(&base, &found))?;
    let passes = names(&rendering["render_passes"]);
    smoke_check!(
        !passes.is_empty(),
        "core_passes_run",
        json!(passes),
        json!("core passes")
    );
    let bricks: Vec<&String> = passes
        .iter()
        .filter(|pass| found.passes.iter().any(|known| *known == pass.as_str()))
        .collect();
    smoke_check!(
        bricks.is_empty(),
        "only_core_passes_run",
        json!(bricks),
        json!([])
    );
    Ok(())
}

pub(crate) fn lit_meshes_frames(session: &mut dyn SmokeSession) -> SmokeResult {
    let base = session.bricks();
    let found = rendered(session);
    let kept = lit();
    smoke_check!(
        kept.iter().all(|brick| found.bricks.contains(brick)),
        "lit_bricks_listed",
        json!(found.bricks),
        json!(kept)
    );
    framed(session, core_only(&base, &found))?;
    let plain = captured(session, "captures_core_only")?;
    let rendering = framed(session, lit_meshes(&base, &found))?;
    let passes = names(&rendering["render_passes"]);
    smoke_check!(
        passes.iter().any(|pass| pass == "opaque"),
        "opaque_pass_runs",
        json!(passes),
        json!("opaque")
    );
    let drawn = captured(session, "captures_lit_meshes")?;
    if let (Some(plain), Some(drawn)) = (&plain, &drawn) {
        let changed = differing(plain, drawn, 8)?;
        smoke_check!(
            changed > 0,
            "lit_meshes_draw",
            json!(changed),
            json!("> 0")
        );
    }
    Ok(())
}
