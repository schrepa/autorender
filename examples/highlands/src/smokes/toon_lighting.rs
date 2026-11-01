use engine_core::smoke::{Checked, ask, capture};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::{aim, open, settle};

const VIEW: &str = "close";

const FRAMES: u32 = 8;

const UNLINKED: [&str; 3] = ["shader_link", "unknown_module", "place_not_filled"];

pub(super) fn standard(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    open(session, false)?;
    looked(session)
}

pub(super) fn toon(session: &mut dyn SmokeSession, standard: Value) -> Result<(), SmokeFailure> {
    open(session, false)?;
    filled(session)?;
    let toon = looked(session)?;
    smoke_eq!(
        toon != standard,
        true,
        "the_toon_lighting_changes_the_frame"
    );
    Ok(())
}

fn filled(session: &mut dyn SmokeSession) -> Checked<()> {
    let listing = ask(session, "describe", json!({}))?;
    let fillers: Vec<&str> = listing["bricks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|brick| fills_lighting(&brick["fills"]))
        .filter_map(|brick| brick["name"].as_str())
        .collect();
    smoke_eq!(
        fillers,
        ["highlands_toon_lighting"],
        "the_toon_lighting_fills_the_lighting_slot"
    );
    Ok(())
}

fn fills_lighting(fills: &Value) -> bool {
    fills
        .as_array()
        .into_iter()
        .flatten()
        .any(|filled| filled == "lighting" || filled["place"] == "lighting")
}

fn looked(session: &mut dyn SmokeSession) -> Checked<Value> {
    aim(session, VIEW)?;
    let mut rendering = Value::Null;
    for _ in 0..FRAMES {
        rendering = rendered(session)?;
    }
    let passes = &rendering["render_passes"];
    smoke_eq!(
        passes.as_array().is_some_and(|passes| !passes.is_empty()),
        true,
        "the_lit_scene_runs_passes"
    );
    settle(session)?;
    capture(session)
}

fn rendered(session: &mut dyn SmokeSession) -> Checked<Value> {
    let reply = session.request("control", json!({"name": "render", "args": {}}));
    let refused = reply
        .as_ref()
        .err()
        .map(|refused| refused["error"].clone())
        .unwrap_or_default();
    let unlinked = UNLINKED.iter().find(|code| refused["code"] == **code);
    smoke_eq!(unlinked, None, "the_lighting_links_with_its_faces");
    smoke_eq!(&refused, &Value::Null, "the_lit_scene_renders");
    Ok(reply
        .map(|mut answered| answered["result"]["rendering"].take())
        .unwrap_or_default())
}
