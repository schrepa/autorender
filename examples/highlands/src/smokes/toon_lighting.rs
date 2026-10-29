use engine::smoke::{SmokeResult, SmokeSession};
use engine::smoke_check;
use engine::terrain::data::smokes::{Checked, ask, capture, control, settle};
use highlands_toon_lighting::ToonLightingBrick;
use serde_json::{Value, json};

use super::{LEFT_OUT, aim, open};
use crate::{Highlands, Options, RockMaps};

const VIEW: &str = "close";

const FRAMES: u32 = 8;

const UNLINKED: [&str; 3] = ["shader_link", "face_missing", "slot_unfilled"];

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    open(session, false)?;
    let standard = looked(session)?;
    let bricks = session
        .bricks()
        .without(LEFT_OUT)
        .fill::<ToonLightingBrick>();
    session.restart(bricks);
    furnished(session)?;
    filled(session)?;
    let toon = looked(session)?;
    smoke_check!(
        toon != standard,
        "the_toon_lighting_changes_the_frame",
        &toon,
        &json!({ "differs_from": standard })
    );
    Ok(())
}

fn furnished(session: &mut dyn SmokeSession) -> Checked<()> {
    let options = Options {
        rocks: false,
        ..Options::default()
    };
    let built = Highlands::furnish(session.app(), &options, &RockMaps::default()).map(drop);
    smoke_check!(
        built.is_ok(),
        "highlands_builds_with_the_toon_lighting",
        &built.as_ref().err(),
        &Value::Null
    );
    control(session, "pause", json!({}))?;
    Ok(())
}

fn filled(session: &mut dyn SmokeSession) -> Checked<()> {
    let listing = ask(session, "describe", json!({}))?;
    let fillers: Vec<&str> = listing["bricks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|brick| brick["slot"] == "lighting")
        .filter_map(|brick| brick["name"].as_str())
        .collect();
    smoke_check!(
        fillers == ["toon_lighting"],
        "the_toon_lighting_fills_the_lighting_slot",
        &fillers,
        &json!(["toon_lighting"])
    );
    Ok(())
}

fn looked(session: &mut dyn SmokeSession) -> Checked<Value> {
    aim(session, VIEW)?;
    let mut rendering = Value::Null;
    for _ in 0..FRAMES {
        rendering = rendered(session)?;
    }
    let passes = &rendering["render_passes"];
    smoke_check!(
        passes.as_array().is_some_and(|passes| !passes.is_empty()),
        "the_lit_scene_runs_passes",
        passes,
        &json!({ "not": [] })
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
    smoke_check!(
        !UNLINKED.iter().any(|code| refused["code"] == *code),
        "the_lighting_links_with_its_faces",
        &refused,
        &json!({ "none_of": UNLINKED })
    );
    smoke_check!(
        reply.is_ok(),
        "the_lit_scene_renders",
        &refused,
        &json!("rendered")
    );
    Ok(reply
        .map(|mut answered| answered["result"]["rendering"].take())
        .unwrap_or_default())
}
