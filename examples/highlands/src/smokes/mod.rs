mod jobs;
mod lod;
mod toon_lighting;
mod pond;
mod torrent;

use engine::host::Jobs;
use engine::smoke::{SmokeEntry, SmokeSession};
use engine::smoke_check;
use engine::terrain::data::smokes::{Checked, control};
use serde_json::{Value, json};

use crate::{Highlands, Options, RockMaps};

const GAME: &str = "highlands";

const LEFT_OUT: [&str; 2] = ["collision", "physics"];

pub fn smokes() -> Vec<SmokeEntry> {
    vec![
        SmokeEntry {
            name: "highlands_jobs",
            brick: GAME,
            gpu: true,
            run: jobs::run,
        },
        SmokeEntry {
            name: "highlands_lod",
            brick: GAME,
            gpu: true,
            run: lod::run,
        },
        SmokeEntry {
            name: "highlands_torrent",
            brick: GAME,
            gpu: true,
            run: torrent::run,
        },
        SmokeEntry {
            name: "highlands_toon_lighting",
            brick: GAME,
            gpu: true,
            run: toon_lighting::run,
        },
        SmokeEntry {
            name: "highlands_pond",
            brick: GAME,
            gpu: true,
            run: pond::run,
        },
    ]
}

fn open(session: &mut dyn SmokeSession, in_place: bool) -> Checked<()> {
    session.restart(session.bricks().without(LEFT_OUT));
    if in_place {
        session.app().insert_resource(Jobs::default());
    }
    let options = Options {
        rocks: false,
        ..Options::default()
    };
    let built = Highlands::furnish(session.app(), &options, &RockMaps::default()).map(drop);
    smoke_check!(
        built.is_ok(),
        "highlands_builds",
        &built.as_ref().err(),
        &Value::Null
    );
    control(session, "pause", json!({}))?;
    Ok(())
}

fn aim(session: &mut dyn SmokeSession, view: &str) -> Checked<()> {
    let aimed = session.request(
        "control",
        json!({"name": "scene_view", "args": {"name": view}}),
    );
    smoke_check!(
        aimed.is_ok(),
        "the_view_is_known",
        &aimed.as_ref().err().map(|refused| &refused["error"]),
        &view
    );
    Ok(())
}
