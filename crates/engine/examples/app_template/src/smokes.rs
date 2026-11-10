use engine_core::{PassName, Smoke, SmokeFailure, SmokeSession, smoke_check};
use engine_smoke::{above, control, counted_in_joined_passes, number, query, settled};
use serde_json::{Value, json};

use crate::bricks::glimmer::{GLIMMER, GlimmerPass, LOOK};

const FRAMES: u32 = 8;

pub(crate) const SMOKES: [Smoke; 3] = [
    Smoke {
        name: "glimmer_counted_in_every_pass",
        run,
        without: &[],
    },
    Smoke {
        name: "dropped_material_leaves_nothing_waiting",
        run: dropped,
        without: &[],
    },
    Smoke {
        name: "app_template_starts_clean",
        run: engine_smoke::starts_clean,
        without: &[],
    },
];

fn scene() -> Value {
    json!({
        "id": "glimmer-passes", "revision": 1,
        "materials": [{
            "id": "disc", "color": [1.0, 1.0, 1.0, 1.0], "slots": {"opacity": LOOK},
            "alpha_mode": "mask", "alpha_cutoff": 0.5, "cull": "none", "lighting": "unlit",
        }],
        "meshes": [],
        "entities": [
            {"id": "spark", "drawn": {"kind": GLIMMER.name(), "material": "disc"},
             "transform": {"translation": [0.0, 1.0, 0.0]}},
            {"id": "camera", "transform": {"translation": [0.0, 3.0, 8.0], "rotation_degrees": [-20.0, 0.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.1, "far": 200.0, "projection": {"Perspective": {"vertical_fov_radians": 0.9}}},
             "view": {}},
        ],
    })
}

fn emptied() -> Value {
    let mut emptied = scene();
    emptied["materials"] = json!([]);
    if let Some(entities) = emptied["entities"].as_array_mut() {
        entities.retain(|entity| entity["id"] != "spark");
    }
    emptied
}

fn glimmer_cells(rendering: &Value) -> Vec<Value> {
    rendering["kind_passes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|cell| cell["kind"] == GLIMMER.name())
        .cloned()
        .collect()
}

fn dropped(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
    control(session, "pause", json!({}))?;
    let drawn = settled(session)?;
    let brick = glimmer_cells(&drawn)
        .first()
        .and_then(|cell| cell["brick"].as_str())
        .unwrap_or_default()
        .to_owned();
    smoke_check!(
        !brick.is_empty(),
        "glimmer_cells_name_their_brick",
        glimmer_cells(&drawn),
        json!([{ "kind": GLIMMER.name(), "brick": "the glimmer brick" }])
    );
    session.source(&emptied().to_string());
    control(session, "pause", json!({}))?;
    let shown = settled(session)?;
    let held: Vec<Value> = glimmer_cells(&shown)
        .into_iter()
        .filter(|cell| cell["state"]["state"] == "waiting" || cell["state"]["state"] == "refused")
        .collect();
    let why = query(session, "why", json!({ "brick": brick }))?;
    let got = json!({
        "cells": held,
        "waiting": why["waiting"],
        "drawing": why["drawing"],
        "failed": why["failed"],
    });
    let wanted = json!({ "cells": [], "waiting": [], "drawing": [], "failed": [] });
    smoke_check!(
        got == wanted,
        "a_dropped_material_leaves_nothing_waiting",
        got,
        wanted
    );
    Ok(())
}

fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    let counted = counted_in_joined_passes(session, GLIMMER.name(), FRAMES)?;
    let own = counted
        .get(GlimmerPass::NAME)
        .cloned()
        .unwrap_or(Value::Null);
    smoke_check!(
        own["state"]["state"] == "drew" && number(&own["state"]["draws"]) > 0.0,
        "glimmer_draws_in_its_own_pass",
        own,
        json!({ "state": "drew", "draws": above(0) })
    );
    Ok(())
}
