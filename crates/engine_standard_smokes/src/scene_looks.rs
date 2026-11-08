use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::{control, grid, inspected, serve};

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let mut document = grid(4, false);
    document["environment"] = json!({ "bloom": { "intensity": 0.3 } });
    document["looks"] =
        json!([{ "name": "bright", "environment": { "bloom": { "intensity": 1.5 } } }]);
    serve(session, &document)?;
    control(
        session,
        "environment",
        json!({ "merge": true, "haze": { "spread": 0.25 } }),
        "sets_haze",
    )?;
    control(
        session,
        "scene_look",
        json!({ "name": "bright" }),
        "shows_look",
    )?;
    smoke_eq!(
        field(session, "bloom", "intensity")?,
        json!(1.5),
        "look_sets_its_sections"
    );
    smoke_eq!(
        field(session, "haze", "spread")?,
        json!(0.25),
        "look_keeps_other_sections"
    );
    control(
        session,
        "scene_look",
        json!({ "name": "scene" }),
        "shows_scene_look",
    )?;
    smoke_eq!(
        field(session, "bloom", "intensity")?,
        json!(0.30000001192092896),
        "scene_look_restores"
    );
    Ok(())
}

fn field(
    session: &mut dyn SmokeSession,
    component: &str,
    name: &str,
) -> Result<Value, SmokeFailure> {
    Ok(inspected(session, component, "inspects_section")?[name].clone())
}
