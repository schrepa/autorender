use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_standard_smokes::requests::{Asked, control, query, seen};
use serde_json::{Value, json};

const GARDEN: &str = include_str!("../../../../web/scripts/fixtures/garden.scene.json");

fn digest_after(session: &mut dyn SmokeSession, revision: &str) -> Asked {
    let args = json!({ "expected_revision": revision, "revision": revision });
    control(session, "reload", args, "reloads")?;
    let walking = json!({ "verbs": { "walk.move": [0, 1] } });
    control(session, "input", walking, "walks")?;
    control(session, "step", json!({ "ticks": 12 }), "steps_walking")?;
    control(session, "input", json!({}), "stops")?;
    Ok(query(session, "digest", json!({}), "digests")?["result"]["components"].take())
}

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let parsed = serde_json::from_str::<Value>(GARDEN);
    let wanted = json!(true);
    smoke_eq!(
        seen(parsed.is_ok(), json!(false), &wanted),
        wanted,
        "garden_parses"
    );
    let source = parsed.unwrap_or_default();
    let revision = source["revision"].as_u64().unwrap_or_default().to_string();
    session.source(&source.to_string());
    control(session, "pause", json!({}), "pauses")?;
    let first = digest_after(session, &revision)?;
    let again = digest_after(session, &revision)?;
    let counted = first
        .as_object()
        .is_some_and(|components| !components.is_empty());
    let wanted = json!({ "components": "counted", "again": first });
    smoke_eq!(
        seen(
            counted && again == first,
            json!({ "components": first, "again": again }),
            &wanted,
        ),
        wanted,
        "same_ticks_same_digest"
    );
    Ok(())
}
