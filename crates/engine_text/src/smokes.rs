use engine_core::smoke::{Checked, inspect, query, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_check};
use serde_json::{Value, json};

const OVERLAY: &str = "overlay";
const FRAMES: u32 = 240;

fn measured(session: &mut dyn SmokeSession, args: Value) -> Checked<(f64, f64)> {
    let size = query(session, "text_measure", args)?;
    Ok((
        size["width"].as_f64().unwrap_or(0.0),
        size["height"].as_f64().unwrap_or(0.0),
    ))
}

pub fn measure(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let hi = measured(session, json!({ "text": "Hi", "size": 20.0 }))?;
    let hello = measured(session, json!({ "text": "Hello", "size": 20.0 }))?;
    smoke_check!(hello.0 > hi.0, "hello_wider_than_hi", &hello.0, &hi.0);
    let line = "Hello Hello Hello Hello";
    let long = measured(session, json!({ "text": line, "size": 20.0 }))?;
    let wrapped = measured(
        session,
        json!({ "text": line, "size": 20.0, "max_width": 80.0 }),
    )?;
    smoke_check!(
        wrapped.1 > long.1,
        "wrapping_makes_it_taller",
        &wrapped.1,
        &long.1
    );
    Ok(())
}

fn overlaid(rendering: &Value) -> bool {
    rendering["render_passes"]
        .as_array()
        .is_some_and(|passes| passes.iter().any(|name| name == OVERLAY))
}

fn scene(entities: &Value) -> String {
    json!({
        "id": "text",
        "revision": 1,
        "textures": [],
        "materials": [],
        "meshes": [],
        "entities": entities,
    })
    .to_string()
}

pub fn drawn(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene(&json!([])));
    let empty = render(session)?;
    smoke_check!(
        !overlaid(&empty),
        "no_text_no_overlay",
        &empty["render_passes"],
        &OVERLAY
    );
    session.source(&scene(&json!([{
        "id": "label",
        "text": { "text": "Hello", "size": 24.0, "offset": [8.0, 8.0] },
    }])));
    let mut rendering = Value::Null;
    let mut stats = Value::Null;
    for _ in 0..FRAMES {
        rendering = render(session)?;
        stats = inspect(session, "text_stats")?;
        if stats["glyphs_drawn"].as_u64().unwrap_or(0) > 0 {
            break;
        }
    }
    smoke_check!(
        stats["glyphs_drawn"].as_u64().unwrap_or(0) > 0,
        "text_glyphs_drawn",
        &stats,
        &"glyphs_drawn"
    );
    smoke_check!(
        overlaid(&rendering),
        "text_overlaid",
        &rendering["render_passes"],
        &OVERLAY
    );
    Ok(())
}
