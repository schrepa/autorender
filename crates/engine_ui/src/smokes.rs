use engine_core::smoke::{Checked, control, inspect, query, refuse, render, renders_until};
use engine_core::{SmokeFailure, SmokeSession, smoke_check, smoke_eq};
use serde_json::{Value, json};

const OVERLAY: &str = "overlay";
const FRAMES: u32 = 240;

fn scene(entities: &Value) -> String {
    json!({
        "id": "ui",
        "revision": 1,
        "textures": [],
        "materials": [],
        "meshes": [],
        "entities": entities,
    })
    .to_string()
}

fn menu() -> Value {
    json!([
        {
            "id": "menu",
            "ui": {
                "root": {
                    "id": "frame",
                    "absolute": { "left": 10.0, "top": 20.0 },
                    "width": { "px": 200.0 },
                    "height": { "px": 100.0 },
                    "padding": [5.0, 5.0, 5.0, 5.0],
                    "gap": 10.0,
                    "color": [0.1, 0.1, 0.1, 0.9],
                    "children": [
                        {
                            "id": "start",
                            "kind": "button",
                            "height": { "px": 30.0 },
                            "text": "Start",
                            "color": [0.2, 0.2, 0.3, 1.0],
                            "action": { "emit": "start" },
                        },
                        {
                            "id": "options",
                            "kind": "button",
                            "height": { "px": 30.0 },
                            "text": "Options",
                            "color": [0.2, 0.2, 0.3, 1.0],
                            "action": { "go": "options" },
                        },
                    ],
                },
            },
        },
        {
            "id": "options_title",
            "ui": {
                "screen": "options",
                "root": { "id": "title", "kind": "label", "text": "Options" },
            },
        },
    ])
}

fn tree(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "ui_tree", json!({}))
}

fn rect_of(tree: &Value, id: &str) -> Value {
    tree["nodes"]
        .as_array()
        .and_then(|nodes| nodes.iter().find(|node| node["id"] == id))
        .map_or(Value::Null, |node| node["rect"].clone())
}

pub fn layout_press(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene(&menu()));
    control(session, "pause", json!({}))?;
    session.step(1);
    let laid = tree(session)?;
    smoke_eq!(
        rect_of(&laid, "frame"),
        json!([10.0, 20.0, 200.0, 100.0]),
        "frame_rect"
    );
    smoke_eq!(
        rect_of(&laid, "start"),
        json!([15.0, 25.0, 190.0, 30.0]),
        "start_rect"
    );
    smoke_eq!(
        rect_of(&laid, "options"),
        json!([15.0, 65.0, 190.0, 30.0]),
        "options_rect"
    );
    smoke_eq!(rect_of(&laid, "title"), Value::Null, "other_screen_hidden");
    let layouts = inspect(session, "ui_stats")?["layouts"].clone();
    session.step(3);
    smoke_eq!(
        inspect(session, "ui_stats")?["layouts"],
        layouts,
        "static_ui_lays_out_once"
    );
    let unknown = refuse(session, "control", "ui_press", json!({ "id": "nowhere" }))?;
    smoke_eq!(
        unknown["code"],
        json!("invalid_value"),
        "unknown_widget_refused"
    );
    control(session, "ui_press", json!({ "id": "start" }))?;
    session.step(1);
    smoke_eq!(tree(session)?["events"], json!(["start"]), "press_emits");
    smoke_eq!(
        inspect(session, "ui_stats")?["events"],
        json!(1),
        "events_counted"
    );
    control(session, "ui_press", json!({ "id": "options" }))?;
    session.step(1);
    smoke_eq!(tree(session)?["events"], json!([]), "events_clear");
    smoke_eq!(
        query(session, "screens", json!({}))?["top"],
        json!("options"),
        "press_changes_screen"
    );
    smoke_eq!(
        rect_of(&tree(session)?, "title") != Value::Null,
        true,
        "new_screen_shown"
    );
    control(session, "screen", json!({ "action": "pop" }))?;
    session.source(&scene(&json!([])));
    session.step(1);
    smoke_eq!(
        inspect(session, "ui_stats")?["nodes"],
        json!(0),
        "removed_ui_has_no_nodes"
    );
    Ok(())
}

fn overlaid(rendering: &Value) -> bool {
    rendering["render_passes"]
        .as_array()
        .is_some_and(|passes| passes.iter().any(|name| name == OVERLAY))
}

pub fn drawn(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene(&json!([])));
    let empty = render(session)?;
    smoke_check!(
        !overlaid(&empty),
        "no_ui_no_overlay",
        &empty["render_passes"],
        &OVERLAY
    );
    session.source(&scene(&menu()));
    let mut rendering = Value::Null;
    let mut stats = Value::Null;
    for _ in 0..FRAMES {
        rendering = render(session)?;
        stats = inspect(session, "ui_stats")?;
        if stats["rects_drawn"].as_u64().unwrap_or(0) > 0 {
            break;
        }
    }
    smoke_check!(
        stats["rects_drawn"].as_u64().unwrap_or(0) > 0,
        "ui_rects_drawn",
        &stats,
        &"rects_drawn"
    );
    smoke_check!(
        overlaid(&rendering),
        "ui_overlaid",
        &rendering["render_passes"],
        &OVERLAY
    );
    session.source(&scene(&json!([])));
    let (gone, after) = renders_until(session, FRAMES, |rendering| !overlaid(rendering))?;
    smoke_check!(
        gone,
        "removed_ui_no_overlay",
        &after["render_passes"],
        &OVERLAY
    );
    Ok(())
}
