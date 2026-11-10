use engine_core::{ColorPass, PassName, Smoke, SmokeFailure, SmokeSession, smoke_check};
use engine_smoke::{above, control, counted_in_joined_passes, number, render, settled};
use engine_sprites::SPRITES;
use serde_json::{Value, json};

const FRAMES: u32 = 8;

const WHITE_SHEET: &str = "/////////////////////w==";

pub(crate) const SMOKES: [Smoke; 1] = [Smoke {
    name: "sprite_and_label_draw",
    run,
}];

fn scene() -> Value {
    json!({
        "id": "flat", "revision": 1,
        "textures": [{"id": "sheet", "width": 2, "height": 2, "rgba8": WHITE_SHEET}],
        "materials": [{
            "id": "glass", "color": [0.4, 0.7, 1.0, 1.0], "opacity": 0.5, "alpha_mode": "blend",
            "cull": "none", "lighting": "unlit",
        }],
        "meshes": [],
        "entities": [
            {"id": "sprite", "sprite": {"sheet": "sheet", "rect": [0.0, 0.0, 1.0, 1.0], "tint": [1.0, 0.5, 0.2, 1.0]},
             "transform": {"translation": [0.0, 0.0, 0.0], "scale": [2.0, 2.0, 2.0]}},
            {"id": "pane", "sprite": {"sheet": "sheet", "rect": [0.0, 0.0, 1.0, 1.0], "tint": [1.0, 1.0, 1.0, 1.0]},
             "drawn": {"kind": SPRITES.name(), "material": "glass"},
             "transform": {"translation": [2.5, 0.0, 1.0], "scale": [1.5, 1.5, 1.5]}},
            {"id": "label", "text": {"text": "Flat", "size": 32.0}},
            {"id": "camera", "transform": {"translation": [0.0, 0.0, 10.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.1, "far": 100.0, "projection": {"Perspective": {"vertical_fov_radians": 0.9}}},
             "view": {}},
        ],
    })
}

fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    let counted = counted_in_joined_passes(session, SPRITES.name(), FRAMES)?;
    let color = counted.get(ColorPass::NAME).cloned().unwrap_or(Value::Null);
    smoke_check!(
        color["state"]["state"] == "drew" && number(&color["state"]["draws"]) > 0.0,
        "sprite_draws",
        color,
        json!({ "state": "drew", "draws": above(0) })
    );
    let text = render(session)?["bricks"]["text_stats"].take();
    smoke_check!(
        number(&text["glyphs_drawn"]) > 0.0,
        "label_draws",
        text,
        json!({ "glyphs_drawn": above(0) })
    );
    Ok(())
}
