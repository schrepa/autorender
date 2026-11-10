use engine_core::{
    Brick, ColorPass, PassName, Smoke, SmokeFailure, SmokeSession, smoke_check, smoke_eq,
};
use engine_particles::{PARTICLES, ParticlesBrick};
use engine_smoke::{above, control, counted_in_joined_passes, number, query, render, settled};
use engine_sprites::SPRITES;
use engine_standard_rendering::{ReflectionProbesBrick, ShadowsBrick, SpritesBrick, TextBrick};
use serde_json::{Value, json};

const FRAMES: u32 = 8;

const WHITE_SHEET: &str = "/////////////////////w==";

pub(crate) const SMOKES: [Smoke; 3] = [
    Smoke {
        name: "sprite_and_label_draw",
        run,
        without: &[],
    },
    Smoke {
        name: "flat_game_starts_clean",
        run: engine_smoke::starts_clean,
        without: &[],
    },
    Smoke {
        name: "kinds_draw_without_shadows_and_probes",
        run: without_shadows_and_probes,
        without: &[],
    },
];

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

fn with_sparks() -> Value {
    let mut document = scene();
    if let Some(entities) = document["entities"].as_array_mut() {
        entities.push(json!({
            "id": "sparks", "particles": {},
            "transform": {"translation": [-2.5, 0.0, 0.0]},
        }));
    }
    document
}

fn package<B: Brick>() -> &'static str {
    B::SOURCE.package()
}

fn listed(status: &Value, brick: &str) -> bool {
    status["features"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|name| name == brick)
}

fn label_glyphs(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let text = render(session)?["bricks"]["text_stats"].take();
    smoke_check!(
        number(&text["glyphs_drawn"]) > 0.0,
        "label_draws",
        text,
        json!({ "glyphs_drawn": above(0) })
    );
    Ok(())
}

fn without_shadows_and_probes(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&with_sparks().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    let status = query(session, "status", json!({}))?;
    smoke_eq!(
        json!({
            "sprites": listed(&status, package::<SpritesBrick>()),
            "text": listed(&status, package::<TextBrick>()),
            "particles": listed(&status, package::<ParticlesBrick>()),
            "shadows": listed(&status, package::<ShadowsBrick>()),
            "probes": listed(&status, package::<ReflectionProbesBrick>()),
        }),
        json!({
            "sprites": true, "text": true, "particles": true, "shadows": false, "probes": false,
        }),
        "the_list_has_sprites_text_and_particles_without_shadows_and_probes"
    );
    smoke_eq!(
        status["unmet"],
        json!([]),
        "the_list_starts_with_no_unmet_need"
    );
    for kind in [SPRITES, PARTICLES] {
        counted_in_joined_passes(session, kind.name(), FRAMES)?;
    }
    label_glyphs(session)
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
    label_glyphs(session)
}
