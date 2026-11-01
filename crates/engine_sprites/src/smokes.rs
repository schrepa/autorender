use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

pub fn drawn(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let scene = json!({
        "id": "sprites",
        "revision": 1,
        "textures": [{ "id": "sheet", "width": 1, "height": 1, "rgba8": "/////w==" }],
        "materials": [],
        "meshes": [],
        "entities": [{
            "id": "sprite",
            "sprite": { "sheet": "sheet", "rect": [0, 0, 1, 1], "tint": [1, 1, 1, 1] },
        }],
    });
    session.source(&scene.to_string());
    smoke_eq!(session.frame(), true, "sprites_frame");
    Ok(())
}
