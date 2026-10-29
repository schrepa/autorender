use engine_core::math::{cos, sin};
use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::scene_kit::{SceneFixture, ask, component, render};

fn ball(rings: u32, segments: u32) -> Value {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for ring in 0..=rings {
        let phi = std::f32::consts::PI * ring as f32 / rings as f32;
        for segment in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * segment as f32 / segments as f32;
            let normal = [sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta)];
            vertices.push(json!({
                "position": normal, "normal": normal,
                "uv": [segment as f32 / segments as f32, ring as f32 / rings as f32],
            }));
        }
    }
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * (segments + 1) + segment;
            let b = a + segments + 1;
            indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    json!({"kind": "inline", "vertices": vertices, "indices": indices})
}

fn fixture() -> Value {
    let ribbon: Vec<Value> = [0.0_f32, 1.0, 2.0]
        .into_iter()
        .flat_map(|y| {
            [-0.3_f32, 0.3].map(
                |x| json!({"position": [x, y, 0], "normal": [0, 0, 1], "uv": [x + 0.3, y / 2.0]}),
            )
        })
        .collect();
    json!({
        "id": "mesh-fixture", "revision": 1,
        "camera": {"position": [0, 1, 4], "target": [0, 1, 0], "fov_degrees": 60, "near": 0.1, "far": 2000},
        "materials": [
            {"id": "plain", "color": [0.8, 0.4, 0.2, 1], "double_sided": true},
            {"id": "stone", "color": [0.5, 0.55, 0.6, 1], "double_sided": true},
        ],
        "meshes": [
            {"id": "ribbon", "source": {"kind": "inline", "vertices": ribbon,
                "indices": [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]}},
            {"id": "ball", "source": ball(24, 32)},
        ],
        "entities": [
            {"id": "limb", "mesh": "ribbon", "material": "plain"},
            {"id": "near-ball", "mesh": "ball", "material": "stone", "transform": {"translation": [2.5, 1, -2]}},
            {"id": "far-ball", "mesh": "ball", "material": "stone", "transform": {"translation": [-40, 1, -300]}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let (_scene, _) = SceneFixture::load(session, &fixture())?;
    ask(session, "control", json!({"name": "pause"}))?;
    render(session)?;
    let drawn = component(session, "limb", "Mesh3d")?;
    smoke_eq!(
        json!({
            "vertices": drawn["mesh"]["vertices"], "indices": drawn["mesh"]["indices"],
            "double_sided": drawn["material"]["double_sided"],
        }),
        json!({"vertices": 6, "indices": 12, "double_sided": true}),
        "inspects"
    );
    let near = component(session, "near-ball", "Mesh3d")?;
    let levels = &near["mesh"]["rendered"]["levels"];
    smoke_check!(
        levels.as_array().is_some_and(|levels| !levels.is_empty()),
        "ball_has_coarser_levels",
        json!(near["mesh"]["rendered"]),
        json!({"levels": "at least one"})
    );
    let near_level = drawn_level(session, "near-ball")?;
    let far_level = drawn_level(session, "far-ball")?;
    smoke_check!(
        near_level < 3,
        "near_ball_draws_as_mesh",
        json!(near_level),
        json!({"below": 3})
    );
    smoke_check!(
        far_level > near_level,
        "far_ball_draws_coarser",
        json!(far_level),
        json!({"above": near_level})
    );
    Ok(())
}

fn drawn_level(session: &mut dyn SmokeSession, entity: &str) -> Result<u64, SmokeFailure> {
    for _ in 0..100 {
        let drawn = session.frame();
        smoke_check!(drawn, "frames_draw", json!(drawn), json!(true));
        if let Some(level) = component(session, entity, "Mesh3d")?["drawn_level"].as_u64() {
            return Ok(level);
        }
    }
    smoke_check!(
        false,
        "drawn_level_reads_back",
        json!(entity),
        json!("a drawn level")
    );
    Ok(0)
}
