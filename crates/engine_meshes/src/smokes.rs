use engine_core::SmokeFailure;
use engine_core::math::{cos, sin};
use engine_core::smoke::{Checked, ask, component, control, environment, inspect, render};
use engine_core::{SmokeSession, smoke_eq};
use serde_json::{Value, json};

const SIDE: usize = 12;
const PAINTS: [[f32; 4]; 4] = [
    [0.85, 0.35, 0.2, 1.0],
    [0.25, 0.6, 0.85, 1.0],
    [0.4, 0.75, 0.3, 1.0],
    [0.9, 0.75, 0.25, 1.0],
];

fn grid() -> Value {
    let mut materials = vec![json!({"id": "ground", "color": [0.55, 0.58, 0.52, 1.0]})];
    materials.extend(
        PAINTS
            .iter()
            .enumerate()
            .map(|(index, paint)| json!({"id": format!("paint-{index}"), "color": paint})),
    );
    let mut entities = vec![json!({
        "id": "ground", "mesh": "box", "material": "ground",
        "transform": {"translation": [0.0, -0.5, 0.0], "scale": [SIDE as f32 * 6.0, 1.0, SIDE as f32 * 6.0]},
    })];
    let middle = (SIDE as f32 - 1.0) / 2.0;
    for index in 0..SIDE * SIDE {
        let (row, column) = ((index / SIDE) as f32, (index % SIDE) as f32);
        let scale = 0.8 + (index % 5) as f32 * 0.3;
        let turn = ((index * 37) % 360) as f32;
        entities.push(json!({
            "id": format!("box-{index}"), "mesh": "box", "material": format!("paint-{}", index % PAINTS.len()),
            "transform": {
                "translation": [(column - middle) * 4.0, scale / 2.0, (row - middle) * 4.0],
                "rotation_degrees": [0.0, turn, 0.0],
                "scale": [scale, scale, scale],
            },
        }));
    }
    let eye = SIDE as f64 * 2.6;
    json!({
        "id": "culling-grid", "revision": 1,
        "camera": {"position": [0.0, 30.0, eye], "target": [0.0, 0.0, 0.0], "fov_degrees": 60, "near": 0.1, "far": 1000.0},
        "materials": materials,
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "entities": entities,
    })
}

fn previewed(session: &mut dyn SmokeSession) -> Checked<(Value, Value)> {
    let captured = ask(session, "capture", json!({ "alpha": 0.5 }))?;
    Ok((captured["rendering"].clone(), inspect(session, "meshes_stats")?))
}

fn passes(rendering: &Value, pass: &str) -> bool {
    rendering["render_passes"]
        .as_array()
        .is_some_and(|passes| passes.iter().any(|name| name == pass))
}

pub(crate) fn culling_settings(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid().to_string());
    control(session, "pause", json!({}))?;
    render(session)?;
    session.step(3);
    let on = json!({ "enabled": true });
    let off = json!({ "enabled": false });
    smoke_eq!(inspect(session, "occlusion_culling")?, on, "occlusion_on_by_default");
    smoke_eq!(inspect(session, "subpixel_culling")?, on, "subpixel_on_by_default");
    let (rendering, stats) = previewed(session)?;
    smoke_eq!(
        stats["occlusion_culling"]["active"],
        json!(true),
        "gpu_frames_cull_occluded"
    );
    smoke_eq!(
        stats["subpixel_culling"]["active"],
        json!(true),
        "gpu_frames_cull_subpixel"
    );
    smoke_eq!(
        passes(&rendering, "occlusion_test"),
        true,
        "occlusion_test_runs"
    );
    let both = json!({ "occlusion_culling": off, "subpixel_culling": off });
    let reply = environment(session, both)?;
    smoke_eq!(
        reply["environment_changed"],
        json!(["occlusion_culling", "subpixel_culling"]),
        "culling_settings_controlled"
    );
    smoke_eq!(inspect(session, "occlusion_culling")?, off, "culling_settings_inspected");
    let (rendering, stats) = previewed(session)?;
    smoke_eq!(
        json!([
            stats["occlusion_culling"]["active"],
            stats["subpixel_culling"]["active"],
            stats["occlusion_culling"]["instances"],
            stats["subpixel_culling"]["instances"],
        ]),
        json!([false, false, null, null]),
        "culling_settings_observed"
    );
    smoke_eq!(
        passes(&rendering, "occlusion_test"),
        false,
        "no_occlusion_test_while_off"
    );
    let phases = json!({ "occlusion_culling": { "enabled": true, "phases": 2 } });
    let refused = session.request("control", json!({ "name": "environment", "args": phases }));
    smoke_eq!(
        refused.err().map(|reply| reply["error"]["code"].clone()),
        Some(json!("unknown_field")),
        "culling_settings_validated"
    );
    environment(session, json!({ "occlusion_culling": on, "subpixel_culling": on }))?;
    let (_, stats) = previewed(session)?;
    smoke_eq!(
        stats["occlusion_culling"]["active"],
        json!(true),
        "culling_runs_again"
    );
    Ok(())
}

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

pub(crate) fn meshes(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    control(session, "pause", json!({}))?;
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
    smoke_eq!(session.frame(), true, "frames_draw");
    Ok(())
}
