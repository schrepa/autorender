use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::scene_kit::{SceneFixture, ask, capture, component, near_all, numbers, render};

const CURVE: [[f64; 3]; 4] = [
    [-12.0, 0.5, 4.0],
    [-4.0, 0.5, -2.0],
    [4.0, 0.5, 3.0],
    [12.0, 0.5, -3.0],
];
const FARTHER: [[f64; 3]; 4] = [
    [40.0, 0.5, 2.0],
    [70.0, 0.5, -2.0],
    [160.0, 0.5, 6.0],
    [290.0, 0.5, -4.0],
];

fn fixture(points: &[[f64; 3]]) -> Value {
    json!({
        "id": "spline-fixture", "revision": 1,
        "camera": {"position": [0, 14, 22], "target": [0, 0, 0], "fov_degrees": 55, "near": 0.1, "far": 500},
        "materials": [
            {"id": "plain", "color": [0.8, 0.3, 0.2, 1]},
            {"id": "ground", "color": [0.3, 0.32, 0.28, 1]},
            {"id": "rail", "color": [0.85, 0.8, 0.7, 1]},
        ],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "entities": [
            {"id": "player", "mesh": "box", "material": "plain", "tags": ["player"], "transform": {"translation": [0, 0.5, 8]}},
            {"id": "floor", "mesh": "box", "material": "ground", "transform": {"translation": [0, -0.5, 0], "scale": [40, 1, 30]}},
            {"id": "rail", "spline": {"points": points, "mesh": "box", "material": "rail"}, "transform": {"scale": [1, 1, 1]}},
        ],
    })
}

fn polyline(points: &[[f64; 3]]) -> f64 {
    points
        .windows(2)
        .map(|pair| {
            pair[0]
                .iter()
                .zip(pair[1])
                .map(|(from, to)| (to - from) * (to - from))
                .sum::<f64>()
                .sqrt()
        })
        .sum()
}

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let source = fixture(&CURVE);
    let (mut scene, _) = SceneFixture::load(session, &source)?;
    ask(session, "control", json!({"name": "pause"}))?;
    let listing = ask(session, "describe", json!({}))?;
    let listed = listing["types"]
        .as_array()
        .is_some_and(|types| types.iter().any(|kind| kind["name"] == "SplineMesh"));
    smoke_check!(listed, "describe_lists_spline", json!(listing["types"]), json!("SplineMesh"));
    let first = render(session)?;
    let built = component(session, "rail", "SplineMesh")?;
    let flat: Vec<f64> = CURVE.iter().flatten().copied().collect();
    let points: Vec<f64> = built["points"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(numbers)
        .collect();
    smoke_check!(
        near_all(&json!(points), &flat, 1e-6),
        "built_points",
        json!(built["points"]),
        json!(CURVE)
    );
    smoke_check!(built["error"].is_null(), "built_without_error", json!(built["error"]), json!(null));
    lengths(&built, &CURVE)?;
    let copies = built["copies"].as_u64().unwrap_or(0);
    smoke_eq!(
        json!([built["chunks"], built["vertices"], first["render_candidates"]]),
        json!([1, copies * 24, 3]),
        "built"
    );

    let with_rail = capture(session)?;
    let mut bare = source.clone();
    if let Some(rail) = bare["entities"][2].as_object_mut() {
        rail.remove("spline");
    }
    reloaded(&mut scene, session, &bare)?;
    let without = capture(session)?;
    smoke_check!(without != with_rail, "draws_and_goes", json!(without), json!({"not": with_rail}));
    let removed = render(session)?;
    smoke_eq!(json!(removed["render_candidates"]), json!(2), "removing_the_field_removes_its_chunks");

    let longer_points: Vec<[f64; 3]> = CURVE.iter().chain(&FARTHER).copied().collect();
    reloaded(&mut scene, session, &fixture(&longer_points))?;
    render(session)?;
    let rebuilt = component(session, "rail", "SplineMesh")?;
    lengths(&rebuilt, &longer_points)?;
    let copies = rebuilt["copies"].as_u64().unwrap_or(0);
    smoke_eq!(json!(rebuilt["chunks"]), json!(copies.div_ceil(256)), "rebuilds");

    let mut invalid = fixture(&longer_points);
    invalid["entities"][2]["spline"]["points"] = json!([[0, 0, 0]]);
    let refused = scene.reload(session, &invalid).err().unwrap_or_default();
    smoke_eq!(
        json!({"code": refused["code"], "path": refused["path"]}),
        json!({"code": "invalid_spline", "path": "$.entities[2].spline.points"}),
        "refused"
    );
    smoke_eq!(
        json!(component(session, "rail", "SplineMesh")?["chunks"]),
        json!(rebuilt["chunks"]),
        "the_scene_stays"
    );
    Ok(())
}

fn lengths(spline: &Value, points: &[[f64; 3]]) -> SmokeResult {
    let length = spline["length"].as_f64().unwrap_or(0.0);
    let straight = polyline(points);
    smoke_check!(
        length >= straight && length < 1.1 * straight,
        "curve_length",
        json!(length),
        json!({"at_least": straight, "below": 1.1 * straight})
    );
    smoke_eq!(
        json!(spline["copies"]),
        json!(length.round() as u64),
        "copies_follow_length"
    );
    Ok(())
}

fn reloaded(
    scene: &mut SceneFixture,
    session: &mut dyn SmokeSession,
    document: &Value,
) -> Result<Value, SmokeFailure> {
    let reloaded = scene.reload(session, document);
    smoke_check!(reloaded.is_ok(), "reload_accepted", json!(reloaded.as_ref().err()), json!(null));
    Ok(reloaded.unwrap_or_default())
}
