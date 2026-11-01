use engine_core::smoke::{Checked, ask, capture, control, not, query, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

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

fn shown(held: bool, got: Value, expected: &Value) -> Value {
    if held { expected.clone() } else { got }
}

fn load(session: &mut dyn SmokeSession, document: &Value) -> Checked<Value> {
    session.source(&document.to_string());
    let loaded = session
        .request("control", json!({"name": "render"}))
        .map(|reply| reply["result"].clone())
        .map_err(|reply| reply["error"].clone());
    smoke_eq!(
        shown(loaded.is_ok(), json!(loaded.as_ref().err()), &Value::Null),
        Value::Null,
        "fixture_loads"
    );
    Ok(loaded.unwrap_or_default())
}

fn reload(session: &mut dyn SmokeSession, document: &Value) -> Result<Value, Value> {
    session.source(&document.to_string());
    let revision = document["revision"].as_u64().unwrap_or_default().to_string();
    let args = json!({"expected_revision": revision, "revision": revision});
    session
        .request("control", json!({"name": "reload", "args": args}))
        .map_err(|reply| reply["error"].clone())?;
    session
        .request("control", json!({"name": "render"}))
        .map(|reply| reply["result"].clone())
        .map_err(|reply| reply["error"].clone())
}

fn reloaded(session: &mut dyn SmokeSession, document: &Value) -> Checked<Value> {
    let reloaded = reload(session, document);
    smoke_eq!(
        shown(reloaded.is_ok(), json!(reloaded.as_ref().err()), &Value::Null),
        Value::Null,
        "reload_accepted"
    );
    Ok(reloaded.unwrap_or_default())
}

fn rail(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "spline_mesh", json!({"entity": "rail"}))
}

fn candidates(rendering: &Value) -> Value {
    rendering["meshes_stats"]["render_candidates"].clone()
}

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let source = fixture(&CURVE);
    load(session, &source)?;
    control(session, "pause", json!({}))?;
    let listing = ask(session, "describe", json!({}))?;
    let listed = listing["types"]
        .as_array()
        .is_some_and(|types| types.iter().any(|kind| kind["name"] == "spline"));
    smoke_eq!(
        shown(listed, json!(listing["types"]), &json!("spline")),
        json!("spline"),
        "describe_lists_spline"
    );
    let first = render(session)?;
    let built = rail(session)?;
    let flat: Vec<f64> = CURVE.iter().flatten().copied().collect();
    let points: Vec<f64> = built["points"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|point| point.as_array().into_iter().flatten())
        .map(|value| value.as_f64().unwrap_or(f64::NAN))
        .collect();
    let near = points.len() == flat.len()
        && points
            .iter()
            .zip(&flat)
            .all(|(got, expected)| got.is_finite() && (got - expected).abs() <= 1e-6);
    smoke_eq!(
        shown(near, json!(built["points"]), &json!(CURVE)),
        json!(CURVE),
        "built_points"
    );
    smoke_eq!(
        shown(built["error"].is_null(), json!(built["error"]), &Value::Null),
        Value::Null,
        "built_without_error"
    );
    lengths(&built, &CURVE)?;
    let copies = built["copies"].as_u64().unwrap_or(0);
    smoke_eq!(
        json!([built["chunks"], built["vertices"], candidates(&first)]),
        json!([1, copies * 24, 3]),
        "built"
    );

    let with_rail = capture(session)?;
    let mut bare = source.clone();
    if let Some(rail) = bare["entities"][2].as_object_mut() {
        rail.remove("spline");
    }
    reloaded(session, &bare)?;
    let without = capture(session)?;
    smoke_eq!(
        shown(without != with_rail, json!(without), &not(&with_rail)),
        not(&with_rail),
        "draws_and_goes"
    );
    let removed = render(session)?;
    smoke_eq!(
        candidates(&removed),
        json!(2),
        "removing_the_field_removes_its_chunks"
    );

    let longer_points: Vec<[f64; 3]> = CURVE.iter().chain(&FARTHER).copied().collect();
    reloaded(session, &fixture(&longer_points))?;
    render(session)?;
    let rebuilt = rail(session)?;
    lengths(&rebuilt, &longer_points)?;
    let copies = rebuilt["copies"].as_u64().unwrap_or(0);
    smoke_eq!(
        json!(rebuilt["chunks"]),
        json!(copies.div_ceil(256)),
        "rebuilds"
    );

    let mut invalid = fixture(&longer_points);
    invalid["entities"][2]["spline"]["points"] = json!([[0, 0, 0]]);
    let refused = reload(session, &invalid).err().unwrap_or_default();
    smoke_eq!(
        json!({"code": refused["code"], "path": refused["path"]}),
        json!({"code": "invalid_spline", "path": "$.entities[2].spline.points"}),
        "refused"
    );
    smoke_eq!(
        json!(rail(session)?["chunks"]),
        json!(rebuilt["chunks"]),
        "the_scene_stays"
    );
    Ok(())
}

fn lengths(spline: &Value, points: &[[f64; 3]]) -> Result<(), SmokeFailure> {
    let length = spline["length"].as_f64().unwrap_or(0.0);
    let straight = polyline(points);
    let bounds = json!({"at_least": straight, "below": 1.1 * straight});
    smoke_eq!(
        shown(
            length >= straight && length < 1.1 * straight,
            json!(length),
            &bounds
        ),
        bounds,
        "curve_length"
    );
    smoke_eq!(
        json!(spline["copies"]),
        json!(length.round() as u64),
        "copies_follow_length"
    );
    Ok(())
}
