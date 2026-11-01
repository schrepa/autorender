use engine_core::smoke::{Checked, ask, capture, control, not, render};
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

fn loaded(session: &mut dyn SmokeSession, document: &Value) -> Checked<Value> {
    session.source(&document.to_string());
    control(
        session,
        "reload",
        json!({"expected_revision": "1", "revision": "1"}),
    )
}

fn splines(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut answered = ask(session, "splines", json!({}))?;
    Ok(answered["splines"].take())
}

fn rail(session: &mut dyn SmokeSession) -> Checked<Value> {
    Ok(splines(session)?.get(0).cloned().unwrap_or_default())
}

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let source = fixture(&CURVE);
    loaded(session, &source)?;
    control(session, "pause", json!({}))?;
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
            .all(|(got, expected)| (got - expected).abs() <= 1e-6);
    smoke_eq!(
        shown(near, built["points"].clone(), &json!(CURVE)),
        json!(CURVE),
        "built_points"
    );
    smoke_eq!(built["error"], Value::Null, "built_without_error");
    lengths(&built, &CURVE)?;
    let copies = built["copies"].as_u64().unwrap_or(0);
    smoke_eq!(
        json!([
            built["chunks"],
            built["vertices"],
            first["render_candidates"]
        ]),
        json!([1, copies * 24, 3]),
        "built"
    );

    let with_rail = capture(session)?;
    let mut bare = source.clone();
    if let Some(rail) = bare["entities"][2].as_object_mut() {
        rail.remove("spline");
    }
    loaded(session, &bare)?;
    let without = capture(session)?;
    smoke_eq!(
        shown(without != with_rail, without, &not(&with_rail)),
        not(&with_rail),
        "draws_and_goes"
    );
    let removed = render(session)?;
    smoke_eq!(
        json!(removed["render_candidates"]),
        json!(2),
        "removing_the_field_removes_its_chunks"
    );
    smoke_eq!(splines(session)?, json!([]), "released");

    let longer_points: Vec<[f64; 3]> = CURVE.iter().chain(&FARTHER).copied().collect();
    loaded(session, &fixture(&longer_points))?;
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
    loaded(session, &invalid)?;
    render(session)?;
    let refused = rail(session)?;
    smoke_eq!(
        json!({"code": refused["error"]["code"], "path": refused["error"]["path"], "chunks": refused["chunks"]}),
        json!({"code": "invalid_value", "path": "$.points", "chunks": 0}),
        "refused"
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
