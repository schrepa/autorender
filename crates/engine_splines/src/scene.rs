use engine_core::{Error, ErrorCode, Expected, MaterialHandle, MeshHandle, Reach};
use engine_places::splines::SplineMesh;
use serde::Serialize;
use serde_json::{Value, json};

use crate::spline_mesh::MAX_SPLINE_POINTS;

#[derive(Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct SceneSpline {
    #[schemars(length(min = 2, max = MAX_SPLINE_POINTS))]
    points: Vec<[f32; 3]>,
    mesh: MeshHandle,
    material: MaterialHandle,
}

impl From<&SplineMesh> for SceneSpline {
    fn from(spline: &SplineMesh) -> Self {
        Self {
            points: spline.points.iter().map(|point| point.to_array()).collect(),
            mesh: spline.mesh,
            material: spline.material,
        }
    }
}

impl SceneSpline {
    fn validate(&self, path: &str) -> Result<(), Error> {
        if !(2..=MAX_SPLINE_POINTS).contains(&self.points.len()) {
            return Err(
                Error::new(ErrorCode::InvalidSpline, format!("{path}.points"))
                    .expected(Expected::Array)
                    .range(2.0, MAX_SPLINE_POINTS as f64)
                    .got(self.points.len()),
            );
        }
        if let Some(index) = self.points.iter().position(|point| {
            !point
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 1.0e7)
        }) {
            return Err(
                Error::new(ErrorCode::InvalidSpline, format!("{path}.points[{index}]"))
                    .expected(Expected::Finite)
                    .range(-1.0e7, 1.0e7)
                    .got(self.points[index]),
            );
        }
        Ok(())
    }
}

pub(crate) fn check_spline(spline: &SplineMesh, path: &str) -> Result<(), Error> {
    SceneSpline::from(spline).validate(path)
}

pub(crate) fn spline_reach(spline: &SplineMesh, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
    let size = match serde_json::to_value(spline.mesh) {
        Ok(Value::String(mesh)) => reach.mesh_size(&mesh),
        _ => 0.0,
    };
    for point in &spline.points {
        extend(
            reach
                .world
                .transform_point3(*point)
                .distance(reach.position)
                + size,
        );
    }
}

pub(crate) fn spline_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(SceneSpline)).unwrap_or(Value::Null)
}

pub(crate) fn spline_example() -> Value {
    json!({
        "points": [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        "mesh": "example",
        "material": "example",
    })
}
