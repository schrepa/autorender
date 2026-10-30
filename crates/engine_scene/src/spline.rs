use engine_core::{
    Error, Expected, MAX_SPLINE_POINTS, MaterialHandle, MeshHandle, SplineMesh,
    bevy_ecs::world::EntityWorldMut, glam::Vec3,
};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

use crate::components::{
    ComponentContext, ComponentKind, Placement, Prepared, PreparedComponent, Reach, admitted,
    no_files, no_paths, parsed, valid_world,
};
use crate::instance::Lookups;
use crate::validation::reference;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneSpline {
    #[cfg_attr(feature = "schema", schemars(length(min = 2, max = MAX_SPLINE_POINTS)))]
    pub points: Vec<[f32; 3]>,
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub mesh: String,
    #[cfg_attr(
        feature = "schema",
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub material: String,
}

impl SceneSpline {
    pub fn from_runtime(
        spline: &SplineMesh,
        mesh: impl Into<String>,
        material: impl Into<String>,
    ) -> Self {
        Self {
            points: spline.points.iter().map(|point| point.to_array()).collect(),
            mesh: mesh.into(),
            material: material.into(),
        }
    }

    pub(crate) fn validate(&self, path: &str) -> Result<(), Error> {
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

    pub(crate) fn runtime(&self, mesh: MeshHandle, material: MaterialHandle) -> SplineMesh {
        SplineMesh {
            points: self.points.iter().copied().map(Vec3::from_array).collect(),
            mesh,
            material,
        }
    }
}

pub(crate) static SPLINE: ComponentKind = ComponentKind {
    name: "spline",
    shows: "SplineMesh",
    placement: Placement::Anywhere,
    schema: crate::component_schema!(SceneSpline),
    example: spline_example,
    validate: validate_spline,
    validate_world: valid_world,
    prepare: prepare_spline,
    files: no_files,
    admit: admitted,
    paths: no_paths,
};

fn spline_example() -> Value {
    json!({
        "points": [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        "mesh": "example",
        "material": "example",
    })
}

fn validate_spline(value: &Value, path: &str) -> Result<(), Error> {
    parsed::<SceneSpline>(value, path)?.validate(path)
}

fn prepare_spline(value: &Value, context: &ComponentContext<'_>) -> Prepared {
    let spline: SceneSpline = parsed(value, context.path)?;
    let document = context.document;
    let meshes: BTreeSet<&str> = document
        .meshes
        .iter()
        .map(|mesh| mesh.id.as_str())
        .collect();
    let materials: BTreeSet<&str> = document
        .materials
        .iter()
        .map(|material| material.id.as_str())
        .collect();
    reference(&spline.mesh, &meshes, &format!("{}.mesh", context.path))?;
    reference(
        &spline.material,
        &materials,
        &format!("{}.material", context.path),
    )?;
    Ok(Box::new(spline))
}

impl PreparedComponent for SceneSpline {
    fn spawn(&self, entity: &mut EntityWorldMut<'_>, lookups: &Lookups) {
        entity.insert(self.runtime(
            lookups.meshes[self.mesh.as_str()],
            lookups.materials[self.material.as_str()],
        ));
    }

    fn reach(&self, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
        let size = reach.mesh_size(&self.mesh);
        for point in &self.points {
            extend(
                reach
                    .world
                    .transform_point3(Vec3::from_array(*point))
                    .distance(reach.position)
                    + size,
            );
        }
    }
}
