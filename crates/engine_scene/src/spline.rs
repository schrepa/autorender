use engine_core::{
    Error, Expected, MAX_SPLINE_POINTS, MaterialHandle, MeshHandle, SplineMesh,
    bevy_ecs::world::EntityWorldMut, glam::Vec3,
};
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};

use crate::components::{Placement, Reach, SceneComponent};
use crate::instance::Lookups;
use crate::{SceneDocument, SceneFiles};

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

impl engine_settings::Setting for SceneSpline {}

impl SceneComponent for SceneSpline {
    type Prepared = Self;

    const PLACEMENT: Placement = Placement::Anywhere;

    fn prepare(&self, _: &str, _: &SceneDocument, _: Option<&SceneFiles>) -> Result<Self, Error> {
        Ok(self.clone())
    }

    fn spawn(prepared: &Self, entity: &mut EntityWorldMut<'_>, lookups: &Lookups) {
        entity.insert(prepared.runtime(
            lookups.meshes[prepared.mesh.as_str()],
            lookups.materials[prepared.material.as_str()],
        ));
    }

    fn reach(prepared: &Self, reach: &Reach<'_>, extend: &mut impl FnMut(f32)) {
        let size = reach.mesh_size(&prepared.mesh);
        for point in &prepared.points {
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
