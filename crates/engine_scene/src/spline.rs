//! Scene version 9 spline meshes (decision 071): an entity field bending
//! copies of one of the document's meshes along a curve.

use engine_core::{MAX_SPLINE_POINTS, MaterialHandle, MeshHandle, SplineMesh, glam::Vec3};
use serde::{Deserialize, Serialize};

use crate::SceneError;

/// A mesh bent along a curve on this entity (decision 071). Requires
/// version 9.
///
/// The curve passes through `points` (2 to 4096, in the entity's space),
/// joined by a centripetal Catmull-Rom spline. Copies of the mesh lie end
/// to end along it, the mesh's x axis along the curve, its y up and its z
/// across, each stretched along x so a whole number of them spans it; the
/// curve's frames keep y up. Chunks of at most 256 meters of curve draw
/// with the material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneSpline {
    #[cfg_attr(feature = "schema", schemars(length(min = 2, max = 4096)))]
    pub points: Vec<[f32; 3]>,
    /// ID of the bent mesh in this document.
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub mesh: String,
    /// ID of the material its chunks draw with in this document.
    #[cfg_attr(
        feature = "schema",
        schemars(length(min = 1, max = 64), regex(pattern = r"^[A-Za-z0-9_.-]+$"))
    )]
    pub material: String,
}

impl SceneSpline {
    /// The field of `spline`, whose mesh and material have authored IDs
    /// `mesh` and `material`.
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

    /// Validates the curve at authored path `path`; its mesh and material
    /// references are checked with the entity.
    pub(crate) fn validate(&self, path: &str) -> Result<(), SceneError> {
        if !(2..=MAX_SPLINE_POINTS).contains(&self.points.len()) {
            return Err(SceneError::new(
                "invalid_spline",
                format!("{path}.points"),
                format!("a curve names 2 to {MAX_SPLINE_POINTS} points"),
            ));
        }
        if let Some(index) = self.points.iter().position(|point| {
            !point
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 1.0e7)
        }) {
            return Err(SceneError::new(
                "invalid_spline",
                format!("{path}.points[{index}]"),
                "points must be finite and within 1e7",
            ));
        }
        Ok(())
    }

    /// The component, with the mesh's and material's resident handles.
    pub(crate) fn runtime(&self, mesh: MeshHandle, material: MaterialHandle) -> SplineMesh {
        SplineMesh {
            points: self.points.iter().copied().map(Vec3::from_array).collect(),
            mesh,
            material,
        }
    }
}
