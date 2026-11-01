use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, Expected, Operation, SceneContext, SceneIds};
use engine_places::splines::SplineMesh;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::spline_mesh::SplineBuilds;

#[derive(Serialize)]
struct SplineMeshView {
    points: Vec<[f32; 3]>,
    mesh: Option<String>,
    material: Option<String>,
    length: Option<f32>,
    copies: Option<u32>,
    chunks: Option<usize>,
    vertices: Option<u32>,
    error: Option<Error>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SplineMeshRequest {
    entity: String,
}

pub(crate) struct InspectSplineMesh;

impl Operation for InspectSplineMesh {
    const NAME: &'static str = "spline_mesh";
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, Entity>,
        Query<'static, 'static, &'static SplineMesh>,
        Res<'static, SplineBuilds>,
        Option<Res<'static, SceneContext>>,
    );
    type Request = SplineMeshRequest;

    fn run(
        (ids, alive, splines, builds, context): SystemParamItem<'_, '_, Self::Params>,
        request: SplineMeshRequest,
    ) -> Result<Value, Error> {
        let id = request.entity;
        let entity = ids.entity(&id).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(&id)
        })?;
        if !alive.contains(entity) {
            return Err(Error::new(ErrorCode::EntityRemoved, "$.entity")
                .expected(Expected::Live)
                .got(&id));
        }
        let Ok(spline) = splines.get(entity) else {
            return Ok(Value::Null);
        };
        let assets = context.as_deref().map(|context| &*context.assets);
        let mesh = assets.and_then(|assets| {
            assets
                .meshes
                .iter()
                .find(|(_, handle)| **handle == spline.mesh)
                .map(|(id, _)| id.clone())
        });
        let material = assets.and_then(|assets| {
            assets
                .materials
                .iter()
                .find(|(_, handle)| **handle == spline.material)
                .map(|(id, _)| id.clone())
        });
        let chunks = builds.of(entity);
        Ok(json!(SplineMeshView {
            points: spline.points.iter().map(|point| point.to_array()).collect(),
            mesh,
            material,
            length: chunks.map(|chunks| chunks.length),
            copies: chunks.map(|chunks| chunks.copies),
            chunks: chunks.map(|chunks| chunks.entities.len()),
            vertices: chunks.map(|chunks| chunks.vertices),
            error: chunks.and_then(|chunks| chunks.error.clone()),
        }))
    }
}
