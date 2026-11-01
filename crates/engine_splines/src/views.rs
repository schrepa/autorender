use bevy_ecs::prelude::*;
use engine_core::{Error, Operation};
use engine_places::splines::SplineMesh;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::spline_mesh::SplineBuilds;

#[derive(Serialize)]
struct SplineMeshView {
    entity: u64,
    points: Vec<[f32; 3]>,
    length: Option<f32>,
    copies: Option<u32>,
    chunks: Option<usize>,
    vertices: Option<u32>,
    error: Option<Error>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SplinesRequest {}

pub(crate) struct InspectSplines;

impl Operation for InspectSplines {
    const NAME: &'static str = "splines";
    type Params = (
        Query<'static, 'static, (Entity, &'static SplineMesh)>,
        Res<'static, SplineBuilds>,
    );
    type Request = SplinesRequest;

    fn run(
        (splines, builds): SystemParamItem<'_, '_, Self::Params>,
        _: SplinesRequest,
    ) -> Result<Value, Error> {
        let mut views: Vec<SplineMeshView> = splines
            .iter()
            .map(|(entity, spline)| {
                let chunks = builds.of(entity);
                SplineMeshView {
                    entity: entity.to_bits(),
                    points: spline.points.iter().map(|point| point.to_array()).collect(),
                    length: chunks.map(|chunks| chunks.length),
                    copies: chunks.map(|chunks| chunks.copies),
                    chunks: chunks.map(|chunks| chunks.meshes.len()),
                    vertices: chunks.map(|chunks| chunks.vertices),
                    error: chunks.and_then(|chunks| chunks.error.clone()),
                }
            })
            .collect();
        views.sort_by_key(|view| view.entity);
        Ok(json!({ "splines": views }))
    }
}
