use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, DrawnPoses, DrawnTable, Error, ErrorCode, Expected, FrameTime, GlobalTransforms,
    GpuBits, Material, Operation, SceneIds, Surface, pose_rows,
};
use serde_json::{Value, json};

use crate::MESH;
use crate::layouts::LodFade;
use crate::presentation::PresentationTotals;

#[derive(Resource, Default)]
pub(crate) struct WatchedSlot {
    pub slot: Option<u32>,
}

#[derive(engine_core::__private::serde::Deserialize)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
pub(crate) struct DrawnRequest {
    entity: String,
}

pub(crate) struct MeshDrawn;

impl Operation for MeshDrawn {
    const NAME: &'static str = "mesh_drawn";
    const CONTROL: bool = false;
    type Params = (
        SceneIds<'static, 'static>,
        Res<'static, DrawnTable>,
        Res<'static, PresentationTotals>,
        Res<'static, FrameTime>,
        ResMut<'static, WatchedSlot>,
        (Res<'static, DrawnPoses>, Res<'static, GlobalTransforms>),
    );
    type Request = DrawnRequest;

    fn run(
        (ids, table, totals, time, mut watched, (poses, globals)): SystemParamItem<
            '_,
            '_,
            Self::Params,
        >,
        request: DrawnRequest,
    ) -> Result<Value, Error> {
        let id = request.entity;
        let entity = ids.entity(&id).ok_or_else(|| {
            Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Entity)
                .got(&id)
        })?;
        let drawn_pose = poses.drawn_pose(&globals, entity).map(pose_rows);
        let Some(slot) = table.slot(entity) else {
            return Ok(json!({
                "slot": null,
                "drawn_level": null,
                "drawn_pose": drawn_pose,
                "read_back": null,
            }));
        };
        watched.slot = Some(slot);
        let watched = totals
            .last()
            .and_then(|last| last.watched)
            .filter(|&(seen, _)| seen == slot)
            .map(|(_, word)| word);
        let age = totals.age(time.frame);
        let pose = watched.and_then(|_| totals.pose());
        Ok(json!({
            "slot": slot,
            "drawn_level": watched.and_then(LodFade::drawn_level),
            "drawn_pose": drawn_pose,
            "read_back": watched.zip(age).map(|(word, age)| json!({
                "level": LodFade::drawn_level(word),
                "word": <LodFade as GpuBits>::decoded(word),
                "pose": pose,
                "age": age,
            })),
        }))
    }
}

#[derive(engine_core::__private::serde::Deserialize)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
pub(crate) struct ListingRequest {}

pub(crate) struct MeshesListing;

impl Operation for MeshesListing {
    const NAME: &'static str = "meshes_listing";
    type Params = (
        Res<'static, DrawnTable>,
        Res<'static, GlobalTransforms>,
        Res<'static, AssetLibrary>,
    );
    type Request = ListingRequest;

    fn run(
        (table, globals, assets): SystemParamItem<'_, '_, Self::Params>,
        _: ListingRequest,
    ) -> Result<Value, Error> {
        let instances: Vec<Value> = (0..table.records().len() as u32)
            .filter_map(|slot| table.drawn(slot))
            .filter(|drawn| drawn.kind == MESH)
            .map(|drawn| {
                let material = drawn.material.and_then(|handle| assets.material(handle));
                let surface = material.map_or(Surface::MATTE, Material::surface);
                let model = globals
                    .get(drawn.node)
                    .and_then(|global| global.matrix())
                    .unwrap_or(glam::Mat4::IDENTITY);
                json!({
                    "mesh": drawn.geometry.map(|mesh| mesh.slot() as u32),
                    "material": drawn.material.map(|material| material.slot() as u32),
                    "model": model.to_cols_array(),
                    "color": material.map_or([1.0; 4], Material::color),
                    "tint": drawn.values.tint,
                    "surface": {
                        "roughness": surface.roughness,
                        "metalness": surface.metalness,
                        "emission": surface.emission,
                        "ripples": surface.ripples,
                    },
                })
            })
            .collect();
        Ok(json!({
            "instances": instances,
            "missing": table.census(MESH).waiting,
        }))
    }
}
