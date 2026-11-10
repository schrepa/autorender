use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, DrawnPoses, DrawnTable, Error, FrameTime, GlobalTransforms, GpuBits, Material,
    Operation, PoseRows, SceneIds, pose_rows,
};
use engine_places::surface::SurfaceValues;

use crate::MESH;
use crate::layouts::{LodFade, LodHistory};
use crate::presentation::PresentationTotals;

#[derive(Resource, Default)]
pub(crate) struct WatchedSlot {
    pub slot: Option<u32>,
}

#[derive(engine_core::__private::serde::Deserialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
pub(crate) struct DrawnRequest {
    entity: String,
}

#[derive(engine_core::__private::serde::Serialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde")]
pub(crate) struct DrawnReply {
    slot: Option<u32>,
    drawn_level: Option<u32>,
    drawn_pose: Option<PoseRows>,
    read_back: Option<ReadBack>,
}

#[derive(engine_core::__private::serde::Serialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde")]
pub(crate) struct FadeWord {
    shown: u32,
    previous: u32,
    progress: u32,
    held: bool,
}

#[derive(engine_core::__private::serde::Serialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde")]
pub(crate) struct ReadBack {
    level: Option<u32>,
    word: Option<FadeWord>,
    pose: Option<PoseRows>,
    age: u64,
}

fn fade_word(word: u32) -> Option<FadeWord> {
    LodFade::unpack(word).map(|fade| FadeWord {
        shown: fade.shown,
        previous: fade.previous,
        progress: fade.progress,
        held: fade.history == LodHistory::Held,
    })
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
    type Reply = DrawnReply;

    fn run(
        (ids, table, totals, time, mut watched, (poses, globals)): SystemParamItem<
            '_,
            '_,
            Self::Params,
        >,
        request: DrawnRequest,
    ) -> Result<DrawnReply, Error> {
        let id = request.entity;
        let entity = ids.named(&id, "$.entity")?;
        let drawn_pose = poses.drawn_pose(&globals, entity).map(pose_rows);
        let Some(slot) = table.slot(entity) else {
            return Ok(DrawnReply {
                slot: None,
                drawn_level: None,
                drawn_pose,
                read_back: None,
            });
        };
        watched.slot = Some(slot);
        let watched = totals
            .last()
            .and_then(|last| last.watched)
            .filter(|&(seen, _)| seen == slot)
            .map(|(_, word)| word);
        let age = totals.age(time.frame);
        let pose = watched.and_then(|_| totals.pose());
        Ok(DrawnReply {
            slot: Some(slot),
            drawn_level: watched.and_then(LodFade::drawn_level),
            drawn_pose,
            read_back: watched.zip(age).map(|(word, age)| ReadBack {
                level: LodFade::drawn_level(word),
                word: fade_word(word),
                pose,
                age,
            }),
        })
    }
}

#[derive(engine_core::__private::serde::Deserialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
pub struct MeshesListing {}

#[derive(engine_core::__private::serde::Serialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde")]
pub struct MeshesListed {
    pub instances: Vec<MeshInstance>,
    pub missing: u32,
}

#[derive(engine_core::__private::serde::Serialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde")]
pub struct MeshSurface {
    pub roughness: f32,
    pub metalness: f32,
    pub emission: f32,
}

#[derive(engine_core::__private::serde::Serialize, schemars::JsonSchema)]
#[serde(crate = "engine_core::__private::serde")]
pub struct MeshInstance {
    pub mesh: Option<u32>,
    pub material: Option<u32>,
    pub model: [f32; 16],
    pub color: [f32; 4],
    pub tint: [f32; 4],
    pub surface: MeshSurface,
}

impl Operation for MeshesListing {
    const NAME: &'static str = "meshes_listing";
    type Params = (
        Res<'static, DrawnTable>,
        Res<'static, GlobalTransforms>,
        Res<'static, AssetLibrary>,
    );
    type Request = Self;
    type Reply = MeshesListed;

    fn run(
        (table, globals, assets): SystemParamItem<'_, '_, Self::Params>,
        _: Self,
    ) -> Result<MeshesListed, Error> {
        let instances: Vec<MeshInstance> = (0..table.records().len() as u32)
            .filter_map(|slot| Some((table.owner(slot)?, table.drawn(slot)?)))
            .filter(|(_, drawn)| drawn.kind == MESH)
            .map(|(entity, drawn)| {
                let material = drawn.material.and_then(|handle| assets.material(handle));
                let surface = SurfaceValues::read(
                    |name| material.and_then(|material| material.surface_value(name)),
                    |name| material.and_then(|material| material.value(name)),
                );
                let model = globals
                    .get(entity)
                    .and_then(|global| global.matrix())
                    .unwrap_or(glam::Mat4::IDENTITY);
                MeshInstance {
                    mesh: drawn.geometry.map(|mesh| mesh.slot() as u32),
                    material: drawn.material.map(|material| material.slot() as u32),
                    model: model.to_cols_array(),
                    color: material.map_or([1.0; 4], Material::color),
                    tint: drawn.values.tint,
                    surface: MeshSurface {
                        roughness: surface.roughness,
                        metalness: surface.metalness,
                        emission: material.map_or(0.0, Material::emission),
                    },
                }
            })
            .collect();
        Ok(MeshesListed {
            instances,
            missing: table.census(MESH).waiting,
        })
    }
}
