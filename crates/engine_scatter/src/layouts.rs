use bevy_ecs::prelude::Resource;
use bytemuck::{Pod, Zeroable};
use engine_core::{GpuBits, gpu_bits, gpu_consts, gpu_struct};
use engine_places::meshes::DEBUG_LEVEL_AT;
use engine_places_rendering::impostors::{DEBUG_CARD, ImpostorRecord};
use glam::Mat4;

use crate::settings::MAX_PLACEMENT_LODS;

gpu_struct! {
    pub(crate) struct CullFrame {
        pub eye: [f32; 4],
        pub counts: [u32; 4],
        pub strides: [u32; 4],
        pub drawn: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct CullView {
        pub eye: [f32; 4],
        pub shape: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct PlacedValues {
        pub tilt: u32,
        pub ground: f32,
        pub reach: f32,
        pub texel: u32,
        pub tint: u32,
        pub sway: f32,
        pub look: [u32; 2],
    }
}

gpu_struct! {
    pub(crate) struct CullGroup {
        pub ids: u32,
        pub count: u32,
        pub levels: u32,
        pub camera: u32,
        pub errors: [f32; 7],
        pub impostor: f32,
        pub record: u32,
        pub parts: u32,
        pub pad: [u32; 2],
    }
}

gpu_struct! {
    pub(crate) struct Snap {
        pub counts: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct CanopyMap {
        pub rect: [f32; 4],
        pub shape: [f32; 4],
        pub layers: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct PlacedGroup {
        pub sphere: [f32; 4],
        pub extent: [f32; 3],
        pub layer: u32,
        pub emissive: [f32; 3],
        pub flags: u32,
        pub color: [u32; 2],
        pub pad: [u32; 2],
    }
}

gpu_struct! {
    #[derive(Resource)]
    pub struct CanopyPiece {
        pub foliage: [f32; 4],
        pub handover: [f32; 4],
        pub light: [f32; 4],
    }
}

gpu_consts! {
    pub(crate) const DEAD: u32 = 0xffffffff;
    pub(crate) const NO_LAYER: u32 = 0x7f;
    pub(crate) const SCATTER_GROUP: u32 = 64;
    pub(crate) const LAID_GROUP: u32 = 8;
    pub(crate) const WIND_REACH: f32 = 4.0;
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScatterTerrain {
    #[bits(9)]
    pub(crate) index: u32,
    #[bits(7)]
    pub(crate) layer: u32,
    #[bits(5)]
    pub(crate) near: u32,
    #[bits(10)]
    pub(crate) rank: u32,
    pub(crate) hidden: bool,
}

pub(crate) const TERRAIN_INDEX: u32 = 0x1ff;

pub(crate) const GENERATED: &[&str] = &[
    WGSL,
    <ScatterTerrain as GpuBits>::WGSL,
    <CullFrame as engine_core::GpuStruct>::WGSL,
    <CullView as engine_core::GpuStruct>::WGSL,
    <PlacedValues as engine_core::GpuStruct>::WGSL,
    <CullGroup as engine_core::GpuStruct>::WGSL,
    <Snap as engine_core::GpuStruct>::WGSL,
    <CanopyMap as engine_core::GpuStruct>::WGSL,
    crate::jobs::canopy::WGSL,
    <PlacedGroup as engine_core::GpuStruct>::WGSL,
];

pub(crate) const LEVELS: u32 = MAX_PLACEMENT_LODS as u32 + 1;
pub(crate) const RANK_MAX: f32 = 1023.0;

pub(crate) const RECORD_LEVELS: u32 = 16;
pub(crate) const RECORD_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawKind {
    Placed,
    PlacedCard,
}

impl DrawKind {
    const ALL: [Self; 2] = [Self::Placed, Self::PlacedCard];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DrawRecord {
    pub(crate) kind: DrawKind,
    pub(crate) level: u32,
}

impl DrawRecord {
    pub(crate) fn of(kind: DrawKind, level: usize) -> Self {
        Self {
            kind,
            level: level.min(RECORD_LEVELS as usize - 1) as u32,
        }
    }

    pub(crate) fn offset(self) -> u64 {
        let kind = DrawKind::ALL
            .iter()
            .position(|kind| *kind == self.kind)
            .unwrap_or_default() as u64;
        (kind * u64::from(RECORD_LEVELS) + u64::from(self.level.min(RECORD_LEVELS - 1)))
            * RECORD_BYTES
    }

    fn word(self) -> u32 {
        let card = if self.kind == DrawKind::PlacedCard {
            DEBUG_CARD
        } else {
            0
        };
        (self.level.min(RECORD_LEVELS - 1) << DEBUG_LEVEL_AT) | card
    }
}

pub(crate) fn record_words() -> Vec<[u32; 4]> {
    let mut words = Vec::new();
    for kind in DrawKind::ALL {
        for level in 0..RECORD_LEVELS {
            let record = DrawRecord { kind, level };
            words.push([record.word(), 0, 0, 0]);
        }
    }
    words
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct PlacedRecord {
    pub center: [f32; 3],
    pub radius: f32,
    pub group: u32,
    pub scale: f32,
    pub terrain: u32,
    pub sink: f32,
    pub pad: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct SlotColor {
    pub color: [u32; 2],
    pub surface: u32,
    pub pad: u32,
}

pub(crate) type PoseData = [[f32; 4]; 3];

pub(crate) type DrawArgs = [u32; 5];

pub(crate) fn draw_args((index_count, first_index, base_vertex): (u32, u32, i32)) -> DrawArgs {
    [
        index_count,
        0,
        first_index,
        u32::from_ne_bytes(base_vertex.to_ne_bytes()),
        0,
    ]
}

pub(crate) fn placed_group(impostor: ImpostorRecord, color: [u32; 2]) -> PlacedGroup {
    PlacedGroup {
        sphere: impostor.sphere,
        extent: impostor.extent,
        layer: impostor.layer,
        emissive: impostor.emissive,
        flags: impostor.flags,
        color,
        pad: [0; 2],
    }
}

pub(crate) fn pose_rows(model: Mat4) -> PoseData {
    let rows = model.transpose();
    [
        rows.x_axis.to_array(),
        rows.y_axis.to_array(),
        rows.z_axis.to_array(),
    ]
}
