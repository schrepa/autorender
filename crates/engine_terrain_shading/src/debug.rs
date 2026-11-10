use std::any::TypeId;

use engine_core::{DrawKind, GpuBits, gpu_bits};
use engine_render::debug::{DEBUG_NO_MATERIAL, DebugRecord};
use engine_render::{DebugMode, DebugShown};
use serde::Serialize;
use serde_json::Value;

use crate::TerrainShadingBrick;
use crate::gpu::{
    DEBUG_SHOWN_DETAIL, DEBUG_SHOWN_MATERIALS, DEBUG_SHOWN_SKY, DEBUG_SHOWN_TILES, ShownMaterials,
    ShownTile,
};

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TerrainDebug {
    #[bits(1, at = 14)]
    pub(crate) tile_edge: bool,
    pub(crate) patch_edge: bool,
    #[bits(4, at = 24)]
    pub(crate) patch_x: u32,
    #[bits(4)]
    pub(crate) patch_y: u32,
}

const _: () = {
    let ours = <TerrainDebug as GpuBits>::FIELDS;
    let record = <DebugRecord as GpuBits>::FIELDS;
    let mut at = 0;
    while at < ours.len() {
        let mut other = 0;
        while other < record.len() {
            assert!(
                (ours[at].mask() << ours[at].at) & (record[other].mask() << record[other].at) == 0,
                "terrain_debug_bits_sit_beside_the_debug_record"
            );
            other += 1;
        }
        at += 1;
    }
};

pub(crate) const GENERATED: &[&str] = &[<TerrainDebug as GpuBits>::WGSL];

pub(crate) const MODES: [(&str, u32); 4] = [
    ("tiles", DEBUG_SHOWN_TILES),
    ("materials", DEBUG_SHOWN_MATERIALS),
    ("sky", DEBUG_SHOWN_SKY),
    ("detail", DEBUG_SHOWN_DETAIL),
];

pub(crate) fn member(name: &'static str) -> DebugMode {
    DebugMode {
        name,
        decode: terrain,
        fields: engine_core::schema_of::<TerrainProbe>,
    }
}

pub(crate) fn shown(shown: &DebugShown) -> u32 {
    if shown.owner != Some(TypeId::of::<TerrainShadingBrick>()) {
        return 0;
    }
    MODES.get(shown.local as usize).map_or(0, |(_, code)| *code)
}

#[derive(Serialize, schemars::JsonSchema)]
struct TerrainProbe {
    terrain: Option<TerrainProbed>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct TerrainProbed {
    tile: [i32; 2],
    patch: [u32; 2],
    materials: Vec<MaterialShare>,
    sky: f32,
    detail: DetailShares,
}

#[derive(Serialize, schemars::JsonSchema)]
struct MaterialShare {
    material: u32,
    share: f32,
}

#[derive(Serialize, schemars::JsonSchema)]
struct DetailShares {
    detailed: f32,
    averaged: f32,
    canopy: f32,
}

fn terrain(kind: Option<DrawKind>, ids: [u32; 4], color: [f32; 4]) -> Value {
    let probe = TerrainProbe {
        terrain: (kind == Some(crate::terrain::TERRAIN)).then(|| terrain_probed(ids, color)),
    };
    serde_json::to_value(probe).unwrap_or(Value::Null)
}

fn terrain_probed(
    [record, object, detail, tiles]: [u32; 4],
    [ids, ..]: [f32; 4],
) -> TerrainProbed {
    let ids = ShownMaterials::unpack(ids.to_bits()).unwrap_or_default();
    let share = |word: u32, byte: u32| f32::from(((word >> (byte * 8)) & 0xff) as u8) / 255.0;
    let mut materials: Vec<(u32, f32)> = [ids.first, ids.second, ids.third, ids.fourth]
        .into_iter()
        .zip(0..)
        .map(|(material, slot)| (u32::from(material), share(detail, slot)))
        .filter(|&(material, share)| material != DEBUG_NO_MATERIAL && share > 0.0)
        .collect();
    materials.sort_by(|a, b| b.1.total_cmp(&a.1));
    let patch = TerrainDebug::unpack(record).map_or([0, 0], |bits| [bits.patch_x, bits.patch_y]);
    let tile = ShownTile::unpack(tiles).unwrap_or_default();
    TerrainProbed {
        tile: [i32::from(tile.x as i16), i32::from(tile.z as i16)],
        patch,
        materials: materials
            .into_iter()
            .map(|(material, share)| MaterialShare { material, share })
            .collect(),
        sky: share(object, 0),
        detail: DetailShares {
            detailed: share(object, 1),
            averaged: share(object, 2),
            canopy: share(object, 3),
        },
    }
}
