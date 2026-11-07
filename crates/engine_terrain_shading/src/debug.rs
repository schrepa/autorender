use std::any::TypeId;

use engine_core::{DrawKind, GpuBits, gpu_bits};
use engine_render::debug::{DEBUG_NO_MATERIAL, DebugRecord};
use engine_render::{DebugMode, DebugShown};
use serde_json::{Value, json};

use crate::TerrainShadingBrick;
use crate::gpu::{DEBUG_SHOWN_DETAIL, DEBUG_SHOWN_MATERIALS, DEBUG_SHOWN_SKY, DEBUG_SHOWN_TILES};

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
    }
}

pub(crate) fn shown(shown: &DebugShown) -> u32 {
    if shown.owner != Some(TypeId::of::<TerrainShadingBrick>()) {
        return 0;
    }
    MODES.get(shown.local as usize).map_or(0, |(_, code)| *code)
}

fn terrain(
    kind: Option<DrawKind>,
    [record, object, detail, tiles]: [u32; 4],
    [ids, ..]: [f32; 4],
) -> Value {
    if kind != Some(crate::terrain::TERRAIN) {
        return json!({ "terrain": null });
    }
    let ids = ids.to_bits();
    let share = |word: u32, byte: u32| f32::from(((word >> (byte * 8)) & 0xff) as u8) / 255.0;
    let mut materials: Vec<(u32, f32)> = (0..4)
        .map(|slot| ((ids >> (slot * 8)) & 0xff, share(detail, slot)))
        .filter(|&(material, share)| material != DEBUG_NO_MATERIAL && share > 0.0)
        .collect();
    materials.sort_by(|a, b| b.1.total_cmp(&a.1));
    let patch = TerrainDebug::unpack(record).map_or([0, 0], |bits| [bits.patch_x, bits.patch_y]);
    json!({
        "terrain": {
            "tile": [
                i32::from((tiles & 0xffff) as u16 as i16),
                i32::from((tiles >> 16) as u16 as i16),
            ],
            "patch": patch,
            "materials": materials
                .iter()
                .map(|&(material, share)| json!({"material": material, "share": share}))
                .collect::<Vec<Value>>(),
            "sky": share(object, 0),
            "detail": {
                "detailed": share(object, 1),
                "averaged": share(object, 2),
                "canopy": share(object, 3),
            },
        },
    })
}
