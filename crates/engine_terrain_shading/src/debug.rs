use std::any::TypeId;

use engine_render::debug::{DEBUG_KIND_TERRAIN, DEBUG_NO_MATERIAL};
use engine_render::{DebugMode, DebugShown};
use serde_json::{Value, json};

use crate::TerrainShadingBrick;
use crate::gpu::{
    DEBUG_PATCH_SHIFT, DEBUG_SHOWN_DETAIL, DEBUG_SHOWN_MATERIALS, DEBUG_SHOWN_SKY,
    DEBUG_SHOWN_TILES,
};

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

fn terrain([record, object, detail, tiles]: [u32; 4], [ids, ..]: [f32; 4]) -> Value {
    if record & 0xf != DEBUG_KIND_TERRAIN {
        return json!({ "terrain": null });
    }
    let ids = ids.to_bits();
    let share = |word: u32, byte: u32| f32::from(((word >> (byte * 8)) & 0xff) as u8) / 255.0;
    let mut materials: Vec<(u32, f32)> = (0..4)
        .map(|slot| ((ids >> (slot * 8)) & 0xff, share(detail, slot)))
        .filter(|&(material, share)| material != DEBUG_NO_MATERIAL && share > 0.0)
        .collect();
    materials.sort_by(|a, b| b.1.total_cmp(&a.1));
    let within = record >> DEBUG_PATCH_SHIFT;
    json!({
        "terrain": {
            "tile": [
                i32::from((tiles & 0xffff) as u16 as i16),
                i32::from((tiles >> 16) as u16 as i16),
            ],
            "patch": [within & 0xf, (within >> 4) & 0xf],
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
