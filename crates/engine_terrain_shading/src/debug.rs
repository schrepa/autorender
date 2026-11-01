use engine_render::debug::{DEBUG_KIND_TERRAIN, DEBUG_NO_MATERIAL};
use serde_json::{Value, json};

use crate::gpu::DEBUG_PATCH_SHIFT;

pub(crate) fn terrain([record, object, detail, tiles]: [u32; 4], color: [f32; 4]) -> Value {
    if record & 0xf != DEBUG_KIND_TERRAIN {
        return json!({ "terrain": null });
    }
    let ids = color[0].to_bits();
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
