use serde_json::{Value, json};

use crate::gpu::{DEBUG_PATCH_EDGE, DEBUG_PATCH_SHIFT, DEBUG_TILE_EDGE};

const LEVEL_SHIFT: u32 = 4;
const LEVEL_MASK: u32 = 0xf;

pub(crate) fn terrain([record, object, detail, tiles]: [u32; 4], _: [f32; 4]) -> Value {
    let share = |word: u32, byte: u32| f32::from(((word >> (byte * 8)) & 0xff) as u8) / 255.0;
    let within = record >> DEBUG_PATCH_SHIFT;
    json!({
        "level": (record >> LEVEL_SHIFT) & LEVEL_MASK,
        "tile_edge": record & DEBUG_TILE_EDGE != 0,
        "patch_edge": record & DEBUG_PATCH_EDGE != 0,
        "tile": [
            i32::from((tiles & 0xffff) as u16 as i16),
            i32::from((tiles >> 16) as u16 as i16),
        ],
        "patch": [within & 0xf, (within >> 4) & 0xf],
        "weights": [share(detail, 0), share(detail, 1), share(detail, 2), share(detail, 3)],
        "sky": share(object, 0),
        "detail": {
            "detailed": share(object, 1),
            "averaged": share(object, 2),
            "canopy": share(object, 3),
        },
    })
}
