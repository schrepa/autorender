import package::face::debug::{DEBUG_KIND_SKY, DEBUG_KIND_TERRAIN, DEBUG_NO_MATERIAL, DEBUG_OTHER,
    DEBUG_SKY, DebugShade, DebugTargets};
import package::generated::pieces::terrain_shading::piece;
import package::generated::terrain_shading::{DEBUG_PATCH_EDGE, DEBUG_SHOWN_DETAIL,
    DEBUG_SHOWN_MATERIALS, DEBUG_SHOWN_SKY, DEBUG_SHOWN_TILES, DEBUG_TILE_EDGE};

fn debug_material_color(material: u32) -> vec3<f32> {
    let hue = fract(f32(material) * 0.618034 + 0.13);
    let k = (vec3<f32>(0.0, 2.0, 4.0) / 6.0 + hue) * 6.0;
    return 0.15 + 0.7 * saturate(abs(k % 6.0 - 3.0) - 1.0);
}

fn debug_materials_color(materials: u32, shares: vec4<f32>) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    for (var slot = 0u; slot < 4u; slot++) {
        let material = (materials >> (slot * 8u)) & 0xffu;
        if material != DEBUG_NO_MATERIAL {
            color += shares[slot] * debug_material_color(material);
        }
    }
    return color;
}

fn debug_detail_color(made: vec4<f32>) -> vec3<f32> {
    let color = vec3<f32>(0.1, 0.65, 0.15) * max(made.y - made.z, 0.0)
        + vec3<f32>(0.85, 0.75, 0.1) * made.z
        + vec3<f32>(0.75, 0.12, 0.1) * (1.0 - made.y);
    return mix(color, vec3<f32>(0.15, 0.3, 0.9), made.w);
}

fn debug_tiles_color(record: u32) -> vec3<f32> {
    if (record & DEBUG_TILE_EDGE) != 0u {
        return vec3<f32>(0.95, 0.15, 0.1);
    }
    if (record & DEBUG_PATCH_EDGE) != 0u {
        return vec3<f32>(0.1, 0.1, 0.1);
    }
    return vec3<f32>(0.6);
}

fn shade_kind(shade: DebugShade, targets: DebugTargets) -> DebugShade {
    let view = piece().debug.x;
    if view == 0u {
        return shade;
    }
    let ids = targets.ids;
    let record = ids.x;
    let kind = record & 0xfu;
    let made = unpack4x8unorm(ids.y);
    var color = vec3<f32>(0.0);
    if kind != DEBUG_KIND_TERRAIN {
        color = select(DEBUG_OTHER, DEBUG_SKY, kind == DEBUG_KIND_SKY);
    } else if view == DEBUG_SHOWN_TILES {
        color = debug_tiles_color(record);
    } else if view == DEBUG_SHOWN_MATERIALS {
        color = debug_materials_color(targets.terrain.y, unpack4x8unorm(ids.z));
    } else if view == DEBUG_SHOWN_SKY {
        color = vec3<f32>(made.x);
    } else if view == DEBUG_SHOWN_DETAIL {
        color = debug_detail_color(made);
    }
    let surface = kind != 0u && kind != DEBUG_KIND_SKY && view != DEBUG_SHOWN_SKY;
    return DebugShade(color, surface);
}
