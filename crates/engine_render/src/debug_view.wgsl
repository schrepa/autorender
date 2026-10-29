struct DebugViewUniform {
    view: vec4<u32>,
    sun: vec4<f32>,
}

@group(0) @binding(0) var<uniform> debug_view: DebugViewUniform;
@group(0) @binding(1) var debug_ids: texture_2d<u32>;
@group(0) @binding(2) var debug_count: texture_2d<f32>;
@group(0) @binding(3) var debug_terrain: texture_2d<u32>;

const DEBUG_VIEW_KINDS: u32 = 1u;
const DEBUG_VIEW_LEVELS: u32 = 2u;
const DEBUG_VIEW_CARDS: u32 = 3u;
const DEBUG_VIEW_SHADOWS: u32 = 4u;
const DEBUG_VIEW_CASTERS: u32 = 5u;
const DEBUG_VIEW_OVERDRAW: u32 = 6u;
const DEBUG_VIEW_NORMALS: u32 = 7u;
const DEBUG_VIEW_TILES: u32 = 8u;
const DEBUG_VIEW_MATERIALS: u32 = 9u;
const DEBUG_VIEW_SKY: u32 = 10u;
const DEBUG_VIEW_DETAIL: u32 = 11u;
const DEBUG_VIEW_WATER_DEPTH: u32 = 12u;
const DEBUG_VIEW_WATER_SPEED: u32 = 13u;
const DEBUG_VIEW_WATER_FROUDE: u32 = 14u;
const DEBUG_VIEW_WATER_TURBULENCE: u32 = 15u;
const DEBUG_VIEW_WATER_WHITE: u32 = 16u;
const DEBUG_VIEW_WATER_REACH: u32 = 17u;
const DEBUG_VIEW_WATER_SPAN: u32 = 19u;
const DEBUG_VIEW_WATER_EDDIES: u32 = 20u;
const DEBUG_VIEW_WATER_ROUGHNESS: u32 = 21u;

const DEBUG_KIND_MESH: u32 = 1u;
const DEBUG_KIND_PLACED: u32 = 2u;
const DEBUG_KIND_CARD: u32 = 3u;
const DEBUG_KIND_PLACED_CARD: u32 = 4u;
const DEBUG_KIND_TERRAIN: u32 = 6u;
const DEBUG_KIND_COVER: u32 = 7u;
const DEBUG_KIND_WATER: u32 = 8u;
const DEBUG_KIND_FALLS: u32 = 10u;
const DEBUG_KIND_SKY: u32 = 11u;
const DEBUG_LEVEL_SHIFT: u32 = 4u;
const DEBUG_FADING: u32 = 0x100u;
const DEBUG_CASTS: u32 = 0x200u;
const DEBUG_SOURCE_SHIFT: u32 = 16u;
const DEBUG_SUN_NONE: u32 = 0u;
const DEBUG_SUN_LEVEL: u32 = 1u;
const DEBUG_SUN_HORIZON: u32 = 21u;
const DEBUG_SUN_BEYOND: u32 = 22u;
const DEBUG_SHADOWED: u32 = 0x80u;
const DEBUG_TILE_EDGE: u32 = 0x800u;
const DEBUG_PATCH_EDGE: u32 = 0x1000u;
const DEBUG_NO_MATERIAL: u32 = 0xffu;

const DEBUG_OTHER = vec3<f32>(0.4, 0.4, 0.42);
const DEBUG_SKY = vec3<f32>(0.12, 0.13, 0.16);
const DEBUG_CARDS = vec3<f32>(0.95, 0.25, 0.85);

const DEBUG_LEVELS = array<vec3<f32>, 8>(
    vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.95, 0.6, 0.15), vec3<f32>(0.9, 0.9, 0.2),
    vec3<f32>(0.3, 0.85, 0.3), vec3<f32>(0.2, 0.75, 0.9), vec3<f32>(0.3, 0.4, 0.95),
    vec3<f32>(0.7, 0.35, 0.95), vec3<f32>(0.85, 0.85, 0.85),
);

const DEBUG_HEAT = array<vec3<f32>, 11>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.05, 0.1, 0.35), vec3<f32>(0.1, 0.25, 0.7),
    vec3<f32>(0.1, 0.55, 0.75), vec3<f32>(0.15, 0.75, 0.35), vec3<f32>(0.6, 0.85, 0.15),
    vec3<f32>(0.95, 0.8, 0.1), vec3<f32>(0.95, 0.45, 0.1), vec3<f32>(0.85, 0.15, 0.15),
    vec3<f32>(0.8, 0.3, 0.7), vec3<f32>(1.0, 1.0, 1.0),
);

const DEBUG_WATER_DEEPEST: f32 = 2.0;
const DEBUG_WATER_FASTEST: f32 = 6.0;
const DEBUG_WATER_FROUDE_TOP: f32 = 3.0;
const DEBUG_WATER_CRITICAL: f32 = 0.04;
const DEBUG_WATER_DEPTHS = array<vec3<f32>, 3>(
    vec3<f32>(0.88, 0.96, 0.88), vec3<f32>(0.2, 0.62, 0.88), vec3<f32>(0.04, 0.09, 0.4),
);
const DEBUG_WATER_SPEEDS = array<vec3<f32>, 3>(
    vec3<f32>(0.08, 0.03, 0.25), vec3<f32>(0.8, 0.15, 0.3), vec3<f32>(1.0, 0.92, 0.35),
);
const DEBUG_WATER_TURBULENCE = array<vec3<f32>, 3>(
    vec3<f32>(0.02, 0.2, 0.22), vec3<f32>(0.1, 0.6, 0.45), vec3<f32>(0.85, 1.0, 0.4),
);
const DEBUG_WATER_CLEAR = vec3<f32>(0.0, 0.12, 0.35);
const DEBUG_SPAN_STEPS: f32 = 254.0;
const DEBUG_WATER_SPANS = array<vec3<f32>, 3>(
    vec3<f32>(0.3, 0.05, 0.45), vec3<f32>(0.1, 0.62, 0.58), vec3<f32>(0.98, 0.95, 0.6),
);
const DEBUG_BREAK_ONSET: f32 = 0.6;
const DEBUG_BREAK_FULL: f32 = 1.2;
const DEBUG_WATER_EDDIES = array<vec3<f32>, 3>(
    vec3<f32>(0.04, 0.05, 0.12), vec3<f32>(0.95, 0.5, 0.1), vec3<f32>(1.0, 1.0, 0.95),
);
const DEBUG_WATER_ROUGHNESS = array<vec3<f32>, 3>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.75, 0.42, 0.22), vec3<f32>(1.0, 1.0, 1.0),
);
const DEBUG_REACHES = array<vec3<f32>, 6>(
    vec3<f32>(0.5, 0.35, 0.65), vec3<f32>(0.15, 0.35, 0.95), vec3<f32>(0.1, 0.8, 0.85),
    vec3<f32>(0.45, 0.85, 0.2), vec3<f32>(0.98, 0.6, 0.1), vec3<f32>(0.9, 0.12, 0.2),
);

fn debug_normal(packed: u32) -> vec3<f32> {
    let encoded = unpack2x16snorm(packed);
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    let normal = vec3<f32>(encoded - shift, z);
    return normal / max(length(normal), 1.0e-6);
}

fn debug_kind_color(kind: u32) -> vec3<f32> {
    switch kind {
        case DEBUG_KIND_MESH: {
            return vec3<f32>(0.55, 0.62, 0.78);
        }
        case DEBUG_KIND_PLACED: {
            return vec3<f32>(0.2, 0.72, 0.66);
        }
        case DEBUG_KIND_CARD, DEBUG_KIND_PLACED_CARD: {
            return DEBUG_CARDS;
        }
        case DEBUG_KIND_TERRAIN: {
            return vec3<f32>(0.62, 0.46, 0.3);
        }
        case DEBUG_KIND_COVER: {
            return vec3<f32>(0.35, 0.8, 0.3);
        }
        case DEBUG_KIND_WATER: {
            return vec3<f32>(0.2, 0.45, 0.95);
        }
        case DEBUG_KIND_FALLS: {
            return vec3<f32>(0.55, 0.75, 1.0);
        }
        case DEBUG_KIND_SKY: {
            return vec3<f32>(0.75, 0.85, 0.95);
        }
        default: {
            return vec3<f32>(0.0);
        }
    }
}

fn debug_level_color(record: u32) -> vec3<f32> {
    let level = min((record >> DEBUG_LEVEL_SHIFT) & 0xfu, 7u);
    switch record & 0xfu {
        case DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_TERRAIN: {
            return DEBUG_LEVELS[level];
        }
        case DEBUG_KIND_CARD, DEBUG_KIND_PLACED_CARD: {
            return DEBUG_CARDS;
        }
        case DEBUG_KIND_SKY: {
            return DEBUG_SKY;
        }
        default: {
            return DEBUG_OTHER;
        }
    }
}

fn debug_card_color(record: u32) -> vec3<f32> {
    switch record & 0xfu {
        case DEBUG_KIND_CARD, DEBUG_KIND_PLACED_CARD: {
            return select(DEBUG_CARDS, vec3<f32>(0.98, 0.65, 0.88), (record & DEBUG_FADING) != 0u);
        }
        case DEBUG_KIND_SKY: {
            return DEBUG_SKY;
        }
        default: {
            return DEBUG_OTHER;
        }
    }
}

fn debug_sun_color(sun: u32) -> vec3<f32> {
    let source = sun & (DEBUG_SHADOWED - 1u);
    var color = vec3<f32>(0.0);
    if source >= DEBUG_SUN_LEVEL && source < DEBUG_SUN_HORIZON {
        let level = f32(source - DEBUG_SUN_LEVEL)
            / f32(DEBUG_SUN_HORIZON - DEBUG_SUN_LEVEL - 1u);
        let green_to_cyan = mix(vec3<f32>(0.3, 1.0, 0.3), vec3<f32>(0.2, 0.9, 0.95),
            saturate(2.0 * level));
        color = mix(green_to_cyan, vec3<f32>(0.25, 0.35, 0.95), saturate(2.0 * level - 1.0));
    } else if source == DEBUG_SUN_HORIZON {
        color = vec3<f32>(0.7, 0.25, 0.7);
    } else if source == DEBUG_SUN_BEYOND {
        color = vec3<f32>(0.5, 0.5, 0.5);
    } else if source == DEBUG_SUN_NONE {
        color = vec3<f32>(0.0);
    }
    return select(color, color * 0.45, (sun & DEBUG_SHADOWED) != 0u);
}

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

fn debug_caster_color(record: u32) -> vec3<f32> {
    let kind = record & 0xfu;
    if kind == DEBUG_KIND_SKY {
        return DEBUG_SKY;
    }
    if kind == DEBUG_KIND_TERRAIN {
        return vec3<f32>(0.62, 0.46, 0.3);
    }
    if (record & DEBUG_CASTS) == 0u {
        return DEBUG_OTHER;
    }
    switch kind {
        case DEBUG_KIND_MESH, DEBUG_KIND_CARD: {
            return vec3<f32>(0.9, 0.25, 0.2);
        }
        case DEBUG_KIND_PLACED: {
            return vec3<f32>(0.95, 0.55, 0.15);
        }
        case DEBUG_KIND_PLACED_CARD: {
            return DEBUG_CARDS;
        }
        default: {
            return DEBUG_OTHER;
        }
    }
}

fn debug_ramp(stops: array<vec3<f32>, 3>, t: f32) -> vec3<f32> {
    let along = saturate(t) * 2.0;
    if along < 1.0 {
        return mix(stops[0], stops[1], along);
    }
    return mix(stops[1], stops[2], along - 1.0);
}

fn debug_froude_color(froude: f32) -> vec3<f32> {
    if abs(froude - 1.0) < DEBUG_WATER_CRITICAL {
        return vec3<f32>(0.0);
    }
    if froude < 1.0 {
        return mix(vec3<f32>(0.08, 0.2, 0.7), vec3<f32>(0.65, 0.92, 1.0), saturate(froude));
    }
    return mix(vec3<f32>(1.0, 0.9, 0.3), vec3<f32>(0.8, 0.08, 0.08),
        saturate((froude - 1.0) / (DEBUG_WATER_FROUDE_TOP - 1.0)));
}

fn debug_eddies_color(froude: f32) -> vec3<f32> {
    let calm = saturate(froude / DEBUG_BREAK_ONSET);
    let breaking = saturate((froude - DEBUG_BREAK_ONSET) / (DEBUG_BREAK_FULL - DEBUG_BREAK_ONSET));
    return debug_ramp(DEBUG_WATER_EDDIES, 0.5 * (calm + breaking));
}

fn debug_water_color(view: u32, ids: vec4<u32>, numbers: vec2<u32>) -> vec3<f32> {
    let motion = unpack2x16float(ids.y);
    let body = unpack4x8unorm(ids.z);
    let froudes = unpack2x16float(numbers.x);
    let state = unpack4x8unorm(numbers.y);
    switch view {
        case DEBUG_VIEW_WATER_DEPTH: {
            return debug_ramp(DEBUG_WATER_DEPTHS, sqrt(max(motion.x, 0.0) / DEBUG_WATER_DEEPEST));
        }
        case DEBUG_VIEW_WATER_SPEED: {
            return debug_ramp(DEBUG_WATER_SPEEDS, motion.y / DEBUG_WATER_FASTEST);
        }
        case DEBUG_VIEW_WATER_FROUDE: {
            return debug_froude_color(froudes.x);
        }
        case DEBUG_VIEW_WATER_TURBULENCE: {
            return debug_ramp(DEBUG_WATER_TURBULENCE, state.x);
        }
        case DEBUG_VIEW_WATER_WHITE: {
            return mix(DEBUG_WATER_CLEAR, vec3<f32>(1.0), state.y);
        }
        case DEBUG_VIEW_WATER_SPAN: {
            let steps = round(body.x * (DEBUG_SPAN_STEPS + 1.0));
            return debug_ramp(DEBUG_WATER_SPANS, max(steps - 1.0, 0.0) / DEBUG_SPAN_STEPS);
        }
        case DEBUG_VIEW_WATER_EDDIES: {
            return debug_eddies_color(froudes.y);
        }
        case DEBUG_VIEW_WATER_ROUGHNESS: {
            return debug_ramp(DEBUG_WATER_ROUGHNESS, body.y);
        }
        default: {
            return DEBUG_REACHES[min(numbers.y >> 24u, 5u)];
        }
    }
}

@fragment
fn fs_debug_view(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let ids = textureLoad(debug_ids, pixel, 0);
    let record = ids.x;
    let kind = record & 0xfu;
    var color = vec3<f32>(0.0);
    var surface = kind != 0u && kind != DEBUG_KIND_SKY;
    switch debug_view.view.x {
        case DEBUG_VIEW_KINDS: {
            color = debug_kind_color(kind);
        }
        case DEBUG_VIEW_LEVELS: {
            color = debug_level_color(record);
        }
        case DEBUG_VIEW_CARDS: {
            color = debug_card_color(record);
        }
        case DEBUG_VIEW_SHADOWS: {
            color = select(debug_sun_color(record >> DEBUG_SOURCE_SHIFT), DEBUG_SKY,
                kind == DEBUG_KIND_SKY);
        }
        case DEBUG_VIEW_CASTERS: {
            color = debug_caster_color(record);
        }
        case DEBUG_VIEW_OVERDRAW: {
            let count = u32(round(textureLoad(debug_count, pixel, 0).r));
            color = DEBUG_HEAT[min(count, 10u)];
            surface = false;
        }
        case DEBUG_VIEW_NORMALS: {
            color = select(debug_normal(ids.w) * 0.5 + 0.5, DEBUG_SKY, !surface);
            surface = false;
        }
        case DEBUG_VIEW_WATER_DEPTH, DEBUG_VIEW_WATER_SPEED, DEBUG_VIEW_WATER_FROUDE,
            DEBUG_VIEW_WATER_TURBULENCE, DEBUG_VIEW_WATER_WHITE, DEBUG_VIEW_WATER_REACH,
            DEBUG_VIEW_WATER_SPAN, DEBUG_VIEW_WATER_EDDIES, DEBUG_VIEW_WATER_ROUGHNESS: {
            color = select(DEBUG_OTHER, DEBUG_SKY, kind == DEBUG_KIND_SKY);
            if kind == DEBUG_KIND_WATER {
                color = debug_water_color(debug_view.view.x, ids,
                    textureLoad(debug_terrain, pixel, 0).xy);
            }
            surface = false;
        }
        default: {
            let view = debug_view.view.x;
            let made = unpack4x8unorm(ids.y);
            if kind != DEBUG_KIND_TERRAIN {
                color = select(DEBUG_OTHER, DEBUG_SKY, kind == DEBUG_KIND_SKY);
            } else if view == DEBUG_VIEW_TILES {
                color = debug_tiles_color(record);
            } else if view == DEBUG_VIEW_MATERIALS {
                color = debug_materials_color(textureLoad(debug_terrain, pixel, 0).y,
                    unpack4x8unorm(ids.z));
            } else if view == DEBUG_VIEW_SKY {
                color = vec3<f32>(made.x);
            } else if view == DEBUG_VIEW_DETAIL {
                color = debug_detail_color(made);
            }
            surface = surface && view != DEBUG_VIEW_SKY;
        }
    }
    if debug_view.view.y != 0u && surface {
        let facing = dot(debug_normal(ids.w), debug_view.sun.xyz);
        color *= 0.45 + 0.55 * saturate(0.5 + 0.5 * facing);
    }
    return vec4<f32>(color, 1.0);
}
