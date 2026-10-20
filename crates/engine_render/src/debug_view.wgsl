// Debug views: each pixel of the debug targets (`meshes/debug.wgsl`) shown
// as the view says. Colors are the canvas's stored values: they need no
// encoding.

struct DebugViewUniform {
    // x: the view (DEBUG_VIEW_*); y: whether colors darken where surfaces
    // turn from the sun.
    view: vec4<u32>,
    // Toward the sun.
    sun: vec4<f32>,
}

@group(0) @binding(0) var<uniform> debug_view: DebugViewUniform;
@group(0) @binding(1) var debug_ids: texture_2d<u32>;
@group(0) @binding(2) var debug_count: texture_2d<f32>;
// Terrain's tiles and materials (`meshes/terrain_debug.wgsl`).
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

// What drew a pixel (`meshes/debug.wgsl`): its kind, level and flags, and
// the sun's lookup above DEBUG_SOURCE_SHIFT.
const DEBUG_KIND_MESH: u32 = 1u;
const DEBUG_KIND_PLACED: u32 = 2u;
const DEBUG_KIND_CARD: u32 = 3u;
const DEBUG_KIND_PLACED_CARD: u32 = 4u;
const DEBUG_KIND_VIRTUAL: u32 = 5u;
const DEBUG_KIND_TERRAIN: u32 = 6u;
const DEBUG_KIND_COVER: u32 = 7u;
const DEBUG_KIND_WATER: u32 = 8u;
const DEBUG_KIND_SEA: u32 = 9u;
const DEBUG_KIND_FALLS: u32 = 10u;
const DEBUG_KIND_SKY: u32 = 11u;
const DEBUG_LEVEL_SHIFT: u32 = 4u;
const DEBUG_FADING: u32 = 0x100u;
// The draw casts sun shadows; placed objects' casters the kept map holds.
const DEBUG_CASTS: u32 = 0x200u;
const DEBUG_KEPT: u32 = 0x400u;
const DEBUG_SOURCE_SHIFT: u32 = 16u;
const DEBUG_SUN_NONE: u32 = 0u;
const DEBUG_SUN_CASCADE: u32 = 1u;
const DEBUG_SUN_LEVEL: u32 = 5u;
const DEBUG_SUN_KEPT: u32 = 8u;
const DEBUG_SUN_MOVING: u32 = 9u;
const DEBUG_SUN_HORIZON: u32 = 10u;
const DEBUG_SUN_BEYOND: u32 = 11u;
const DEBUG_SHADOWED: u32 = 0x10u;
// Terrain pixels: on a tile's edge, on a patch's, in their record, with the
// patch within its tile above DEBUG_PATCH_SHIFT; a material slot holding
// none.
const DEBUG_TILE_EDGE: u32 = 0x800u;
const DEBUG_PATCH_EDGE: u32 = 0x1000u;
const DEBUG_PATCH_SHIFT: u32 = 24u;
const DEBUG_NO_MATERIAL: u32 = 0xffu;

// What neither a view's subject nor the sky is shown as.
const DEBUG_OTHER = vec3<f32>(0.4, 0.4, 0.42);
const DEBUG_SKY = vec3<f32>(0.12, 0.13, 0.16);
const DEBUG_CARDS = vec3<f32>(0.95, 0.25, 0.85);

// Levels of detail from the finest: red, orange, yellow, green, cyan, blue,
// violet, gray.
const DEBUG_LEVELS = array<vec3<f32>, 8>(
    vec3<f32>(0.9, 0.2, 0.2), vec3<f32>(0.95, 0.6, 0.15), vec3<f32>(0.9, 0.9, 0.2),
    vec3<f32>(0.3, 0.85, 0.3), vec3<f32>(0.2, 0.75, 0.9), vec3<f32>(0.3, 0.4, 0.95),
    vec3<f32>(0.7, 0.35, 0.95), vec3<f32>(0.85, 0.85, 0.85),
);

// Fragments per pixel, none to ten or more: dark blue through green and
// yellow to red, pink and white.
const DEBUG_HEAT = array<vec3<f32>, 11>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.05, 0.1, 0.35), vec3<f32>(0.1, 0.25, 0.7),
    vec3<f32>(0.1, 0.55, 0.75), vec3<f32>(0.15, 0.75, 0.35), vec3<f32>(0.6, 0.85, 0.15),
    vec3<f32>(0.95, 0.8, 0.1), vec3<f32>(0.95, 0.45, 0.1), vec3<f32>(0.85, 0.15, 0.15),
    vec3<f32>(0.8, 0.3, 0.7), vec3<f32>(1.0, 1.0, 1.0),
);

// The unit normal packed as two signed fractions of its octahedral
// coordinates.
fn debug_normal(packed: u32) -> vec3<f32> {
    let encoded = unpack2x16snorm(packed);
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    let normal = vec3<f32>(encoded - shift, z);
    return normal / max(length(normal), 1.0e-6);
}

// A color of its own for each value: clusters.
fn debug_hashed(value: u32) -> vec3<f32> {
    var h = value * 0x9e3779b9u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = h ^ (h >> 15u);
    let bytes = vec3<f32>(f32(h & 0xffu), f32((h >> 8u) & 0xffu), f32((h >> 16u) & 0xffu));
    return 0.25 + 0.7 * bytes / 255.0;
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
        case DEBUG_KIND_VIRTUAL: {
            return vec3<f32>(0.62, 0.42, 0.95);
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
        case DEBUG_KIND_SEA: {
            return vec3<f32>(0.1, 0.28, 0.75);
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

// Levels of meshes, placed objects and terrain; impostor cards; virtual
// geometry's clusters (`cluster`).
fn debug_level_color(record: u32, cluster: u32) -> vec3<f32> {
    let level = min((record >> DEBUG_LEVEL_SHIFT) & 0xfu, 7u);
    switch record & 0xfu {
        case DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_TERRAIN: {
            return DEBUG_LEVELS[level];
        }
        case DEBUG_KIND_CARD, DEBUG_KIND_PLACED_CARD: {
            return DEBUG_CARDS;
        }
        case DEBUG_KIND_VIRTUAL: {
            return debug_hashed(cluster);
        }
        case DEBUG_KIND_SKY: {
            return DEBUG_SKY;
        }
        default: {
            return DEBUG_OTHER;
        }
    }
}

// Impostor cards, fading ones lighter, against everything else.
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

// The sun's lookup (`meshes/debug.wgsl`'s `debug_sun`), darker where it
// leaves the point in shadow.
fn debug_sun_color(sun: u32) -> vec3<f32> {
    let source = sun & 0xfu;
    var color = vec3<f32>(0.0);
    if source >= DEBUG_SUN_CASCADE && source < DEBUG_SUN_LEVEL {
        // Greens, darker from the first cascade.
        let cascade = f32(source - DEBUG_SUN_CASCADE);
        color = vec3<f32>(0.3, 1.0, 0.3) * (1.0 - 0.2 * cascade);
    } else if source >= DEBUG_SUN_LEVEL && source < DEBUG_SUN_KEPT {
        // Cyan to blue from the finest kept level.
        let level = f32(source - DEBUG_SUN_LEVEL);
        color = mix(vec3<f32>(0.2, 0.9, 0.95), vec3<f32>(0.25, 0.35, 0.95), level * 0.5);
    } else if source == DEBUG_SUN_KEPT {
        color = vec3<f32>(0.4, 0.25, 0.8);
    } else if source == DEBUG_SUN_MOVING {
        color = vec3<f32>(1.0, 0.55, 0.1);
    } else if source == DEBUG_SUN_HORIZON {
        color = vec3<f32>(0.7, 0.25, 0.7);
    } else if source == DEBUG_SUN_BEYOND {
        color = vec3<f32>(0.5, 0.5, 0.5);
    } else if source == DEBUG_SUN_NONE {
        color = vec3<f32>(0.0);
    }
    return select(color, color * 0.45, (sun & DEBUG_SHADOWED) != 0u);
}

// A distinct color for terrain's surface material `material`.
fn debug_material_color(material: u32) -> vec3<f32> {
    let hue = fract(f32(material) * 0.618034 + 0.13);
    let k = (vec3<f32>(0.0, 2.0, 4.0) / 6.0 + hue) * 6.0;
    return 0.15 + 0.7 * saturate(abs(k % 6.0 - 3.0) - 1.0);
}

// Terrain's surface materials (`materials`, a byte each) in their shares
// (`shares`, a byte each).
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

// How much detail terrain's ground shows, from its shares (full, colored,
// canopy): green where all its maps show, yellow where their color alone
// does, red where their means stand in, blue toward its cover's canopy.
fn debug_detail_color(made: vec4<f32>) -> vec3<f32> {
    let color = vec3<f32>(0.1, 0.65, 0.15) * made.y
        + vec3<f32>(0.85, 0.75, 0.1) * max(made.z - made.y, 0.0)
        + vec3<f32>(0.75, 0.12, 0.1) * (1.0 - made.z);
    return mix(color, vec3<f32>(0.15, 0.3, 0.9), made.w);
}

// Terrain's tiles and patches as drawn, from its record: tile edges red,
// patch edges dark, the ground between gray.
fn debug_tiles_color(record: u32) -> vec3<f32> {
    if (record & DEBUG_TILE_EDGE) != 0u {
        return vec3<f32>(0.95, 0.15, 0.1);
    }
    if (record & DEBUG_PATCH_EDGE) != 0u {
        return vec3<f32>(0.1, 0.1, 0.1);
    }
    return vec3<f32>(0.6);
}

// How a pixel's draw casts sun shadows.
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
            return select(vec3<f32>(0.95, 0.55, 0.15), vec3<f32>(0.95, 0.85, 0.2),
                (record & DEBUG_KEPT) != 0u);
        }
        case DEBUG_KIND_PLACED_CARD: {
            return DEBUG_CARDS;
        }
        case DEBUG_KIND_VIRTUAL: {
            return vec3<f32>(0.62, 0.42, 0.95);
        }
        default: {
            return DEBUG_OTHER;
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
            color = debug_level_color(record, ids.z);
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
        default: {
            // Terrain's views: its ground against everything else.
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
            // The sky view shows the share unlit.
            surface = surface && view != DEBUG_VIEW_SKY;
        }
    }
    if debug_view.view.y != 0u && surface {
        let facing = dot(debug_normal(ids.w), debug_view.sun.xyz);
        color *= 0.45 + 0.55 * saturate(0.5 + 0.5 * facing);
    }
    return vec4<f32>(color, 1.0);
}
