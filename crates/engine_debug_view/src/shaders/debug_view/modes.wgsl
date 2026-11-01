import package::debug_view::shown::{DEBUG_OTHER, DEBUG_SKY, debug_normal};
import package::generated::debug_view::{DEBUG_VIEW_CARDS, DEBUG_VIEW_KINDS, DEBUG_VIEW_LEVELS,
    DEBUG_VIEW_NORMALS, DEBUG_VIEW_OVERDRAW};
import package::generated::pieces::debug_view::piece;
import package::place::debug::{DEBUG_FADING, DEBUG_KIND_CARD, DEBUG_KIND_COVER, DEBUG_KIND_FALLS,
    DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD, DEBUG_KIND_SKY, DEBUG_KIND_TERRAIN,
    DEBUG_KIND_WATER, DEBUG_LEVEL_SHIFT, DebugShade, DebugTargets};

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

fn shade_kind(shade: DebugShade, targets: DebugTargets) -> DebugShade {
    let ids = targets.ids;
    let record = ids.x;
    let kind = record & 0xfu;
    let surface = kind != 0u && kind != DEBUG_KIND_SKY;
    var shaded = shade;
    switch piece().view.x {
        case DEBUG_VIEW_KINDS: {
            shaded = DebugShade(debug_kind_color(kind), surface);
        }
        case DEBUG_VIEW_LEVELS: {
            shaded = DebugShade(debug_level_color(record), surface);
        }
        case DEBUG_VIEW_CARDS: {
            shaded = DebugShade(debug_card_color(record), surface);
        }
        case DEBUG_VIEW_OVERDRAW: {
            let count = u32(round(targets.count));
            shaded = DebugShade(DEBUG_HEAT[min(count, 10u)], false);
        }
        case DEBUG_VIEW_NORMALS: {
            shaded = DebugShade(select(debug_normal(ids.w) * 0.5 + 0.5, DEBUG_SKY, !surface), false);
        }
        default: {}
    }
    return shaded;
}
