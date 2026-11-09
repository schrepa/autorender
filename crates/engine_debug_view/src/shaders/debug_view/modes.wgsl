import package::debug_view::shown::{DEBUG_EMPTY, DEBUG_OTHER, debug_normal};
import package::engine::debug::debug_kind_color;
import package::generated::debug_view::{DEBUG_VIEW_CARDS, DEBUG_VIEW_KINDS, DEBUG_VIEW_LEVELS,
    DEBUG_VIEW_NORMALS, DEBUG_VIEW_OVERDRAW};
import package::face::impostors::DEBUG_CARD;
import package::face::instances::{DEBUG_FADING, DEBUG_LEVEL_AT, DEBUG_LEVEL_MASK};
import package::generated::pieces::debug_view::piece;
import package::place::debug::{DebugShade, DebugTargets, debug_record_kind};

const DEBUG_CARDS = vec3<f32>(0.95, 0.25, 0.85);

fn debug_record_card(record: u32) -> bool {
    return (record & DEBUG_CARD) != 0u;
}

fn debug_record_fading(record: u32) -> bool {
    return (record & DEBUG_FADING) != 0u;
}

fn debug_record_level(record: u32) -> u32 {
    return (record >> DEBUG_LEVEL_AT) & DEBUG_LEVEL_MASK;
}

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

fn debug_kind_shade(record: u32) -> vec3<f32> {
    let kind = debug_record_kind(record);
    if kind == 0u {
        return DEBUG_EMPTY;
    }
    if debug_record_card(record) {
        return DEBUG_CARDS;
    }
    return debug_kind_color(kind);
}

fn debug_level_color(record: u32) -> vec3<f32> {
    if debug_record_kind(record) == 0u {
        return DEBUG_EMPTY;
    }
    if debug_record_card(record) {
        return DEBUG_CARDS;
    }
    return DEBUG_LEVELS[min(debug_record_level(record), 7u)];
}

fn debug_card_color(record: u32) -> vec3<f32> {
    if debug_record_kind(record) == 0u {
        return DEBUG_EMPTY;
    }
    if debug_record_card(record) {
        return select(DEBUG_CARDS, vec3<f32>(0.98, 0.65, 0.88), debug_record_fading(record));
    }
    return DEBUG_OTHER;
}

fn shade_kind(shade: DebugShade, targets: DebugTargets) -> DebugShade {
    let ids = targets.ids;
    let record = ids.x;
    let surface = debug_record_kind(record) != 0u;
    var shaded = shade;
    switch piece().view.x {
        case DEBUG_VIEW_KINDS: {
            shaded = DebugShade(debug_kind_shade(record), surface);
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
            shaded = DebugShade(select(debug_normal(ids.w) * 0.5 + 0.5, DEBUG_EMPTY, !surface), false);
        }
        default: {}
    }
    return shaded;
}
