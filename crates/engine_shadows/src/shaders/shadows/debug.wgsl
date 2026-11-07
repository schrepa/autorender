import package::face::debug::{DEBUG_CARDS, DEBUG_EMPTY, DEBUG_OTHER, DebugShade, DebugTargets,
    debug_record_card, debug_record_kind, debug_record_source};
import package::engine::debug::debug_kind_color;
import package::engine::drawn::drawn_records;
import package::generated::kinds::KIND_MESH;
import package::generated::passes::PASS_SHADOW;
import package::generated::pieces::shadows::piece;
import package::generated::shadows::{DEBUG_SHOWN_CASTERS, DEBUG_SHOWN_SHADOWS};
import package::place::sun_blockers::{DEBUG_SHADOWED, DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON,
    DEBUG_SUN_LEVEL, DEBUG_SUN_NONE};

fn debug_sun_color(sun: u32) -> vec3<f32> {
    let source = sun & (DEBUG_SHADOWED - 1u);
    var color = vec3<f32>(0.0);
    if source >= DEBUG_SUN_LEVEL {
        let level = f32(source - DEBUG_SUN_LEVEL)
            / f32(max(u32(piece().level_state.x), 2u) - 1u);
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

fn drawn_casts(kind: u32, object: u32) -> bool {
    if kind != KIND_MESH {
        return true;
    }
    return object < arrayLength(&drawn_records) && (drawn_records[object].passes & PASS_SHADOW) != 0u;
}

fn debug_caster_color(record: u32, object: u32) -> vec3<f32> {
    let kind = debug_record_kind(record);
    if kind == 0u {
        return DEBUG_EMPTY;
    }
    if !drawn_casts(kind, object) {
        return DEBUG_OTHER;
    }
    if debug_record_card(record) {
        return DEBUG_CARDS;
    }
    return debug_kind_color(kind);
}

fn shade_kind(shade: DebugShade, targets: DebugTargets) -> DebugShade {
    let record = targets.ids.x;
    let kind = debug_record_kind(record);
    let surface = kind != 0u;
    var shaded = shade;
    switch u32(piece().state.z) {
        case DEBUG_SHOWN_SHADOWS: {
            shaded = DebugShade(select(debug_sun_color(debug_record_source(record)), DEBUG_EMPTY,
                !surface), surface);
        }
        case DEBUG_SHOWN_CASTERS: {
            shaded = DebugShade(debug_caster_color(record, targets.ids.y), surface);
        }
        default: {}
    }
    return shaded;
}
