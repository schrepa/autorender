import package::face::debug::{DEBUG_CARDS, DEBUG_CASTS, DEBUG_KIND_CARD, DEBUG_KIND_MESH,
    DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD, DEBUG_KIND_SKY, DEBUG_KIND_TERRAIN, DEBUG_OTHER,
    DEBUG_SKY, DEBUG_SOURCE_SHIFT, DebugShade, DebugTargets};
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

fn shade_kind(shade: DebugShade, targets: DebugTargets) -> DebugShade {
    let record = targets.ids.x;
    let kind = record & 0xfu;
    let surface = kind != 0u && kind != DEBUG_KIND_SKY;
    var shaded = shade;
    switch u32(piece().state.z) {
        case DEBUG_SHOWN_SHADOWS: {
            shaded = DebugShade(select(debug_sun_color(record >> DEBUG_SOURCE_SHIFT), DEBUG_SKY,
                kind == DEBUG_KIND_SKY), surface);
        }
        case DEBUG_SHOWN_CASTERS: {
            shaded = DebugShade(debug_caster_color(record), surface);
        }
        default: {}
    }
    return shaded;
}
