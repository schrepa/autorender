import package::engine::debug::{DEBUG_CASTS, DEBUG_FADING, DEBUG_KIND_CARD, DEBUG_KIND_COVER,
    DEBUG_KIND_FALLS, DEBUG_KIND_MESH, DEBUG_KIND_PLACED, DEBUG_KIND_PLACED_CARD, DEBUG_KIND_SKY,
    DEBUG_KIND_TERRAIN, DEBUG_KIND_WATER, DEBUG_LEVEL_SHIFT, DEBUG_NO_MATERIAL, DEBUG_SOURCE_SHIFT};

struct DebugTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) terrain: vec2<u32>,
}

struct DebugShade {
    color: vec3<f32>,
    surface: bool,
}

const DEBUG_OTHER: vec3<f32> = vec3<f32>(0.4, 0.4, 0.42);
const DEBUG_SKY: vec3<f32> = vec3<f32>(0.12, 0.13, 0.16);
const DEBUG_CARDS: vec3<f32> = vec3<f32>(0.95, 0.25, 0.85);

fn debug_targets(record: u32, object: u32, detail: u32, normal: vec3<f32>, sun: u32) -> DebugTargets {
    var targets: DebugTargets;
    targets.ids = vec4<u32>(record, object, detail, 0u);
    targets.count = 1.0;
    return targets;
}

fn shown(shade: DebugShade, targets: DebugTargets) -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
