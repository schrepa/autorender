import package::engine::debug::{DEBUG_NO_MATERIAL, debug_record, debug_record_kind,
    debug_record_source, debug_record_with_kind, debug_record_with_source};

struct DebugTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) words: vec2<u32>,
}

struct DebugShade {
    color: vec3<f32>,
    surface: bool,
}

const DEBUG_OTHER: vec3<f32> = vec3<f32>(0.0);
const DEBUG_EMPTY: vec3<f32> = vec3<f32>(0.0);

fn debug_targets(record: u32, object: u32, detail: u32, normal: vec3<f32>, source: u32) -> DebugTargets {
    var targets: DebugTargets;
    targets.ids = vec4<u32>(record, object, detail, 0u);
    targets.count = 1.0;
    return targets;
}

fn shown(shade: DebugShade, targets: DebugTargets) -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
