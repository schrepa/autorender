import package::engine::view::stable_normalize;
import package::face::sun::sun_direction;
import package::generated::pieces::debug_view::piece;
import package::place::debug::{DEBUG_SOURCE_SHIFT, DebugShade, DebugTargets};

const DEBUG_OTHER: vec3<f32> = vec3<f32>(0.4, 0.4, 0.42);
const DEBUG_SKY: vec3<f32> = vec3<f32>(0.12, 0.13, 0.16);

fn debug_octahedral(normal: vec3<f32>) -> vec2<f32> {
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let sign = select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
    return select(n.xy, (vec2<f32>(1.0) - abs(n.yx)) * sign, n.z < 0.0);
}

fn debug_targets(record: u32, object: u32, detail: u32, normal: vec3<f32>, sun: u32)
        -> DebugTargets {
    var targets: DebugTargets;
    targets.ids = vec4<u32>(record | (sun << DEBUG_SOURCE_SHIFT), object, detail,
        pack2x16snorm(debug_octahedral(stable_normalize(normal))));
    targets.count = 1.0;
    targets.terrain = vec2<u32>(0u);
    return targets;
}

fn debug_normal(packed: u32) -> vec3<f32> {
    let encoded = unpack2x16snorm(packed);
    let z = 1.0 - abs(encoded.x) - abs(encoded.y);
    let fold = max(-z, 0.0);
    let shift = vec2<f32>(select(-fold, fold, encoded.x >= 0.0),
        select(-fold, fold, encoded.y >= 0.0));
    let normal = vec3<f32>(encoded - shift, z);
    return normal / max(length(normal), 1.0e-6);
}

fn shown(shade: DebugShade, targets: DebugTargets) -> vec4<f32> {
    var color = shade.color;
    if piece().view.y != 0u && shade.surface {
        let facing = dot(debug_normal(targets.ids.w), sun_direction());
        color *= 0.45 + 0.55 * saturate(0.5 + 0.5 * facing);
    }
    return vec4<f32>(color, 1.0);
}
