import package::engine::shading::ViewPixel;

struct DebugTargets {
    color: vec4<f32>,
    ids: vec4<u32>,
}

fn shown(color: vec4<f32>, at: ViewPixel) -> vec4<f32> {
    return color;
}
