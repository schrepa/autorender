import package::engine::math::maximum_abs;

@group(0) @binding(0) var<uniform> camera: CameraUniform;

override PASS_CLAMP_NEAR: bool = false;

fn view_clip(world: vec3<f32>) -> vec4<f32> {
    let clip = camera.view_projection * vec4<f32>(world, 1.0);
    if PASS_CLAMP_NEAR {
        return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
    }
    return clip;
}

fn stable_normalize(value: vec3<f32>) -> vec3<f32> {
    let scale = maximum_abs(value);
    if scale == 0.0 { // optics: a zero vector's guard, whatever it measures
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(value / scale);
}
