import package::engine::shading::SurfaceInput;
import package::engine::surface::opacity_cut;
import package::engine::view::camera;
import package::face::instances::instance_color;

fn project(position: vec3<f32>) -> vec4<f32> {
    let clip = camera.view_projection * vec4<f32>(position, 1.0);
    return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
}

@vertex
fn vs_caster(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return project(position);
}

struct MaskedOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_alpha: vec3<f32>,
}

@vertex
fn vs_caster_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(3) slot: u32) -> MaskedOutput {
    var output: MaskedOutput;
    output.clip = project(position);
    output.uv_alpha = vec3<f32>(uv, unpack2x16unorm(instance_color(slot).color.y).y);
    return output;
}

@fragment
fn fs_caster_masked(input: MaskedOutput) {
    var surface: SurfaceInput;
    surface.color = vec4<f32>(1.0, 1.0, 1.0, input.uv_alpha.z);
    surface.uv = input.uv_alpha.xy;
    surface.uv1 = input.uv_alpha.xy;
    surface.front = true;
    if opacity_cut(surface) {
        discard;
    }
}
