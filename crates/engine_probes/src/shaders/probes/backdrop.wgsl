import package::engine::view::camera;
import package::face::sky::environment_sample;

struct BackdropOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_backdrop(@builtin(vertex_index) index: u32) -> BackdropOutput {
    let ndc = vec2<f32>(f32((index << 1u) & 2u) * 2.0 - 1.0, f32(index & 2u) * 2.0 - 1.0);
    var output: BackdropOutput;
    output.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    output.ndc = ndc;
    return output;
}

@fragment
fn fs_backdrop(input: BackdropOutput) -> @location(0) vec4<f32> {
    let far = camera.inverse_view_projection * vec4<f32>(input.ndc, 0.0, 1.0);
    let direction = normalize(far.xyz / far.w - camera.position.xyz);
    return vec4<f32>(environment_sample(direction, 0.0), 1.0);
}
