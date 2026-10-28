struct View {
    clip: mat4x4<f32>,
}
@group(0) @binding(9) var<uniform> view: View;

@vertex
fn vs_terrain_shadow(input: PatchInput) -> @builtin(position) vec4<f32> {
    var clip = view.clip * vec4<f32>(terrain_vertex(input) + terrain.eye.xyz, 1.0);
    clip.z = max(clip.z, 0.0);
    return clip;
}
