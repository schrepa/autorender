import package::engine::view::camera;
import package::terrain_shading::faces::heightfield::{terrain, use_terrain};
import package::terrain_shading::field::{PatchInput, terrain_vertex};

@vertex
fn vs_terrain_shadow(input: PatchInput) -> @builtin(position) vec4<f32> {
    use_terrain(input.terrain);
    var clip = camera.view_projection * vec4<f32>(terrain_vertex(input) + terrain.eye.xyz, 1.0);
    clip.z = max(clip.z, 0.0);
    return clip;
}
