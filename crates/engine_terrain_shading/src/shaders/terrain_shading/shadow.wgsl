import package::engine::view::view_clip;
import package::terrain_shading::faces::heightfield::{terrain, use_terrain};
import package::terrain_shading::field::{PatchInput, terrain_vertex};

@vertex
fn vs_terrain_shadow(input: PatchInput) -> @builtin(position) vec4<f32> {
    use_terrain(input.terrain);
    return view_clip(terrain_vertex(input) + terrain.eye.xyz);
}
