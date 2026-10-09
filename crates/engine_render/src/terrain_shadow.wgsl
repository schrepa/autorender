// Terrain in shadow maps, after `meshes/terrain_vertex.wgsl`:
// the camera's patches drawn depth-only into each map, at the positions the
// camera sees, so terrain shadows itself without acne.

struct View {
    // World to this map's clip space.
    clip: mat4x4<f32>,
}
@group(0) @binding(9) var<uniform> view: View;

@vertex
fn vs_terrain_shadow(input: PatchInput) -> @builtin(position) vec4<f32> {
    let vertex = terrain_vertex(input);
    var clip = view.clip * vec4<f32>(vertex.relative + terrain.eye.xyz, 1.0);
    // Casters between the sun and the map's near plane still darken its
    // receivers: flatten them onto it.
    clip.z = max(clip.z, 0.0);
    return clip;
}
