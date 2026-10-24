// Terrain in shadow maps, after `meshes/terrain_vertex.wgsl` and
// `meshes/terrain_surface.wgsl`: the leaves of the camera's subdivision
// drawn depth-only into each map drawn every frame, on the true surface
// without the displacement the camera's passes add, so terrain its
// materials do not displace shadows itself without acne; and the ground a
// kept level holds, at a level of detail of its own and unmorphed, through a
// view moving it away from the sun as far as the ground its receivers draw
// may lie below it.

struct View {
    // World to this map's clip space.
    clip: mat4x4<f32>,
}
@group(0) @binding(9) var<uniform> view: View;

// Casters between the sun and the map's near plane still darken its
// receivers: flattened onto it.
fn terrain_cast(relative: vec3<f32>) -> vec4<f32> {
    var clip = view.clip * vec4<f32>(relative + terrain.eye.xyz, 1.0);
    clip.z = max(clip.z, 0.0);
    return clip;
}

@vertex
fn vs_terrain_shadow(input: PatchInput) -> @builtin(position) vec4<f32> {
    return terrain_cast(terrain_vertex(input));
}

@vertex
fn vs_terrain_surface_shadow(@builtin(instance_index) leaf: u32,
        @builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let local = surface_vertex(surface_leaf(leaf), vertex);
    return terrain_cast(surface_relative(local, terrain_true_height(local)));
}
