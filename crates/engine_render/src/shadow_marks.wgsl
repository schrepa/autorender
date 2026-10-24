// The shadow clipmap's tiles the frame's pixels read, after the scene
// declarations (`shadows/mod.rs`): each pixel of the depth buffer the frame
// drew marks the tile holding its point at the level a pixel's size picks
// there (`shadow_level`), and at the next coarser where lookups blend it
// in; where a level's window or band does not hold the point, the first
// coarser level that does, which lookups then read. Read back a few frames
// later, the marks give those tiles slots and draws.

// The depth buffer the frame drew (reversed: 0 where only sky is), and the
// marks, a bit a cell of every level (cell c of level l at bit
// l * side * side + c).
@group(0) @binding(24) var shadow_depth: texture_depth_2d;
@group(0) @binding(25) var<storage, read_write> shadow_marks: array<atomic<u32>>;

// Marks the tile holding light-space `light` in level `level`, or in the
// first coarser level whose window and band hold it.
fn mark_level(level: u32, light: vec3<f32>) {
    let side = u32(shadow.atlas.x);
    for (var held = level; held < u32(shadow.level_state.x); held++) {
        let at = level_point(held, light);
        if !at.inside {
            continue;
        }
        let bit = (held * side + at.cell.y) * side + at.cell.x;
        let mask = 1u << (bit % 32u);
        if (atomicLoad(&shadow_marks[bit / 32u]) & mask) == 0u {
            atomicOr(&shadow_marks[bit / 32u], mask);
        }
        return;
    }
}

@compute @workgroup_size(8, 8)
fn mark_shadow_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
    // The pixels of the frame's render size, from the targets' top-left.
    let size = vec2<u32>(camera.region.xy);
    if any(id.xy >= size) || shadow.state.x == 0.0 {
        return;
    }
    let depth = textureLoad(shadow_depth, id.xy, 0);
    if depth <= 0.0 {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / camera.region.xy;
    let clip = camera.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = clip.xyz / clip.w;
    let pick = shadow_level(world);
    let light = light_space(world);
    mark_level(pick.level, light);
    if pick.blend > 0.0 {
        mark_level(pick.level + 1u, light);
    }
}
