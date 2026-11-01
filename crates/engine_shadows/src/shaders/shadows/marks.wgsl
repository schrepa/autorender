import package::engine::view::camera;
import package::generated::pieces::shadows::piece;
import package::generated::shadows::MARK_GROUP;
import package::shadows::sample::{level_point, light_space, shadow_level};

@group(1) @binding(0) var shadow_depth: texture_depth_2d;
@group(1) @binding(1) var<storage, read_write> shadow_marks: array<atomic<u32>>;

fn mark_level(level: u32, light: vec3<f32>) {
    let side = u32(piece().atlas.x);
    for (var held = level; held < u32(piece().level_state.x); held++) {
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

@compute @workgroup_size(MARK_GROUP, MARK_GROUP)
fn mark_shadow_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(camera.region.xy);
    if any(id.xy >= size) || piece().state.x == 0.0 {
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
