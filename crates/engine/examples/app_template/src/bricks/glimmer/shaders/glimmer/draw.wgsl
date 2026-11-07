import package::engine::drawn::drawn_shown;
import package::engine::shading::{flat, flat_point};
import package::engine::view::{camera, view_pixel};
import package::generated::glimmer::GlimmerInstance;
import package::generated::targets::{ViewTargets, view_targets};
@group(1) @binding(0) var<storage, read> glimmers: array<GlimmerInstance>;

const GLIMMER_UP: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
const GLIMMER_SIDE: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);
const GLIMMER_RING: f32 = 1.75;
const GLIMMER_RING_INNER: f32 = 0.8;

struct Corner { @builtin(position) clip: vec4<f32>, @location(0) unit: vec2<f32>, @location(1) color: vec4<f32>, @location(2) previous: vec4<f32> }

fn placed(corner: u32, glimmer: GlimmerInstance, grow: f32, color: vec4<f32>) -> Corner {
    if !drawn_shown(glimmer.drawn.x) {
        var hidden: Corner;
        hidden.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return hidden;
    }
    let unit = vec2<f32>(f32((50u >> corner) & 1u), f32((44u >> corner) & 1u)) * 2.0 - 1.0;
    let toward = normalize(camera.position.xyz - glimmer.place.xyz);
    let across = cross(GLIMMER_UP, toward);
    let right = select(GLIMMER_SIDE, normalize(across), dot(across, across) > 1.0e-6);
    let up = cross(toward, right);
    let offset = (right * unit.x + up * unit.y) * glimmer.place.w * grow;
    var out: Corner;
    out.clip = camera.view_projection * vec4<f32>(glimmer.place.xyz + offset, 1.0);
    out.previous = camera.previous_view_projection * vec4<f32>(glimmer.previous.xyz + offset, 1.0);
    out.unit = unit;
    out.color = color;
    return out;
}

@vertex fn vs_glimmer(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32) -> Corner {
    let glimmer = glimmers[index];
    return placed(corner, glimmer, 1.0, glimmer.color);
}

@vertex fn vs_ring(@builtin(vertex_index) corner: u32, @builtin(instance_index) index: u32) -> Corner {
    let glimmer = glimmers[index];
    return placed(corner, glimmer, GLIMMER_RING, glimmer.ring);
}

@fragment fn fs_glimmer(corner: Corner) -> ViewTargets {
    if dot(corner.unit, corner.unit) > 1.0 { discard; }
    return view_targets(flat_point(corner.color), flat(corner.color),
        view_pixel(corner.clip, corner.previous));
}

@fragment fn fs_ring(corner: Corner) -> ViewTargets {
    let reach = length(corner.unit);
    if reach > 1.0 || reach < GLIMMER_RING_INNER { discard; }
    return view_targets(flat_point(corner.color), flat(corner.color),
        view_pixel(corner.clip, corner.previous));
}
