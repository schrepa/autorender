import package::engine::math::maximum_abs;
import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};
import package::engine::slots::shade;
import package::engine::view::{FLOAT_GAP, camera, eye_world, fade_noise, view_pixel};
import package::generated::terrain_shading::{RUNG_WORDS, TABLE_RUNGS};
import package::generated::writes::{DepthOutputs, Outputs, depth_outputs, outputs};
import package::place::ground_tiles::PALETTE_ENTRIES;
import package::terrain_shading::draw::{TERRAIN_BAND_PIXELS, TerrainLook, terrain_detailed,
    terrain_palette, terrain_point_seen, terrain_seen};
import package::terrain_shading::faces::heightfield::{terrain, terrain_look, terrain_view,
    use_terrain};
import package::terrain_shading::tables::table_floats;

@group(1) @binding(0) var terrain_ids_texture: texture_2d<u32>;

// optics: TERRAIN_SPLIT_ROUNDINGS counts the float gaps that can make a pixel beyond the split
// optics: work out a smaller size than the split assumed: log2 is within 2^-21 near 1 (8), the
// optics: fade's product and quotient (2), the footprint's facing rounding above 1 (4), the size
// optics: from the detail eye (8), the point rebuilt from its depth (10), this split's sums (10).
const TERRAIN_SPLIT_ROUNDINGS: f32 = 64.0;

fn terrain_relative_at(pixel: vec2<f32>, depth: f32, size: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(pixel.x * 2.0 / size.x - 1.0, 1.0 - pixel.y * 2.0 / size.y);
    let point = terrain_view.inverse_relative_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w;
}

fn terrain_ray_at(pixel: vec2<f32>, size: vec2<f32>) -> vec3<f32> {
    let relative = terrain_relative_at(pixel, 1.0, size);
    return relative / max(length(relative), 1.0e-30);
}

fn terrain_split_finest() -> f32 {
    var finest = 0.0;
    for (var entry = 0u; entry < PALETTE_ENTRIES; entry++) {
        let held = terrain_palette[entry];
        if held.params.w > 0.0 {
            let tilings = table_floats(TABLE_RUNGS + entry * RUNG_WORDS).xy;
            finest = max(finest, max(held.params.x, max(tilings.x, tilings.y)));
        }
    }
    return finest;
}

fn terrain_split_depth(distance: f32) -> f32 {
    let projection = terrain_view.relative_view_projection;
    let toward = vec3<f32>(projection[0].w, projection[1].w, projection[2].w);
    let clip = projection * vec4<f32>(normalize(toward) * distance, 1.0);
    return select(1.0, clamp(clip.z / clip.w, 0.0, 1.0), clip.w > 0.0);
}

fn terrain_split_depths() -> vec2<f32> {
    let angle = camera.pixel_angle;
    let pixels = terrain_look.reach.z;
    if !(angle > 0.0 && pixels > 0.0) {
        return vec2<f32>(0.0);
    }
    let gap = TERRAIN_SPLIT_ROUNDINGS * FLOAT_GAP;
    let least = max(terrain.relief.x / TERRAIN_BAND_PIXELS, terrain_split_finest() / pixels);
    let eye = camera.detail.xyz;
    let far = max(least * (1.0 + gap) - camera.texel, 0.0) / angle
        + length(camera.position.xyz - eye);
    let skip = far + gap * (far + maximum_abs(camera.position.xyz) + maximum_abs(eye));
    return vec2<f32>(terrain_split_depth(far), terrain_split_depth(skip));
}

struct TerrainResolveInput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) terrain: u32,
    @location(1) @interpolate(flat) skip: f32,
}

fn terrain_corner(vertex: u32, depth: f32) -> vec4<f32> {
    let x = f32((vertex << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, depth, 1.0);
}

@vertex
fn vs_terrain_resolve(@builtin(vertex_index) vertex: u32, @builtin(instance_index) index: u32)
        -> TerrainResolveInput {
    use_terrain(index);
    let skip = select(terrain_split_depths().y, 0.0, vertex >= 3u);
    return TerrainResolveInput(terrain_corner(vertex % 3u, 0.0), index, skip);
}

@vertex
fn vs_terrain_resolve_far(@builtin(vertex_index) vertex: u32,
        @builtin(instance_index) index: u32) -> TerrainResolveInput {
    use_terrain(index);
    return TerrainResolveInput(terrain_corner(vertex, terrain_split_depths().x), index, 0.0);
}

struct TerrainResolved {
    point: SurfacePoint,
    shaded: Shaded,
    at: ViewPixel,
    depth: f32,
}

fn terrain_resolved(input: TerrainResolveInput, skip: f32) -> TerrainResolved {
    use_terrain(input.terrain);
    let dither = select(0.5, fade_noise(input.position.xy), terrain_view.eye_offset.w > 0.5);
    let position = input.position;
    let size = camera.region.xy;
    let texel = textureLoad(terrain_ids_texture, vec2<i32>(position.xy), 0);
    let mine = texel.x == terrain.directory.x + 1u;
    if !mine {
        discard;
    }
    let depth = bitcast<f32>(texel.y);
    let relative = terrain_relative_at(position.xy, depth, size);
    let away = length(relative);
    let here = select(0.0, 1.0, mine);
    let held = select(0.0, 1.0, mine && !(depth < skip));
    let reach = 0.05 * away + 0.5;
    let others = vec2<f32>(dpdxFine(here), dpdyFine(here));
    let farther = vec2<f32>(dpdxFine(away), dpdyFine(away));
    let same = (others == vec2<f32>(0.0)) & (abs(farther) < vec2<f32>(reach));
    let row = max(held, abs(dpdxFine(held)));
    let quad = max(row, abs(dpdyFine(row))) > 0.0;
    if !quad {
        discard;
    }
    let ray = terrain_ray_at(position.xy, size);
    let look = TerrainLook(dpdxCoarse(ray), dpdyCoarse(ray), dither);
    var point: SurfacePoint;
    var beyond = 0.0;
    if quad {
        let seen = terrain_seen(relative, look);
        beyond = seen.beyond;
        point = terrain_point_seen(relative, seen, true);
    }
    point.normal_dx = select(vec3<f32>(0.0), dpdxFine(point.normal), same.x);
    point.normal_dy = select(vec3<f32>(0.0), dpdyFine(point.normal), same.y);
    point.world_dx = select(point.world_dx, dpdxFine(relative), same.x);
    point.world_dy = select(point.world_dy, dpdyFine(relative), same.y);
    var shaded: Shaded;
    var previous = vec4<f32>(0.0);
    var written = depth;
    if quad {
        shaded = shade(point, position.xy);
        let met = relative + relative * (beyond / max(away, 1.0e-6));
        let moved = (camera.previous_view_projection * vec4<f32>(eye_world(met), 1.0)).xyw;
        previous = vec4<f32>(moved.xy, 0.0, moved.z);
        if beyond != 0.0 {
            let clip = terrain_view.relative_view_projection * vec4<f32>(met, 1.0);
            written = select(1.0, saturate(clip.z / clip.w), clip.w > 0.0);
        }
    }
    return TerrainResolved(point, shaded, view_pixel(position, previous), written);
}

@fragment
fn fs_terrain_resolve(input: TerrainResolveInput) -> DepthOutputs {
    let held = terrain_resolved(input, input.skip);
    return depth_outputs(held.point, held.shaded, held.at, held.depth);
}

@fragment
fn fs_terrain_resolve_far(input: TerrainResolveInput) -> Outputs {
    terrain_detailed = false;
    let held = terrain_resolved(input, 0.0);
    return outputs(held.point, held.shaded, held.at);
}
