import package::engine::color::decode_srgb;
import package::engine::mesh::declarations::MASK_MIP_SCALE;
import package::engine::mesh::impostor_frames::{
    IMPOSTOR_CUTOFF, IMPOSTOR_HEMISPHERE, impostor_basis, impostor_facing,
    impostor_frame_direction, impostor_frames, impostor_grid, impostor_met,
};
import package::engine::mesh::vertex::octahedral_normal;
import package::engine::post::frame::{depth_texture, drawn_uv, glow, linear, post};
import package::engine::screen_points::{depth_normal, world_at};
import package::engine::shading::{BODY_SOLID, BODY_THIN, BODY_WATER};
import package::engine::view::{camera, stable_normalize};
import package::face::bounce::bounce_light;
import package::face::cover::{cover_growth, cover_seen, cover_tint};
import package::face::ground_tiles::{
    LevelTile, NOT_RESIDENT, PEAK_CELLS, PEAK_MIPS, TILE_QUADS, ground_colors,
    ground_colors_mip, ground_extent, ground_of, ground_sky, ground_tile, level_tiles,
    macro_color, material_look, terrain_info, tile_height, tile_layer, tile_material, tile_peak,
    triangle_height,
};
import package::face::horizons::terrain_visibility;
import package::face::placed_tiles::{
    Pose, placed_color, placed_frame, placed_frame_layers, placed_impostor, placed_pose,
    placed_record, placed_shown,
};
import package::face::shadows::sun_shadowed;
import package::face::sky::{
    ambient as sky_ambient, atmospheric, cloud_passing, sky_above, sky_average,
};
import package::face::sun::{sun_color, sun_direction};
import package::face::tonemap::tonemap;
import package::generated::pieces::reflections::piece;
import package::generated::reflections::{NO_PLACED, TraceTerrain};
import package::generated::targets::output_surface;
import package::reflections::surface::MIRRORED_BITS;

@group(1) @binding(0) var nearest_depth: texture_2d<f32>;
@group(1) @binding(1) var scene_levels: texture_2d<f32>;
@group(1) @binding(2) var<storage, read> world_terrains: array<TraceTerrain>;
@group(1) @binding(3) var<storage, read> world_placed: array<u32>;

const REFLECTION_STEPS: u32 = 96u;
const WORLD_STEPS: u32 = 2u * PEAK_CELLS;
const WORLD_PAST: f32 = 1.0e-3;
const REFLECTION_THIN: f32 = 0.05;
const REFLECTION_CONE: f32 = 1.19;
const REFLECTION_LOBE_EDGE: f32 = 0.244;
const REFLECTION_TAPS: u32 = 8u;
const REFLECTION_FACING: f32 = 0.05;
const REFLECTION_END: f32 = 0.1;
const FLOAT_MAX: f32 = 3.4028235e38f;

fn nearest_depth_at(cell: vec2<i32>, mip: i32) -> f32 {
    if mip == 0 {
        let last = vec2<i32>(post.size.xy) - 1;
        return textureLoad(depth_texture, clamp(cell, vec2<i32>(0), last), 0);
    }
    let span = 1 << u32(mip);
    let last = max((vec2<i32>(post.size.xy) + span - 1) / span, vec2<i32>(1)) - 1;
    return textureLoad(nearest_depth, clamp(cell, vec2<i32>(0), last), mip - 1).r;
}

fn screen_point(world: vec3<f32>) -> vec4<f32> {
    let clip = post.view_projection * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec4<f32>((ndc.x * 0.5 + 0.5) * post.size.x, (0.5 - ndc.y * 0.5) * post.size.y,
        ndc.z, clip.w);
}

fn world_point(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel * post.size.zw;
    let world = post.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return world.xyz / world.w;
}

struct Mirroring {
    normal: vec3<f32>,
    roughness: vec2<f32>,
    mirrored: f32,
}

fn mirroring_at(pixel: vec2<i32>) -> Mirroring {
    let words = textureLoad(output_surface, pixel, 0);
    let most = (1u << MIRRORED_BITS) - 1u;
    var mirroring: Mirroring;
    mirroring.normal = normalize(octahedral_normal(vec2<f32>(words.xy) / 65535.0 * 2.0 - 1.0));
    mirroring.roughness = unpack4x8unorm(words.z).xy;
    mirroring.mirrored = f32(words.w & most) / f32(most);
    return mirroring;
}

fn body_at(pixel: vec2<i32>) -> u32 {
    return textureLoad(output_surface, pixel, 0).w >> MIRRORED_BITS;
}

fn blurred_scene(at: vec2<f32>, level: f32) -> vec3<f32> {
    return textureSampleLevel(scene_levels, linear, drawn_uv(at, post), level).rgb;
}

fn lobe_light(hit: vec2<f32>, way: vec2<f32>, lengths: vec2<f32>, p: vec3<f32>,
        plane: vec3<f32>) -> vec3<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let squared = dot(way, way);
    let along = select(vec2<f32>(0.0, 1.0), way * inverseSqrt(max(squared, 1.0e-30)),
        squared > 1.0e-12);
    let count = u32(clamp(ceil(lengths.x / max(lengths.y, 0.5)), 1.0, f32(REFLECTION_TAPS)));
    let spacing = 2.0 * lengths.x / f32(count);
    let level = log2(max(max(2.0 * lengths.y, spacing), 1.0));
    if count == 1u {
        return blurred_scene(hit, level);
    }
    let no_plane = all(plane == vec3<f32>(0.0));
    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var tap = 0u; tap < count; tap++) {
        let offset = f32(2u * tap + 1u) / f32(count) - 1.0;
        let at = hit + along * (offset * lengths.x);
        let seen = world_at(clamp(vec2<i32>(floor(at)), vec2<i32>(0), last));
        let above = dot(seen.xyz - p, plane) > distance(seen.xyz, post.eye.xyz) / post.eye.w;
        if seen.w != 0.0 && !above && !no_plane {
            continue;
        }
        let weight = pow(REFLECTION_LOBE_EDGE, offset * offset);
        sum += blurred_scene(at, level) * weight;
        total += weight;
    }
    if total <= 0.0 {
        return blurred_scene(hit, level);
    }
    return sum / total;
}

fn surface_span(pixel: vec2<i32>, axis: vec2<i32>, depth: f32, at: vec2<f32>, shown: vec3<f32>)
        -> f32 {
    let before = nearest_depth_at(pixel - axis, 0);
    let after = nearest_depth_at(pixel + axis, 0);
    if before <= 0.0 || after <= 0.0 || (before - depth) * (after - depth) >= 0.0 {
        return 0.0;
    }
    let smaller = select(after, before, abs(before - depth) < abs(after - depth));
    return distance(world_point(at, smaller), shown);
}

fn reflects_itself(pixel: vec2<i32>, p: vec3<f32>, plane: vec3<f32>, at: vec2<f32>) -> bool {
    let last = vec2<i32>(post.size.xy) - 1;
    let cell = clamp(vec2<i32>(floor(at)), vec2<i32>(0), last);
    if all(plane == vec3<f32>(0.0)) || body_at(cell) != body_at(pixel) {
        return false;
    }
    let depth = nearest_depth_at(cell, 0);
    let center = vec2<f32>(cell) + 0.5;
    let seen = world_point(center, depth);
    let spans = max(surface_span(cell, vec2<i32>(1, 0), depth, center, seen),
        surface_span(cell, vec2<i32>(0, 1), depth, center, seen));
    let sight = normalize(seen - post.eye.xyz);
    return abs(dot(seen - p, plane)) <= spans * abs(dot(sight, plane));
}

struct Unposed {
    x: vec3<f32>,
    y: vec3<f32>,
    z: vec3<f32>,
    origin: vec3<f32>,
}

fn unposed_pose(pose: Pose) -> Unposed {
    let x = vec3<f32>(pose.r0.x, pose.r1.x, pose.r2.x);
    let y = vec3<f32>(pose.r0.y, pose.r1.y, pose.r2.y);
    let z = vec3<f32>(pose.r0.z, pose.r1.z, pose.r2.z);
    let yz = cross(y, z);
    let zx = cross(z, x);
    let xy = cross(x, y);
    let determinant = dot(x, yz);
    let inverse = select(0.0, 1.0 / determinant, abs(determinant) > 1.0e-30);
    return Unposed(yz * inverse, zx * inverse, xy * inverse,
        vec3<f32>(pose.r0.w, pose.r1.w, pose.r2.w));
}

fn unposed_offset(unposed: Unposed, offset: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(unposed.x, offset), dot(unposed.y, offset), dot(unposed.z, offset));
}

fn object_holds(slot: u32, front: vec3<f32>, back: vec3<f32>) -> bool {
    let record = placed_record(slot);
    if !placed_shown(record)
        || distance(front, record.center) > record.radius
        || distance(back, record.center) > record.radius {
        return false;
    }
    let impostor = placed_impostor(record.group);
    let unposed = unposed_pose(placed_pose(slot));
    let near = unposed_offset(unposed, front - unposed.origin) - impostor.sphere.xyz;
    let far = unposed_offset(unposed, back - unposed.origin) - impostor.sphere.xyz;
    return impostor.sphere.w > 0.0 && all(abs(near) <= impostor.extent)
        && all(abs(far) <= impostor.extent);
}

fn block_holds(block: u32, first: vec2<f32>, size: f32, local: vec2<f32>, front: vec3<f32>,
        back: vec3<f32>) -> bool {
    let cells = world_placed[block];
    let apron = bitcast<f32>(world_placed[block + 1u]);
    let entries = block + world_placed[block + 2u];
    let width = (size + 2.0 * apron) / f32(cells);
    let at = floor((local - (first - vec2<f32>(apron))) / width);
    if any(at < vec2<f32>(0.0)) || any(at >= vec2<f32>(f32(cells))) {
        return false;
    }
    let cell = vec2<u32>(at);
    let ends = block + 4u + (4u * cells * cells - 1u) / 3u;
    let index = cell.y * cells + cell.x;
    let listed = select(world_placed[ends + index - 1u], 0u, index == 0u);
    let listed_end = world_placed[ends + index];
    for (var entry = listed; entry < listed_end; entry++) {
        if object_holds(world_placed[entries + entry], front, back) {
            return true;
        }
    }
    return false;
}

fn crown_holds(front: vec3<f32>, back: vec3<f32>) -> bool {
    let count = u32(piece().trace.w);
    for (var index = 0u; index < count; index++) {
        let terrain = world_terrains[index];
        let held = terrain.maps.y;
        let info = terrain_info(terrain.maps.x);
        let ground = ground_of(info);
        let local = front.xz - info.origin.xz;
        let apron = terrain.reach.x;
        for (var level = 0u; level < info.levels.y; level++) {
            if ((held >> level) & 1u) == 0u {
                continue;
            }
            let size = f32(TILE_QUADS) * info.origin.w * f32(1u << level);
            let tiles = vec2<i32>(level_tiles(ground, level));
            let low = max(vec2<i32>(floor((local - apron) / size)), vec2<i32>(0));
            let high = min(vec2<i32>(floor((local + apron) / size)), tiles - 1);
            for (var z = low.y; z <= high.y; z++) {
                for (var x = low.x; x <= high.x; x++) {
                    let layer = tile_layer(ground, level, vec2<u32>(u32(x), u32(z)));
                    if layer != NOT_RESIDENT && world_placed[layer] != NO_PLACED
                        && block_holds(world_placed[layer], vec2<f32>(f32(x), f32(z)) * size,
                            size, local, front, back) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

struct ScreenWalk {
    met: bool,
    point: vec3<f32>,
    clear: f32,
}

fn screen_walk(origin: vec3<f32>, ray: vec3<f32>) -> ScreenWalk {
    var walk: ScreenWalk;
    let moving = abs(ray.xy) > vec2<f32>(1.0e-9);
    let forward = ray.xy >= vec2<f32>(0.0);
    let rate = select(vec2<f32>(1.0), ray.xy, moving);
    let ahead = select(vec2<f32>(0.0), vec2<f32>(1.0), forward);
    let past = select(vec2<f32>(-0.005), vec2<f32>(0.005), forward);
    let top = i32(textureNumLevels(nearest_depth));
    let along = max(max(abs(ray.x), abs(ray.y)), 1.0e-6);
    let exits = select(vec2<f32>(FLOAT_MAX), (floor(origin.xy) + ahead + past - origin.xy) / rate,
        moving);
    var t = min(exits.x, exits.y);
    var point = origin + ray * t;
    var mip = 0;
    walk.clear = min(t, 1.0);
    var entered = vec2<f32>(-1.0);
    for (var step = 0u; step < REFLECTION_STEPS; step++) {
        if t > 1.0 || any(point.xy < vec2<f32>(0.0)) || any(point.xy >= post.size.xy) {
            break;
        }
        let size = exp2(f32(mip));
        let cell = floor(point.xy / size);
        let surface = nearest_depth_at(vec2<i32>(cell), mip);
        let edges = select(vec2<f32>(FLOAT_MAX), ((cell + ahead) * size + past - origin.xy) / rate,
            moving);
        let leave = min(edges.x, edges.y);
        if surface < point.z {
            if entered.x >= 0.0 {
                break;
            }
            var meet = FLOAT_MAX;
            if ray.z < 0.0 {
                meet = (surface - origin.z) / ray.z;
            }
            t = min(leave, meet);
            point = origin + ray * t;
            walk.clear = min(t, 1.0);
            if leave < meet {
                mip = min(mip + 1, top);
            } else if mip > 0 {
                mip -= 1;
            } else {
                walk.met = t <= 1.0 && surface > 0.0;
                walk.point = point;
                return walk;
            }
            continue;
        }
        if mip > 0 {
            mip -= 1;
            continue;
        }
        if surface <= 0.0 {
            break;
        }
        let here = vec2<i32>(cell);
        let body = body_at(here);
        if body == BODY_WATER {
            walk.met = true;
            walk.point = point;
            return walk;
        }
        let ray_point = world_point(point.xy, point.z);
        let shown = world_point(point.xy, surface);
        let stride = distance(world_point(point.xy + ray.xy / along, point.z + ray.z / along),
            ray_point);
        let inside = select(1.0 - fract(point.xy), fract(point.xy), forward);
        let axis = select(vec2<i32>(0, 1), vec2<i32>(1, 0), inside.x < inside.y);
        let spans = stride + surface_span(here, axis, surface, point.xy, shown);
        var thickness = 0.0;
        if body == BODY_THIN {
            thickness = REFLECTION_THIN;
        }
        if body == BODY_SOLID {
            if entered.x < 0.0 {
                entered = point.xy;
            }
            thickness = distance(point.xy, entered) * distance(shown, post.eye.xyz) / post.eye.w;
        }
        if distance(ray_point, shown) <= thickness + spans
            || (body == BODY_THIN && crown_holds(shown, ray_point)) {
            walk.met = true;
            walk.point = point;
            return walk;
        }
        if body != BODY_SOLID {
            break;
        }
        t = leave;
        point = origin + ray * t;
    }
    return walk;
}

fn box_span(start: vec2<f32>, rate: vec2<f32>, low: vec2<f32>, high: vec2<f32>, nearest: f32,
        farthest: f32) -> vec2<f32> {
    let moving = abs(rate) > vec2<f32>(1.0e-12);
    let inverse = 1.0 / select(vec2<f32>(1.0), rate, moving);
    let a = (low - start) * inverse;
    let b = (high - start) * inverse;
    let inside = (start >= low) & (start <= high);
    let enters = select(select(vec2<f32>(FLOAT_MAX), vec2<f32>(-FLOAT_MAX), inside), min(a, b),
        moving);
    let leaves = select(select(vec2<f32>(-FLOAT_MAX), vec2<f32>(FLOAT_MAX), inside), max(a, b),
        moving);
    return vec2<f32>(max(nearest, max(enters.x, enters.y)),
        min(farthest, min(leaves.x, leaves.y)));
}

fn square_exit(start: vec2<f32>, rate: vec2<f32>, first: vec2<f32>, width: f32) -> f32 {
    let forward = rate >= vec2<f32>(0.0);
    let moving = abs(rate) > vec2<f32>(1.0e-12);
    let edge = first + select(vec2<f32>(-WORLD_PAST), vec2<f32>(width + WORLD_PAST), forward);
    let exits = select(vec2<f32>(FLOAT_MAX), (edge - start) / select(vec2<f32>(1.0), rate, moving),
        moving);
    return min(exits.x, exits.y);
}

fn in_cutout(local: vec2<f32>, cutout: vec4<f32>) -> bool {
    return cutout.z > cutout.x && all(local >= cutout.xy) && all(local <= cutout.zw);
}

struct WorldHit {
    met: bool,
    s: f32,
    normal: vec3<f32>,
    ground: bool,
    terrain: u32,
    level: u32,
    slot: u32,
    held: vec3<f32>,
}

fn ray_footprint(spread: vec2<f32>, s: f32) -> f32 {
    return max((spread.x + s) / post.eye.w, spread.y * s);
}

fn ground_meet(tile: LevelTile, first: vec2<u32>, o: vec3<f32>, direction: vec3<f32>,
        span: vec2<f32>, spacing: f32, cutout: vec4<f32>) -> vec4<f32> {
    var h: array<f32, 9>;
    for (var z = 0u; z < 3u; z++) {
        for (var x = 0u; x < 3u; x++) {
            h[z * 3u + x] = tile_height(tile, first + vec2<u32>(x, z));
        }
    }
    let start = o.xz / spacing - vec2<f32>(first);
    let rate = direction.xz / spacing;
    var s = span.x;
    for (var crossed = 0u; crossed < 3u; crossed++) {
        if s >= span.y {
            break;
        }
        let quad = min(vec2<u32>(max(floor(start + rate * s), vec2<f32>(0.0))), vec2<u32>(1u));
        let leave = min(square_exit(start, rate, vec2<f32>(quad), 1.0), span.y);
        let corner = quad.y * 3u + quad.x;
        let a = h[corner];
        let b = h[corner + 1u];
        let c = h[corner + 3u];
        let d = h[corner + 4u];
        let f0 = start - vec2<f32>(quad);
        let across = rate.x + rate.y;
        var split = leave;
        if abs(across) > 1.0e-12 {
            split = clamp((1.0 - f0.x - f0.y) / across, s, leave);
        }
        var low = s;
        for (var piece = 0u; piece < 2u; piece++) {
            let high = select(leave, split, piece == 0u);
            if high > low {
                let fa = clamp(f0 + rate * low, vec2<f32>(0.0), vec2<f32>(1.0));
                let fb = clamp(f0 + rate * high, vec2<f32>(0.0), vec2<f32>(1.0));
                let above_low = o.y + direction.y * low - triangle_height(a, b, c, d, fa);
                let above_high = o.y + direction.y * high - triangle_height(a, b, c, d, fb);
                if above_low <= 0.0 || above_high <= 0.0 {
                    var meet = low;
                    if above_low > 0.0 {
                        meet = low + (high - low) * above_low / (above_low - above_high);
                    }
                    let f = clamp(f0 + rate * meet, vec2<f32>(0.0), vec2<f32>(1.0));
                    let local = (vec2<f32>(first + quad) + f) * spacing;
                    if !in_cutout(local, cutout) {
                        let rise = vec2<f32>(mix(b - a, d - c, f.y), mix(c - a, d - b, f.x))
                            / spacing;
                        return vec4<f32>(normalize(vec3<f32>(-rise.x, 1.0, -rise.y)), meet);
                    }
                }
            }
            low = max(high, low);
        }
        s = leave;
    }
    return vec4<f32>(0.0, 1.0, 0.0, -1.0);
}

fn trace_ground(index: u32, p: vec3<f32>, direction: vec3<f32>, bounds: vec2<f32>,
        hit: ptr<function, WorldHit>) {
    let terrain = world_terrains[index];
    let info = terrain_info(terrain.maps.x);
    let ground = ground_of(info);
    let o = p - info.origin.xyz;
    let span = box_span(o.xz, direction.xz, vec2<f32>(0.0), ground_extent(ground), bounds.x,
        min(bounds.y, (*hit).s));
    var s = span.x;
    var steps = 0u;
    loop {
        let end = min(span.y, (*hit).s);
        if s >= end || steps >= WORLD_STEPS {
            break;
        }
        steps++;
        let at = o + direction * s;
        let found = ground_tile(ground, at.xz, 0u);
        if found.tile.layer == NOT_RESIDENT {
            let size = f32(TILE_QUADS) * ground.spacing * f32(1u << (ground.levels - 1u));
            s = square_exit(o.xz / size, direction.xz / size, floor(at.xz / size), 1.0);
            continue;
        }
        let spacing = ground.spacing * f32(1u << found.level);
        let quads = select(TILE_QUADS, TILE_QUADS / 2u, found.level > 0u);
        let top = select(PEAK_MIPS - 1u, PEAK_MIPS - 2u, found.level > 0u);
        let region = floor(at.xz / spacing / f32(quads)) * f32(quads);
        let offset = vec2<u32>(region) - found.tile.first;
        let start = o.xz / spacing - region;
        let rate = direction.xz / spacing;
        let leaves = min(square_exit(start, rate, vec2<f32>(0.0), f32(quads)), end);
        var mip = top;
        loop {
            if s >= leaves || steps >= WORLD_STEPS {
                break;
            }
            steps++;
            let width = f32(2u << mip);
            let cells = quads >> (mip + 1u);
            let cell = min(vec2<u32>(max(floor((start + rate * s) / width), vec2<f32>(0.0))),
                vec2<u32>(cells - 1u));
            let first = vec2<f32>(cell) * width;
            let peak = tile_peak(found.tile, offset / (2u << mip) + cell, mip);
            let exit = min(square_exit(start, rate, first, width), leaves);
            if min(o.y + direction.y * s, o.y + direction.y * exit) > peak {
                s = exit;
                mip = min(mip + 1u, top);
                continue;
            }
            if mip > 0u {
                mip -= 1u;
                continue;
            }
            let met = ground_meet(found.tile, vec2<u32>(region + first), o, direction,
                vec2<f32>(s, exit), spacing, terrain.cutout);
            if met.w >= 0.0 {
                (*hit).met = true;
                (*hit).s = met.w;
                (*hit).normal = met.xyz;
                (*hit).ground = true;
                (*hit).terrain = index;
                (*hit).level = found.level;
                return;
            }
            s = exit;
            mip = min(mip + 1u, top);
        }
    }
}

fn cube_span(start: vec3<f32>, rate: vec3<f32>, half: vec3<f32>, nearest: f32, farthest: f32)
        -> vec2<f32> {
    let moving = abs(rate) > vec3<f32>(1.0e-12);
    let inverse = 1.0 / select(vec3<f32>(1.0), rate, moving);
    let a = (-half - start) * inverse;
    let b = (half - start) * inverse;
    let inside = abs(start) <= half;
    let enters = select(select(vec3<f32>(FLOAT_MAX), vec3<f32>(-FLOAT_MAX), inside), min(a, b),
        moving);
    let leaves = select(select(vec3<f32>(-FLOAT_MAX), vec3<f32>(FLOAT_MAX), inside), max(a, b),
        moving);
    return vec2<f32>(max(nearest, max(max(enters.x, enters.y), enters.z)),
        min(farthest, min(min(leaves.x, leaves.y), leaves.z)));
}

fn test_placed(slot: u32, p: vec3<f32>, direction: vec3<f32>, nearest: f32, spread: vec2<f32>,
        hit: ptr<function, WorldHit>, index: u32) {
    let record = placed_record(slot);
    if !placed_shown(record) {
        return;
    }
    let impostor = placed_impostor(record.group);
    let texels = impostor.flags & (IMPOSTOR_HEMISPHERE - 1u);
    let radius = impostor.sphere.w;
    if texels == 0u || radius <= 0.0 {
        return;
    }
    let unposed = unposed_pose(placed_pose(slot));
    let o = unposed_offset(unposed, p - unposed.origin) - impostor.sphere.xyz;
    let way = unposed_offset(unposed, direction);
    let rate = length(way);
    let span = cube_span(o, way, impostor.extent, nearest, (*hit).s);
    if rate <= 0.0 || span.x >= span.y {
        return;
    }
    let toward = -way / rate;
    let hemisphere = (impostor.flags & IMPOSTOR_HEMISPHERE) != 0u;
    let frames = impostor_frames(impostor.flags);
    let grid = (impostor_grid(toward, hemisphere) * 0.5 + 0.5) * f32(frames - 1u);
    let cell = min(vec2<u32>(round(grid)), vec2<u32>(frames - 1u));
    let seen_from = impostor_frame_direction(cell.y * frames + cell.x, frames, hemisphere);
    let basis = impostor_basis(seen_from);
    let s0 = dot(o, toward) / rate;
    let crossing = (o + way * s0) / radius;
    let image = vec2<f32>(dot(crossing, basis.right), -dot(crossing, basis.up));
    let slope = vec2<f32>(dot(toward, basis.right), -dot(toward, basis.up));
    let depth = dot(crossing, seen_from);
    let facing = impostor_facing(slope);
    let size = f32(frames * texels);
    let coarsest = f32(countTrailingZeros(texels) - 1u);
    let footprint = ray_footprint(spread, max(s0, nearest)) * rate;
    let level = clamp(log2(max(footprint * f32(texels) / (2.0 * radius), 1.0e-6)), 0.0,
        coarsest);
    let inset = 0.5 * exp2(ceil(level)) / size;
    let span_uv = 1.0 / f32(frames);
    let first = vec2<f32>(cell) * span_uv;
    let crossed = first + (0.5 + 0.5 * image) * span_uv;
    let depths = 2u * placed_frame_layers() + impostor.layer;
    let scale = 1.0 + MASK_MIP_SCALE * level;
    var at = clamp(crossed, first + inset, first + span_uv - inset);
    let probed = placed_frame(at, impostor.layer, level).a;
    if probed * scale < 1.0 / 255.0 {
        return;
    }
    var along = clamp(impostor_met(placed_frame(at, depths, level).w, probed, depth, facing)
        / probed, -1.0, 1.0);
    at = clamp(crossed + slope * (0.5 * span_uv * along), first + inset,
        first + span_uv - inset);
    let covered = placed_frame(at, impostor.layer, level).a;
    if covered * scale < IMPOSTOR_CUTOFF {
        return;
    }
    along = clamp(impostor_met(placed_frame(at, depths, level).w,
        covered, depth, facing) / covered, -1.0, 1.0);
    let s = clamp(s0 - along * radius / rate, span.x, span.y);
    if s >= (*hit).s {
        return;
    }
    (*hit).met = true;
    (*hit).s = s;
    (*hit).ground = false;
    (*hit).terrain = index;
    (*hit).slot = slot;
    (*hit).held = vec3<f32>(at, level);
}

fn walk_block(block: u32, first: vec2<f32>, size: f32, o: vec3<f32>, p: vec3<f32>,
        direction: vec3<f32>, bounds: vec2<f32>, spread: vec2<f32>,
        hit: ptr<function, WorldHit>, index: u32, steps: ptr<function, u32>) {
    let cells = world_placed[block];
    let apron = bitcast<f32>(world_placed[block + 1u]);
    let entries = block + world_placed[block + 2u];
    let width = (size + 2.0 * apron) / f32(cells);
    let start = (o.xz - (first - vec2<f32>(apron))) / width;
    let rate = direction.xz / width;
    let span = box_span(start, rate, vec2<f32>(0.0), vec2<f32>(f32(cells)), bounds.x,
        min(bounds.y, (*hit).s));
    let top = firstTrailingBit(cells);
    let ends = block + 4u + (4u * cells * cells - 1u) / 3u;
    var s = span.x;
    var mip = top;
    loop {
        if s >= min(span.y, (*hit).s) || *steps >= WORLD_STEPS {
            break;
        }
        *steps += 1u;
        let side = cells >> mip;
        let wide = f32(1u << mip);
        let cell = min(vec2<u32>(max(floor((start + rate * s) / wide), vec2<f32>(0.0))),
            vec2<u32>(side - 1u));
        let tops = block + 4u + 4u * (cells * cells - side * side) / 3u;
        let peak = bitcast<f32>(world_placed[tops + cell.y * side + cell.x]);
        let leave = min(square_exit(start, rate, vec2<f32>(cell) * wide, wide), span.y);
        if min(p.y + direction.y * s, p.y + direction.y * leave) > peak {
            s = leave;
            mip = min(mip + 1u, top);
            continue;
        }
        if mip > 0u {
            mip -= 1u;
            continue;
        }
        let at = cell.y * cells + cell.x;
        let listed = select(world_placed[ends + at - 1u], 0u, at == 0u);
        let listed_end = world_placed[ends + at];
        for (var entry = listed; entry < listed_end; entry++) {
            test_placed(world_placed[entries + entry], p, direction, bounds.x, spread, hit,
                index);
        }
        if (*hit).s <= leave {
            return;
        }
        s = leave;
        mip = min(mip + 1u, top);
    }
}

fn line_shift(line: u32) -> f32 {
    return select(select(1.0, -1.0, line == 1u), 0.0, line == 0u);
}

fn trace_placed(index: u32, p: vec3<f32>, direction: vec3<f32>, bounds: vec2<f32>,
        spread: vec2<f32>, hit: ptr<function, WorldHit>) {
    let terrain = world_terrains[index];
    let held = terrain.maps.y;
    if held == 0u {
        return;
    }
    let info = terrain_info(terrain.maps.x);
    let ground = ground_of(info);
    let o = p - info.origin.xyz;
    let apron = terrain.reach.x;
    let span = box_span(o.xz, direction.xz, vec2<f32>(-apron), ground_extent(ground) + apron,
        bounds.x, min(bounds.y, (*hit).s));
    if span.x >= span.y {
        return;
    }
    let flat = length(direction.xz);
    let side = select(vec2<f32>(1.0, 0.0), vec2<f32>(-direction.z, direction.x) / flat,
        flat > 1.0e-6);
    for (var level = 0u; level < info.levels.y; level++) {
        if ((held >> level) & 1u) == 0u {
            continue;
        }
        var steps = 0u;
        let size = f32(TILE_QUADS) * info.origin.w * f32(1u << level);
        let tiles = level_tiles(ground, level);
        let rate = direction.xz / size;
        for (var line = 0u; line < 3u; line++) {
            let start = (o.xz + side * (apron * line_shift(line))) / size;
            var s = span.x;
            loop {
                if s >= min(span.y, (*hit).s) || steps >= WORLD_STEPS {
                    break;
                }
                steps++;
                let tile = floor(start + rate * s);
                let leave = square_exit(start, rate, tile, 1.0);
                var fresh = all(tile >= vec2<f32>(0.0)) && all(tile < vec2<f32>(tiles));
                for (var earlier = 0u; earlier < line; earlier++) {
                    let before = (o.xz + side * (apron * line_shift(earlier))) / size;
                    let entered = box_span(before, rate, tile, tile + 1.0, span.x, span.y);
                    fresh = fresh && entered.x >= entered.y;
                }
                if fresh {
                    let layer = tile_layer(ground, level, vec2<u32>(tile));
                    if layer != NOT_RESIDENT && world_placed[layer] != NO_PLACED {
                        walk_block(world_placed[layer], tile * size, size, o, p, direction,
                            span, spread, hit, index, &steps);
                    }
                }
                s = leave;
            }
        }
    }
}

fn world_trace(p: vec3<f32>, direction: vec3<f32>, nearest: f32, farthest: f32,
        spread: vec2<f32>) -> WorldHit {
    var hit: WorldHit;
    hit.s = farthest;
    let count = u32(piece().trace.w);
    for (var index = 0u; index < count; index++) {
        var below = hit.s;
        if direction.y > 0.0 {
            below = min(below, (world_terrains[index].reach.z - p.y) / direction.y);
        }
        if below > nearest {
            trace_ground(index, p, direction, vec2<f32>(nearest, below), &hit);
            trace_placed(index, p, direction, vec2<f32>(nearest, below), spread, &hit);
        }
    }
    return hit;
}

fn ground_albedo(terrain: TraceTerrain, local: vec2<f32>, normal: vec3<f32>, toward: vec3<f32>,
        footprint: f32, finest: u32) -> vec3<f32> {
    let info = terrain_info(terrain.maps.x);
    let ground = ground_of(info);
    let asked = u32(clamp(floor(log2(max(footprint / ground.spacing, 1.0))), 0.0,
        f32(ground.levels - 1u)));
    var found = ground_tile(ground, local, max(finest, asked));
    if found.tile.layer == NOT_RESIDENT {
        found = ground_tile(ground, local, finest);
    }
    let samples = local / (ground.spacing * f32(1u << found.level));
    let material = tile_material(found.tile, vec2<u32>(round(max(samples, vec2<f32>(0.0)))));
    let share = f32(material.share) / 255.0;
    let colors = ground_colors(ground, info.levels.w, local, ground_colors_mip(ground, footprint));
    let growth = cover_growth(colors.a);
    let sine = max(abs(dot(toward, normal)), 0.03);
    var albedo = vec3<f32>(0.0);
    var vegetated = 0.0;
    for (var slot = 0u; slot < 2u; slot++) {
        let weight = select(share, 1.0 - share, slot == 0u);
        let entry = material_look(info.levels.z,
            select(material.first, material.second, slot == 1u));
        var color = entry.far.rgb;
        if entry.canopy.w > 0.0 {
            let depth = 0.5 * entry.canopy.w * growth.height * growth.density * terrain.reach.y
                / sine;
            if depth > 0.0 {
                let look = mix(color, entry.canopy.rgb, cover_seen(depth).tip);
                color = mix(color, look, 1.0 - exp(-depth));
            }
            vegetated += weight;
        }
        albedo += weight * color;
    }
    return albedo * (macro_color(colors) * cover_tint(colors.a, vegetated));
}

struct PlacedSurface {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    occlusion: f32,
    sky: f32,
}

fn placed_surface(hit: WorldHit) -> PlacedSurface {
    let group = placed_record(hit.slot).group;
    let layers = placed_frame_layers();
    let layer = placed_impostor(group).layer;
    let albedo = placed_frame(hit.held.xy, layer, hit.held.z);
    let geometry = placed_frame(hit.held.xy, 2u * layers + layer, hit.held.z);
    let surface = placed_frame(hit.held.xy, 3u * layers + layer, hit.held.z);
    let unpremultiply = 1.0 / max(albedo.a, 1.0e-4);
    let tint = placed_color(group);
    let color = vec3<f32>(unpack2x16unorm(tint.x), unpack2x16unorm(tint.y).x);
    let mesh_normal = geometry.xyz * unpremultiply * 2.0 - 1.0;
    let unposed = unposed_pose(placed_pose(hit.slot));
    var placed: PlacedSurface;
    placed.albedo = decode_srgb(albedo.rgb) * unpremultiply * color;
    placed.normal = stable_normalize(unposed.x * mesh_normal.x + unposed.y * mesh_normal.y
        + unposed.z * mesh_normal.z);
    placed.occlusion = surface.z * unpremultiply;
    placed.sky = saturate(surface.w * unpremultiply);
    return placed;
}

fn world_light(hit: WorldHit, p: vec3<f32>, direction: vec3<f32>, footprint: f32) -> vec3<f32> {
    let terrain = world_terrains[hit.terrain];
    let world = p + direction * hit.s;
    var normal = hit.normal;
    var albedo: vec3<f32>;
    var ambient: vec3<f32>;
    var bounce: vec3<f32>;
    var unshaded = 1.0;
    if hit.ground {
        let local = world.xz - terrain_info(terrain.maps.x).origin.xz;
        albedo = ground_albedo(terrain, local, normal, direction, footprint, hit.level);
        let sky = ground_sky(terrain.maps.x, local);
        ambient = sky_average(normalize(normal + (sky.gba * 2.0 - 1.0))) * sky.r;
        bounce = bounce_light(world, normal, true).light;
    } else {
        let placed = placed_surface(hit);
        albedo = placed.albedo;
        normal = placed.normal;
        unshaded = placed.sky;
        let around = bounce_light(world, normal, false);
        ambient = mix(sky_average(normal), sky_above(normal) * around.sky, around.weight)
            * placed.occlusion;
        bounce = around.light * (around.weight * placed.occlusion);
    }
    let sky_light = albedo * (ambient * sky_ambient() + bounce);
    let sun = sun_direction();
    let wrapped = max((dot(normal, sun) + 0.25) / 1.25, 0.0);
    var sun_light = vec3<f32>(0.0);
    if wrapped > 0.0 {
        var visible = cloud_passing(world) * unshaded;
        if sun_shadowed() {
            visible *= terrain_visibility(world, hit.ground);
        }
        sun_light = albedo * sun_color() * (wrapped * visible);
    }
    let eye = camera.position.xyz;
    return tonemap(atmospheric(sky_light + sun_light, eye, world, stable_normalize(world - eye)).rgb);
}

fn reach_fade(travelled: f32, reach: f32) -> f32 {
    return saturate((1.0 - min(travelled / max(reach, 1.0e-6), 1.0)) / REFLECTION_END);
}

@vertex
fn vs_trace(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let reflectivity = textureLoad(glow, pixel, 0).g;
    if reflectivity < 0.01 {
        return vec4<f32>(0.0);
    }
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec4<f32>(0.0);
    }
    let p = center.xyz;
    let mirroring = mirroring_at(pixel);
    let to_eye = normalize(post.eye.xyz - p);
    let plane = depth_normal(pixel, p);
    let facing = min(REFLECTION_FACING, max(dot(plane, to_eye), 0.0));
    var normal = mirroring.normal;
    normal = normalize(normal + max(facing - dot(normal, to_eye), 0.0) * to_eye);
    let direction = reflect(-to_eye, normal);
    let reach = piece().trace.x;
    let start = screen_point(p);
    let rows = transpose(post.view_projection);
    let w_rate = dot(rows[3].xyz, direction);
    var shown = reach;
    if w_rate < 0.0 {
        shown = min(reach, 0.9 * start.w / -w_rate);
    }
    let end = screen_point(p + direction * shown);
    if !(start.w > 0.0 && end.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let origin = start.xyz;
    let ray = end.xyz - start.xyz;
    let alphas = mirroring.roughness * mirroring.roughness;
    let walk = screen_walk(origin, ray);
    if walk.met {
        if reflects_itself(pixel, p, plane, walk.point.xy) {
            return vec4<f32>(0.0);
        }
        let reached = world_point(walk.point.xy, walk.point.z);
        let travelled = distance(reached, p);
        let weight = reach_fade(travelled, reach);
        let pixels = travelled * post.eye.w / max(distance(reached, post.eye.xyz), 1.0e-4);
        let lengths = REFLECTION_CONE * pixels
            * vec2<f32>(alphas.x, alphas.y * max(dot(normal, to_eye), 0.0));
        let color = lobe_light(walk.point.xy, ray.xy, lengths, p, plane);
        return vec4<f32>(color * (reflectivity * weight), mirroring.mirrored * weight);
    }
    let clear = origin + ray * walk.clear;
    let nearest = distance(world_point(clear.xy, clear.z), p);
    if !(nearest < reach) {
        return vec4<f32>(0.0);
    }
    let spread = vec2<f32>(distance(p, post.eye.xyz), 2.0 * REFLECTION_CONE * alphas.x);
    let hit = world_trace(p, direction, nearest, reach, spread);
    if !hit.met {
        return vec4<f32>(0.0);
    }
    let weight = reach_fade(hit.s, reach);
    let color = world_light(hit, p, direction, ray_footprint(spread, hit.s));
    return vec4<f32>(color * (reflectivity * weight), mirroring.mirrored * weight);
}
