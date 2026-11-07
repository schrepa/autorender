import package::engine::mesh::vertex::octahedral_normal;
import package::engine::post::frame::{depth_texture, drawn_uv, linear, post};
import package::engine::screen_points::{depth_normal, world_at};
import package::face::trace_depths::{TRACE_DEPTH_OPAQUE, TRACE_DEPTH_SOLID, TRACE_DEPTH_THIN};
import package::face::trace_source::{trace_holds, trace_world};
import package::generated::pieces::reflections::piece;
import package::generated::targets::{output_reflectivity, output_surface};
import package::reflections::surface::MIRRORED_BITS;

@group(1) @binding(0) var nearest_depth: texture_2d<f32>;
@group(1) @binding(1) var scene_levels: texture_2d<f32>;

const REFLECTION_STEPS: u32 = 96u;
const REFLECTION_THIN: f32 = 0.05;
const REFLECTION_CONE: f32 = 1.19;
const REFLECTION_LOBE_EDGE: f32 = 0.244;
const REFLECTION_TAPS: u32 = 8u;
const REFLECTION_FACING: f32 = 0.05;
const REFLECTION_END: f32 = 0.1;
const REFLECTION_FADE: f32 = 0.2;
const FLOAT_MAX: f32 = 3.4028233e38f;

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
        if body == TRACE_DEPTH_OPAQUE {
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
        if body == TRACE_DEPTH_THIN {
            thickness = REFLECTION_THIN;
        }
        if body == TRACE_DEPTH_SOLID {
            if entered.x < 0.0 {
                entered = point.xy;
            }
            thickness = distance(point.xy, entered) * distance(shown, post.eye.xyz) / post.eye.w;
        }
        if distance(ray_point, shown) <= thickness + spans
            || (body == TRACE_DEPTH_THIN && trace_holds(shown, ray_point)) {
            walk.met = true;
            walk.point = point;
            return walk;
        }
        if body != TRACE_DEPTH_SOLID {
            break;
        }
        t = leave;
        point = origin + ray * t;
    }
    return walk;
}

fn reach_fade(travelled: f32, reach: f32) -> f32 {
    return saturate((1.0 - min(travelled / max(reach, 1.0e-6), 1.0)) / REFLECTION_END);
}

@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let stored = textureLoad(output_reflectivity, pixel, 0).x;
    if stored < 0.01 {
        return vec4<f32>(0.0);
    }
    let mirroring = mirroring_at(pixel);
    let smooth_enough = saturate((1.0 - mirroring.roughness.y / piece().surface.x)
        / REFLECTION_FADE);
    let reflectivity = stored * smooth_enough;
    if reflectivity < 0.01 {
        return vec4<f32>(0.0);
    }
    let mirrored = mirroring.mirrored * smooth_enough;
    let center = world_at(pixel);
    if center.w == 0.0 {
        return vec4<f32>(0.0);
    }
    let p = center.xyz;
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
        return vec4<f32>(color * (reflectivity * weight), mirrored * weight);
    }
    let clear = origin + ray * walk.clear;
    let nearest = distance(world_point(clear.xy, clear.z), p);
    if !(nearest < reach) {
        return vec4<f32>(0.0);
    }
    let spread = vec2<f32>(distance(p, post.eye.xyz), 2.0 * REFLECTION_CONE * alphas.x);
    let hit = trace_world(p, direction, nearest, reach, spread);
    if !hit.met {
        return vec4<f32>(0.0);
    }
    let weight = reach_fade(hit.s, reach);
    return vec4<f32>(hit.light * (reflectivity * weight), mirrored * weight);
}
