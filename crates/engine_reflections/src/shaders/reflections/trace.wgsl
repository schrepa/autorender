import package::engine::geometry::vertex::octahedral_normal;
import package::engine::post::frame::{depth_texture, drawn_uv, linear, post};
import package::engine::screen_points::{depth_normal, world_at};
import package::engine::transparency::cover;
import package::face::sky::{environment_levels, environment_sample};
import package::face::trace_depths::{TRACE_DEPTH_OPAQUE, TRACE_DEPTH_SOLID, TRACE_DEPTH_THIN};
import package::face::trace_source::{trace_holds, trace_world};
import package::generated::pieces::reflections::piece;

@group(1) @binding(0) var nearest_depth: texture_2d<f32>;
@group(1) @binding(1) var scene_levels: texture_2d<f32>;
@group(1) @binding(2) var surface_target: texture_2d<u32>;
@group(1) @binding(3) var reflected_target: texture_2d<f32>;
@group(1) @binding(4) var far_cells: texture_2d<u32>;
@group(1) @binding(5) var far_light: texture_2d<f32>;
@group(1) @binding(6) var far_reach: texture_2d<f32>;
@group(1) @binding(7) var far_traced: texture_2d<f32>;
@group(1) @binding(8) var blended_mask: texture_2d<f32>;

const REFLECTION_STEPS: u32 = 96u;
const REFLECTION_THIN: f32 = 0.05;
const REFLECTION_MOST_TAPS: u32 = 8u;
const REFLECTION_FACING: f32 = 0.05;
const REFLECTION_END: f32 = 0.1;
const FLOAT_MAX: f32 = 3.4028233e38f;
const FAR_MOST: f32 = 6.0e4;
const FAR_SAME: f32 = 0.02;
const FAR_LEAST: f32 = 1.0e-3;
const FAR_NONE: f32 = -1.0;
const FAR_ALIKE: f32 = -2.0;
const WALK_FORWARD: u32 = 0u;
const WALK_EDGE: u32 = 1u;

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
    kept: f32,
}

fn mirroring_at(pixel: vec2<i32>) -> Mirroring {
    let words = textureLoad(surface_target, pixel, 0);
    var mirroring: Mirroring;
    mirroring.normal = normalize(octahedral_normal(vec2<f32>(words.xy) / 65535.0 * 2.0 - 1.0));
    mirroring.roughness = unpack4x8unorm(words.z).xy;
    mirroring.kept = unpack4x8unorm(words.w >> 8u).x;
    return mirroring;
}

fn body_at(pixel: vec2<i32>) -> u32 {
    return textureLoad(surface_target, pixel, 0).w & 0xffu;
}

fn blurred_scene(at: vec2<f32>, level: f32) -> vec4<f32> {
    return textureSampleLevel(scene_levels, linear, drawn_uv(at, post), level);
}

struct Facets {
    along: vec3<f32>,
    normal: vec3<f32>,
    view: vec2<f32>,
    stretched: vec2<f32>,
    alphas: vec2<f32>,
    masked: f32,
}

struct Tap {
    toward: vec3<f32>,
    weight: f32,
    width: f32,
}

struct Lobe {
    facets: Facets,
    low: Tap,
    high: Tap,
    mean: vec3<f32>,
    across: f32,
    spread: f32,
}

fn smith_lambda(way: vec2<f32>, alpha: f32) -> f32 {
    let slope = alpha * way.x / max(way.y, 1.0e-6);
    return 0.5 * (sqrt(1.0 + slope * slope) - 1.0);
}

fn facets_of(normal: vec3<f32>, to_eye: vec3<f32>, alphas: vec2<f32>) -> Facets {
    let laid = normal * dot(to_eye, normal) - to_eye;
    let squared = dot(laid, laid);
    let side = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.x) > 0.5);
    let any_way = normalize(side - normal * dot(side, normal));
    var facets: Facets;
    facets.along = select(any_way, laid * inverseSqrt(max(squared, 1.0e-30)), squared > 1.0e-12);
    facets.normal = normal;
    facets.view = vec2<f32>(dot(to_eye, facets.along), max(dot(to_eye, normal), 1.0e-4));
    facets.stretched = normalize(vec2<f32>(alphas.x * facets.view.x, facets.view.y));
    facets.alphas = alphas;
    facets.masked = 1.0 + smith_lambda(facets.view, alphas.x);
    return facets;
}

fn tap_offset(tap: u32, count: u32) -> f32 {
    return f32(2u * tap + 1u) / f32(count) - 1.0;
}

fn tap_at(facets: Facets, offset: f32) -> Tap {
    let stretched = facets.stretched;
    let shown = 0.5 * (1.0 + stretched.y);
    let lifted = (1.0 - shown) + shown * offset;
    let level = sqrt(max(1.0 - lifted * lifted, 0.0));
    let facet = vec2<f32>(facets.alphas.x * (lifted * stretched.y + level * stretched.x),
        max(level * stretched.y - lifted * stretched.x, 0.0));
    let scale = inverseSqrt(max(dot(facet, facet), 1.0e-30));
    let turned = dot(facets.view, facet) * scale;
    let way = 2.0 * turned * scale * facet - facets.view;
    let chord = sqrt(max(1.0 - offset * offset, 0.0));
    var tap: Tap;
    tap.toward = facets.along * way.x + facets.normal * way.y;
    tap.weight = 0.0;
    if way.y > 0.0 {
        tap.weight = chord * facets.masked / (facets.masked + smith_lambda(way, facets.alphas.x));
    }
    tap.width = 4.0 * turned * facets.alphas.y * chord * scale;
    return tap;
}

fn lobe_of(normal: vec3<f32>, to_eye: vec3<f32>, roughness: vec2<f32>) -> Lobe {
    var lobe: Lobe;
    lobe.facets = facets_of(normal, to_eye, roughness * roughness);
    lobe.low = tap_at(lobe.facets, tap_offset(0u, 2u));
    lobe.high = tap_at(lobe.facets, tap_offset(1u, 2u));
    let sum = lobe.low.toward * lobe.low.weight + lobe.high.toward * lobe.high.weight;
    lobe.mean = reflect(-to_eye, normal);
    if dot(sum, sum) > 1.0e-12 {
        lobe.mean = normalize(sum);
    }
    lobe.across = sqrt(lobe.low.width * lobe.high.width);
    lobe.spread = lobe_length(distance(lobe.low.toward, lobe.high.toward));
    return lobe;
}

fn lobe_length(between: f32) -> f32 {
    let offset = tap_offset(1u, 2u);
    return 2.0 * sqrt(1.0 - offset * offset) * between;
}

fn roughness_spanning(angle: f32) -> f32 {
    let offset = tap_offset(1u, 2u);
    let level = sqrt(1.0 - offset * offset);
    return sqrt(tan(angle / (8.0 * level)) * level / offset);
}

struct Sight {
    shown: f32,
    end: vec4<f32>,
}

fn sight(p: vec3<f32>, direction: vec3<f32>, nearness: f32, reach: f32) -> Sight {
    let rows = transpose(post.view_projection);
    let w_rate = dot(rows[3].xyz, direction);
    var seen: Sight;
    seen.shown = reach;
    if w_rate < 0.0 {
        seen.shown = min(reach, 0.9 * nearness / -w_rate);
    }
    seen.end = screen_point(p + direction * seen.shown);
    return seen;
}

fn sight_travelled(seen: Sight, nearness: f32, part: f32) -> f32 {
    return seen.shown * part * nearness
        / max(seen.end.w * (1.0 - part) + nearness * part, 1.0e-30);
}

fn mirror_of(plane: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return select(plane, normal, all(plane == vec3<f32>(0.0)));
}

fn facing_normal(normal: vec3<f32>, plane: vec3<f32>, to_eye: vec3<f32>) -> vec3<f32> {
    let facing = min(REFLECTION_FACING, max(dot(plane, to_eye), 0.0));
    return normalize(normal + max(facing - dot(normal, to_eye), 0.0) * to_eye);
}

fn narrowed(leaning: vec3<f32>, width: f32, weight: f32, wide: f32, count: u32) -> f32 {
    if !(wide > 0.0 && weight > 0.0) {
        return 1.0;
    }
    let lean = leaning / weight;
    return clamp(sqrt(max(width / weight - dot(lean, lean), 0.0) / wide), 1.0 / f32(count), 1.0);
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

fn plane_depths(p: vec3<f32>, plane: vec3<f32>) -> vec4<f32> {
    let shown = vec4<f32>(plane, -dot(plane, p)) * post.inverse_view_projection;
    if shown.z == 0.0 {
        return vec4<f32>(0.0, 0.0, -1.0, 0.0);
    }
    let slope = vec2<f32>(-2.0 * shown.x * post.size.z, 2.0 * shown.y * post.size.w) / shown.z;
    return vec4<f32>(slope, (shown.x - shown.y - shown.w) / shown.z, abs(slope.x) + abs(slope.y));
}

struct ScreenWalk {
    met: bool,
    point: vec3<f32>,
    clear: f32,
    solid: f32,
    rung: i32,
    turns: u32,
}

fn screen_walk(origin: vec3<f32>, ray: vec3<f32>, mode: u32, rung: i32, budget: u32,
        shelf: vec4<f32>) -> ScreenWalk {
    var walk: ScreenWalk;
    let moving = abs(ray.xy) > vec2<f32>(1.0e-9);
    let forward = ray.xy >= vec2<f32>(0.0);
    let rate = select(vec2<f32>(1.0), ray.xy, moving);
    let ahead = select(vec2<f32>(0.0), vec2<f32>(1.0), forward);
    let past = select(vec2<f32>(-0.005), vec2<f32>(0.005), forward);
    let rising = select(vec2<f32>(0.0), vec2<f32>(1.0), shelf.xy > vec2<f32>(0.0));
    let top = i32(textureNumLevels(nearest_depth));
    let along = max(max(abs(ray.x), abs(ray.y)), 1.0e-6);
    let exits = select(vec2<f32>(FLOAT_MAX), (floor(origin.xy) + ahead + past - origin.xy) / rate,
        moving);
    var t = 0.0;
    if mode == WALK_FORWARD {
        t = min(exits.x, exits.y);
    }
    var point = origin + ray * t;
    var mip = rung;
    walk.clear = min(t, 1.0);
    var entered = vec2<f32>(-1.0);
    var behind = 0u;
    var asked = false;
    for (; walk.turns < budget; walk.turns += 1u) {
        if t > 1.0 || any(point.xy < vec2<f32>(0.0)) || any(point.xy >= post.size.xy) {
            break;
        }
        let size = exp2(f32(mip));
        let cell = floor(point.xy / size);
        let surface = nearest_depth_at(vec2<i32>(cell), mip);
        let edges = select(vec2<f32>(FLOAT_MAX), ((cell + ahead) * size + past - origin.xy) / rate,
            moving);
        let leave = min(edges.x, edges.y);
        var level = 0.0;
        if mode == WALK_EDGE {
            level = max(dot(shelf.xy, (cell + rising) * size) + shelf.z + shelf.w, 0.0);
        }
        if surface < point.z {
            if surface > 0.0 && surface <= level {
                break;
            }
            entered = vec2<f32>(-1.0);
            var meet = FLOAT_MAX;
            if ray.z < 0.0 {
                meet = (surface - origin.z) / ray.z;
            }
            if surface > 0.0 {
                walk.solid = leave;
                walk.rung = mip;
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
        if mode == WALK_EDGE {
            walk.met = surface > level;
            walk.point = vec3<f32>(point.xy, surface);
            break;
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
        let gap = distance(ray_point, shown);
        var held = gap <= thickness + spans;
        if !held && body == TRACE_DEPTH_THIN && !asked {
            asked = true;
            held = trace_holds(shown, ray_point);
        }
        if held {
            walk.met = true;
            walk.point = point;
            return walk;
        }
        behind += 1u;
        if (body != TRACE_DEPTH_SOLID && body != TRACE_DEPTH_THIN)
            || behind >= piece().counts.z {
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

fn inside_screen(at: vec2<f32>) -> f32 {
    let size = post.size.xy;
    return min(min(at.x, at.y), min(size.x - at.x, size.y - at.y));
}

fn held_at(at: vec2<f32>, p: vec3<f32>, plane: vec3<f32>, toward: vec3<f32>, angle: f32)
        -> vec3<f32> {
    let last = vec2<i32>(post.size.xy) - 1;
    let seen = world_at(clamp(vec2<i32>(floor(at)), vec2<i32>(0), last));
    if seen.w == 0.0 {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    let offset = seen.xyz - p;
    let grain = distance(seen.xyz, post.eye.xyz) / post.eye.w;
    let along = dot(offset, toward);
    if along <= 0.0 || (any(plane != vec3<f32>(0.0)) && dot(offset, plane) <= grain) {
        return vec3<f32>(-1.0, along, length(offset));
    }
    let aside = length(offset - toward * along);
    return vec3<f32>(saturate(2.0 - aside / max(along * angle, grain)), along, length(offset));
}

fn far_read(lobe: Lobe, p: vec3<f32>, plane: vec3<f32>, reach: f32) -> vec4<f32> {
    let world = p + lobe.mean * reach;
    let spot = screen_point(world);
    if !(spot.w > 0.0) {
        return vec4<f32>(0.0);
    }
    let angle = max(lobe.across, lobe.spread);
    let footprint = max(angle * reach * post.eye.w
        / max(distance(world, post.eye.xyz), 1.0e-4), 1.0);
    let edge = saturate(inside_screen(spot.xy) / footprint + 0.5);
    if edge <= 0.0 {
        return vec4<f32>(0.0);
    }
    let held = held_at(spot.xy, p, plane, lobe.mean, angle);
    let fit = held.x * reach_fade(held.z, reach) * edge;
    if fit <= 0.0 {
        return vec4<f32>(0.0);
    }
    return blurred_scene(spot.xy, log2(footprint)) * fit;
}

struct Gathered {
    sum: vec4<f32>,
    leaning: vec3<f32>,
    leaning_open: vec3<f32>,
    total: f32,
    open: f32,
    width: f32,
    width_open: f32,
    count: u32,
}

struct Beyond {
    light: vec3<f32>,
    cover: f32,
    s: f32,
    known: bool,
}

fn lobe_taps(lobe: Lobe, count: u32) -> array<Tap, REFLECTION_MOST_TAPS> {
    var taps: array<Tap, REFLECTION_MOST_TAPS>;
    for (var index = 0u; index < count; index++) {
        taps[index] = Tap(lobe.mean, 1.0, lobe.across);
        if count > 1u {
            taps[index] = tap_at(lobe.facets, tap_offset(index, count));
        }
    }
    return taps;
}

fn tap_angle(lobe: Lobe, before: Tap, tap: Tap, after: Tap, between: u32) -> f32 {
    if between > 0u {
        return max(tap.width, distance(after.toward, before.toward) / f32(between));
    }
    return max(lobe.across, lobe.spread);
}

struct FarNear {
    valid: u32,
    hits: u32,
    near: f32,
    far: f32,
    dim: f32,
    bright: f32,
}

fn far_near(at: vec2<f32>, shelf: vec4<f32>) -> FarNear {
    var found: FarNear;
    found.near = FLOAT_MAX;
    found.dim = FLOAT_MAX;
    let side = piece().counts.y;
    let scale = f32(side);
    let base = vec2<i32>(floor(at / scale - 0.5));
    let last = (vec2<i32>(post.size.xy) - 1) / i32(side);
    for (var corner = 0u; corner < 4u; corner++) {
        let cell = base + vec2<i32>(vec2<u32>(corner & 1u, corner >> 1u));
        let words = textureLoad(far_cells, clamp(cell, vec2<i32>(0), last), 0);
        let depth = bitcast<f32>(words.w);
        let expected = dot(shelf.xy, (vec2<f32>(cell) + 0.5) * scale) + shelf.z;
        if depth <= 0.0 || abs(depth - expected) > scale * shelf.w + FAR_SAME * depth {
            continue;
        }
        found.valid += 1u;
        let s = bitcast<f32>(words.z);
        if s > 0.0 {
            let light = vec3<f32>(unpack2x16float(words.x), unpack2x16float(words.y).x);
            let tone = light.r + light.g + light.b;
            found.hits += 1u;
            found.near = min(found.near, s);
            found.far = max(found.far, s);
            found.dim = min(found.dim, tone);
            found.bright = max(found.bright, tone);
        }
    }
    return found;
}

fn far_share(lobe: Lobe) -> f32 {
    let refine = piece().trace.y;
    if !(refine > 0.0) {
        return 0.0;
    }
    let widest = max(lobe.across, lobe.spread) * post.eye.w / f32(piece().counts.y);
    return saturate(2.0 - 2.0 * widest / refine);
}

fn far_level(cells: f32, least: f32) -> f32 {
    return min(max(log2(max(cells, 1.0)), least), f32(textureNumLevels(far_reach) - 1u));
}

fn far_uv(at: vec2<f32>) -> vec2<f32> {
    let scale = f32(piece().counts.y);
    let drawn = ceil(post.size.xy / scale);
    return clamp(at / scale, vec2<f32>(0.5), drawn - 0.5)
        / vec2<f32>(textureDimensions(far_reach, 0));
}

fn far_distance(closeness: f32, distant: f32, reach: f32) -> f32 {
    return clamp(1.0 / max(closeness, 1.0e-12) - distant, 0.0, reach);
}

fn far_spread(pixel: vec2<i32>, p: vec3<f32>, mirror: vec3<f32>, lobe: Lobe, reach: f32,
        least: f32) -> Beyond {
    var beyond: Beyond;
    let scale = f32(piece().counts.y);
    let eye = post.eye.xyz;
    let distant = distance(p, eye);
    let widest = max(lobe.across, lobe.spread) * post.eye.w / scale;
    let around = textureSampleLevel(far_reach, linear, far_uv(vec2<f32>(pixel) + 0.5),
        far_level(widest, least)).rg;
    if around.g < FAR_LEAST {
        return beyond;
    }
    let s = far_distance(around.r / around.g, distant, reach);
    let pixels = s * post.eye.w / (distant + s);
    let low = screen_point(p + reflect(lobe.low.toward, mirror) * s);
    let high = screen_point(p + reflect(lobe.high.toward, mirror) * s);
    let wide = lobe.across * pixels;
    var long = lobe.spread * pixels;
    if low.w > 0.0 && high.w > 0.0 {
        long = lobe_length(distance(low.xy, high.xy));
    }
    let middle = screen_point(p + reflect(lobe.mean, mirror) * s);
    if middle.w > 0.0 {
        let whole = far_level(2.0 * max(long, wide) / scale, 1.0);
        let within = far_uv(middle.xy);
        if textureSampleLevel(far_reach, linear, within, whole).g < FAR_LEAST {
            return beyond;
        }
        if textureSampleLevel(far_light, linear, within, whole).a <= 0.0 {
            beyond.known = true;
            return beyond;
        }
    }
    let most = clamp(piece().counts.x, 1u, REFLECTION_MOST_TAPS);
    let count = clamp(u32(ceil(long / max(wide, scale * exp2(least)))), 1u, most);
    var taps = lobe_taps(lobe, count);
    var lit = vec4<f32>(0.0);
    var held = 0.0;
    var known = 0.0;
    var total = 0.0;
    for (var index = 0u; index < count; index++) {
        let tap = taps[index];
        if tap.weight <= 0.0 {
            continue;
        }
        total += tap.weight;
        let world = p + reflect(tap.toward, mirror) * s;
        let spot = screen_point(world);
        if !(spot.w > 0.0 && dot(tap.toward, mirror) > 0.0) {
            continue;
        }
        let before = max(index, 1u) - 1u;
        let after = min(index + 1u, count - 1u);
        let footprint = tap_angle(lobe, taps[before], tap, taps[after], after - before) * s
            * post.eye.w / max(distance(world, eye), 1.0e-4);
        let level = far_level(footprint / scale, least);
        let at = far_uv(spot.xy);
        let reached = textureSampleLevel(far_reach, linear, at, level).rg;
        lit += textureSampleLevel(far_light, linear, at, level) * tap.weight;
        held += reached.r * tap.weight;
        known += reached.g * tap.weight;
    }
    if known < FAR_LEAST * total {
        return beyond;
    }
    beyond.known = true;
    if lit.a > 0.0 {
        beyond.s = far_distance(held / lit.a, distant, reach);
        beyond.light = lit.rgb / lit.a;
        beyond.cover = saturate(lit.a / max(known, 0.5 * total))
            * reach_fade(beyond.s, reach);
    }
    return beyond;
}

fn far_at(pixel: vec2<i32>, p: vec3<f32>, mirror: vec3<f32>, lobe: Lobe, reach: f32) -> Beyond {
    let own = far_share(lobe);
    var least = 1.0 - own;
    var traced: Beyond;
    if own > 0.0 {
        let held = textureLoad(far_traced, pixel, 0);
        traced.known = held.a >= 0.0;
        if held.a > 0.0 {
            traced.light = held.rgb;
            traced.cover = reach_fade(held.a, reach);
            traced.s = held.a;
        }
        if own >= 1.0 && traced.known {
            return traced;
        }
        if held.a == FAR_NONE {
            least = 1.0;
        }
    }
    let spread = far_spread(pixel, p, mirror, lobe, reach, least);
    if !traced.known {
        return spread;
    }
    if !spread.known {
        return traced;
    }
    var beyond: Beyond;
    beyond.known = true;
    beyond.cover = mix(spread.cover, traced.cover, own);
    if beyond.cover > 0.0 {
        beyond.light = mix(spread.light * spread.cover, traced.light * traced.cover, own)
            / beyond.cover;
        beyond.s = select(spread.s, traced.s, traced.cover > 0.0);
    }
    return beyond;
}

fn gather(lobe: Lobe, p: vec3<f32>, plane: vec3<f32>, travelled: f32, center: vec2<f32>,
        met: bool, reach: f32) -> Gathered {
    let eye = post.eye.xyz;
    let low = screen_point(p + lobe.low.toward * travelled);
    let high = screen_point(p + lobe.high.toward * travelled);
    let pixels = travelled * post.eye.w / max(distance(p + lobe.mean * travelled, eye), 1.0e-4);
    let long = lobe_length(distance(low.xy, high.xy));
    let wide = lobe.across * pixels;
    var longer = lobe.spread / max(lobe.across, 1.0e-6);
    if low.w > 0.0 && high.w > 0.0 {
        longer = long / max(wide, 1.0);
    }
    let most = clamp(piece().counts.x, 1u, REFLECTION_MOST_TAPS);
    let count = clamp(u32(ceil(longer)), 1u, most);
    var taps = lobe_taps(lobe, count);
    let no_plane = all(plane == vec3<f32>(0.0));
    let fade = reach_fade(travelled, reach);
    var gathered: Gathered;
    gathered.count = count;
    var central = 0.0;
    for (var index = 0u; index < count; index++) {
        let tap = taps[index];
        if tap.weight <= 0.0 {
            continue;
        }
        let before = max(index, 1u) - 1u;
        let after = min(index + 1u, count - 1u);
        let angle = tap_angle(lobe, taps[before], tap, taps[after], after - before);
        let lean = tap.toward - lobe.mean;
        let spanned = dot(lean, lean);
        var covered = vec4<f32>(0.0);
        var own = 0.0;
        let world = p + tap.toward * travelled;
        let spot = screen_point(world);
        if spot.w > 0.0 && (no_plane || dot(tap.toward, plane) > 0.0) {
            var at = spot.xy;
            if count == 1u && met {
                at = center;
            }
            let footprint = max(angle * travelled * post.eye.w
                / max(distance(world, eye), 1.0e-4), 1.0);
            let share = saturate(inside_screen(at) / footprint + 0.5);
            if share > 0.0 {
                let held = held_at(at, p, plane, tap.toward, angle);
                if held.x >= 0.0 {
                    var jump = 0.0;
                    var deeper = vec2<f32>(0.0);
                    var breadth = 1.0;
                    if held.x < 1.0 && held.y > travelled {
                        let beyond = p + tap.toward * held.y;
                        let spotted = screen_point(beyond);
                        if spotted.w > 0.0 && inside_screen(spotted.xy) > 0.0 {
                            let deep = held_at(spotted.xy, p, plane, tap.toward, angle);
                            if deep.z > 0.0 {
                                deeper = spotted.xy;
                                breadth = max(angle * held.y * post.eye.w
                                    / max(distance(beyond, eye), 1.0e-4), 1.0);
                                jump = (1.0 - held.x) * max(deep.x, 0.0);
                            }
                        }
                    }
                    if held.x > 0.0 {
                        covered = blurred_scene(at, log2(footprint)) * (fade * held.x);
                    }
                    if jump > 0.0 {
                        covered += blurred_scene(deeper, log2(breadth))
                            * (reach_fade(held.y, reach) * jump);
                    }
                    covered *= share * tap.weight;
                    own = share * tap.weight * (held.x + jump);
                }
            }
        }
        var rest = own - covered.a;
        if met {
            central += tap.weight - own;
        } else {
            rest += tap.weight - own;
        }
        gathered.sum += covered;
        gathered.total += tap.weight;
        gathered.open += rest;
        gathered.leaning += lean * tap.weight;
        gathered.leaning_open += lean * rest;
        gathered.width += spanned * tap.weight;
        gathered.width_open += spanned * rest;
    }
    if met && gathered.total <= 0.0 {
        gathered.total = 1.0;
        gathered.count = 1u;
        central = 1.0;
    }
    if central > 0.0 {
        let middle = blurred_scene(center, log2(max(max(wide, long), 1.0))) * (fade * central);
        gathered.sum += middle;
        gathered.open += central - middle.a;
    }
    return gathered;
}

struct Answer {
    light: vec3<f32>,
    cover: f32,
    sky: vec3<f32>,
}

fn answer_of(gathered: Gathered, beyond: vec3<f32>, answered: f32, lobe: Lobe, coarsest: f32,
        strength: f32) -> Answer {
    var answer: Answer;
    var open = 1.0;
    if gathered.total > 0.0 {
        answer.light = gathered.sum.rgb / gathered.total;
        answer.cover = gathered.sum.a / gathered.total;
        open = gathered.open / gathered.total;
    }
    answer.light += beyond * (answered * open);
    answer.cover += answered * open;
    let share = open * (1.0 - answered);
    if gathered.count > 1u && gathered.total > 0.0 && answer.cover > 0.01
        && share * strength >= 0.01 {
        let whole = gathered.leaning / gathered.total;
        let wide = gathered.width / gathered.total - dot(whole, whole);
        if wide > 0.0 {
            let deepest = f32(environment_levels() - 1u);
            let part = narrowed(gathered.leaning_open, gathered.width_open, gathered.open, wide,
                gathered.count);
            let blur = min(roughness_spanning(lobe.spread), coarsest) * deepest;
            let level_open = min(roughness_spanning(lobe.spread * part), coarsest) * deepest;
            let reference = environment_sample(normalize(lobe.mean + whole), blur);
            let found = environment_sample(
                normalize(lobe.mean + gathered.leaning_open / gathered.open), level_open);
            answer.sky = select(share * (found / max(reference, vec3<f32>(1.0e-12)) - 1.0),
                vec3<f32>(0.0), reference <= vec3<f32>(0.0));
        }
    }
    return answer;
}

fn added(pixel: vec2<i32>, stored: vec4<f32>, traced: vec4<f32>) -> vec4<f32> {
    let seen = (1.0 - cover(blended_mask, pixel)) * piece().applied.y;
    return vec4<f32>((traced.rgb - stored.rgb * traced.a) * seen, 0.0);
}

@fragment
fn fs_reflections(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let stored = textureLoad(reflected_target, pixel, 0);
    if stored.a < 0.01 {
        discard;
    }
    let mirroring = mirroring_at(pixel);
    let reflectivity = stored.a * mirroring.kept;
    if reflectivity < 0.01 {
        discard;
    }
    let limit = piece().surface;
    let rough = 1.0 - saturate((1.0 - mirroring.roughness.x / limit.x) / limit.y);
    let center = world_at(pixel);
    if center.w == 0.0 {
        discard;
    }
    let p = center.xyz;
    let eye = post.eye.xyz;
    let to_eye = normalize(eye - p);
    let plane = depth_normal(pixel, p);
    let normal = facing_normal(mirroring.normal, plane, to_eye);
    let lobe = lobe_of(normal, to_eye, mirroring.roughness);
    let reach = piece().trace.x;
    let start = screen_point(p);
    let seen = sight(p, lobe.mean, start.w, reach);
    if !(start.w > 0.0 && seen.end.w > 0.0) {
        discard;
    }
    let origin = start.xyz;
    let ray = seen.end.xyz - origin;
    let run = max(length(ray.xy), 1.0e-6);
    let walk = screen_walk(origin, ray, WALK_FORWARD, 0, REFLECTION_STEPS, vec4<f32>(0.0));
    let coarsest = mirroring.roughness.x;
    let mirror = mirror_of(plane, normal);
    if walk.met {
        if reflects_itself(pixel, p, plane, walk.point.xy) {
            discard;
        }
        let travelled = distance(world_point(walk.point.xy, walk.point.z), p);
        let gathered = gather(lobe, p, plane, travelled, walk.point.xy, true, reach);
        var beyond: Beyond;
        if gathered.open * reflectivity >= 0.01 * gathered.total {
            beyond = far_spread(pixel, p, mirror, lobe, reach, 1.0 - far_share(lobe));
        }
        let found = answer_of(gathered, beyond.light, beyond.cover, lobe, coarsest,
            reflectivity);
        return added(pixel, stored, vec4<f32>(
            found.light * reflectivity + stored.rgb * found.sky * mirroring.kept,
            mirroring.kept * found.cover));
    }
    let sharp = 1.0 - rough;
    var light = vec3<f32>(0.0);
    var cover = 0.0;
    var sky = vec3<f32>(0.0);
    if reflectivity * sharp >= 0.01 {
        var gathered: Gathered;
        var beyond: Beyond;
        var settled = false;
        if walk.clear < 1.0 || seen.shown < reach {
            beyond = far_at(pixel, p, mirror, lobe, reach);
            let left = origin + ray * walk.clear;
            if seen.shown < reach || inside_screen(left.xy) < 1.0 {
                settled = beyond.known && beyond.cover <= 0.0;
                if beyond.cover > 0.0 {
                    let world = p + lobe.mean * beyond.s;
                    let spot = screen_point(world);
                    let wide = max(lobe.across, lobe.spread) * beyond.s * post.eye.w
                        / max(distance(world, eye), 1.0e-4);
                    if spot.w > 0.0 && inside_screen(spot.xy) > -wide {
                        gathered = gather(lobe, p, plane, beyond.s, vec2<f32>(0.0), false,
                            reach);
                    }
                }
            }
        }
        let most = clamp(piece().counts.x, 1u, REFLECTION_MOST_TAPS);
        if !settled && gathered.total <= 0.0
            && (1.0 - beyond.cover) * reflectivity * sharp >= 0.01 && most > 1u
            && walk.turns < REFLECTION_STEPS
            && lobe.spread * post.eye.w > 1.0 && any(plane != vec3<f32>(0.0)) {
            let lowest = tap_at(lobe.facets, tap_offset(most - 1u, most)).toward;
            let above = tap_at(lobe.facets, tap_offset(most - 2u, most)).toward;
            let under = normalize(2.0 * lowest - above);
            let probe = normalize(under - plane * min(dot(under, plane), 0.0));
            let far = sight(p, probe, start.w, reach);
            let edge = far.end.xyz - origin;
            let along = dot(edge.xy, ray.xy) / run;
            let begin = min(min(walk.solid, walk.clear) * run / max(along, 1.0e-6), 1.0);
            if far.end.w > 0.0 && edge.z < 0.0 && along > 1.0e-6 && begin > 0.0 {
                let scout = screen_walk(origin + edge * begin, -edge * begin, WALK_EDGE,
                    walk.rung, REFLECTION_STEPS - walk.turns, plane_depths(p, plane));
                let depth = scout.point.z;
                let part = (depth - origin.z) / edge.z;
                if scout.met && part > 0.0 && part <= 1.0 {
                    let shown = world_point(floor(scout.point.xy) + 0.5, depth);
                    if dot(shown - p, plane) > distance(shown, eye) / post.eye.w {
                        let blocker = world_point(origin.xy + edge.xy * part, depth);
                        gathered = gather(lobe, p, plane, distance(blocker, p),
                            vec2<f32>(0.0), false, reach);
                    }
                }
            }
        }
        let found = answer_of(gathered, beyond.light, beyond.cover, lobe, coarsest,
            reflectivity * sharp);
        light = found.light * sharp;
        cover = found.cover * sharp;
        sky = found.sky * sharp;
    }
    if reflectivity * rough >= 0.01 {
        let found = far_read(lobe, p, plane, reach);
        light += found.rgb * rough;
        cover += found.a * rough;
    }
    return added(pixel, stored, vec4<f32>(light * reflectivity + stored.rgb * sky * mirroring.kept,
        mirroring.kept * cover));
}

@fragment
fn fs_far(@builtin(position) position: vec4<f32>) -> @location(0) vec4<u32> {
    let side = piece().counts.y;
    let last = vec2<i32>(post.size.xy) - 1;
    let block = vec2<i32>(position.xy) * i32(side);
    let looks = min(side, 2u);
    let apart = max(side / 2u, 1u);
    var pixel = min(block, last);
    var shine = 0.0;
    for (var at = 0u; at < looks * looks; at++) {
        let cell = min(block
            + vec2<i32>(vec2<u32>(at % looks, at / looks) * apart + side / 4u), last);
        let stored = textureLoad(reflected_target, cell, 0).a;
        if stored > shine {
            shine = stored;
            pixel = cell;
        }
    }
    if shine < 0.01 {
        return vec4<u32>(0u);
    }
    let mirroring = mirroring_at(pixel);
    let limit = piece().surface;
    let sharp = saturate((1.0 - mirroring.roughness.x / limit.x) / limit.y);
    let center = world_at(pixel);
    if shine * mirroring.kept * sharp < 0.01 || center.w == 0.0 {
        return vec4<u32>(0u);
    }
    let p = center.xyz;
    let eye = post.eye.xyz;
    let to_eye = normalize(eye - p);
    let plane = depth_normal(pixel, p);
    let heading = reflect(-to_eye,
        mirror_of(plane, facing_normal(mirroring.normal, plane, to_eye)));
    let reach = piece().trace.x;
    let start = screen_point(p);
    let seen = sight(p, heading, start.w, reach);
    if !(start.w > 0.0 && seen.end.w > 0.0) {
        return vec4<u32>(0u);
    }
    let walk = screen_walk(start.xyz, seen.end.xyz - start.xyz, WALK_FORWARD, 0,
        REFLECTION_STEPS, vec4<f32>(0.0));
    if walk.met {
        return vec4<u32>(0u);
    }
    var light = vec3<f32>(0.0);
    var s = 0.0;
    var held = 0.0;
    if walk.clear < 1.0 || seen.shown < reach {
        let distant = distance(p, eye);
        let grain = f32(side) / post.eye.w;
        let traced = trace_world(p, heading,
            max(sight_travelled(seen, start.w, walk.clear), grain * distant), reach,
            vec2<f32>(f32(side) * distant, grain));
        if traced.met {
            light = clamp(traced.light, vec3<f32>(0.0), vec3<f32>(FAR_MOST));
            s = traced.s;
            held = 1.0 / (distant + s);
        }
    }
    return vec4<u32>(pack2x16float(light.rg), pack2x16float(vec2<f32>(light.b, held)),
        bitcast<u32>(s), bitcast<u32>(textureLoad(depth_texture, pixel, 0)));
}

@fragment
fn fs_far_own(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let shine = textureLoad(reflected_target, pixel, 0).a;
    if shine < 0.01 || !(piece().trace.y > 0.0) {
        return vec4<f32>(0.0);
    }
    let mirroring = mirroring_at(pixel);
    let limit = piece().surface;
    let sharp = saturate((1.0 - mirroring.roughness.x / limit.x) / limit.y);
    let center = world_at(pixel);
    if shine * mirroring.kept * sharp < 0.01 || center.w == 0.0 {
        return vec4<f32>(0.0);
    }
    let p = center.xyz;
    let to_eye = normalize(post.eye.xyz - p);
    let plane = depth_normal(pixel, p);
    let normal = facing_normal(mirroring.normal, plane, to_eye);
    let lobe = lobe_of(normal, to_eye, mirroring.roughness);
    if far_share(lobe) <= 0.0 {
        return vec4<f32>(0.0);
    }
    let reach = piece().trace.x;
    let distant = distance(p, post.eye.xyz);
    let shelf = plane_depths(p, plane);
    var near = far_near(vec2<f32>(pixel) + 0.5, shelf);
    var guess = reach;
    if near.hits > 0u {
        guess = near.near;
    }
    let spot = screen_point(p + reflect(lobe.mean, mirror_of(plane, normal)) * guess);
    if spot.w > 0.0 {
        let turned = far_near(spot.xy, shelf);
        if turned.valid > 0u {
            near = turned;
        }
    }
    if near.valid == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, FAR_NONE);
    }
    if near.hits == 0u {
        return vec4<f32>(0.0);
    }
    let margin = piece().trace.z * f32(piece().counts.y) * (distant + near.near) / post.eye.w;
    var farthest = reach;
    if near.hits == near.valid {
        if near.far - near.near <= margin
            && near.bright - near.dim <= piece().trace.w * (near.bright + near.dim) {
            return vec4<f32>(0.0, 0.0, 0.0, FAR_ALIKE);
        }
        farthest = min(2.0 * near.far - near.near + margin, reach);
    }
    let traced = trace_world(p, lobe.mean,
        max(2.0 * near.near - near.far - margin, distant / post.eye.w), farthest,
        vec2<f32>(distant, sqrt(lobe.spread * lobe.across)));
    if !traced.met {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(clamp(traced.light, vec3<f32>(0.0), vec3<f32>(FAR_MOST)),
        min(traced.s, FAR_MOST));
}
