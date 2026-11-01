import package::cover::faces::cover::{COVER_FAR_SIDE, COVER_ROUGHNESS};
import package::cover::laws::{COVER_LEAF_STEPS, COVER_ROOT_OCCLUSION, STRAIGHT_FIRST_VERTEX,
    STRAIGHT_JOINTS};
import package::engine::shading::{BODY_THIN, SurfacePoint, ViewPixel};
import package::engine::view::{PI, camera, stable_normalize};
import package::face::dither::stored;
import package::face::lighting::{finish, local_light, widened};
import package::face::foliage::foliage;
import package::face::sun::sun_direction;
import package::generated::cover::{CoverBlade, CoverFrame};
import package::generated::targets::{ViewTargets, view_targets};

fn blade_facing(blade: CoverBlade) -> f32 {
    return f32(blade.facing & 0xffffu) / 65535.0 * (2.0 * PI);
}

@group(1) @binding(0) var<storage, read> cover_blades: array<CoverBlade>;
@group(1) @binding(1) var<uniform> cover_frame: CoverFrame;

const COVER_SEGMENTS: u32 = 4u;

struct CoverLeaf {
    root: vec3<f32>,
    height: f32,
    width: f32,
    turn: f32,
    curl: f32,
    lean: vec2<f32>,
    sway: f32,
}

const COVER_GOLDEN_ANGLE: f32 = 2.3999632;

fn cover_leaf(blade: CoverBlade, leaf: u32) -> CoverLeaf {
    let size = unpack2x16float(blade.size);
    let steps = blade.leaves & 0xffu;
    let whole = steps / COVER_LEAF_STEPS;
    var stands = select(0.0, 1.0, leaf < whole);
    if leaf == whole {
        stands = f32(steps % COVER_LEAF_STEPS) / f32(COVER_LEAF_STEPS);
    }
    var made: CoverLeaf;
    made.turn = blade_facing(blade) + f32(leaf) * COVER_GOLDEN_ANGLE;
    made.root = blade.root;
    made.height = size.x * stands;
    made.width = size.y;
    made.curl = 0.25 + 0.2 * fract(made.turn * 3.1);
    if leaf > 0u {
        made.root += vec3<f32>(cos(made.turn), 0.0, sin(made.turn)) * (0.5 * size.y);
        made.height *= 0.85 + 0.3 * fract(made.turn * 5.3);
        made.curl += 0.2;
    }
    made.lean = unpack4x8unorm(blade.leaves).yz * 2.0 - 1.0;
    made.sway = unpack4x8unorm(blade.tip_color).w;
    return made;
}

struct BladeVertex {
    blade: CoverBlade,
    leaf: CoverLeaf,
    along: f32,
    across: f32,
}

fn blade_vertex(instance: u32, vertex: u32) -> BladeVertex {
    var at: BladeVertex;
    var joints = COVER_SEGMENTS + 1u;
    var index = vertex;
    if vertex >= STRAIGHT_FIRST_VERTEX {
        joints = STRAIGHT_JOINTS;
        index = vertex - STRAIGHT_FIRST_VERTEX;
        at.blade = cover_blades[cover_frame.blades.x - 1u - instance];
    } else {
        at.blade = cover_blades[instance];
    }
    let per_leaf = 2u * joints + 2u;
    let place = index % per_leaf;
    let next = place > 2u * joints;
    let within = select(min(place, 2u * joints - 1u), 0u, next);
    at.leaf = cover_leaf(at.blade, index / per_leaf + u32(next));
    at.along = f32(within / 2u) / f32(joints - 1u);
    at.across = f32(within & 1u) * 2.0 - 1.0;
    return at;
}

struct CoverOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec3<f32>,
    @location(4) along: f32,
    @location(5) @interpolate(flat) ground: u32,
    @location(6) previous: vec4<f32>,
}

struct CoverPoint {
    world: vec3<f32>,
    normal: vec3<f32>,
}

fn cover_point(leaf: CoverLeaf, along: f32, across: f32, time: f32) -> CoverPoint {
    let facing = vec3<f32>(cos(leaf.turn), 0.0, sin(leaf.turn));
    let side = vec3<f32>(-facing.z, 0.0, facing.x);
    let wind = vec3<f32>(cover_frame.wind.x, 0.0, cover_frame.wind.y);
    let speed = length(wind);
    let toward = select(facing, wind / max(speed, 1.0e-4), speed > 1.0e-4);
    let travel = dot(leaf.root, toward);
    let gust = 0.55 + 0.3 * sin(time * 1.3 - travel * 0.12)
        + 0.15 * sin(time * 3.7 - travel * 0.6 + leaf.turn);
    let lean_wind = leaf.sway * saturate(speed * 0.08) * gust;
    let own = facing * leaf.curl + vec3<f32>(leaf.lean.x, 0.0, leaf.lean.y);
    let lean = toward * lean_wind + own * (1.0 - lean_wind);
    let bend = min(length(lean), 0.95);
    let direction = select(facing, lean / max(length(lean), 1.0e-4), length(lean) > 1.0e-4);
    let height = leaf.height;
    let tip = leaf.root + vec3<f32>(0.0, height * (1.0 - 0.45 * bend * bend), 0.0)
        + direction * height * bend * 0.75;
    let middle = leaf.root + vec3<f32>(0.0, height * 0.6, 0.0);
    let t = along;
    let spine = (1.0 - t) * (1.0 - t) * leaf.root + 2.0 * t * (1.0 - t) * middle + t * t * tip;
    let tangent = 2.0 * (1.0 - t) * (middle - leaf.root) + 2.0 * t * (tip - middle);
    let width = leaf.width * (1.0 - pow(along, 1.5));
    var point: CoverPoint;
    point.world = spine + side * (0.5 * across * width);
    let face = normalize(cross(side, tangent));
    point.normal = normalize(face + side * (0.45 * across));
    return point;
}

fn cover_color(blade: CoverBlade, along: f32) -> vec3<f32> {
    let root = unpack4x8unorm(blade.root_color).rgb;
    let tip = unpack4x8unorm(blade.tip_color).rgb;
    let linear_root = root * root;
    let linear_tip = tip * tip;
    return mix(linear_root, linear_tip, smoothstep(0.0, 0.7, along));
}

fn cover_vertex_point(at: BladeVertex, time: f32) -> CoverPoint {
    if at.leaf.height <= 0.0 {
        return CoverPoint(at.leaf.root, vec3<f32>(0.0, 1.0, 0.0));
    }
    return cover_point(at.leaf, at.along, at.across, time);
}

fn cover_output(instance: u32, vertex: u32, time: f32) -> CoverOutput {
    let at = blade_vertex(instance, vertex);
    let point = cover_vertex_point(at, time);
    var output: CoverOutput;
    output.clip_position = camera.view_projection * vec4<f32>(point.world, 1.0);
    output.world = point.world;
    output.normal = point.normal;
    output.color = cover_color(at.blade, at.along);
    output.along = at.along;
    if cover_frame.ground.x != 0.0 {
        output.ground = (at.blade.facing & 0xffff0000u)
            | (pack2x16float(vec2<f32>(at.leaf.height, 0.0)) & 0xffffu);
    }
    return output;
}

@vertex
fn vs_cover(@builtin(instance_index) instance: u32, @builtin(vertex_index) vertex: u32)
        -> CoverOutput {
    var output = cover_output(instance, vertex, cover_frame.wind.z);
    let previous = cover_vertex_point(blade_vertex(instance, vertex), cover_frame.wind.w);
    output.previous = camera.previous_view_projection * vec4<f32>(previous.world, 1.0);
    return output;
}

fn cover_surface(input: CoverOutput, front: bool) -> SurfacePoint {
    var face = normalize(input.normal);
    if !front {
        face = -face;
    }
    let through = dot(face, sun_direction()) < 0.0;
    let lit = select(face, -face, through);
    var normal = normalize(mix(lit, vec3<f32>(0.0, 1.0, 0.0), 0.6));
    if foliage().w > 0.0 && input.ground != 0u {
        let tilt = vec2<f32>(f32((input.ground >> 16u) & 0xffu), f32(input.ground >> 24u))
            / 255.0 * 2.0 - 1.0;
        let ground = vec3<f32>(tilt.x, sqrt(max(1.0 - dot(tilt, tilt), 0.0)), tilt.y);
        let above = input.along * unpack2x16float(input.ground & 0xffffu).x;
        normal = normalize(mix(normal, ground, 1.0 - smoothstep(0.0, foliage().w, above)));
    }
    let relative = input.world - camera.position.xyz;
    let albedo = input.color * select(1.0, COVER_FAR_SIDE, through);
    var point: SurfacePoint;
    point.albedo = albedo;
    point.leaf = 1.0;
    point.body = BODY_THIN;
    point.opacity = 1.0;
    point.occlusion = mix(COVER_ROOT_OCCLUSION, 1.0, input.along);
    point.roughness = COVER_ROUGHNESS;
    point.alpha = COVER_ROUGHNESS * COVER_ROUGHNESS;
    point.f0 = vec3<f32>(0.04);
    point.diffuse_color = albedo;
    point.geometric = normal;
    point.normal = normal;
    point.distance = length(relative);
    point.view = stable_normalize(-relative);
    point.n_v = max(dot(normal, point.view), 1.0e-4);
    point.world = input.world;
    point.world_dx = dpdx(input.world);
    point.world_dy = dpdy(input.world);
    return widened(point, dpdx(normal), dpdy(normal));
}

@fragment
fn fs_cover(input: CoverOutput, @builtin(front_facing) front: bool) -> ViewTargets {
    let point = cover_surface(input, front);
    var shaded = finish(point, local_light(point, input.clip_position.xy));
    shaded.color = stored(shaded.color, input.clip_position.xy);
    return view_targets(shaded, ViewPixel(input.clip_position, input.previous));
}
