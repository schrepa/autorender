// Ground cover in the mesh passes: the blades and tufts a compute pass
// grew this frame (`cover_grow.wgsl`), each a strip of its leaves curving up
// from its root, leaning in the wind and the way its clump leans them, their
// roots the ground's color and their tips their cover's.

// One blade or tuft: its root (world); its height and its leaves' width at
// their roots (meters, half floats, the height in the low half); its facing
// word (`blade_facing`: its first leaf's flat side faces that way); its
// leaves (`cover_leaf`): how many in steps of COVER_LEAF_STEPS a leaf (the
// low byte), the last past the whole ones standing its share of one of its
// height, and the way its clump leans them along x and z (the next bytes,
// 8-bit fractions of -1 to 1); and its root color (RGB8) and tip color with
// how far it sways (RGBA8).
struct CoverBlade {
    root: vec3<f32>,
    size: u32,
    facing: u32,
    leaves: u32,
    root_color: u32,
    tip_color: u32,
}

// The frame's cover: the wind (x and z, meters per second), and the
// presentation time now and at the previous frame (seconds); whether
// blades keep their ground's normal (x, 1: foliage's ground blend); the
// blades' capacity, whose last slots the straight ones fill backward (x).
struct CoverFrame {
    wind: vec4<f32>,
    ground: vec4<f32>,
    blades: vec4<u32>,
}

// A blade's facing (radians): its word's low 16 bits, a share of a turn,
// below its ground's normal while blades keep it (`cover_grow.wgsl`'s
// `grounded_tilt`).
fn blade_facing(blade: CoverBlade) -> f32 {
    return f32(blade.facing & 0xffffu) / 65535.0 * (2.0 * PI);
}

@group(1) @binding(33) var<storage, read> cover_blades: array<CoverBlade>;
@group(1) @binding(34) var<uniform> cover_frame: CoverFrame;

// Segments along a leaf; its strip has two vertices at each joint.
const COVER_SEGMENTS: u32 = 4u;
// Joints along a leaf too short on screen to show its curve, drawn by the
// blades' second draw (`cover_grow.wgsl`'s `finish_cover`), whose vertices
// count from STRAIGHT_FIRST_VERTEX.
const STRAIGHT_JOINTS: u32 = 3u;
const STRAIGHT_FIRST_VERTEX: u32 = 1024u;

// A leaf of a blade or tuft as `cover_point` draws it: its root (world),
// its height and its width at its root (meters), its turn (radians: its
// flat side faces that way, and it curls that way), how far it curls, its
// clump's lean (x and z) and how far it sways.
struct CoverLeaf {
    root: vec3<f32>,
    height: f32,
    width: f32,
    turn: f32,
    curl: f32,
    lean: vec2<f32>,
    sway: f32,
}

// The golden angle (radians), by which each leaf of a tuft turns from the
// one before, so however many it holds they face every way about evenly.
const COVER_GOLDEN_ANGLE: f32 = 2.3999632;

// Leaf `leaf` of blade `blade`: its first is the blade, curling its own
// way; each further one turns from the one before by the golden angle,
// splays further out, and stands from 0.85 to 1.15 times as tall, rooted
// half its width out from the first's, so the tuft fans out from its root.
// Those past its leaves' count stand none of its height, and the one they
// end in its share of one.
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
    // Each leaf's own curl, by its turn.
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

// A vertex of a strip: its blade or tuft, the leaf it lies on, and how far
// along the leaf (0 at the root, 1 at the tip) and across it (-1 or 1).
struct BladeVertex {
    blade: CoverBlade,
    leaf: CoverLeaf,
    along: f32,
    across: f32,
}

// Vertex `vertex` of blade or tuft `instance` of its draw: whole ones fill
// the list from its front, straight ones from its back. Its strip holds its
// leaves one after another, two vertices at each joint, then each leaf's
// last vertex and the next leaf's first again, so the triangles joining one
// leaf to the next have no area and every leaf's strip starts on an even
// vertex, facing as the first leaf's does.
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
    // From 0 at the root to 1 at the tip.
    @location(4) along: f32,
    // While blades keep their ground's normal: its x and z as 8-bit
    // fractions in the high half, the leaf's height (meters, half float)
    // in the low.
    @location(5) @interpolate(flat) ground: u32,
}

struct CoverTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec3<f32>,
    @location(4) along: f32,
    @location(5) @interpolate(flat) ground: u32,
    @location(6) previous: vec3<f32>,
}

// A point of a leaf: where it lies and the leaf's normal there.
struct CoverPoint {
    world: vec3<f32>,
    normal: vec3<f32>,
}

// Leaf `leaf`'s point `along` its length (0 to 1) and `across` its
// width (-1 to 1) at time `time`: a quadratic curve from the root, rising
// and leaning along the wind as gusts travel across the field, with its
// own curl and its clump's lean the rest of the way.
fn cover_point(leaf: CoverLeaf, along: f32, across: f32, time: f32) -> CoverPoint {
    let facing = vec3<f32>(cos(leaf.turn), 0.0, sin(leaf.turn));
    let side = vec3<f32>(-facing.z, 0.0, facing.x);
    let wind = vec3<f32>(cover_frame.wind.x, 0.0, cover_frame.wind.y);
    let speed = length(wind);
    let toward = select(facing, wind / max(speed, 1.0e-4), speed > 1.0e-4);
    // Gusts: waves running with the wind, slower ripples within them.
    let travel = dot(leaf.root, toward);
    let gust = 0.55 + 0.3 * sin(time * 1.3 - travel * 0.12)
        + 0.15 * sin(time * 3.7 - travel * 0.6 + leaf.turn);
    let lean_wind = leaf.sway * saturate(speed * 0.08) * gust;
    let own = facing * leaf.curl + vec3<f32>(leaf.lean.x, 0.0, leaf.lean.y);
    let lean = toward * lean_wind + own * (1.0 - lean_wind);
    let bend = min(length(lean), 0.95);
    let direction = select(facing, lean / max(length(lean), 1.0e-4), length(lean) > 1.0e-4);
    let height = leaf.height;
    // Control points: the root, straight up, and the tip leaning over
    // while keeping the leaf's length.
    let tip = leaf.root + vec3<f32>(0.0, height * (1.0 - 0.45 * bend * bend), 0.0)
        + direction * height * bend * 0.75;
    let middle = leaf.root + vec3<f32>(0.0, height * 0.6, 0.0);
    let t = along;
    let spine = (1.0 - t) * (1.0 - t) * leaf.root + 2.0 * t * (1.0 - t) * middle + t * t * tip;
    let tangent = 2.0 * (1.0 - t) * (middle - leaf.root) + 2.0 * t * (tip - middle);
    // Tapering to a point.
    let width = leaf.width * (1.0 - pow(along, 1.5));
    var point: CoverPoint;
    point.world = spine + side * (0.5 * across * width);
    // The face's normal, turned across the width so the leaf reads round.
    let face = normalize(cross(side, tangent));
    point.normal = normalize(face + side * (0.45 * across));
    return point;
}

// The color along a blade: the ground's at the root, its cover's at the tip.
fn cover_color(blade: CoverBlade, along: f32) -> vec3<f32> {
    let root = unpack4x8unorm(blade.root_color).rgb;
    let tip = unpack4x8unorm(blade.tip_color).rgb;
    let linear_root = root * root;
    let linear_tip = tip * tip;
    return mix(linear_root, linear_tip, smoothstep(0.0, 0.7, along));
}

// The point of vertex `at` at time `time`: on its leaf, or at the leaf's
// root, every vertex of a leaf standing none of its height alike, so it
// draws nothing.
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
    return cover_output(instance, vertex, cover_frame.wind.z);
}

// Blades move with the wind: their previous positions are last frame's
// sway seen by last frame's camera.
@vertex
fn vs_cover_temporal(@builtin(instance_index) instance: u32,
        @builtin(vertex_index) vertex: u32) -> CoverTemporalOutput {
    let output = cover_output(instance, vertex, cover_frame.wind.z);
    let previous = cover_vertex_point(blade_vertex(instance, vertex), cover_frame.wind.w);
    var temporal: CoverTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.world = output.world;
    temporal.normal = output.normal;
    temporal.color = output.color;
    temporal.along = output.along;
    temporal.ground = output.ground;
    temporal.previous = (camera.previous_view_projection * vec4<f32>(previous.world, 1.0)).xyw;
    return temporal;
}

// The surface a blade fragment shows; darker toward the root, where blades
// shade each other. Blades are thin and pass light through: either side
// lights as the side facing the sun does, the far side seen
// through the blade a little dimmer, and their normal leans most of the way
// up, so they light much as the ground they stand on. The ground's canopy
// shows their average where their tufts retire (`terrain_variety.wgsl`'s
// `cover_seen`).
fn cover_surface(input: CoverOutput, front: bool) -> SurfacePoint {
    var face = normalize(input.normal);
    if !front {
        face = -face;
    }
    let through = dot(face, lighting.sun.xyz) < 0.0;
    let lit = select(face, -face, through);
    var normal = normalize(mix(lit, vec3<f32>(0.0, 1.0, 0.0), 0.6));
    // With foliage's ground blend, bases take the ground's normal, so
    // they light as the ground's canopy around them does.
    if lighting.foliage.w > 0.0 && input.ground != 0u {
        let tilt = vec2<f32>(f32((input.ground >> 16u) & 0xffu), f32(input.ground >> 24u))
            / 255.0 * 2.0 - 1.0;
        let ground = vec3<f32>(tilt.x, sqrt(max(1.0 - dot(tilt, tilt), 0.0)), tilt.y);
        let above = input.along * unpack2x16float(input.ground & 0xffffu).x;
        normal = normalize(mix(normal, ground, 1.0 - smoothstep(0.0, lighting.foliage.w, above)));
    }
    let relative = input.world - camera.position.xyz;
    let albedo = input.color * select(1.0, COVER_FAR_SIDE, through);
    var point: SurfacePoint;
    point.albedo = albedo;
    // Blades are leaf throughout, and sheets.
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

fn cover_temporal_output(input: CoverTemporalOutput) -> CoverOutput {
    var output: CoverOutput;
    output.clip_position = input.clip_position;
    output.world = input.world;
    output.normal = input.normal;
    output.color = input.color;
    output.along = input.along;
    output.ground = input.ground;
    return output;
}

@fragment
fn fs_cover(input: CoverOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(finish(cover_surface(input, front), vec3<f32>(0.0)).color,
        input.clip_position.xy), 1.0);
}

@fragment
fn fs_cover_lit(input: CoverOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let point = cover_surface(input, front);
    return vec4<f32>(encoded(finish(point, local_light(point, input.clip_position.xy)).color,
        input.clip_position.xy), 1.0);
}

@fragment
fn fs_cover_post(input: CoverOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(finish(cover_surface(input, front), vec3<f32>(0.0)),
        input.clip_position.xy);
}

@fragment
fn fs_cover_post_lit(input: CoverOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    let point = cover_surface(input, front);
    return scene_targets(finish(point, local_light(point, input.clip_position.xy)),
        input.clip_position.xy);
}

@fragment
fn fs_cover_temporal(input: CoverTemporalOutput, @builtin(front_facing) front: bool)
        -> TemporalTargets {
    return temporal_targets(finish(cover_surface(cover_temporal_output(input), front),
        vec3<f32>(0.0)), input.clip_position, input.previous);
}

@fragment
fn fs_cover_temporal_lit(input: CoverTemporalOutput, @builtin(front_facing) front: bool)
        -> TemporalTargets {
    let point = cover_surface(cover_temporal_output(input), front);
    return temporal_targets(finish(point, local_light(point, input.clip_position.xy)),
        input.clip_position, input.previous);
}
