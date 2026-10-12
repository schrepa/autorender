// Ground cover in the mesh passes: the blades a compute pass
// grew this frame (`cover_grow.wgsl`), each a strip curving up from its
// root, leaning in the wind, its root the ground's color and its tip its
// cover's.

// One blade: its root (world), height (meters), facing (radians: its flat
// side faces that way), width at the root (meters), and its root color
// (RGB8) and tip color with how far it sways (RGBA8).
struct CoverBlade {
    root: vec3<f32>,
    height: f32,
    facing: f32,
    width: f32,
    root_color: u32,
    tip_color: u32,
}

// The frame's cover: the wind (x and z, meters per second), and the
// presentation time now and at the previous frame (seconds).
struct CoverFrame {
    wind: vec4<f32>,
}

@group(1) @binding(33) var<storage, read> cover_blades: array<CoverBlade>;
@group(1) @binding(34) var<uniform> cover_frame: CoverFrame;

// Segments along a blade; its strip has two vertices at each joint.
const COVER_SEGMENTS: u32 = 4u;

struct CoverOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec3<f32>,
    // From 0 at the root to 1 at the tip.
    @location(4) along: f32,
}

struct CoverTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) world: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec3<f32>,
    @location(4) along: f32,
    @location(5) previous: vec3<f32>,
}

// A point of a blade: where it lies and the blade's normal there.
struct CoverPoint {
    world: vec3<f32>,
    normal: vec3<f32>,
}

// Blade `blade`'s point `along` its length (0 to 1) and `across` its
// width (-1 to 1) at time `time`: a quadratic curve from the root, rising
// and leaning along the wind as gusts travel across the field, with its
// own curl the rest of the way.
fn cover_point(blade: CoverBlade, along: f32, across: f32, time: f32) -> CoverPoint {
    let facing = vec3<f32>(cos(blade.facing), 0.0, sin(blade.facing));
    let side = vec3<f32>(-facing.z, 0.0, facing.x);
    let sway = unpack4x8unorm(blade.tip_color).w;
    let wind = vec3<f32>(cover_frame.wind.x, 0.0, cover_frame.wind.y);
    let speed = length(wind);
    let toward = select(facing, wind / max(speed, 1.0e-4), speed > 1.0e-4);
    // Gusts: waves running with the wind, slower ripples within them.
    let travel = dot(blade.root, toward);
    let gust = 0.55 + 0.3 * sin(time * 1.3 - travel * 0.12)
        + 0.15 * sin(time * 3.7 - travel * 0.6 + blade.facing);
    let lean_wind = sway * saturate(speed * 0.08) * gust;
    // Each blade's own curl, from its facing.
    let curl = 0.25 + 0.2 * fract(blade.facing * 3.1);
    let lean = toward * lean_wind + facing * curl * (1.0 - lean_wind);
    let bend = min(length(lean), 0.95);
    let direction = select(facing, lean / max(length(lean), 1.0e-4), length(lean) > 1.0e-4);
    let height = blade.height;
    // Control points: the root, straight up, and the tip leaning over
    // while keeping the blade's length.
    let tip = blade.root + vec3<f32>(0.0, height * (1.0 - 0.45 * bend * bend), 0.0)
        + direction * height * bend * 0.75;
    let middle = blade.root + vec3<f32>(0.0, height * 0.6, 0.0);
    let t = along;
    let spine = (1.0 - t) * (1.0 - t) * blade.root + 2.0 * t * (1.0 - t) * middle + t * t * tip;
    let tangent = 2.0 * (1.0 - t) * (middle - blade.root) + 2.0 * t * (tip - middle);
    // Tapering to a point.
    let width = blade.width * (1.0 - pow(along, 1.5));
    var point: CoverPoint;
    point.world = spine + side * (0.5 * across * width);
    // The face's normal, turned across the width so the blade reads round.
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

fn cover_output(instance: u32, vertex: u32, time: f32) -> CoverOutput {
    let blade = cover_blades[instance];
    let along = f32(vertex / 2u) / f32(COVER_SEGMENTS);
    let across = f32(vertex & 1u) * 2.0 - 1.0;
    let point = cover_point(blade, along, across, time);
    var output: CoverOutput;
    output.clip_position = camera.view_projection * vec4<f32>(point.world, 1.0);
    output.world = point.world;
    output.normal = point.normal;
    output.color = cover_color(blade, along);
    output.along = along;
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
    let blade = cover_blades[instance];
    let previous = cover_point(blade, output.along, f32(vertex & 1u) * 2.0 - 1.0,
        cover_frame.wind.w);
    var temporal: CoverTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.world = output.world;
    temporal.normal = output.normal;
    temporal.color = output.color;
    temporal.along = output.along;
    temporal.previous = (camera.previous_view_projection * vec4<f32>(previous.world, 1.0)).xyw;
    return temporal;
}

// The surface a blade fragment shows; darker toward the root, where blades
// shade each other. Blades are thin and pass light through: either side
// lights as the side facing the sun does, the far side seen
// through the blade a little dimmer, and their normal leans most of the way
// up, so they light much as the ground they stand on.
fn cover_surface(input: CoverOutput, front: bool) -> SurfacePoint {
    var face = normalize(input.normal);
    if !front {
        face = -face;
    }
    let through = dot(face, lighting.sun.xyz) < 0.0;
    let lit = select(face, -face, through);
    let normal = normalize(mix(lit, vec3<f32>(0.0, 1.0, 0.0), 0.6));
    let relative = input.world - camera.position.xyz;
    let albedo = input.color * select(1.0, 0.8, through);
    var point: SurfacePoint;
    point.albedo = albedo;
    point.opacity = 1.0;
    point.occlusion = mix(0.35, 1.0, input.along);
    point.roughness = 0.7;
    point.alpha = 0.49;
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
    return output;
}

@fragment
fn fs_cover(input: CoverOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(finish(cover_surface(input, front), vec3<f32>(0.0)).color), 1.0);
}

@fragment
fn fs_cover_lit(input: CoverOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let point = cover_surface(input, front);
    return vec4<f32>(encoded(finish(point, local_light(point, input.clip_position.xy)).color),
        1.0);
}

@fragment
fn fs_cover_post(input: CoverOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(finish(cover_surface(input, front), vec3<f32>(0.0)));
}

@fragment
fn fs_cover_post_lit(input: CoverOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    let point = cover_surface(input, front);
    return scene_targets(finish(point, local_light(point, input.clip_position.xy)));
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
