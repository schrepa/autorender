import package::engine::color::luminance;
import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::view::{camera, detail_resolved, detail_span, eye_world, stable_normalize,
    view_pixel};
import package::face::dither::stored;
import package::face::heightfield::{terrain_uniform, use_terrain};
import package::face::lighting::{environment_brdf, finish, sun_visibility};
import package::face::sun::{sun_color, sun_direction};
import package::generated::targets::{ViewTargets, view_targets};
import package::place::water::WATER_HELD;
import package::water::footprint::WATER_SMOOTHEST;
import package::water::surface::{water_body, water_ground, water_pixel, water_scattered,
    water_scene, water_transmitted};

struct FallInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sheet: vec4<f32>,
    @location(3) state: vec2<f32>,
    @location(4) sky: f32,
    @location(5) terrain: u32,
}

struct FallOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sheet: vec4<f32>,
    @location(3) state: vec2<f32>,
    @location(4) sky: f32,
    @location(6) @interpolate(flat) terrain: u32,
}

struct FallTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sheet: vec4<f32>,
    @location(3) state: vec2<f32>,
    @location(4) sky: f32,
    @location(5) previous: vec3<f32>,
    @location(6) @interpolate(flat) terrain: u32,
}

fn fall_output(input: FallInput) -> FallOutput {
    var output: FallOutput;
    output.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.relative = input.position - camera.position.xyz;
    output.normal = input.normal;
    output.sheet = input.sheet;
    output.state = input.state;
    output.sky = input.sky;
    output.terrain = input.terrain;
    return output;
}

@vertex
fn vs_fall_temporal(input: FallInput) -> FallTemporalOutput {
    let output = fall_output(input);
    var temporal: FallTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.normal = output.normal;
    temporal.sheet = output.sheet;
    temporal.state = output.state;
    temporal.sky = output.sky;
    temporal.terrain = output.terrain;
    temporal.previous = (camera.previous_view_projection * vec4<f32>(input.position, 1.0)).xyw;
    return temporal;
}

fn fall_temporal_output(input: FallTemporalOutput) -> FallOutput {
    var output: FallOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.normal = input.normal;
    output.sheet = input.sheet;
    output.state = input.state;
    output.sky = input.sky;
    output.terrain = input.terrain;
    return output;
}

const FALL_ALBEDO: f32 = 0.8;
const FALL_SCATTERING: f32 = 500.0;
const FALL_FORWARD: f32 = 0.85;

fn fall_water(across: f32, half: f32, spread: f32) -> vec2<f32> {
    let band = max(max(2.0 * spread, fwidth(across)), 1.0e-6); // filter: the water's edge as sharp as a pixel resolves it
    let inside = (half - abs(across)) / band;
    let covered = min(saturate(0.5 + inside), 2.0 * half / band);
    let spray = saturate(0.5 - inside) * min(2.0 * spread / band, 1.0);
    return vec2<f32>(covered, spray);
}

fn fall_white(path: f32, white: f32) -> vec3<f32> {
    let depth = FALL_SCATTERING * white * path;
    let passed = 1.0 / (1.0 + 0.5 * (1.0 - FALL_FORWARD) * depth);
    let unscattered = exp(-depth);
    return vec3<f32>(unscattered, 1.0 - passed, passed - unscattered);
}

fn fall_clear(world: vec3<f32>, thickness: f32) -> f32 {
    let at = water_pixel(world.xz - terrain_uniform().origin.xz,
        detail_resolved(detail_span(world, vec3<f32>(0.0, 1.0, 0.0)).xy));
    let surface = terrain_uniform().origin.y + at.surface;
    let under = select(0.0, 1.0, at.depth > 0.0 && at.held >= WATER_HELD);
    return mix(1.0, smoothstep(thickness, 2.0 * thickness, world.y - surface), under);
}

struct FallShade {
    shaded: Shaded,
    shown: f32,
    coverage: f32,
}

fn shade_fall(input: FallOutput) -> FallShade {
    let world = eye_world(input.relative);
    let thickness = input.sheet.w;
    let clear = fall_clear(world, thickness);
    if clear < 0.01 {
        discard;
    }
    let view = stable_normalize(-input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let normal = normalize(input.normal);
    let facing = dot(normal, view); // optics: the face of the sheet the eye sees
    let seen = select(-normal, normal, facing >= 0.0);
    let squarely = max(abs(facing), 0.05);
    let crossing = thickness / squarely;
    let behind = water_ground(input.clip_position.xy);
    let held = saturate((length(behind) - length(input.relative)) / crossing); // optics: what the view meets behind the sheet
    let path = crossing * held;
    let shares = fall_water(input.sheet.x, input.sheet.y, input.sheet.z);
    let white = 1.0 - (1.0 - input.state.x) * (1.0 - input.state.y) * (1.0 - shares.y);
    let shown = shares.x * held * clear;
    if shown < 0.01 {
        discard;
    }
    let scattering = fall_white(path, white);
    let clarity = water_transmitted(path, 0.0);
    let coverage = shares.x * clear * (1.0 - luminance(clarity) * scattering.x);
    let sun = sun_direction();
    let backlit = dot(sun, seen) < 0.0;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = sun_color() * sun_visibility(world, select(seen, -seen, backlit),
            world_dx, world_dy, length(input.relative), false, 0.0);
    }
    var glow = vec3<f32>(0.0);
    if backlit {
        glow = sunlight * (FALL_ALBEDO * scattering.z * -dot(sun, seen));
    }
    var point: SurfacePoint;
    point.albedo = vec3<f32>(FALL_ALBEDO);
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = input.sky;
    point.bounce = vec3<f32>(0.0);
    point.roughness = WATER_SMOOTHEST;
    point.alpha = point.roughness * point.roughness;
    point.f0 = vec3<f32>(0.02 * (1.0 - white));
    point.diffuse_color = vec3<f32>(FALL_ALBEDO * scattering.y);
    point.geometric = seen;
    point.normal = seen;
    point.distance = length(input.relative);
    point.view = view;
    point.n_v = max(abs(facing), 1.0e-4);
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = false;
    point.sun_shade = 0.0;
    let reflectance = luminance(environment_brdf(point.f0, point.roughness, point.n_v));
    let body = water_body(0.0) * (vec3<f32>(1.0) - clarity)
        * water_scattered(view, sunlight, seen)
        * ((1.0 - reflectance) * scattering.x * input.sky);
    var shade: FallShade;
    shade.shaded = finish(point, body + glow);
    shade.shown = shown;
    shade.coverage = coverage;
    return shade;
}

fn fall_mirroring(shade: FallShade, pixel: vec2<f32>) -> FallShade {
    var result = shade;
    let own = luminance(shade.shaded.color) * shade.shown;
    let behind = luminance(textureLoad(water_scene, vec2<i32>(pixel), 0).rgb)
        * (1.0 - shade.coverage);
    result.shaded.mirrored *= own / max(own + behind, 1.0e-6);
    result.shaded.reflect *= shade.shown;
    return result;
}

@fragment
fn fs_fall(input: FallTemporalOutput) -> ViewTargets {
    use_terrain(input.terrain);
    let pixel = input.clip_position.xy;
    let shade = fall_mirroring(shade_fall(fall_temporal_output(input)), pixel);
    let previous = input.previous;
    var targets = view_targets(shade.shaded,
        view_pixel(input.clip_position, vec4<f32>(previous.xy, 0.0, previous.z)));
    targets.color = vec4<f32>(stored(shade.shaded.color, pixel) * shade.shown, shade.coverage);
    return targets;
}
