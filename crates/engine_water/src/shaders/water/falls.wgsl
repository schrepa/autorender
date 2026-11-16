import package::engine::color::luminance;
import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::slots::{derived, set_added_light, set_metalness, set_occlusion,
    set_roughness, set_transmission, shade as shade_point, standard_point};
import package::engine::surfaces::SURFACE_WATER;
import package::engine::view::{camera, detail_across, eye_world, stable_normalize, view_pixel};
import package::face::ground_tiles::{detail_resolved, detail_span};
import package::face::heightfield::{terrain_uniform, use_terrain};
import package::face::lighting::{environment_brdf, given_reflectance, sun_visibility};
import package::face::sun::{sun_color, sun_direction};
import package::face::sun_blockers::SunAt;
import package::generated::water::{WATER_NOISE_REACH, WATER_NOISE_WRAP};
import package::generated::writes::{Outputs, covered_outputs};
import package::place::water::WATER_HELD;
import package::water::flow::{water_noise, water_shown};
import package::water::surface::{water_body, water_ground, water_pixel, water_scattered,
    water_transmitted};
import package::water::uniforms::water;

const FALL_NOISE_DEVIATION: f32 = 0.43;
const FALL_THINNEST: f32 = 1.0e-4;

const FALL_WANDER_SALT: u32 = 0x3c6ef372u;
const FALL_CURTAIN_SALT: u32 = 0xa54ff53au;
const FALL_ROPE_SALT: u32 = 0x510e527fu;
const FALL_STRAND_SALT: u32 = 0x9b05688cu;

struct FallInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sheet: vec4<f32>,
    @location(3) state: vec2<f32>,
    @location(4) sky: f32,
    @location(5) terrain: u32,
    @location(6) flight: vec4<f32>,
}

struct FallOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) sheet: vec4<f32>,
    @location(3) state: vec2<f32>,
    @location(4) sky: f32,
    @location(6) @interpolate(flat) terrain: u32,
    @location(7) flight: vec4<f32>,
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
    @location(7) flight: vec4<f32>,
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
    output.flight = input.flight;
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
    temporal.flight = output.flight;
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
    output.flight = input.flight;
    return output;
}

fn fall_shows(scale: vec2<f32>, speed: f32, pixel: vec2<f32>) -> f32 {
    return water_shown(1.0, max(pixel.x / scale.x, pixel.y / (speed * scale.y)));
}

fn fall_scale(at: vec2<f32>, scale: vec2<f32>, share: f32, time: f32, salt: u32) -> vec3<f32> {
    let wrap = f32(WATER_NOISE_WRAP);
    let moved = fract(time / (scale.y * wrap)) * wrap;
    let noise = water_noise(vec2<f32>(moved - at.x / scale.y, at.y / scale.x), salt);
    return (share / WATER_NOISE_REACH)
        * vec3<f32>(noise.x, noise.z / scale.x, -noise.y / scale.y);
}

struct FallWater {
    cover: f32,
    thick: f32,
    ridge: f32,
    slope: vec2<f32>,
    unseen: f32,
}

fn fall_water(input: FallOutput, aired: f32, pixel: vec2<f32>, time: f32) -> FallWater {
    let fallen = input.flight.x;
    let speed = max(input.flight.y, 1.0e-3);
    let reach = input.sheet.y + input.sheet.z;
    let whole = water.falls.scale_shares.xyz;
    let shares = whole * vec3<f32>(fall_shows(water.falls.curtain_scale, speed, pixel),
        fall_shows(water.falls.rope_scale, speed, pixel),
        fall_shows(water.falls.strand_scale, speed, pixel));
    var across = input.sheet.x;
    if shares.x > 0.0 {
        across += saturate(1.0 - abs(across) / max(reach, 1.0e-6))
            * fall_scale(vec2<f32>(fallen, across), water.falls.wander_scale,
                water.falls.wander_reach, time, FALL_WANDER_SALT).x;
    }
    let at = vec2<f32>(fallen, across);
    var pattern = vec3<f32>(0.0);
    if shares.x > 0.0 {
        pattern += fall_scale(at, water.falls.curtain_scale, shares.x, time, FALL_CURTAIN_SALT);
    }
    if shares.y > 0.0 {
        pattern += fall_scale(at, water.falls.rope_scale, shares.y, time, FALL_ROPE_SALT);
    }
    if shares.z > 0.0 {
        pattern += fall_scale(at, water.falls.strand_scale, shares.z, time, FALL_STRAND_SALT);
    }
    let gathers = water.falls.gathering * max(aired, water.falls.glassy_gathering);
    let swing = gathers * FALL_NOISE_DEVIATION;
    let unseen = swing * length(whole - shares);
    let edge = saturate((reach - abs(across))
        / max(water.falls.edge_thinning * input.sheet.y + 2.0 * input.sheet.z, 1.0e-6));
    let mean = max(max(input.flight.z, 0.0) / speed, 1.0e-9);
    let film = log(mean / water.falls.tear_thickness);
    let stood = mix(film, min(film, water.falls.broken_stands), saturate(input.state.y));
    let gathered = gathers * pattern.x - 0.5 * swing * swing * dot(shares, shares);
    let level = stood + log(max(edge, 1.0e-6)) + gathered;
    let grade = vec2<f32>(pattern.y, pattern.z / speed);
    let band = water.falls.tear_softness + 2.0 * unseen + gathers * length(grade * pixel); // filter: a tear edge is never sharper than a pixel
    var falling: FallWater;
    falling.cover = smoothstep(-band, band, level);
    falling.thick = mean * exp(max(level, 0.0) - stood);
    falling.ridge = min(exp(gathered), 1.0);
    falling.slope = (0.5 * gathers * falling.thick * falling.cover) * grade;
    falling.unseen = unseen;
    return falling;
}

fn fall_white(path: f32, white: f32) -> vec3<f32> {
    let depth = water.falls.scattering * white * path;
    let passed = 1.0 / (1.0 + 0.5 * (1.0 - water.falls.forward) * depth);
    let unscattered = exp(-depth);
    return vec3<f32>(unscattered, 1.0 - passed, passed - unscattered);
}

fn fall_clear(world: vec3<f32>, thickness: f32) -> f32 {
    let at = water_pixel(world.xz - terrain_uniform().origin.xz,
        detail_resolved(detail_span(world, vec3<f32>(0.0, 1.0, 0.0)).xy));
    let surface = terrain_uniform().origin.y + at.surface;
    let wet = saturate(at.held / WATER_HELD - 1.0);
    let plunged = smoothstep(water.falls.plunge_drowns.x * thickness,
        water.falls.plunge_drowns.y * thickness, world.y - surface);
    return mix(1.0, plunged, wet);
}

struct FallShade {
    point: SurfacePoint,
    shaded: Shaded,
    shown: f32,
    coverage: f32,
}

fn fall_facing(input: FallOutput) -> f32 {
    return dot(normalize(input.normal), stable_normalize(-input.relative)); // optics: the face of the sheet the eye sees
}

fn fall_crossing(input: FallOutput) -> f32 {
    return input.sheet.w / max(abs(fall_facing(input)), 0.05);
}

fn fall_held(input: FallOutput) -> f32 {
    let behind = water_ground(input.clip_position.xy);
    return saturate((length(behind) - length(input.relative)) / fall_crossing(input)); // optics: what the view meets behind the sheet
}

fn shade_fall(input: FallOutput, held: f32) -> FallShade {
    let world = eye_world(input.relative);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let clear = fall_clear(world, input.sheet.w);
    if clear < water.falls.faintest {
        discard;
    }
    let view = stable_normalize(-input.relative);
    let normal = normalize(input.normal);
    let facing = fall_facing(input);
    let seen = select(-normal, normal, facing >= 0.0);
    let turned = max(abs(facing), 0.05);
    let aired = 1.0 - (1.0 - saturate(input.state.x)) * (1.0 - saturate(input.state.y));
    let level = vec3<f32>(-normal.z, 0.0, normal.x);
    let across = level * inverseSqrt(max(dot(level, level), 1.0e-12));
    let lean = 1.0 - facing * facing;
    let sideways = dot(view, across);
    let leaning = vec2<f32>(sideways * sideways, max(lean - sideways * sideways, 0.0));
    let grazing = (1.0 / (turned * turned) - 1.0) / max(lean, 1.0e-6);
    let pixel = detail_across(world) * sqrt(vec2<f32>(1.0) + grazing * leaning); // filter: a pixel reaches further over the sheet only the way the view leans
    let falling = fall_water(input, aired, pixel, camera.position.w);
    let shown = falling.cover * held * clear;
    if shown < water.falls.faintest {
        discard;
    }
    let path = falling.thick / turned * held;
    let scattering = fall_white(path, aired);
    let clarity = water_transmitted(path, 0.0);
    let coverage = falling.cover * clear * (1.0 - luminance(clarity) * scattering.x);
    let steep = falling.slope / water.falls.relief_steepest;
    let tilt = falling.slope * inverseSqrt(1.0 + dot(steep, steep));
    let shaped = normalize(seen - across * tilt.x - cross(normal, across) * tilt.y);
    let sun = sun_direction();
    let backlit = dot(sun, seen) < 0.0;
    var sunlight = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = sun_color() * sun_visibility(SunAt(world, select(seen, -seen, backlit),
            world_dx, world_dy, 0.0, SURFACE_WATER, 0u, vec4<f32>(0.0)));
    }
    var glow = vec3<f32>(0.0);
    if backlit {
        glow = sunlight * (water.falls.albedo * scattering.z * -dot(sun, seen));
    }
    let own = water_body(0.0);
    let hue = own / max(max(own.r, max(own.g, own.b)), 1.0e-4);
    let albedo = water.falls.albedo
        * mix(water.falls.thin_tint * hue, vec3<f32>(1.0), scattering.y);
    let roughness = mix(water.relief.roughness.x, water.falls.matte_roughness,
        max(scattering.y, saturate(falling.unseen)));
    let f0 = vec3<f32>(0.02 * (1.0 - aired));
    var point = standard_point();
    point.base_color = albedo;
    point.emissive = vec3<f32>(0.0);
    point.geometric = seen;
    point.normal = shaped;
    point.distance = length(input.relative);
    point.view = view;
    point.world = world;
    point.world_dx = world_dx;
    point.world_dy = world_dy;
    point.kind = SURFACE_WATER;
    point = set_roughness(point, roughness);
    point = set_metalness(point, 0.0);
    point = set_occlusion(point, input.sky * mix(water.falls.furrow_shade, 1.0, falling.ridge));
    point = set_transmission(point, 0.0);
    point = given_reflectance(derived(point), f0, albedo * scattering.y);
    point.n_v = max(abs(dot(shaped, view)), 1.0e-4);
    let reflectance = luminance(environment_brdf(f0, roughness, point.n_v));
    let body = own * (vec3<f32>(1.0) - clarity)
        * water_scattered(view, sunlight, seen)
        * ((1.0 - reflectance) * scattering.x * input.sky);
    point = set_added_light(point, body + glow);
    var shade: FallShade;
    shade.point = point;
    shade.shaded = shade_point(point, input.clip_position.xy);
    shade.shown = shown;
    shade.coverage = coverage;
    return shade;
}

@fragment
fn fs_fall(input: FallTemporalOutput) -> Outputs {
    use_terrain(input.terrain);
    let fall = fall_temporal_output(input);
    let shade = shade_fall(fall, fall_held(fall));
    var shaded = shade.shaded;
    shaded.opacity = shade.coverage;
    shaded.linear = shade.shaded.linear * (shade.shown / max(shade.coverage, FALL_THINNEST));
    let previous = input.previous;
    return covered_outputs(shade.point, shaded,
        view_pixel(input.clip_position, vec4<f32>(previous.xy, 0.0, previous.z)));
}
