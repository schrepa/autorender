import package::engine::color::luminance;
import package::engine::math::{PI, pcg, unit};
import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::slots::{derived, set_added_light, set_direct_blocked, set_metalness,
    set_mirrored_rough, set_occlusion, set_roughness, set_smoother_across, set_transmission,
    shade as shade_point, standard_point};
import package::engine::surfaces::SURFACE_WATER;
import package::engine::view::{camera, detail_across, detail_level, eye_world, stable_normalize,
    view_pixel};
import package::face::heightfield::{SURFACE_LEAF_INDICES, surface_drawn, surface_leaf,
    surface_leaf_count, surface_vertex, terrain_level_height, terrain_uniform,
    terrain_view_uniform, use_terrain};
import package::face::lighting::{distribution, environment_brdf, fresnel, given_reflectance,
    highlights, specular, sun_visibility, visibility};
import package::face::shadows::air_shadow;
import package::face::sky::{ambient, sky_average};
import package::face::sun::{sun_color, sun_direction};
import package::face::sun_blockers::SunAt;
import package::generated::writes::{Outputs, outputs};
import package::generated::water::{FROUDE_GUARD_DEPTH, GRAVITY, NO_WATER_BLOCK, WATER_LEAF_FIRST};
import package::place::water::{WATER_HELD, WaterAt, WaterCell};
import package::water::faces::water::{water_cell, water_read};
import package::water::flow::{water_relief, water_stirring, water_stream};
import package::water::foam::{water_blocking, water_contact, water_filming, water_foam,
    water_sheer, water_white, water_whitening};
import package::water::footprint::{water_footprint, water_roughness, water_turning};
import package::water::refraction::refraction_sampler;
import package::water::uniforms::water;
import package::water::waves::{WaterSwell, water_calm, water_piece, water_swell};

@group(1) @binding(57) var<storage, read> water_leaf_spans: array<u32>;
@group(2) @binding(22) var water_depth: texture_depth_2d;
@group(2) @binding(23) var water_scene: texture_2d<f32>;

fn water_ground(pixel: vec2<f32>) -> vec3<f32> {
    let size = camera.region.xy;
    let depth = textureLoad(water_depth, vec2<i32>(pixel), 0);
    let ndc = vec2<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0)
        - camera.jitter.xy;
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w - camera.position.xyz;
}

const WATER_IOR: f32 = 1.333;
const WATER_SUN_TAPS: u32 = 4u;

struct WaterOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) @interpolate(flat) terrain: u32,
    @location(2) depth: f32,
}

fn water_leaf_terrain(instance: u32) -> vec2<u32> {
    let last = max(water.terrains.x, 1u) - 1u;
    for (var terrain = 0u; terrain < last; terrain += 1u) {
        let span = WATER_LEAF_FIRST + 2u * terrain;
        if instance < water_leaf_spans[span] + water_leaf_spans[span + 1u] {
            return vec2<u32>(terrain, instance - water_leaf_spans[span]);
        }
    }
    return vec2<u32>(last, instance - water_leaf_spans[WATER_LEAF_FIRST + 2u * last]);
}

struct WaterStored {
    at: WaterAt,
    found: bool,
}

fn water_stored_read(local: vec2<f32>, cell: WaterCell) -> WaterStored {
    var stored: WaterStored;
    if cell.block != NO_WATER_BLOCK {
        stored.at = water_read(cell);
        stored.found = true;
        return stored;
    }
    let ground = terrain_level_height(local,
        select(0u, cell.level, cell.level < terrain_uniform().directory.y));
    if ground.found {
        stored.at.surface = ground.height;
        stored.found = true;
    }
    return stored;
}

fn water_stored(local: vec2<f32>, seen: f32) -> WaterStored {
    let picked = detail_level(seen, terrain_uniform().anchor.w, terrain_uniform().directory.y);
    let cell = water_cell(local, picked.level);
    let fine = water_stored_read(local, cell);
    if !fine.found || picked.blend <= 0.0 || cell.level != picked.level {
        return fine;
    }
    let coarser = water_stored_read(local, water_cell(local, picked.level + 1u));
    if !coarser.found {
        return fine;
    }
    return WaterStored(water_blend(fine.at, coarser.at, picked.blend), true);
}

fn water_blend(first: WaterAt, second: WaterAt, toward: f32) -> WaterAt {
    var at = first;
    if toward >= 0.5 {
        at = second;
    }
    at.surface = mix(first.surface, second.surface, toward);
    at.depth = mix(first.depth, second.depth, toward);
    at.held = mix(first.held, second.held, toward);
    at.words.flow = mix(first.words.flow, second.words.flow, toward);
    at.words.span = mix(first.words.span, second.words.span, toward);
    at.words.state = mix(first.words.state, second.words.state, toward);
    at.slope = mix(first.slope, second.slope, toward);
    return at;
}

fn water_pixel(local: vec2<f32>, seen: f32) -> WaterAt {
    let stored = water_stored(local, seen);
    var at: WaterAt;
    if stored.found {
        at = stored.at;
    }
    return at;
}

fn water_drawn(drawn: vec3<f32>) -> WaterAt {
    let world = eye_world(drawn);
    return water_pixel(world.xz - terrain_uniform().origin.xz, detail_across(world));
}

fn water_on_surface(drawn: vec3<f32>, at: WaterAt) -> vec3<f32> {
    let above = eye_world(drawn).y - terrain_uniform().origin.y - at.surface;
    return drawn - vec3<f32>(0.0, above, 0.0);
}

fn water_relative(local: vec2<f32>, height: f32) -> vec3<f32> {
    let from_eye = terrain_uniform().origin.xz - terrain_uniform().eye.xz;
    return vec3<f32>(local.x + from_eye.x, height + terrain_uniform().anchor.y,
        local.y + from_eye.y) + terrain_view_uniform().eye_offset.xyz;
}

fn water_output(vertex: u32, instance: u32) -> WaterOutput {
    let leaf = water_leaf_terrain(instance);
    use_terrain(leaf.x);
    let triangle = surface_leaf(leaf.y);
    let local = surface_vertex(triangle, vertex);
    var output: WaterOutput;
    output.terrain = leaf.x;
    let ground = terrain_level_height(local, 0u);
    if !ground.found {
        output.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return output;
    }
    let seen = detail_across(eye_world(water_relative(local, ground.height)));
    let stored = water_stored(local, seen);
    if !stored.found {
        output.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return output;
    }
    let at = stored.at;
    let bed = surface_drawn(triangle, local) + terrain_view_uniform().eye_offset.xyz;
    let still = water_relative(local, at.surface).y;
    var swell: WaterSwell;
    let calm = water_calm(length(at.words.flow));
    let room = min(at.depth, still - bed.y);
    if calm > 0.0 && room > 0.0 {
        let allowed = water.waves.mesh_error * camera.pixel_error * detail_across(eye_world(bed));
        let finest = terrain_uniform().anchor.w
            * f32(1u << min(water_cell(local, 0u).level, 30u));
        swell = water_swell(bed.xz + (camera.position.xz - water.waves.anchor), vec2<f32>(0.0),
            vec2<f32>(0.0), water.waves.shown.x * max(water_piece(allowed), finest), room, calm);
    }
    let relative = vec3<f32>(bed.x, still + swell.height, bed.z);
    output.clip_position = terrain_view_uniform().relative_view_projection
        * vec4<f32>(relative, 1.0);
    output.relative = relative;
    output.depth = relative.y - bed.y;
    return output;
}

@vertex
fn vs_water(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> WaterOutput {
    return water_output(vertex, instance);
}

fn water_previous(relative: vec3<f32>, velocity: vec3<f32>, share: f32) -> vec3<f32> {
    let seconds = clamp(camera.position.w - camera.previous.w, 0.0, WATER_LONGEST_FRAME);
    let before = eye_world(relative) - velocity * (seconds * share);
    return (camera.previous_view_projection * vec4<f32>(before, 1.0)).xyw;
}

const WATER_LONGEST_FRAME: f32 = 0.1;

fn water_extinction(aerated: f32) -> vec3<f32> {
    return water.absorption.rgb + water.sediment.w + water.optics.bubbles * aerated;
}

fn water_transmitted(meters: f32, aerated: f32) -> vec3<f32> {
    return exp(-water_extinction(aerated) * meters);
}

fn water_body(aerated: f32) -> vec3<f32> {
    let own = luminance(water.absorption.rgb);
    let sediment = water.sediment.w;
    let bubbles = water.optics.bubbles * aerated;
    return (water.color.rgb * own + water.sediment.rgb * sediment)
        / max(own + sediment + bubbles, 1.0e-6)
        + vec3<f32>(water.optics.bubble_albedo * bubbles)
        / max(water_extinction(aerated), vec3<f32>(1.0e-6));
}

fn water_facing(normal: vec3<f32>, view: vec3<f32>, plane: vec3<f32>) -> vec3<f32> {
    let mirrored = reflect(-view, normal);
    if dot(view, plane) <= 0.0 || mirrored.y >= 0.0 {
        return normal;
    }
    return normalize(view + vec3<f32>(mirrored.x, -mirrored.y, mirrored.z));
}

fn water_heading(view: vec3<f32>) -> vec2<f32> {
    let away = -view.xz;
    let squared = dot(away, away);
    let laid = away * inverseSqrt(max(squared, 1.0e-30));
    return select(vec2<f32>(1.0, 0.0), laid, squared > 1.0e-12); // optics: the view's frame on water
}

fn water_view_tangent(normal: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let laid = normal * dot(view, normal) - view;
    let squared = dot(laid, laid);
    let side = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.x) > 0.5);
    let tangent = laid * inverseSqrt(max(squared, 1.0e-30));
    let any_tangent = normalize(side - normal * dot(side, normal));
    return select(any_tangent, tangent, squared > 1.0e-12); // optics: the view's frame on water
}

fn water_distribution(along: f32, across: f32, n_h: f32, alphas: vec2<f32>) -> f32 {
    let a2 = alphas.x * alphas.y;
    let v = vec3<f32>(alphas.y * along, alphas.x * across, a2 * n_h);
    let w2 = a2 / dot(v, v);
    return a2 * w2 * w2 / PI;
}

fn water_visibility(n_v: f32, n_l: f32, view_tangents: vec2<f32>, light_tangents: vec2<f32>,
        alphas: vec2<f32>) -> f32 {
    let lambda_v = n_l * length(vec3<f32>(alphas * view_tangents, n_v));
    let lambda_l = n_v * length(vec3<f32>(alphas * light_tangents, n_l));
    return 0.5 / (lambda_v + lambda_l);
}

fn water_phase(cosine: f32) -> f32 {
    let g = water.optics.phase_forward;
    let denominator = 1.0 + g * g - 2.0 * g * cosine;
    return (1.0 - g * g) / (denominator * sqrt(denominator)) / water.optics.phase_mean;
}

fn water_scattered(view: vec3<f32>, sunlight: vec3<f32>, plane: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sky = sky_average(up) * ambient();
    let sun = sun_direction();
    let sun_in = dot(sun, plane);
    if sun_in <= 0.0 || dot(view, plane) <= 0.0 {
        return sky;
    }
    let entering = refract(-sun, plane, 1.0 / WATER_IOR);
    let leaving = -refract(-view, plane, 1.0 / WATER_IOR);
    let into = 1.0 - fresnel(vec3<f32>(0.02), sun_in).r;
    return sky + sunlight * (sun_in * into * water_phase(dot(entering, leaving)));
}

fn water_sunned(world: vec3<f32>, view: vec3<f32>, plane: vec3<f32>, path: vec2<f32>,
        pixel: vec2<f32>) -> f32 {
    let bent = refract(-view, plane, 1.0 / WATER_IOR);
    let sinking = -dot(bent, plane);
    let falling = -dot(refract(-sun_direction(), plane, 1.0 / WATER_IOR), plane);
    let fading = path.y * (1.0 + max(sinking, 0.0) / max(falling, 1.0e-3)); // optics: light dims on its way down to a point and again on the view's way out
    let reached = 1.0 - exp(-fading * path.x);
    let jitter = unit(pcg(u32(pixel.x) ^ pcg(u32(pixel.y))));
    var sum = 0.0;
    for (var tap = 0u; tap < WATER_SUN_TAPS; tap++) {
        let share = (f32(tap) + jitter) / f32(WATER_SUN_TAPS);
        let along = -log(1.0 - share * reached) / max(fading, 1.0e-6); // optics: equal shares of the light scattered back along the view's way through the water
        sum += air_shadow(world + bent * along);
    }
    return sum / f32(WATER_SUN_TAPS);
}

struct WaterBehind {
    color: vec3<f32>,
    ground: vec3<f32>,
}

fn water_behind(relative: vec3<f32>, normal: vec3<f32>, view: vec3<f32>, pixel: vec2<f32>,
        ground: vec3<f32>, plane: vec3<f32>) -> WaterBehind {
    let size = camera.region.xy;
    var behind: WaterBehind;
    behind.ground = ground;
    var at = pixel;
    let bent = refract(-view, normal, 1.0 / WATER_IOR);
    let sinking = -dot(bent, plane);
    if dot(view, plane) > 0.0 && sinking > 0.0 {
        let below = max(dot(relative - ground, plane), 0.0);
        let reach = min(below / max(sinking, 0.1), water.optics.refraction_reach);
        let clip = camera.view_projection
            * vec4<f32>(eye_world(relative + bent * reach), 1.0);
        if clip.w > 1.0e-6 {
            let moved = (vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5) * size;
            if all(moved >= vec2<f32>(0.0)) && all(moved < size)
                && textureLoad(water_depth, vec2<i32>(moved), 0) > 0.0 {
                let there = water_ground(moved);
                if dot(there, there) > dot(relative, relative)
                    && dot(relative - there, plane) > 0.0 {
                    behind.ground = there;
                    at = moved;
                }
            }
        }
    }
    behind.color = textureSampleLevel(water_scene, refraction_sampler,
        min(at, size - 0.5) / vec2<f32>(textureDimensions(water_depth)), 0.0).rgb;
    return behind;
}

struct WaterSurface {
    relative: vec3<f32>,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    normal: vec3<f32>,
    roughness: vec2<f32>,
    plane: vec3<f32>,
    foam_shade: f32,
    foam_thin: f32,
    foam: f32,
    scatter: vec3<f32>,
    transmit: vec3<f32>,
    path: vec2<f32>,
    calm: f32,
}

struct WaterLit {
    point: SurfacePoint,
    shaded: Shaded,
    reflectance: f32,
    sunny: f32,
}

fn water_light(surface: WaterSurface, pixel: vec2<f32>) -> WaterLit {
    let world = eye_world(surface.relative);
    let distance = length(surface.relative);
    let view = stable_normalize(-surface.relative);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let normal = water_facing(surface.normal, view, surface.plane);
    let foam_roughness = mix(water.foam.rough.y, water.foam.rough.x, surface.foam_thin);
    let roughness = mix(surface.roughness, vec2<f32>(foam_roughness), surface.foam);
    let alphas = roughness * roughness;
    let n_v = max(dot(normal, view), 1.0e-4);
    let f0 = vec3<f32>(0.02);
    let reflectance = saturate(luminance(environment_brdf(f0, roughness.x, n_v)));
    let sun = sun_direction();
    var sunlight = vec3<f32>(0.0);
    var sunk = vec3<f32>(0.0);
    if sun.y > 0.0 {
        sunlight = sun_color()
            * sun_visibility(SunAt(world, up, surface.world_dx, surface.world_dy, 0.0,
                SURFACE_WATER, 0u, vec4<f32>(0.0)));
        sunk = sun_color() * water_sunned(world, view, surface.plane, surface.path, pixel);
    }
    var glint = vec3<f32>(0.0);
    let n_l = dot(normal, sun);
    if n_l > 0.0 && any(sunlight > vec3<f32>(0.0)) {
        let half_vector = normalize(view + sun);
        let v_h = max(dot(view, half_vector), 1.0e-4);
        let disc = 0.5 * highlights().y;
        let widened = sqrt(alphas * alphas + vec2<f32>(disc * disc, disc * disc / (v_h * v_h)));
        let along = water_view_tangent(normal, view);
        let across = cross(normal, along);
        let tangents = mat2x3<f32>(along, across);
        glint = fresnel(f0, v_h) * sunlight
            * (water_distribution(dot(along, half_vector), dot(across, half_vector),
                max(dot(normal, half_vector), 0.0), widened)
                * water_visibility(n_v, n_l, view * tangents, sun * tangents, widened) * n_l
                * (specular() * PI));
    }
    let body = surface.scatter * (vec3<f32>(1.0) - surface.transmit)
        * water_scattered(view, sunk, surface.plane) * (1.0 - reflectance);
    let sky = sky_average(up) * ambient();
    let around = sun_color() * (water.foam.around.x * saturate(sun.y))
        + sky * water.foam.around.y;
    let foam_n_l = dot(surface.normal, sun);
    var foam_glint = vec3<f32>(0.0);
    if foam_n_l > 0.0 && any(sunlight > vec3<f32>(0.0)) {
        let foam_alpha = foam_roughness * foam_roughness;
        let half_vector = normalize(view + sun);
        foam_glint = fresnel(f0, max(dot(view, half_vector), 0.0)) * sunlight
            * (distribution(max(dot(surface.normal, half_vector), 0.0), foam_alpha)
                * visibility(max(dot(surface.normal, view), 1.0e-4), foam_n_l, foam_alpha)
                * foam_n_l * (specular() * PI));
    }
    let foam = water.foam.albedo * (1.0 - surface.foam_shade)
        * (mix(around, sky, water.foam.sky) + sunlight * max(foam_n_l, 0.0)) + foam_glint;
    var point = standard_point();
    point.base_color = surface.scatter;
    point.emissive = vec3<f32>(0.0);
    point = set_smoother_across(point, roughness.x - roughness.y);
    point.geometric = up;
    point.normal = normal;
    point.distance = distance;
    point.view = view;
    point.world = world;
    point.world_dx = surface.world_dx;
    point.world_dy = surface.world_dy;
    point.kind = SURFACE_WATER;
    point = set_direct_blocked(point, 1.0);
    point = set_added_light(point, mix(body + glint, foam, surface.foam));
    point = set_roughness(point, roughness.x);
    point = set_mirrored_rough(point, surface.calm);
    point = set_metalness(point, 0.0);
    point = set_occlusion(point, 1.0);
    point = set_transmission(point, 0.0);
    point.mirrored = saturate(1.0 - surface.foam / water.foam.mirrors);
    point = given_reflectance(derived(point), f0, vec3<f32>(0.0));
    var lit: WaterLit;
    lit.point = point;
    lit.shaded = shade_point(point, pixel);
    lit.reflectance = reflectance;
    let sunlit = luminance(sunlight) * max(sun.y, 0.0);
    lit.sunny = sunlit / max(sunlit + luminance(sky), 1.0e-6);
    return lit;
}

struct WaterShade {
    point: SurfacePoint,
    shaded: Shaded,
    coverage: f32,
    behind: vec3<f32>,
    fade: f32,
    velocity: vec3<f32>,
}

fn water_seen(pixel: vec2<f32>) -> vec3<f32> {
    let last = vec2<i32>(textureDimensions(water_scene)) - 1;
    return textureLoad(water_scene, min(vec2<i32>(pixel), last), 0).rgb;
}

fn water_covering(shade: WaterShade, seen: vec3<f32>) -> vec4<f32> {
    return vec4<f32>((shade.shaded.linear + shade.behind) * shade.fade
        + seen * (1.0 - shade.fade), 1.0);
}

fn water_caustics(world: vec3<f32>, normal: vec3<f32>, bed: f32) -> f32 {
    let sun = sun_direction();
    let into = refract(-sun, normal, 1.0 / WATER_IOR);
    let level = refract(-sun, vec3<f32>(0.0, 1.0, 0.0), 1.0 / WATER_IOR);
    let landing = world.xz + into.xz * (bed / max(-into.y, 0.1));
    let level_landing = world.xz + level.xz * (bed / max(-level.y, 0.1));
    let spread = abs(determinant(mat2x2<f32>(dpdx(landing), dpdy(landing))));
    let level_spread = abs(determinant(mat2x2<f32>(dpdx(level_landing), dpdy(level_landing))));
    if level_spread <= 0.0 {
        return 1.0;
    }
    let gathered = level_spread / max(spread, level_spread * 1.0e-3);
    return min(pow(gathered, water.optics.caustic_contrast), water.optics.caustic_most);
}

fn water_edge(meters: f32, change: f32) -> f32 {
    return max(saturate(meters / max(change * water.shore.edge_pixels, 1.0e-6)),
        smoothstep(0.0, water.shore.edge, meters));
}

fn water_ends(held: f32, change: f32) -> f32 {
    return saturate((held - WATER_HELD) / max(change * water.shore.edge_pixels, 1.0e-6));
}

fn water_ramp(meters: f32, width: f32, change: f32) -> f32 {
    let half = min(0.5 * change, max(meters, 0.0));
    if half < 1.0e-3 * width {
        return smoothstep(0.0, width, meters);
    }
    return (water_ramped(meters + half, width) - water_ramped(meters - half, width))
        / (2.0 * half);
}

fn water_ramped(meters: f32, width: f32) -> f32 {
    let u = saturate(meters / width);
    return width * u * u * u * (1.0 - 0.5 * u) + max(meters - width, 0.0);
}

struct WaterFace {
    point: vec3<f32>,
    plane: vec3<f32>,
    standing: f32,
    standing_change: f32,
    depth: f32,
    world_dx: vec3<f32>,
    world_dy: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    roughness: vec2<f32>,
    foam: vec2<f32>,
    pores: f32,
    rate: f32,
    calm: f32,
}

fn water_face(still: vec3<f32>, at: WaterAt, own: vec3<f32>) -> WaterFace {
    let world = eye_world(still);
    if at.held < WATER_HELD {
        discard;
    }
    let plane = normalize(vec3<f32>(at.slope.x, 1.0, at.slope.y));
    let resting = dot(still - own, plane) / plane.y; // optics: what the view meets behind the pixel
    let flow = at.words.flow;
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let speed = length(flow);
    var swell: WaterSwell;
    let calm = water_calm(speed);
    let room = min(at.depth, resting);
    if calm > 0.0 && room > 0.0 {
        swell = water_swell(still.xz + (camera.position.xz - water.waves.anchor), world_dx.xz,
            world_dy.xz, 0.0, room, calm);
    }
    let standing = resting + swell.height;
    if standing <= 0.0 {
        discard;
    }
    let point = still + vec3<f32>(0.0, swell.height, 0.0);
    let standing_change = fwidth(standing);
    let depth = max(at.depth + swell.height, 0.0);
    let turbulence = at.words.state.x;
    let dropping = length(at.slope);
    let stirring = water_stirring(speed, water_eddies(speed, dropping, depth), turbulence);
    let stream = water_stream(world.xz, flow, depth, stirring, turbulence, detail_across(world),
        camera.position.w, world_dx.xz, world_dy.xz);
    let contact = water_contact(standing, standing_change,
        water_footprint(world_dx.xz, world_dy.xz), speed, water_blocking(standing, depth));
    let whitening = water_whitening(stream, at.words.state.y, dropping, contact);
    let relief = water_relief(stream, 1.0 - whitening, swell.slope, swell.hidden);
    let white = water_white(stream, relief.bumps, whitening, water_filming(stream, standing),
        at.words.state.z, water_sheer(standing, dropping), world_dx.xz, world_dy.xz);
    let fall = at.slope + relief.slope + white.slope;
    let normal = normalize(vec3<f32>(fall.x, 1.0, fall.y));
    let view = stable_normalize(-point);
    let roughness = water_roughness(relief.hidden + water_turning(fall), water_heading(view));
    return WaterFace(point, plane, standing, standing_change, depth, world_dx, world_dy, normal,
        view, roughness, white.layers, white.pores, swell.rate, calm);
}

fn shade_water(pixel: vec2<f32>, at: WaterAt, face: WaterFace, behind: WaterBehind,
        layers: vec2<f32>) -> WaterShade {
    let point = face.point;
    let world = eye_world(point);
    let plane = face.plane;
    let standing = face.standing;
    let standing_change = face.standing_change;
    let depth = face.depth;
    let speed = length(at.words.flow);
    let normal = face.normal;
    let view = face.view;
    let bed_depth = min(depth, standing);
    let gathered = water_caustics(world, normal, bed_depth);
    let racing = smoothstep(water.shore.racing.x, water.shore.racing.y, speed);
    let shallows = mix(water_ramp(standing, water.shore.fade, standing_change), 1.0, racing);
    let fade = min(water_edge(standing, standing_change), water_ends(at.held, fwidth(at.held)))
        * shallows;
    let through = min(max(length(behind.ground - point), 0.0), // optics: the view's way through the water
        depth * plane.y / max(dot(view, plane), 0.1)); // optics: the view's way through the water
    let down = clamp(dot(point - behind.ground, plane) / plane.y, 0.0, through); // optics: the light's way down to what the view meets
    let white = max(layers.x, 0.0) + face.foam.x;
    let foam = water_foam(white, layers.y + face.foam.y, face.pores);
    let cover = foam.cover;
    let aerated = 1.0 - exp(-white);
    let transmit = water_transmitted(through + down, aerated);
    var surface: WaterSurface;
    surface.relative = point;
    surface.world_dx = face.world_dx;
    surface.world_dy = face.world_dy;
    surface.normal = normal;
    surface.roughness = face.roughness;
    surface.plane = plane;
    surface.foam_shade = foam.shade;
    surface.foam_thin = foam.thin;
    surface.foam = cover;
    surface.scatter = water_body(aerated);
    surface.transmit = transmit;
    surface.path = vec2<f32>(through, luminance(water_extinction(aerated)));
    surface.calm = face.calm;
    let lit = water_light(surface, pixel);
    let caustic = 1.0 + (gathered - 1.0) * lit.sunny
        * exp(-bed_depth * (1.0 / water.optics.caustic_depth + water.optics.bubbles * aerated));
    let wet = water_ramp(standing, water.shore.soaked, standing_change) * water.shore.wet;
    let let_through = (1.0 - lit.reflectance) * (1.0 - cover);
    var result: WaterShade;
    result.point = lit.point;
    result.shaded = lit.shaded;
    result.coverage = max(max(1.0 - (1.0 - lit.reflectance) * luminance(transmit), wet),
        cover) * fade;
    result.behind = behind.color * transmit * (caustic * let_through * (1.0 - wet));
    result.fade = fade;
    result.velocity = vec3<f32>(at.words.flow.x, face.rate, at.words.flow.y);
    return water_mirroring(result);
}

fn water_mirroring(shade: WaterShade) -> WaterShade {
    var result = shade;
    result.point.opacity = shade.fade;
    return result;
}

fn shade_mesh(mesh: vec3<f32>, pixel: vec2<f32>) -> WaterShade {
    let at = water_drawn(mesh);
    let own = water_ground(pixel);
    let face = water_face(water_on_surface(mesh, at), at, own);
    let behind = water_behind(face.point, face.normal, face.view, pixel, own, face.plane);
    return shade_water(pixel, at, face, behind, vec2<f32>(0.0));
}

@fragment
fn fs_water(input: WaterOutput) -> Outputs {
    if input.depth <= 0.0 {
        discard;
    }
    use_terrain(input.terrain);
    let shade = shade_mesh(input.relative, input.clip_position.xy);
    let previous = water_previous(input.relative, shade.velocity, saturate(shade.coverage));
    var targets = outputs(shade.point, shade.shaded,
        view_pixel(input.clip_position, vec4<f32>(previous.xy, 0.0, previous.z)));
    targets.color = water_covering(shade, water_seen(input.clip_position.xy));
    return targets;
}


fn water_eddies(speed: f32, slope: f32, depth: f32) -> f32 {
    let stirring = GRAVITY * speed * slope * max(depth, FROUDE_GUARD_DEPTH);
    return select(0.0, pow(stirring, 1.0 / 3.0), stirring > 0.0);
}

fn water_eddy_froude(speed: f32, slope: f32, depth: f32) -> f32 {
    return water_eddies(speed, slope, depth)
        / sqrt(GRAVITY * max(depth, FROUDE_GUARD_DEPTH));
}

@group(1) @binding(0) var<storage, read_write> water_leaf_args: array<u32>;

@compute @workgroup_size(1)
fn cs_water_leaf_args() {
    var total = 0u;
    for (var terrain = 0u; terrain < water.terrains.x; terrain++) {
        use_terrain(terrain);
        let count = surface_leaf_count();
        water_leaf_args[WATER_LEAF_FIRST + 2u * terrain] = total;
        water_leaf_args[WATER_LEAF_FIRST + 2u * terrain + 1u] = count;
        total += count;
    }
    water_leaf_args[0] = SURFACE_LEAF_INDICES;
    water_leaf_args[1] = total;
    water_leaf_args[2] = 0u;
    water_leaf_args[3] = 0u;
    water_leaf_args[4] = 0u;
}
