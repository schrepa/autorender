import package::engine::color::luminance;
import package::engine::drawn::drawn_opaque_pyramid;
import package::engine::math::{PI, pcg, unit};
import package::engine::presentation::ROUNDING_MARGIN;
import package::engine::shading::{Shaded, SurfacePoint};
import package::engine::slots::{derived, set_added_light, set_direct_blocked, set_metalness,
    set_mirrored_rough, set_occlusion, set_roughness, set_smoother_across, set_transmission,
    shade as shade_point, standard_point};
import package::engine::surfaces::SURFACE_WATER;
import package::engine::view::{camera, detail_across, detail_level, eye_world, stable_normalize,
    view_pixel};
import package::face::ground_tiles::{GroundTerrain, NOT_RESIDENT, level_tiles, tile_layer};
import package::face::heightfield::{SURFACE_LEAF_INDICES, TERRAIN_GRID, surface_drawn_at,
    surface_leaf, surface_leaf_count, surface_vertex, terrain_level_height, terrain_uniform,
    terrain_view_uniform, use_terrain};
import package::face::lighting::{distribution, environment_brdf, fresnel, given_reflectance,
    highlights, specular, sun_visibility, visibility};
import package::face::shadows::air_shadow;
import package::face::sky::{ambient, sky_average};
import package::face::sun::{sun_color, sun_direction};
import package::face::sun_blockers::SunAt;
import package::generated::writes::{Outputs, outputs};
import package::generated::water::{FROUDE_GUARD_DEPTH, GRAVITY, NO_WATER_BLOCK, WATER_LEAF_DISPATCH,
    WATER_LEAF_FIRST, WATER_LEAF_TOTAL, WATER_LEAF_UNLISTED, WATER_LIST_GROUP, WATER_LIST_GROUPS,
    WATER_LIST_ORDER, WATER_RAW_VERTEX, WATER_SCAN_GROUP, WATER_TILE_PATCHES};
import package::place::water::{WATER_HELD, WaterAt, WaterCell};
import package::water::faces::water::{water_cell, water_read};
import package::water::flow::{water_relief, water_stirring, water_stream};
import package::water::foam::{WaterWhite, water_blocking, water_contact, water_filming,
    water_foam, water_sheer, water_white, water_whitening};
import package::water::footprint::{water_footprint, water_roughness, water_turning};
import package::water::refraction::refraction_sampler;
import package::water::uniforms::water;
import package::water::waves::{WaterSwell, water_calm, water_piece, water_swell};

@group(1) @binding(57) var<storage, read> water_leaf_spans: array<u32>;
@group(1) @binding(59) var<storage, read> water_leaf_order: array<u32>;
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

fn water_leaf_grounded(low: vec2<f32>, high: vec2<f32>) -> bool {
    let terrain = terrain_uniform();
    let ground = GroundTerrain(terrain.directory.x, terrain.directory.y, terrain.anchor.w);
    let level = ground.levels - 1u;
    let tiles = level_tiles(ground, level);
    let span = ground.spacing * f32(1u << level) * f32(WATER_TILE_PATCHES * (TERRAIN_GRID - 1u));
    let slack = ROUNDING_MARGIN * max(max(abs(low.x), abs(low.y)), max(abs(high.x), abs(high.y)));
    let first = min(vec2<u32>(max(low - slack, vec2<f32>(0.0)) / span), tiles - vec2<u32>(1u));
    let last = min(vec2<u32>(max(high + slack, vec2<f32>(0.0)) / span), tiles - vec2<u32>(1u));
    for (var z = first.y; z <= last.y; z++) {
        for (var x = first.x; x <= last.x; x++) {
            if tile_layer(ground, level, vec2<u32>(x, z)) == NOT_RESIDENT {
                return false;
            }
        }
    }
    return true;
}

fn water_leaf_seen(corner_low: vec2<f32>, corner_high: vec2<f32>) -> bool {
    if water.reach.z == 0.0 {
        return true;
    }
    let terrain = terrain_uniform();
    if terrain.directory.y == 0u || !water_leaf_grounded(corner_low, corner_high) {
        return true;
    }
    let placed = abs(terrain.origin.xz) + abs(terrain.eye.xz) + max(abs(corner_low),
        abs(corner_high));
    let moved = vec2<f32>(abs(terrain.relief.x) * (1.0 + ROUNDING_MARGIN)
        + ROUNDING_MARGIN * (placed.x + placed.y));
    let low = water_relative(corner_low - moved, water.reach.x - terrain.origin.y);
    let high = water_relative(corner_high + moved, water.reach.y - terrain.origin.y);
    let clip = terrain_view_uniform().relative_view_projection;
    return !(water_beyond(vec4<f32>(1.0, 0.0, 0.0, 1.0) * clip, low, high)
        || water_beyond(vec4<f32>(-1.0, 0.0, 0.0, 1.0) * clip, low, high)
        || water_beyond(vec4<f32>(0.0, 1.0, 0.0, 1.0) * clip, low, high)
        || water_beyond(vec4<f32>(0.0, -1.0, 0.0, 1.0) * clip, low, high)
        || water_beyond(vec4<f32>(0.0, 0.0, -1.0, 1.0) * clip, low, high)
        || water_beyond(vec4<f32>(0.0, 0.0, 1.0, 0.0) * clip, low, high)
        || water_behind_opaque(clip, low, high));
}

const WATER_LARGEST: f32 = 3.4028233e38;
const WATER_HIDDEN_SCALE: f32 = 0.9999847412109375;
const WATER_NEAREST_W: f32 = 1.0e-6;

fn water_behind_opaque(clip: mat4x4<f32>, low: vec3<f32>, high: vec3<f32>) -> bool {
    var first = vec2<f32>(WATER_LARGEST);
    var last = vec2<f32>(-WATER_LARGEST);
    var nearest = -WATER_LARGEST;
    for (var corner = 0u; corner < 8u; corner++) {
        let at = select(low, high, vec3<bool>((corner & 1u) != 0u, (corner & 2u) != 0u,
            (corner & 4u) != 0u));
        let projected = clip * vec4<f32>(at, 1.0);
        if !(projected.w > WATER_NEAREST_W) {
            return false;
        }
        let ndc = projected.xyz / projected.w;
        first = min(first, ndc.xy);
        last = max(last, ndc.xy);
        nearest = max(nearest, ndc.z);
    }
    if !(all(abs(vec4<f32>(first, last)) < vec4<f32>(WATER_LARGEST)) && nearest < WATER_LARGEST) {
        return false;
    }
    let size = camera.region.xy;
    let corner_low = vec2<f32>(first.x * 0.5 + 0.5, 0.5 - last.y * 0.5) * size;
    let corner_high = vec2<f32>(last.x * 0.5 + 0.5, 0.5 - first.y * 0.5) * size;
    let edge = size - 1.0;
    var a = vec2<u32>(clamp(floor(corner_low) - 1.0, vec2<f32>(0.0), edge)) / 2u;
    var b = vec2<u32>(clamp(floor(corner_high) + 1.0, vec2<f32>(0.0), edge)) / 2u;
    let levels = textureNumLevels(drawn_opaque_pyramid);
    var level = 0u;
    loop {
        if (b.x - a.x <= 1u && b.y - a.y <= 1u) || level + 1u >= levels {
            break;
        }
        a = a / 2u;
        b = b / 2u;
        level += 1u;
    }
    if b.x - a.x > 1u || b.y - a.y > 1u {
        return false;
    }
    let farthest = min(
        min(textureLoad(drawn_opaque_pyramid, a, level).r,
            textureLoad(drawn_opaque_pyramid, vec2<u32>(b.x, a.y), level).r),
        min(textureLoad(drawn_opaque_pyramid, vec2<u32>(a.x, b.y), level).r,
            textureLoad(drawn_opaque_pyramid, b, level).r));
    return nearest < farthest * WATER_HIDDEN_SCALE;
}

fn water_beyond(plane: vec4<f32>, low: vec3<f32>, high: vec3<f32>) -> bool {
    let support = select(low, high, plane.xyz >= vec3<f32>(0.0));
    let size = max(abs(low), abs(high));
    let margin = ROUNDING_MARGIN * (dot(abs(plane.xyz), size) + abs(plane.w));
    return dot(plane.xyz, support) + plane.w < -margin;
}

fn water_output(index: u32, instance: u32) -> WaterOutput {
    let raw = index >= WATER_RAW_VERTEX;
    let vertex = select(index, index - WATER_RAW_VERTEX, raw);
    var listed = instance;
    if !raw {
        listed = water_leaf_order[WATER_LIST_ORDER + instance];
    }
    let leaf = water_leaf_terrain(listed);
    use_terrain(leaf.x);
    let triangle = surface_leaf(leaf.y);
    var output: WaterOutput;
    let local = surface_vertex(triangle, vertex);
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
    let bed = surface_drawn_at(triangle, local, ground.height)
        + terrain_view_uniform().eye_offset.xyz;
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

fn water_scattered(view: vec3<f32>, sunlight: vec3<f32>, plane: vec3<f32>, sky: vec3<f32>)
        -> vec3<f32> {
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
    let sky = sky_average(up) * ambient();
    let body = surface.scatter * (vec3<f32>(1.0) - surface.transmit)
        * water_scattered(view, sunk, surface.plane, sky) * (1.0 - reflectance);
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

struct WaterRise {
    plane: vec3<f32>,
    swell: WaterSwell,
    calm: f32,
    standing: f32,
    point: vec3<f32>,
}

fn water_rise(still: vec3<f32>, at: WaterAt, own: vec3<f32>, world_dx: vec3<f32>,
        world_dy: vec3<f32>) -> WaterRise {
    let plane = normalize(vec3<f32>(at.slope.x, 1.0, at.slope.y));
    let resting = dot(still - own, plane) / plane.y; // optics: what the view meets behind the pixel
    let speed = length(at.words.flow);
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
    return WaterRise(plane, swell, calm, standing, still + vec3<f32>(0.0, swell.height, 0.0));
}

struct WaterFlowing {
    depth: f32,
    hidden: vec3<f32>,
    white: WaterWhite,
    fall: vec2<f32>,
}

fn water_flowing(world: vec3<f32>, at: WaterAt, rise: WaterRise, world_dx: vec3<f32>,
        world_dy: vec3<f32>, standing_change: f32) -> WaterFlowing {
    let flow = at.words.flow;
    let speed = length(flow);
    let swell = rise.swell;
    let standing = rise.standing;
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
    return WaterFlowing(depth, relief.hidden, white, at.slope + relief.slope + white.slope);
}

fn water_face(rise: WaterRise, flowing: WaterFlowing, world_dx: vec3<f32>, world_dy: vec3<f32>,
        standing_change: f32, turning: vec3<f32>) -> WaterFace {
    let fall = flowing.fall;
    let normal = normalize(vec3<f32>(fall.x, 1.0, fall.y));
    let view = stable_normalize(-rise.point);
    let roughness = water_roughness(flowing.hidden + turning, water_heading(view));
    return WaterFace(rise.point, rise.plane, rise.standing, standing_change, flowing.depth,
        world_dx, world_dy, normal, view, roughness, flowing.white.layers, flowing.white.pores,
        rise.swell.rate, rise.calm);
}

fn water_gathered(face: WaterFace) -> f32 {
    return water_caustics(eye_world(face.point), face.normal, min(face.depth, face.standing));
}

fn water_quad_any(here: bool) -> bool {
    let own = select(0.0, 1.0, here);
    let across = dpdxFine(own);
    let down = dpdyFine(own);
    let diagonal = dpdyFine(across);
    return here | (across != 0.0) | (down != 0.0) | (diagonal != 0.0);
}

fn shade_water(pixel: vec2<f32>, at: WaterAt, face: WaterFace, behind: WaterBehind,
        layers: vec2<f32>, gathered: f32, held_change: f32) -> WaterShade {
    let point = face.point;
    let plane = face.plane;
    let standing = face.standing;
    let standing_change = face.standing_change;
    let depth = face.depth;
    let speed = length(at.words.flow);
    let normal = face.normal;
    let view = face.view;
    let bed_depth = min(depth, standing);
    let racing = smoothstep(water.shore.racing.x, water.shore.racing.y, speed);
    let shallows = mix(water_ramp(standing, water.shore.fade, standing_change), 1.0, racing);
    let fade = min(water_edge(standing, standing_change), water_ends(at.held, held_change))
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

@fragment
fn fs_water(input: WaterOutput) -> Outputs {
    let shallow = input.depth <= 0.0;
    if shallow {
        discard;
    }
    let pixel = input.clip_position.xy;
    var at: WaterAt;
    var still = vec3<f32>(0.0);
    if water_quad_any(!shallow) {
        use_terrain(input.terrain);
        at = water_drawn(input.relative);
        still = water_on_surface(input.relative, at);
    }
    let dry = at.held < WATER_HELD;
    if dry {
        discard;
    }
    let live = water_quad_any(!shallow && !dry);
    let world = eye_world(still);
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    var rise: WaterRise;
    if live {
        rise = water_rise(still, at, water_ground(pixel), world_dx, world_dy);
    }
    let standing_change = fwidth(rise.standing);
    let risen = water_quad_any(live && rise.standing > 0.0);
    var flowing: WaterFlowing;
    if risen {
        flowing = water_flowing(world, at, rise, world_dx, world_dy, standing_change);
    }
    let face = water_face(rise, flowing, world_dx, world_dy, standing_change,
        water_turning(flowing.fall));
    let gathered = water_gathered(face);
    let held_change = fwidth(at.held);
    var targets: Outputs;
    if risen {
        let behind = water_behind(face.point, face.normal, face.view, pixel, water_ground(pixel),
            face.plane);
        let shade = shade_water(pixel, at, face, behind, vec2<f32>(0.0), gathered, held_change);
        let previous = water_previous(input.relative, shade.velocity, saturate(shade.coverage));
        targets = outputs(shade.point, shade.shaded,
            view_pixel(input.clip_position, vec4<f32>(previous.xy, 0.0, previous.z)));
        targets.color = water_covering(shade, water_seen(pixel));
    }
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
@group(1) @binding(60) var<storage, read_write> water_leaf_list: array<u32>;

fn water_leaf_draw(at: u32, instances: u32, first: u32) {
    water_leaf_args[at] = SURFACE_LEAF_INDICES;
    water_leaf_args[at + 1u] = instances;
    water_leaf_args[at + 2u] = first;
    water_leaf_args[at + 3u] = 0u;
    water_leaf_args[at + 4u] = 0u;
}

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
    let groups = total / WATER_LIST_GROUP + u32(total % WATER_LIST_GROUP != 0u);
    let listed = total <= water.terrains.y && groups <= WATER_LIST_GROUPS;
    water_leaf_draw(0u, total, SURFACE_LEAF_INDICES);
    water_leaf_draw(WATER_LEAF_UNLISTED, select(total, 0u, listed), SURFACE_LEAF_INDICES);
    water_leaf_args[WATER_LEAF_TOTAL] = total;
    water_leaf_args[WATER_LEAF_DISPATCH] = select(0u, groups, listed);
    water_leaf_args[WATER_LEAF_DISPATCH + 1u] = 1u;
    water_leaf_args[WATER_LEAF_DISPATCH + 2u] = 1u;
}

fn water_leaf_drawn(leaf: u32) -> bool {
    let at = water_leaf_terrain(leaf);
    use_terrain(at.x);
    let triangle = surface_leaf(at.y);
    return water_leaf_seen(min(min(triangle.a, triangle.b), triangle.c),
        max(max(triangle.a, triangle.b), triangle.c));
}

fn water_list_masks() -> u32 {
    return WATER_LIST_ORDER + water.terrains.y;
}

fn water_list_offsets() -> u32 {
    return water_list_masks() + 2u * (water.terrains.y / WATER_LIST_GROUP);
}

var<workgroup> water_marks: array<atomic<u32>, 2>;

@compute @workgroup_size(WATER_LIST_GROUP)
fn cs_water_leaf_mark(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    let leaf = group.x * WATER_LIST_GROUP + lane;
    if leaf < water_leaf_spans[WATER_LEAF_TOTAL] && water_leaf_drawn(leaf) {
        atomicOr(&water_marks[lane / 32u], 1u << (lane % 32u));
    }
    workgroupBarrier();
    if lane < 2u {
        water_leaf_list[water_list_masks() + 2u * group.x + lane] = atomicLoad(&water_marks[lane]);
    }
}

var<workgroup> water_scan_chunks: u32;
var<workgroup> water_sums: array<u32, WATER_SCAN_GROUP>;

fn water_chunk_marked(chunk: u32) -> u32 {
    let masks = water_list_masks() + 2u * chunk;
    return countOneBits(water_leaf_list[masks]) + countOneBits(water_leaf_list[masks + 1u]);
}

fn water_listed_draw(instances: u32) {
    water_leaf_list[0] = SURFACE_LEAF_INDICES;
    water_leaf_list[1] = instances;
    water_leaf_list[2] = 0u;
    water_leaf_list[3] = 0u;
    water_leaf_list[4] = 0u;
}

@compute @workgroup_size(WATER_SCAN_GROUP)
fn cs_water_leaf_scan(@builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        water_scan_chunks = water_leaf_spans[WATER_LEAF_DISPATCH];
    }
    let chunks = workgroupUniformLoad(&water_scan_chunks);
    if chunks == 0u {
        if lane == 0u {
            water_listed_draw(0u);
        }
        return;
    }
    let per = chunks / WATER_SCAN_GROUP + u32(chunks % WATER_SCAN_GROUP != 0u);
    let start = min(lane * per, chunks);
    let end = min(start + per, chunks);
    var sum = 0u;
    for (var chunk = start; chunk < end; chunk++) {
        sum += water_chunk_marked(chunk);
    }
    water_sums[lane] = sum;
    workgroupBarrier();
    for (var step = 1u; step < WATER_SCAN_GROUP; step <<= 1u) {
        let added = select(0u, water_sums[max(lane, step) - step], lane >= step);
        workgroupBarrier();
        water_sums[lane] += added;
        workgroupBarrier();
    }
    let offsets = water_list_offsets();
    var offset = water_sums[lane] - sum;
    for (var chunk = start; chunk < end; chunk++) {
        water_leaf_list[offsets + chunk] = offset;
        offset += water_chunk_marked(chunk);
    }
    if lane == WATER_SCAN_GROUP - 1u {
        water_listed_draw(water_sums[lane]);
    }
}

@compute @workgroup_size(WATER_LIST_GROUP)
fn cs_water_leaf_list(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    let masks = water_list_masks() + 2u * group.x;
    let low = water_leaf_list[masks];
    let high = water_leaf_list[masks + 1u];
    let bit = lane % 32u;
    let below = (1u << bit) - 1u;
    if ((select(low, high, lane >= 32u) >> bit) & 1u) == 0u {
        return;
    }
    let before = select(countOneBits(low & below), countOneBits(low) + countOneBits(high & below),
        lane >= 32u);
    let at = WATER_LIST_ORDER + water_leaf_list[water_list_offsets() + group.x] + before;
    water_leaf_list[at] = group.x * WATER_LIST_GROUP + lane;
}
