struct Camera {
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    position: vec4<f32>,
    jitter: vec4<f32>,
    options: vec4<f32>,
    wind: vec4<f32>,
    region: vec4<f32>,
    detail: vec4<f32>,
}
struct Lighting {
    sun: vec4<f32>,
    sun_color: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
    fog: vec4<f32>,
    effects: vec4<f32>,
    grade: vec4<f32>,
    shadow_tint: vec4<f32>,
    highlight_tint: vec4<f32>,
    refraction: vec4<f32>,
    aerial: vec4<f32>,
    aerial_color: vec4<f32>,
    occupancy: vec4<f32>,
    occupancy_maps: vec4<f32>,
    occupancy_centers: array<vec4<f32>, 4>,
    cloud_shadow: vec4<f32>,
    cloud_layer: vec4<f32>,
    night: vec4<f32>,
    star_turn: vec4<f32>,
    moon: vec4<f32>,
    foliage: vec4<f32>,
    highlights: vec4<f32>,
    windows: vec4<f32>,
    night_shape: vec4<f32>,
    haze_march: vec4<f32>,
}
struct Shadow {
    light: array<vec4<f32>, 3>,
    state: vec4<f32>,
    level_state: vec4<f32>,
    atlas: vec4<f32>,
    windows: array<vec4<f32>, SHADOW_LEVELS>,
    bands: array<vec4<f32>, SHADOW_LEVELS>,
}
const SHADOW_LEVELS: u32 = 20u;
const SHADOW_APRON: u32 = 1u;
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<uniform> shadow: Shadow;
@group(0) @binding(3) var shadow_atlas: texture_depth_2d;
@group(0) @binding(4) var shadow_tiles: texture_2d<u32>;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
@group(0) @binding(7) var shadow_texels: sampler;
@group(0) @binding(14) var terrain_horizons: texture_2d_array<f32>;
const HORIZON_RECORDS: u32 = 384u;
@group(0) @binding(15) var<uniform> terrain_horizon_records: array<vec4<f32>, HORIZON_RECORDS>;
@group(0) @binding(16) var terrain_horizon_sampler: sampler;
@group(0) @binding(17) var occupancy_coverage: texture_2d_array<f32>;
@group(0) @binding(18) var occupancy_heights: texture_2d_array<f32>;
@group(0) @binding(11) var environment_maps: texture_cube_array<f32>;
@group(0) @binding(12) var environment_sampler: sampler;

override SHADOW_MAPS: bool = true;
override TERRAIN_HORIZONS: bool = true;
override CLOUD_SHADOWS: bool = true;
override OCCUPANCY: bool = true;
override PROBES: bool = true;
override WINDOWS: bool = true;
override BOUNCE: bool = true;

fn occupancy_cover(world: vec3<f32>, map: u32) -> f32 {
    let inverse = lighting.occupancy.x / f32(1u << map);
    let centers = lighting.occupancy_centers[map / 2u];
    let center = select(centers.xy, centers.zw, (map & 1u) == 1u);
    let local = (world.xz - center) * inverse + 0.5;
    if any(local <= vec2<f32>(0.0)) || any(local >= vec2<f32>(1.0)) {
        return -1.0;
    }
    let size = lighting.occupancy.y;
    let uv = fract(world.xz * inverse);
    let at = uv * size - vec2<f32>(0.5);
    let corner = floor(at);
    var coverage: f32;
    if all(corner >= vec2<f32>(0.0)) && all(corner < vec2<f32>(size - 1.0)) {
        coverage = textureSampleLevel(occupancy_coverage, terrain_horizon_sampler, uv, map,
            0.0).r;
    } else {
        let texels = vec2<i32>(i32(size));
        let low = (vec2<i32>(corner) + texels) % texels;
        let high = (vec2<i32>(corner) + vec2<i32>(1) + texels) % texels;
        let share = at - corner;
        let below = mix(textureLoad(occupancy_coverage, low, map, 0).r,
            textureLoad(occupancy_coverage, vec2<i32>(high.x, low.y), map, 0).r, share.x);
        let above = mix(textureLoad(occupancy_coverage, vec2<i32>(low.x, high.y), map, 0).r,
            textureLoad(occupancy_coverage, high, map, 0).r, share.x);
        coverage = mix(below, above, share.y);
    }
    coverage = min(coverage, 1.0);
    if coverage <= 0.0 {
        return 0.0;
    }
    let texel = min(vec2<u32>(uv * size), vec2<u32>(u32(size) - 1u));
    let heights = textureLoad(occupancy_heights, texel, map, 0).rg;
    let height = world.y - lighting.occupancy_maps.y;
    let top = heights.x;
    let bottom = -heights.y;
    if height >= top {
        return 0.0;
    }
    return coverage * saturate((top - height) / max(top - bottom, 0.5));
}

fn occupancy_visibility(world: vec3<f32>) -> f32 {
    let strength = lighting.occupancy.w;
    if !OCCUPANCY || strength <= 0.0 {
        return 1.0;
    }
    let maps = u32(lighting.occupancy_maps.x);
    let picked = detail_level(detail_distance(world), 0.5 * lighting.occupancy.z, maps + 1u);
    var map = picked.level;
    var blend = picked.blend;
    var hidden = -1.0;
    for (; map < maps; map++) {
        hidden = occupancy_cover(world, map);
        if hidden >= 0.0 {
            break;
        }
        blend = 0.0;
    }
    if hidden < 0.0 {
        return 1.0;
    }
    if blend > 0.0 {
        var next = 0.0;
        if map + 1u < maps {
            next = occupancy_cover(world, map + 1u);
        }
        if next >= 0.0 {
            hidden = mix(hidden, next, blend);
        }
    }
    return 1.0 - strength * hidden;
}

const PI: f32 = 3.14159265;

fn eye_world(relative: vec3<f32>) -> vec3<f32> {
    return relative + camera.position.xyz;
}

const DETAIL_EDGE_ON: f32 = 0.0625;

const DETAIL_ANISOTROPY: f32 = 16.0;

fn detail_distance(world: vec3<f32>) -> f32 {
    return length(world - camera.detail.xyz);
}

fn detail_across(world: vec3<f32>) -> f32 {
    return detail_distance(world) * camera.options.y;
}

fn detail_span(world: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let across = detail_across(world);
    let toward = world - camera.detail.xyz;
    let ray = toward / max(length(toward), 1.0e-6);
    let cosine = dot(normal, ray);
    let facing = max(abs(cosine), DETAIL_EDGE_ON);
    let laid = ray - normal * cosine;
    let way = laid.xz * inverseSqrt(max(dot(laid, laid), 1.0e-12));
    return vec4<f32>(across, across / facing, way);
}

fn detail_resolved(span: vec2<f32>) -> f32 {
    return max(min(span.x, span.y), max(span.x, span.y) / DETAIL_ANISOTROPY);
}

struct DetailLevel {
    level: u32,
    blend: f32,
}

fn detail_level(span: f32, base: f32, levels: u32) -> DetailLevel {
    let coarsest = f32(max(levels, 1u) - 1u);
    let at = clamp(log2(max(span / max(base, 1.0e-30), 1.0e-30)), 0.0, coarsest);
    var chosen: DetailLevel;
    chosen.level = u32(floor(at));
    chosen.blend = saturate(2.0 * (at - floor(at)) - 1.0);
    return chosen;
}

fn material_bias() -> f32 {
    return camera.options.w;
}

fn material_detail() -> f32 {
    return camera.region.z;
}

fn stable_normalize(value: vec3<f32>) -> vec3<f32> {
    let scale = maximum_abs(value);
    if scale == 0.0 { // optics: a zero vector's guard, whatever it measures
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(value / scale);
}

fn dithering() -> bool {
    return lighting.grade.w != 0.0;
}

fn encoded(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let value = encode_srgb(color);
    if !dithering() {
        return value;
    }
    return dithered(value, pixel, camera.options.z, 0u);
}

fn stored(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    if !dithering() {
        return color;
    }
    return decode_srgb(dithered(encode_srgb(color), pixel, camera.options.z, 0u));
}

fn haze(direction: vec3<f32>) -> vec3<f32> {
    return haze_sky(direction) + sun_halo(direction);
}

fn haze_sky(direction: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    let color = mix(lighting.horizon.rgb, lighting.zenith.rgb, sqrt(max(up, 0.0)));
    return mix(color, lighting.ground.rgb, min(max(-up, 0.0) * 1.5, 0.35));
}

fn sun_halo(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    let square = toward * toward;
    return lighting.sun_color.rgb * (0.12 * square * square * square);
}

fn sky(direction: vec3<f32>) -> vec3<f32> {
    return haze(direction) + sun_disc(direction);
}

fn sun_disc(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    return lighting.sun_color.rgb * (6.0 * pow(toward, 900.0));
}

const CLOUD_HORIZON: f32 = 0.05;

fn cloud_dome_uv(direction: vec3<f32>) -> vec2<f32> {
    let around = atan2(direction.z, direction.x) * (1.0 / 6.2831853);
    let low = -asin(CLOUD_HORIZON);
    let elevation = asin(clamp(direction.y, -1.0, 1.0));
    return vec2<f32>(fract(around), saturate((elevation - low) / (1.5707963 - low)));
}

struct CloudDomes {
    older: vec4<f32>,
    newer: vec4<f32>,
    older_mirror: vec4<f32>,
    newer_mirror: vec4<f32>,
    eye: vec4<f32>,
    view: vec4<f32>,
}

const CLOUD_PARALLAX_REACH: f32 = 60000.0;

fn cloud_dome_direction(domes: CloudDomes, origin: vec3<f32>, start: vec3<f32>,
        direction: vec3<f32>) -> vec3<f32> {
    let rise = domes.eye.w - start.y;
    if direction.y * rise <= 0.0 || abs(direction.y) < 1.0e-5 {
        return direction;
    }
    let along = min(rise / direction.y, CLOUD_PARALLAX_REACH);
    let toward = start + direction * along - origin;
    let reach = length(toward);
    if reach < 1.0e-3 {
        return direction;
    }
    return toward / reach;
}

fn behind_clouds(light: vec3<f32>, cloud: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
    let shown = smoothstep(-CLOUD_HORIZON, -0.4 * CLOUD_HORIZON, direction.y);
    return mix(light, light * cloud.a + cloud.rgb, shown);
}

fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    return mix(sky_above(normal), lighting.ground.rgb, max(-normal.y, 0.0));
}

fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(lighting.horizon.rgb, lighting.zenith.rgb, 0.5);
    return mix(side, lighting.zenith.rgb, max(normal.y, 0.0));
}

fn height_depth(eye: vec3<f32>, point: vec3<f32>, start: f32, density: f32, falloff: f32)
        -> f32 {
    let ray = point - eye;
    let distance = length(ray);
    if distance <= start {
        return 0.0;
    }
    let from_height = max(eye.y + ray.y * (start / distance), -1000.0);
    let to_height = max(point.y, -1000.0);
    let rise = falloff * (to_height - from_height);
    let near = exp(-falloff * from_height);
    let far = exp(-falloff * to_height);
    let mean = select(near, (near - far) / rise, abs(rise) > 1.0e-4);
    return density * (distance - start) * mean;
}

fn fog(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let depth = height_depth(eye, point, lighting.fog.x, lighting.fog.y, lighting.fog.z);
    return min(1.0 - exp(-depth), lighting.fog.w);
}

struct AirLight {
    fog: vec3<f32>,
    fog_sun: vec3<f32>,
    air: vec3<f32>,
    air_sun: vec3<f32>,
}

fn air_light(direction: vec3<f32>) -> AirLight {
    var light: AirLight;
    if hazy() {
        let behind = sky_behind(direction);
        let glow = min(sky_glow(direction), behind);
        light.fog = behind - glow;
        light.fog_sun = glow;
        light.air = light.fog;
        light.air_sun = glow;
        return light;
    }
    light.fog = haze_sky(direction);
    light.fog_sun = sun_halo(direction);
    light.air = lighting.aerial_color.rgb;
    light.air_sun = air_glow(direction);
    return light;
}

fn air_glow(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    let broad = toward * toward;
    let lobe = broad * broad;
    return lighting.sun_color.rgb * (lighting.aerial.w * lobe * (0.25 + 0.75 * lobe));
}

fn sky_glow(direction: vec3<f32>) -> vec3<f32> {
    if lighting.grade.z > 0.5 {
        return air_glow(direction);
    }
    return sun_halo(direction);
}

fn marched_reach() -> f32 {
    return select(lighting.haze_march.x, 0.0, camera.options.x != 0.0);
}

const HAZE_BLUR: f32 = 0.3;
const HAZE_LIFT: f32 = 0.02;

fn hazy() -> bool {
    return lighting.aerial_color.w != 0.0;
}

fn sky_behind(direction: vec3<f32>) -> vec3<f32> {
    let lifted = normalize(vec3<f32>(direction.x, max(direction.y, HAZE_LIFT), direction.z));
    let level = HAZE_BLUR * f32(textureNumLevels(environment_maps) - 1u);
    return textureSampleLevel(environment_maps, environment_sampler, lifted, 0, level).rgb;
}

fn atmospheric(light: vec3<f32>, eye: vec3<f32>, point: vec3<f32>, direction: vec3<f32>)
        -> vec4<f32> {
    let air_lit = air_light(direction);
    let ray = point - eye;
    let distance = length(ray);
    let reached = eye + ray * (min(marched_reach(), distance) / max(distance, 1.0e-6));
    let fogged = fog(eye, point);
    var color = light;
    if fogged != 0.0 {
        color = mix(light, air_lit.fog, fogged) + air_lit.fog_sun * (fogged - fog(eye, reached));
    }
    var covered = fogged;
    if lighting.aerial.y > 0.0 {
        let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
            lighting.aerial.z);
        if depth != 0.0 {
            let air = 1.0 - exp(-depth);
            let marched = 1.0 - exp(-height_depth(eye, reached, lighting.aerial.x,
                lighting.aerial.y, lighting.aerial.z));
            color = mix(color, air_lit.air, air) + air_lit.air_sun * (air - marched);
            covered = fogged + (1.0 - fogged) * air;
        }
    }
    return vec4<f32>(color, covered);
}

fn atmospheric_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let fogged = fog(eye, point);
    if lighting.aerial.y <= 0.0 {
        return fogged;
    }
    let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
        lighting.aerial.z);
    return fogged + (1.0 - fogged) * (1.0 - exp(-depth));
}

fn grade(color: vec3<f32>) -> vec3<f32> {
    let exposed = max(color, vec3<f32>(0.0)) * lighting.horizon.w;
    let peak = max(exposed.r, max(exposed.g, exposed.b));
    let kept = exposed * (filmic(vec3<f32>(peak)).r / max(peak, 1.0e-6));
    let curved = mix(filmic(exposed), kept, 0.5);
    let root = sqrt(curved);
    let perceptual = root + lighting.grade.x * (root * root * (3.0 - 2.0 * root) - root);
    var mapped = perceptual * perceptual;
    let tone = smoothstep(0.2, 0.8, sqrt(luminance(mapped)));
    mapped *= mix(lighting.shadow_tint.rgb, lighting.highlight_tint.rgb, tone);
    let top = max(mapped.r, max(mapped.g, mapped.b));
    let bottom = min(mapped.r, min(mapped.g, mapped.b));
    let muted = 1.0 - (top - bottom) / max(top, 1.0e-6);
    let amount = lighting.ground.w + lighting.grade.y * muted;
    return saturate(mix(vec3<f32>(luminance(mapped)), mapped, amount));
}

fn glow_weight(color: vec3<f32>) -> f32 {
    let bright = max(luminance(color) * lighting.horizon.w - lighting.effects.y, 0.0);
    return bright / (1.0 + bright);
}

const RECEIVER_MARGIN: f32 = 2.0 / 65535.0;

struct MapPoint {
    texel: vec2<f32>,
    depth: f32,
    slope: vec2<f32>,
    soft: f32,
}

var<private> crown_light: f32;

fn crown_softness(per_meter: f32) -> f32 {
    return crown_light * shadow.state.y * abs(per_meter);
}

fn texel_slope(map_dx: vec4<f32>, map_dy: vec4<f32>, scale: vec2<f32>, limit: f32) -> vec2<f32> {
    let texel_dx = map_dx.xy * scale;
    let texel_dy = map_dy.xy * scale;
    let det = texel_dx.x * texel_dy.y - texel_dx.y * texel_dy.x;
    var slope = vec2<f32>(0.0);
    if abs(det) > 1.0e-12 {
        slope = vec2<f32>(map_dx.z * texel_dy.y - map_dy.z * texel_dx.y,
            map_dy.z * texel_dx.x - map_dx.z * texel_dy.x) / det;
    }
    return clamp(slope, vec2<f32>(-limit), vec2<f32>(limit));
}

fn atlas_depths(point: MapPoint) -> vec4<f32> {
    let corner = floor(point.texel - vec2<f32>(0.5)) + vec2<f32>(1.0);
    return textureGather(shadow_atlas, shadow_texels, corner * shadow.atlas.zw);
}

fn plane_visibility(depths: vec4<f32>, point: MapPoint) -> f32 {
    let corner = point.texel - vec2<f32>(0.5);
    let f = corner - floor(corner);
    let du = vec4<f32>(-f.x, 1.0 - f.x, 1.0 - f.x, -f.x);
    let dv = vec4<f32>(1.0 - f.y, 1.0 - f.y, -f.y, -f.y);
    let plane = point.depth + point.slope.x * du + point.slope.y * dv - RECEIVER_MARGIN;
    var lit = select(vec4<f32>(0.0), vec4<f32>(1.0), plane <= depths);
    if point.soft > 0.0 {
        lit = exp2(-max(plane - depths, vec4<f32>(0.0)) / point.soft);
    }
    return mix(mix(lit.w, lit.z, f.x), mix(lit.x, lit.y, f.x), f.y);
}

fn light_space(world: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(shadow.light[0].xyz, world), dot(shadow.light[1].xyz, world),
        dot(shadow.light[2].xyz, world));
}

fn shadow_level(world: vec3<f32>) -> DetailLevel {
    return detail_level(detail_across(world), shadow.level_state.y, u32(shadow.level_state.x));
}

struct LevelPoint {
    cell: vec2<u32>,
    across: vec2<f32>,
    depth: f32,
    inside: bool,
}

fn level_point(level: u32, light: vec3<f32>) -> LevelPoint {
    let window = shadow.windows[level];
    let band = shadow.bands[level];
    let side = shadow.atlas.x;
    let tiles = (light.xy - window.xy) / window.z;
    let tile = floor(tiles);
    var point: LevelPoint;
    point.depth = (light.z - band.x) * band.y;
    point.inside = all(tile >= vec2<f32>(0.0)) && all(tile < vec2<f32>(side))
        && point.depth > 0.0 && point.depth < 1.0;
    let first = vec2<i32>(round(window.xy / window.z));
    point.cell = vec2<u32>((first + vec2<i32>(tile)) & vec2<i32>(i32(side) - 1));
    point.across = tiles - tile;
    return point;
}

fn level_slots(level: u32, point: LevelPoint) -> vec2<u32> {
    let side = u32(shadow.atlas.x);
    let entry = textureLoad(shadow_tiles, vec2<u32>(point.cell.x, level * side + point.cell.y),
        0).r;
    return vec2<u32>(entry & 0xffffu, entry >> 16u);
}

fn slot_texel(slot: u32, across: vec2<f32>) -> vec2<f32> {
    let texels = shadow.level_state.z;
    let side = texels + 2.0 * f32(SHADOW_APRON);
    let per_row = u32(shadow.atlas.y);
    let corner = vec2<f32>(vec2<u32>(slot % per_row, slot / per_row)) * side;
    return corner + vec2<f32>(f32(SHADOW_APRON)) + vec2<f32>(across.x, 1.0 - across.y) * texels;
}

struct KeptLift {
    first: u32,
    lifts: vec4<f32>,
}
var<private> kept_lift: KeptLift;

fn level_lift(level: u32) -> f32 {
    if level < kept_lift.first || level - kept_lift.first >= 4u {
        return 0.0;
    }
    return kept_lift.lifts[level - kept_lift.first];
}

const CROWN_TAPS: u32 = 4u;
const CROWN_REACH: f32 = 2.0;
const CROWN_SPIRAL_TURN: f32 = 2.39996323;

fn crown_spread(level: u32, light: vec3<f32>, point: MapPoint, center: LevelPoint, slot: u32,
        visible: f32) -> f32 {
    let depth_per_meter = max(shadow.bands[level].y, 1.0e-12);
    let crown = point.soft / depth_per_meter;
    let turn = fade_noise(floor(point.texel) + vec2<f32>(0.5)) * 2.0 * PI;
    var way = vec2<f32>(cos(turn), sin(turn));
    let step = vec2<f32>(cos(CROWN_SPIRAL_TURN), sin(CROWN_SPIRAL_TURN));
    var lit = visible;
    for (var tap = 0u; tap < CROWN_TAPS; tap++) {
        let across = CROWN_REACH * crown * sqrt((f32(tap) + 0.5) / f32(CROWN_TAPS));
        let at = level_point(level, light + vec3<f32>(way, 0.0) * across);
        way = vec2<f32>(way.x * step.x - way.y * step.y, way.x * step.y + way.y * step.x);
        if !at.inside {
            continue;
        }
        var held = slot;
        if any(at.cell != center.cell) {
            held = level_slots(level, at).x;
            if held == 0u {
                continue;
            }
        }
        let caster = textureLoad(shadow_atlas, vec2<i32>(slot_texel(held - 1u, at.across)), 0);
        let down = max(point.depth - caster, 0.0) / depth_per_meter;
        lit = max(lit, exp2(-(across + down) / crown));
    }
    return lit;
}

fn level_visibility(level: u32, world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>, spread: bool) -> f32 {
    let window = shadow.windows[level];
    let band = shadow.bands[level];
    let up = vec3<f32>(shadow.light[0].y, shadow.light[1].y, shadow.light[2].y);
    let light = light_space(world + normal * band.z) + up * level_lift(level);
    let at = level_point(level, light);
    if !at.inside {
        return -1.0;
    }
    let slots = level_slots(level, at);
    if slots.x == 0u {
        return -1.0;
    }
    var point: MapPoint;
    point.texel = slot_texel(slots.x - 1u, at.across);
    point.depth = at.depth;
    let per_world = vec3<f32>(1.0, 1.0, band.y);
    point.slope = texel_slope(vec4<f32>(light_space(dx) * per_world, 0.0),
        vec4<f32>(light_space(dy) * per_world, 0.0), vec2<f32>(1.0, -1.0) / window.w, band.w);
    point.soft = crown_softness(band.y);
    var visible = plane_visibility(atlas_depths(point), point);
    if spread && point.soft > 0.0 && visible < 1.0 {
        visible = crown_spread(level, light, point, at, slots.x, visible);
    }
    if slots.y != 0u {
        var moving = point;
        moving.texel = slot_texel(slots.y - 1u, at.across);
        visible = min(visible, plane_visibility(atlas_depths(moving), moving));
    }
    return visible;
}

const HORIZON_DIRECTIONS: u32 = 16u;
const HORIZON_LAYERS: u32 = 9u;
const HORIZON_RECORD_VECTORS: u32 = 6u;

struct HorizonTurn {
    first: u32,
    second: u32,
    between: f32,
}

fn horizon_turn(heading: vec2<f32>) -> HorizonTurn {
    let turn = fract(atan2(heading.y, heading.x) / (2.0 * PI) + 1.0) * f32(HORIZON_DIRECTIONS);
    var found: HorizonTurn;
    found.first = min(u32(turn), HORIZON_DIRECTIONS - 1u);
    found.second = (found.first + 1u) % HORIZON_DIRECTIONS;
    found.between = turn - f32(found.first);
    return found;
}

struct HorizonCrest {
    slope: f32,
    distance: f32,
    elevation: f32,
    inside: bool,
}

fn horizon_crest(at: u32, turn: HorizonTurn, world: vec3<f32>, on_terrain: bool)
        -> HorizonCrest {
    var crest: HorizonCrest;
    let place = terrain_horizon_records[at];
    let shape = terrain_horizon_records[at + 1u];
    let uv = (world.xz - place.xy) * place.zw;
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        return crest;
    }
    crest.inside = true;
    let base = u32(shape.x);
    let near = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
        base + turn.first / 2u, 0.0);
    var far = near;
    if (turn.first & 1u) == 1u {
        far = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
            base + turn.second / 2u, 0.0);
    }
    let odd = (turn.first & 1u) == 1u;
    let heights = vec2<f32>(select(near.r, near.g, odd), select(far.g, far.r, odd));
    let distances = vec2<f32>(select(near.b, near.a, odd), select(far.a, far.b, odd));
    let slopes = max(heights / max(distances, vec2<f32>(1.0e-3)), vec2<f32>(-1.0));
    crest.slope = mix(slopes.x, slopes.y, turn.between);
    crest.distance = mix(distances.x, distances.y, turn.between);
    if !on_terrain {
        let ground = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
            base + HORIZON_LAYERS - 1u, 0.0).r;
        crest.elevation = world.y - shape.w - ground;
    }
    return crest;
}

fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
    if !TERRAIN_HORIZONS {
        return 1.0;
    }
    let sun = lighting.sun.xyz;
    let level = length(sun.xz);
    if level < 1.0e-4 {
        return 1.0;
    }
    let rise = sun.y / level;
    let turn = horizon_turn(sun.xz);
    let count = min(u32(shadow.level_state.w), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
    var visible = 1.0;
    for (var index = 0u; index < count; index++) {
        let at = HORIZON_RECORD_VECTORS * index;
        let crest = horizon_crest(at, turn, world, on_terrain);
        if !crest.inside {
            continue;
        }
        let clearance = crest.distance * (crest.slope - rise);
        let shape = terrain_horizon_records[at + 1u];
        let edge = max(crest.distance * shape.y, 0.5);
        visible = min(visible, smoothstep(-edge, edge, crest.elevation + shape.z - clearance));
        if visible <= 0.0 {
            break;
        }
    }
    return visible;
}

fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32, on_terrain: bool) -> f32 {
    if !SHADOW_MAPS || shadow.state.x == 0.0 {
        return cloud_passing(world);
    }
    let terrain = terrain_visibility(world, on_terrain);
    if terrain <= 0.0 {
        return 0.0;
    }
    let visible = min(terrain, map_visibility(world, normal, dx, dy));
    if visible == 0.0 {
        return 0.0;
    }
    return visible * cloud_passing(world);
}

const BOUNCE_LOW: f32 = 1.5;
const BOUNCE_HIGH: f32 = 24.0;
const BOUNCE_WINDOW_ROW: u32 = 4096u;

struct BounceLight {
    light: vec3<f32>,
    sky: f32,
    weight: f32,
}

fn bounce_layer(texel: vec2<f32>, first: u32, normal: vec3<f32>) -> vec4<f32> {
    let mean = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first, 0.0);
    let x = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 1u,
        0.0);
    let y = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 2u,
        0.0);
    let z = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 3u,
        0.0);
    return mean + x * normal.x + y * normal.y + z * normal.z;
}

fn bounce_texel(at: u32, level: u32, uv: vec2<f32>) -> vec2<f32> {
    let header = terrain_horizon_records[at + 2u];
    let arranged = terrain_horizon_records[at + 3u];
    let columns = f32(u32(header.y) >> level);
    let window = min(header.w, columns);
    let packed = u32(terrain_horizon_records[at + 4u + level / 4u][level % 4u]);
    let first = vec2<f32>(f32(packed % BOUNCE_WINDOW_ROW), f32(packed / BOUNCE_WINDOW_ROW));
    let column = clamp(uv * columns - 0.5, vec2<f32>(0.0), vec2<f32>(columns - 1.0)) - first;
    if any(column < vec2<f32>(0.0)) || any(column > vec2<f32>(window - 1.0)) {
        return vec2<f32>(-1.0);
    }
    let row = u32(arranged.z);
    let region = vec2<f32>(f32(level % row), f32(level / row)) * header.w;
    return (region + column + 0.5) * arranged.x;
}

fn bounce_held(texel: vec2<f32>, first: u32, normal: vec3<f32>, above: f32) -> vec4<f32> {
    let held = bounce_layer(texel, first, normal);
    if above <= 0.0 {
        return held;
    }
    return mix(held, bounce_layer(texel, first + 4u, normal), above);
}

struct BounceAt {
    first: u32,
    texel: vec2<f32>,
    next: vec2<f32>,
    blend: f32,
    layer: f32,
}

fn bounce_at(world: vec3<f32>, on_terrain: bool) -> BounceAt {
    var found: BounceAt;
    found.texel = vec2<f32>(-1.0);
    if !BOUNCE || !TERRAIN_HORIZONS {
        return found;
    }
    let count = min(u32(shadow.level_state.w), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
    for (var index = 0u; index < count; index++) {
        let at = HORIZON_RECORD_VECTORS * index;
        let header = terrain_horizon_records[at + 2u];
        if header.x < 0.0 {
            continue;
        }
        let place = terrain_horizon_records[at];
        let uv = (world.xz - place.xy) * place.zw;
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            continue;
        }
        let arranged = terrain_horizon_records[at + 3u];
        let levels = u32(header.z);
        let picked = detail_level(detail_distance(world), 0.5 * arranged.y, levels);
        var level = max(picked.level, u32(arranged.w));
        var blend = select(0.0, picked.blend, level == picked.level);
        var texel = bounce_texel(at, level, uv);
        for (; texel.x < 0.0 && level + 1u < levels; level++) {
            texel = bounce_texel(at, level + 1u, uv);
            blend = 0.0;
        }
        if texel.x < 0.0 {
            continue;
        }
        if !on_terrain {
            let shape = terrain_horizon_records[at + 1u];
            let ground = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
                u32(shape.x) + HORIZON_LAYERS - 1u, 0.0).r;
            found.layer = (world.y - shape.w - ground - BOUNCE_LOW)
                / (BOUNCE_HIGH - BOUNCE_LOW);
        }
        found.first = u32(header.x);
        found.texel = texel;
        if blend > 0.0 && level + 1u < levels {
            found.next = bounce_texel(at, level + 1u, uv);
            found.blend = select(0.0, blend, found.next.x >= 0.0);
        }
        return found;
    }
    return found;
}

fn bounce_light(world: vec3<f32>, normal: vec3<f32>, on_terrain: bool) -> BounceLight {
    var found: BounceLight;
    found.sky = 1.0;
    let at = bounce_at(world, on_terrain);
    if at.texel.x < 0.0 {
        return found;
    }
    let above = saturate(at.layer);
    var held = bounce_held(at.texel, at.first, normal, above);
    if at.blend > 0.0 {
        held = mix(held, bounce_held(at.next, at.first, normal, above), at.blend);
    }
    found.light = max(held.rgb, vec3<f32>(0.0));
    found.sky = saturate(held.a);
    found.weight = 1.0;
    return found;
}

fn bounce_seen(at: BounceAt) -> f32 {
    var lower = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, at.texel,
        at.first, 0.0).a;
    var upper = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, at.texel,
        at.first + 4u, 0.0).a;
    if at.blend > 0.0 {
        lower = mix(lower, textureSampleLevel(terrain_horizons, terrain_horizon_sampler,
            at.next, at.first, 0.0).a, at.blend);
        upper = mix(upper, textureSampleLevel(terrain_horizons, terrain_horizon_sampler,
            at.next, at.first + 4u, 0.0).a, at.blend);
    }
    return mix(lower, upper, saturate(at.layer));
}

const CLOUD_SHADOW_CELLS: u32 = 128u;
@group(0) @binding(19) var<storage, read> cloud_shadows: array<u32, CLOUD_SHADOW_CELLS * CLOUD_SHADOW_CELLS / 2u>;

fn cloud_shadow_cell(x: u32, z: u32) -> f32 {
    let index = z * CLOUD_SHADOW_CELLS + x;
    return unpack2x16float(cloud_shadows[index / 2u])[index % 2u];
}

fn cloud_passing(world: vec3<f32>) -> f32 {
    let strength = lighting.cloud_shadow.w;
    if !CLOUD_SHADOWS || strength <= 0.0 || lighting.sun.y <= 0.01 {
        return 1.0;
    }
    let rise = max(lighting.cloud_layer.x - world.y, 0.0);
    let base = world.xz + lighting.sun.xz * (rise / lighting.sun.y);
    let size = f32(CLOUD_SHADOW_CELLS);
    let cells = base * lighting.cloud_shadow.z - 0.5;
    let local = cells - lighting.cloud_shadow.xy;
    if any(local < vec2<f32>(0.0)) || any(local > vec2<f32>(size - 1.0)) {
        return 1.0;
    }
    let count = i32(CLOUD_SHADOW_CELLS);
    let first = vec2<i32>(floor(cells));
    let a = vec2<u32>(first & vec2<i32>(count - 1));
    let b = vec2<u32>((first + vec2<i32>(1)) & vec2<i32>(count - 1));
    let f = cells - floor(cells);
    let passing = mix(
        mix(cloud_shadow_cell(a.x, a.y), cloud_shadow_cell(b.x, a.y), f.x),
        mix(cloud_shadow_cell(a.x, b.y), cloud_shadow_cell(b.x, b.y), f.x), f.y);
    let edge = min(local, vec2<f32>(size - 1.0) - local) / (0.1 * size);
    let fade = smoothstep(0.0, 1.0, saturate(min(edge.x, edge.y)));
    return mix(1.0, passing, strength * fade);
}

fn fade_noise(position: vec2<f32>) -> f32 {
    return fract(fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))))
        + camera.options.z);
}

fn map_lookup(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> vec2<f32> {
    let pick = shadow_level(world);
    let count = u32(shadow.level_state.x);
    for (var level = pick.level; level < count; level++) {
        let visible = level_visibility(level, world, normal, dx, dy, true);
        if visible < 0.0 {
            continue;
        }
        if level > pick.level || pick.blend <= 0.0 {
            return vec2<f32>(visible, f32(level));
        }
        for (var next = level + 1u; next < count; next++) {
            let coarser = level_visibility(next, world, normal, dx, dy, false);
            if coarser >= 0.0 {
                return vec2<f32>(mix(visible, coarser, pick.blend), f32(level));
            }
        }
        return vec2<f32>(visible, f32(level));
    }
    return vec2<f32>(1.0, -1.0);
}

fn map_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> f32 {
    return map_lookup(world, normal, dx, dy).x;
}

fn air_visibility(world: vec3<f32>) -> f32 {
    let pick = shadow_level(world);
    let light = light_space(world);
    for (var level = pick.level; level < u32(shadow.level_state.x); level++) {
        let at = level_point(level, light);
        if !at.inside {
            continue;
        }
        let slots = level_slots(level, at);
        if slots.x == 0u {
            continue;
        }
        let still = slot_texel(slots.x - 1u, at.across) * shadow.atlas.zw;
        var visible = textureSampleCompareLevel(shadow_atlas, shadow_sampler, still, at.depth);
        if slots.y != 0u {
            let moving = slot_texel(slots.y - 1u, at.across) * shadow.atlas.zw;
            visible = min(visible,
                textureSampleCompareLevel(shadow_atlas, shadow_sampler, moving, at.depth));
        }
        return visible;
    }
    return 1.0;
}

fn distribution(n_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let d = n_h * n_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    return 0.5 / mix(2.0 * n_l * n_v, n_l + n_v, alpha);
}

fn fresnel(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    let m = 1.0 - cosine;
    let m2 = m * m;
    return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

fn environment_brdf(f0: vec3<f32>, roughness: f32, n_v: f32) -> vec3<f32> {
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022)
        + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

fn pixel_noise(position: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))));
}

struct TransparentTargets {
    @location(0) accumulated: vec4<f32>,
    @location(1) revealed: vec2<f32>,
}

fn weighted(color: vec3<f32>, opacity: f32, distance: f32) -> TransparentTargets {
    let near = distance / 5.0;
    let far = distance / 200.0;
    let far3 = far * far * far;
    let weight = opacity * clamp(10.0 / (1.0e-5 + near * near + far3 * far3), 0.01, 3000.0);
    var targets: TransparentTargets;
    targets.accumulated = vec4<f32>(color * opacity, opacity) * weight;
    targets.revealed = vec2<f32>(opacity, 0.0);
    return targets;
}

fn weighted_particle(color: vec3<f32>, opacity: f32, distance: f32) -> TransparentTargets {
    var targets = weighted(color, opacity, distance);
    targets.revealed.y = opacity;
    return targets;
}
