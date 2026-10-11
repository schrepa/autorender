// Shared scene declarations, prepended to the mesh, particle and sun shaft
// shaders: the first bind group's camera, atmosphere and sun
// shadows, and the lighting functions every path shades with.

// Uniform: three column-major mat4 and four vec4, 16-byte alignment,
// exactly 256 bytes. `view_projection` is the rasterized projection; temporally
// antialiased frames jitter it by `jitter.xy` clip units per w.
// Their motion uses the previous frame's unjittered `previous_view_projection`
// and `jitter.zw`, two over the viewport size in pixels. The unjittered
// inverse turns pixels into sky rays; `position.xyz` is the eye and
// `position.w` presentation time in seconds. `options.x` is 1
// while a reflection probe captures a cube face; `options.y`
// is the angle a pixel spans at the view's center (radians), which
// `pixel_footprint` measures pixels by. `wind` is
// the wind's velocity (x and z), gust strength and the previous frame's
// presentation time, which temporal frames' motion sways by.
struct Camera {
    view_projection: mat4x4<f32>,
    previous_view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    position: vec4<f32>,
    jitter: vec4<f32>,
    options: vec4<f32>,
    wind: vec4<f32>,
}
// The scene's atmosphere in linear color; scalars ride in w.
struct Lighting {
    // Toward the sun; w: rim strength.
    sun: vec4<f32>,
    // w: specular scale (1 is physically based).
    sun_color: vec4<f32>,
    // w: ambient scale.
    zenith: vec4<f32>,
    // w: exposure.
    horizon: vec4<f32>,
    // w: saturation.
    ground: vec4<f32>,
    // Start, density, height falloff and maximum.
    fog: vec4<f32>,
    // Emission scale, the exposed brightness above which
    // light glows, the roughness beyond which surfaces
    // reflect only the sky, and 1 when screen-space reflections run.
    effects: vec4<f32>,
    // Contrast and vibrance.
    grade: vec4<f32>,
    // Split toning: luminance-normalized gains of shadows and highlights.
    shadow_tint: vec4<f32>,
    highlight_tint: vec4<f32>,
    // Refraction through glass: the bend's strength (0 without
    // refraction) and the blur radius in pixels of fully rough glass.
    refraction: vec4<f32>,
    // Aerial perspective: start, density at height zero,
    // height falloff and the sun's glow; the air's own color.
    aerial: vec4<f32>,
    aerial_color: vec4<f32>,
    // Occupancy: the map's center (x and z), the inverse of
    // its side and the strength covers darken by (0 without a map); then
    // the reference height its heights are above, its texels a side, the
    // meters from the eye it fades out by and the far map's
    // reference height; the far map's center, the inverse of its side and
    // its reach (0 without one).
    occupancy: vec4<f32>,
    occupancy_heights: vec4<f32>,
    occupancy_far: vec4<f32>,
    // Under the sea: the light its water scatters toward the
    // eye (rgb) and the meters through which half the light passes (w);
    // zero above it.
    underwater: vec4<f32>,
    // The sea: its still level, the meters through which half the light
    // passes, and 1 while it stands.
    sea: vec4<f32>,
    // Clouds' shadows: the grid's first cell's corner (x and
    // z), cells per meter and the shadows' strength (0 without clouds);
    // then the clouds' base height.
    cloud_shadow: vec4<f32>,
    cloud_layer: vec4<f32>,
}
// Sun shadows: world to each near cascade and to both
// far maps (x and y in [-1, 1] across a map, z in [0, 1]).
struct Shadow {
    near: array<mat4x4<f32>, 4>,
    far: mat4x4<f32>,
    // x: shadows on; y: the kept far map holds casters; z: near cascades in
    // use; w: distance where shadows begin to fade.
    state: vec4<f32>,
    // View distance each near cascade reaches; unreachable beyond those in
    // use.
    reach: vec4<f32>,
    // Normal offsets (world units) of lookups in each near cascade.
    near_offsets: vec4<f32>,
    // Steepest receiver plane (depth per texel) of each near cascade.
    near_limits: vec4<f32>,
    // x: distance where shadows are gone; y: normal offset of far lookups;
    // z: steepest receiver plane of the far maps.
    offsets: vec4<f32>,
    // Texels a side of the near cascades (x) and the far maps (y).
    sizes: vec4<f32>,
    // World to light space: the sun's right, up and
    // away-from-sun axes, which every kept level scales.
    light: array<vec4<f32>, 3>,
    // Each kept level's light space: one over its width (x and y in level
    // widths, whose fractions place a point in its map, which wraps), and
    // its depth band's nearest depth and one over its depth (z in [0, 1]).
    level_scales: array<vec4<f32>, 3>,
    // Each level's window: its first corner in level widths (x and y), the
    // normal offset of its lookups (z, world units) and its steepest
    // receiver plane (w).
    level_windows: array<vec4<f32>, 3>,
    // The tiles of each level holding their casters: a bit per slot, two
    // vectors a level.
    level_valid: array<vec4<u32>, 6>,
    // x: levels in use; y: their texels a side; z: a bit for each level
    // with tiles not yet drawn; w: the terrains whose horizon records
    // lookups loop over.
    level_state: vec4<f32>,
}
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<uniform> shadow: Shadow;
// The near cascades, one layer each.
@group(0) @binding(3) var near_shadows: texture_depth_2d_array;
// The far map of still casters, and the one redrawn every frame.
@group(0) @binding(4) var kept_shadows: texture_depth_2d;
@group(0) @binding(5) var moving_shadows: texture_depth_2d;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
// Unfiltered reads of four shadow texels, for receiver planes.
@group(0) @binding(7) var shadow_texels: sampler;
// Terrain's long sun shadows: each terrain's horizon map, a
// grid aligned with the sun: x how far above the ground the sun clears the
// terrain toward it, y how far toward the sun the slope setting that lies, z
// the ground's height.
@group(0) @binding(14) var terrain_horizons: texture_2d_array<f32>;
// Two texels per terrain casting: its map's center (world x and z) and unit
// direction toward the sun; the map's half extents along and across the
// sun, how fast its shadows soften per meter of that distance, and the
// meters receivers rise before comparing. One unused texel without any.
@group(0) @binding(15) var terrain_horizon_records: texture_2d<f32>;
@group(0) @binding(16) var terrain_horizon_sampler: sampler;
// What placed objects cover around the camera: the densest
// cover over each texel, and the covers' highest top and lowest bottom
// (negated), above the map's reference height.
@group(0) @binding(17) var occupancy_coverage: texture_2d_array<f32>;
@group(0) @binding(18) var occupancy_heights: texture_2d_array<f32>;
// The kept shadow levels, one layer each.
@group(0) @binding(8) var kept_levels: texture_depth_2d_array;

// How much the covers in occupancy layer `layer` (centered at `center`,
// `inverse` over its side, heights above `reference`) hide the sky from
// `world`: their cover there, the more the deeper below their top; none
// above them or outside the map.
fn occupancy_cover(world: vec3<f32>, layer: u32, center: vec2<f32>, inverse: f32,
        reference: f32) -> f32 {
    let uv = (world.xz - center) * inverse + 0.5;
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
        return 0.0;
    }
    let coverage = min(textureSampleLevel(occupancy_coverage, terrain_horizon_sampler, uv,
        layer, 0.0).r, 1.0);
    if coverage <= 0.0 {
        return 0.0;
    }
    let size = lighting.occupancy_heights.y;
    let texel = min(vec2<u32>(uv * size), vec2<u32>(u32(size) - 1u));
    let heights = textureLoad(occupancy_heights, texel, layer, 0).rg;
    let height = world.y - reference;
    let top = heights.x;
    let bottom = -heights.y;
    if height >= top {
        return 0.0;
    }
    return coverage * saturate((top - height) / max(top - bottom, 0.5));
}

// The share of the sky light placed objects' covers leave at `world`:
// under a cover, its density's worth less; within it, less
// the deeper below its top; above it or outside the maps, all of it. Covers
// darken less over the outer two fifths of the map's reach around the eye,
// so their edge is never seen, where the far map's take
// over, fading out over the outer two fifths of theirs.
fn occupancy_visibility(world: vec3<f32>) -> f32 {
    let strength = lighting.occupancy.w;
    if strength <= 0.0 {
        return 1.0;
    }
    let reach = lighting.occupancy_heights.z;
    let away = length(world.xz - camera.position.xz);
    let near = 1.0 - smoothstep(0.6 * reach, reach, away);
    let far_reach = lighting.occupancy_far.w;
    let far = (1.0 - near) * select(0.0, 1.0 - smoothstep(0.6 * far_reach, far_reach, away),
        far_reach > 0.0);
    var hidden = 0.0;
    if near > 0.0 {
        hidden += near * occupancy_cover(world, 0u, lighting.occupancy.xy, lighting.occupancy.z,
            lighting.occupancy_heights.x);
    }
    if far > 0.0 {
        hidden += far * occupancy_cover(world, 1u, lighting.occupancy_far.xy,
            lighting.occupancy_far.z, lighting.occupancy_heights.w);
    }
    return 1.0 - strength * hidden;
}

const PI: f32 = 3.14159265;

// The meters a pixel spans on a surface at `world` facing `normal`, the
// mean of its two sides: from the angle a pixel spans
// (`camera.options.y`), the point's depth along the view and how obliquely
// the view meets the surface, so it changes smoothly over curved surfaces
// where screen derivatives step from one triangle to the next. Views more
// grazing than a sixteenth count as that.
fn pixel_footprint(world: vec3<f32>, normal: vec3<f32>) -> f32 {
    let relative = world - camera.position.xyz;
    let distance = max(length(relative), 1.0e-6);
    let view = camera.view_projection;
    let depth = max(dot(vec3<f32>(view[0].w, view[1].w, view[2].w), relative), 1.0e-6);
    let facing = max(abs(dot(normal, relative)) / distance, 0.0625);
    return depth * camera.options.y * sqrt(depth / (distance * facing));
}

fn stable_normalize(value: vec3<f32>) -> vec3<f32> {
    let scale = maximum_abs(value);
    // Interpolation between opposing vertex normals can produce a zero vector.
    if scale == 0.0 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(value / scale);
}

// Graded linear color as the canvas stores it: WebGPU canvases are UNORM,
// so shaders drawing into them encode sRGB themselves.
fn encoded(color: vec3<f32>) -> vec3<f32> {
    return encode_srgb(color);
}

// The sky's light in `direction` without the sun's disc: horizon to zenith
// above; below, haze darkening only partway toward the ground, as distant
// ground seen through air does; and a wide halo toward the sun. Fog and
// reflections take this color.
fn haze(direction: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    var color = mix(lighting.horizon.rgb, lighting.zenith.rgb, sqrt(max(up, 0.0)));
    color = mix(color, lighting.ground.rgb, min(max(-up, 0.0) * 1.5, 0.35));
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    let square = toward * toward;
    return color + lighting.sun_color.rgb * (0.12 * square * square * square);
}

// The sky itself: its haze and the sun's small disc.
fn sky(direction: vec3<f32>) -> vec3<f32> {
    // Under the sea, nothing shows past the water.
    if lighting.underwater.w > 0.0 {
        return underwater_light(direction);
    }
    return open_sky(direction);
}

// The light the sea's water scatters toward an eye under it looking along
// `direction`: brightest looking up toward the light coming
// down through it, dim looking down into the deep.
fn underwater_light(direction: vec3<f32>) -> vec3<f32> {
    let up = saturate(direction.y * 0.5 + 0.5);
    return lighting.underwater.rgb * (0.35 + 1.3 * up * up);
}

// The sky over the open air: its haze and the sun's disc.
fn open_sky(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    return haze(direction) + lighting.sun_color.rgb * (6.0 * pow(toward, 900.0));
}

// Rays below the horizon by this much (the sine of their elevation) see no
// clouds (`clouds.wgsl`'s `HORIZON`).
const CLOUD_HORIZON: f32 = 0.05;

// The clouds' dome texel looking along `direction`: around by its azimuth,
// up by its elevation above `-asin(CLOUD_HORIZON)`, every row as many
// degrees.
fn cloud_dome_uv(direction: vec3<f32>) -> vec2<f32> {
    let around = atan2(direction.z, direction.x) * (1.0 / 6.2831853);
    let low = -asin(CLOUD_HORIZON);
    let elevation = asin(clamp(direction.y, -1.0, 1.0));
    return vec2<f32>(fract(around), saturate((elevation - low) / (1.5707963 - low)));
}

// The clouds' shown domes: the older's origin (world) and
// whether domes show (1); the newer's origin and its share of the blend;
// the eye lookups correct for, and the cloud layer's middle (world y);
// and whether the view's own march shows the sky's clouds (1), so the view's sky shows no dome.
struct CloudDomes {
    older: vec4<f32>,
    newer: vec4<f32>,
    eye: vec4<f32>,
    view: vec4<f32>,
}

// Meters along a view the domes' parallax correction reaches at most.
const CLOUD_PARALLAX_REACH: f32 = 60000.0;

// The direction in which the dome marched from `origin` holds what the view
// along `direction` from the domes' eye sees of the clouds:
// toward where the view meets the cloud layer's middle. Views never
// reaching it look along themselves.
fn cloud_dome_direction(domes: CloudDomes, origin: vec3<f32>, direction: vec3<f32>)
        -> vec3<f32> {
    let rise = domes.eye.w - domes.eye.y;
    if direction.y * rise <= 0.0 || abs(direction.y) < 1.0e-5 {
        return direction;
    }
    let along = min(rise / direction.y, CLOUD_PARALLAX_REACH);
    let toward = domes.eye.xyz + direction * along - origin;
    let reach = length(toward);
    if reach < 1.0e-3 {
        return direction;
    }
    return toward / reach;
}

// `light` from the sky along `direction` behind the dome's `cloud`: the
// light it scatters toward the eye, over the share of the sky passing it;
// none below the horizon.
fn behind_clouds(light: vec3<f32>, cloud: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
    let shown = smoothstep(-CLOUD_HORIZON, -0.4 * CLOUD_HORIZON, direction.y);
    return mix(light, light * cloud.a + cloud.rgb, shown);
}

// How much of what lies `distance` meters away the sea's water hides from
// an eye under it.
fn underwater_cover(distance: f32) -> f32 {
    return 1.0 - exp2(-distance / lighting.underwater.w);
}

// The sky's average light over the hemisphere around `normal`: the zenith
// from above, half horizon and half zenith from the side (cooler than the
// sun, so shade contrasts with light) and the ground from below.
fn sky_average(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(lighting.horizon.rgb, lighting.zenith.rgb, 0.5);
    let above = mix(side, lighting.zenith.rgb, max(normal.y, 0.0));
    return mix(above, lighting.ground.rgb, max(-normal.y, 0.0));
}

// Optical depth from `eye` to `point` beyond `start` of a medium of
// `density` per unit of distance at height zero, decaying exponentially by
// `falloff` per unit of height, integrated along the ray.
fn height_depth(eye: vec3<f32>, point: vec3<f32>, start: f32, density: f32, falloff: f32)
        -> f32 {
    let ray = point - eye;
    let distance = length(ray);
    if distance <= start {
        return 0.0;
    }
    // Far below the ground the density would overflow; it is saturated there.
    let from_height = max(eye.y + ray.y * (start / distance), -1000.0);
    let to_height = max(point.y, -1000.0);
    let rise = falloff * (to_height - from_height);
    let near = exp(-falloff * from_height);
    let far = exp(-falloff * to_height);
    // The mean density along the ray; its limit where the ray is level.
    let mean = select(near, (near - far) / rise, abs(rise) > 1.0e-4);
    return density * (distance - start) * mean;
}

// Fog covering `point` seen from `eye`: exponential in the distance beyond
// the start, with a density decaying exponentially with height, integrated
// along the ray, and at most its maximum.
fn fog(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let depth = height_depth(eye, point, lighting.fog.x, lighting.fog.y, lighting.fog.z);
    return min(1.0 - exp(-depth), lighting.fog.w);
}

// The light air scatters toward the eye looking along `direction`:
// its own color, and toward the sun a glow of the sun's
// color, a broad lobe around a narrower one.
fn air_light(direction: vec3<f32>) -> vec3<f32> {
    let toward = max(dot(direction, lighting.sun.xyz), 0.0);
    let broad = toward * toward;
    let lobe = broad * broad;
    return lighting.aerial_color.rgb
        + lighting.sun_color.rgb * (lighting.aerial.w * lobe * (0.25 + 0.75 * lobe));
}

// What fog and air leave of light `lit` from `point` seen
// from `eye` looking along `direction`, and how much of it they cover
// together (w).
fn atmospheric(lit: vec3<f32>, eye: vec3<f32>, point: vec3<f32>, direction: vec3<f32>)
        -> vec4<f32> {
    // Light reaching what lies under the sea dims through the water above
    // it.
    var light = lit;
    if lighting.sea.z > 0.0 && point.y < lighting.sea.x {
        light *= exp2((point.y - lighting.sea.x) / lighting.sea.y);
    }
    // Under the sea, its water stands in for fog and air.
    if lighting.underwater.w > 0.0 {
        let covered = underwater_cover(length(point - eye));
        return vec4<f32>(mix(light, underwater_light(direction), covered), covered);
    }
    let fogged = basin_cover(eye, point);
    // Without fog, mixing in none of its haze leaves the light.
    var color = light;
    if fogged != 0.0 {
        color = mix(light, haze(direction), fogged);
    }
    var covered = fogged;
    if lighting.aerial.y > 0.0 {
        let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
            lighting.aerial.z);
        // Where the air is no depth, mixing in none of it leaves the color.
        if depth != 0.0 {
            let air = 1.0 - exp(-depth);
            color = mix(color, air_light(direction), air);
            covered = fogged + (1.0 - fogged) * air;
        }
    }
    return vec4<f32>(color, covered);
}

// How much of what lies at `point` seen from `eye` fog and air cover
// together.
fn atmospheric_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    if lighting.underwater.w > 0.0 {
        return underwater_cover(length(point - eye));
    }
    let fogged = basin_cover(eye, point);
    if lighting.aerial.y <= 0.0 {
        return fogged;
    }
    let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
        lighting.aerial.z);
    return fogged + (1.0 - fogged) * (1.0 - exp(-depth));
}

// Narkowicz's fit of the ACES filmic curve.
fn filmic(x: vec3<f32>) -> vec3<f32> {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// The grade of every shaded color: exposure; the
// filmic curve, half per channel and half on the brightest channel, which
// keeps the hue and saturation of bright colors; contrast as an S-curve in a
// perceptual (square root) space that keeps black and white, so dark shade
// deepens without crushing; split toning of shadows and highlights; then
// saturation, and vibrance that lifts muted colors more than vivid ones.
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

// Glow weight of light `color`: its exposed brightness above the bloom
// threshold, as x / (1 + x).
fn glow_weight(color: vec3<f32>) -> f32 {
    let bright = max(luminance(color) * lighting.horizon.w - lighting.effects.y, 0.0);
    return bright / (1.0 + bright);
}

// Visibility of the sun at `world` in the far maps: the kept one of still
// casters and the one redrawn every frame; 1 outside them.
fn far_visibility(world: vec3<f32>) -> f32 {
    let clip = shadow.far * vec4<f32>(world, 1.0);
    if any(abs(clip.xy) >= vec2<f32>(1.0)) || clip.z >= 1.0 {
        return 1.0;
    }
    let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
    // Where the kept map shadows, the moving one leaves the minimum at 0.
    var kept = 1.0;
    if shadow.state.y != 0.0 {
        kept = textureSampleCompareLevel(kept_shadows, shadow_sampler, uv, clip.z);
        if kept == 0.0 {
            return 0.0;
        }
    }
    return min(textureSampleCompareLevel(moving_shadows, shadow_sampler, uv, clip.z), kept);
}

// Receiver planes: a surface's lookup compares each of the
// four texels around it with the surface's own plane at that texel's center.
// A tilted surface then matches its own depths, so lookups need no margin
// that grows with texel size, which would keep every caster's lowest part
// (all of a low one) from shadowing what it stands on. Coarser maps then
// only make shadow edges blockier. The margin left covers depth rounding.
const RECEIVER_MARGIN: f32 = 2.0 / 65535.0;

// A receiver in a shadow map: its position in texels, its depth, and how its
// depth changes per texel along the map's axes.
struct MapPoint {
    texel: vec2<f32>,
    depth: f32,
    slope: vec2<f32>,
}

// How depth changes per texel along a map's axes where the surface moves
// by `map_dx` and `map_dy` per pixel in the map's coordinates, `scale`
// texels per unit of x and y; within `limit` depth per texel where the
// surface turns edge-on to the sun.
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

// `world` in the map `clip_from_world`, `size` texels a side. The plane
// comes from how the surface moves per pixel (`dx` and `dy`); its slope stays
// within `limit` depth per texel where the surface turns edge-on to the sun.
fn map_point(clip_from_world: mat4x4<f32>, world: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        size: f32, limit: f32) -> MapPoint {
    let clip = clip_from_world * vec4<f32>(world, 1.0);
    // Texel rows run downward.
    var point: MapPoint;
    point.texel = (clip.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5)) * size;
    point.depth = clip.z;
    point.slope = texel_slope(clip_from_world * vec4<f32>(dx, 0.0),
        clip_from_world * vec4<f32>(dy, 0.0), vec2<f32>(0.5, -0.5) * size, limit);
    return point;
}

// Where to gather the four texels around `point` in a map `size` texels a
// side: their shared corner, so the hardware's limited subtexel precision
// cannot pick a neighboring block near texel edges.
fn gather_uv(point: MapPoint, size: f32) -> vec2<f32> {
    return (floor(point.texel - vec2<f32>(0.5)) + vec2<f32>(1.0)) / size;
}

// Visibility of `point` from the four texel depths around it, gathered at
// `gather_uv`, in gather order (least u and greatest v, both greatest,
// greatest u and least v, both least), each compared with the receiver's
// plane at the texel's center and weighted bilinearly.
fn plane_visibility(depths: vec4<f32>, point: MapPoint) -> f32 {
    let corner = point.texel - vec2<f32>(0.5);
    let f = corner - floor(corner);
    let du = vec4<f32>(-f.x, 1.0 - f.x, 1.0 - f.x, -f.x);
    let dv = vec4<f32>(1.0 - f.y, 1.0 - f.y, -f.y, -f.y);
    let plane = point.depth + point.slope.x * du + point.slope.y * dv - RECEIVER_MARGIN;
    let lit = select(vec4<f32>(0.0), vec4<f32>(1.0), plane <= depths);
    return mix(mix(lit.w, lit.z, f.x), mix(lit.x, lit.y, f.x), f.y);
}

// Visibility of the surface at `world` in near cascade `cascade`,
// looked up `normal` times its offset away; -1 where the cascade does
// not cover the lookup.
fn cascade_visibility(cascade: u32, world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    let size = shadow.sizes.x;
    let point = map_point(shadow.near[cascade], world + normal * shadow.near_offsets[cascade],
        dx, dy, size, shadow.near_limits[cascade]);
    let uv = point.texel / size;
    if any(abs(uv * 2.0 - vec2<f32>(1.0)) >= vec2<f32>(0.98)) || point.depth >= 1.0 {
        return -1.0;
    }
    let corner = gather_uv(point, size);
    return plane_visibility(textureGather(near_shadows, shadow_texels, corner, cascade), point);
}

// Kept levels hand over to the next coarser map across this
// share of their width inside their windows' edges.
const LEVEL_HANDOVER: f32 = 0.08;
// The last near cascade hands over to the far maps across this share of its
// reach.
const CASCADE_HANDOVER: f32 = 0.3;

// The four depths around `point` in kept level `level`, `size` texels a
// side, in gather order. The map wraps: texels past one edge are those at
// the other.
fn level_depths(level: u32, point: MapPoint, size: f32) -> vec4<f32> {
    let corner = floor(point.texel - vec2<f32>(0.5));
    if all(corner >= vec2<f32>(0.0)) && all(corner < vec2<f32>(size - 1.0)) {
        return textureGather(kept_levels, shadow_texels, (corner + vec2<f32>(1.0)) / size, level);
    }
    let last = vec2<i32>(i32(size) - 1);
    let low = vec2<i32>(corner) & last;
    let high = (vec2<i32>(corner) + vec2<i32>(1)) & last;
    return vec4<f32>(textureLoad(kept_levels, vec2<i32>(low.x, high.y), level, 0),
        textureLoad(kept_levels, high, level, 0),
        textureLoad(kept_levels, vec2<i32>(high.x, low.y), level, 0),
        textureLoad(kept_levels, low, level, 0));
}

// Visibility of still casters' sun at the surface point `world` in the kept
// levels, each looked up its offset along `normal`: from the
// finest level whose window and band hold it on a tile holding its casters,
// handing over to the next across the window's edge and as its texels
// shrink below a pixel's footprint. x: the visibility found,
// weighted; y: the weight left to coarser maps (1 where no level holds the
// point).
fn level_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> vec2<f32> {
    let count = u32(shadow.level_state.x);
    if count == 0u {
        return vec2<f32>(0.0, 1.0);
    }
    // Light space once; each level scales it.
    let axes = mat3x3<f32>(shadow.light[0].xyz, shadow.light[1].xyz, shadow.light[2].xyz);
    let light = world * axes;
    let light_normal = normal * axes;
    let light_dx = dx * axes;
    let light_dy = dy * axes;
    let size = shadow.level_state.y;
    let pending = u32(shadow.level_state.z);
    // A level whose texels are under half a pixel's footprint shows no more
    // than the coarser maps after it, which read cheaper: lookups hand over
    // to the next map as its texels shrink from 0.7 to 0.35 of the
    // footprint, as textures blend mipmaps. Windows' offsets are half their
    // texels.
    let footprint = max(pixel_footprint(world, normal), 1.0e-6);
    var visible = 0.0;
    var left = 1.0;
    for (var level = 0u; level < count; level++) {
        let window = shadow.level_windows[level];
        let fine = saturate((2.0 * window.z / footprint - 0.35) / 0.35);
        if fine <= 0.0 {
            continue;
        }
        let scales = shadow.level_scales[level];
        let at = light + light_normal * window.z;
        let position = vec3<f32>(at.xy * scales.x, (at.z - scales.y) * scales.z);
        let local = position.xy - window.xy;
        let margin = 2.0 / size;
        if any(local < vec2<f32>(margin)) || any(local > vec2<f32>(1.0 - margin))
            || position.z <= 0.0 || position.z >= 1.0 {
            continue;
        }
        if (pending & (1u << level)) != 0u {
            let slot = vec2<u32>(vec2<i32>(floor(position.xy * 16.0)) & vec2<i32>(15));
            let bit = slot.y * 16u + slot.x;
            let words = shadow.level_valid[level * 2u + bit / 128u];
            if (words[(bit / 32u) % 4u] & (1u << (bit % 32u))) == 0u {
                continue;
            }
        }
        // Texel rows run downward, and the map wraps.
        var point: MapPoint;
        point.texel = vec2<f32>(fract(position.x), 1.0 - fract(position.y)) * size;
        point.depth = position.z;
        let level_scale = vec3<f32>(scales.x, scales.x, scales.z);
        point.slope = texel_slope(vec4<f32>(light_dx * level_scale, 0.0),
            vec4<f32>(light_dy * level_scale, 0.0), vec2<f32>(1.0, -1.0) * size, window.w);
        let here = plane_visibility(level_depths(level, point, size), point);
        let edge = min(min(local.x, local.y), min(1.0 - local.x, 1.0 - local.y));
        let weight = (1.0 - saturate(1.0 - edge / LEVEL_HANDOVER)) * fine;
        visible += left * weight * here;
        left *= 1.0 - weight;
        if left <= 0.0 {
            break;
        }
    }
    return vec2<f32>(visible, left);
}

// Visibility of the surface at `world` beyond the near cascades: of the
// moving map's casters and of still ones, which the kept levels
// hold where they reach and the kept far map elsewhere. Far maps are
// looked up their offset along `normal` and see nothing outside them.
fn far_surface_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>) -> f32 {
    let size = shadow.sizes.y;
    let point = map_point(shadow.far, world + normal * shadow.offsets.y, dx, dy, size,
        shadow.offsets.z);
    let uv = point.texel / size;
    let inside = all(abs(uv * 2.0 - vec2<f32>(1.0)) < vec2<f32>(1.0)) && point.depth < 1.0;
    let corner = gather_uv(point, size);
    var still = 1.0;
    if shadow.state.y != 0.0 {
        // Still casters first: where they leave nothing, the moving ones
        // cannot leave less.
        let levels = level_visibility(world, normal, dx, dy);
        still = levels.x;
        if levels.y > 0.0 {
            var kept = 1.0;
            if inside {
                kept = plane_visibility(textureGather(kept_shadows, shadow_texels, corner), point);
            }
            still += levels.y * kept;
        }
        if still == 0.0 {
            return 0.0;
        }
    }
    var moving = 1.0;
    if inside {
        moving = plane_visibility(textureGather(moving_shadows, shadow_texels, corner), point);
    }
    if shadow.state.y == 0.0 {
        return moving;
    }
    return min(moving, still);
}

// The share of the sun terrains leave unblocked at `world`:
// below a terrain's horizon there, none; above it, all; between, a soft edge
// growing with the distance to the slope casting it. Heights count from the
// map's ground, which terrain's own surfaces (`on_terrain`) lie on whatever
// detail they draw at.
fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
    // Their count: the records' texels over two (`level_state.w`).
    let count = u32(shadow.level_state.w);
    var visible = 1.0;
    for (var index = 0u; index < count; index++) {
        let place = textureLoad(terrain_horizon_records, vec2<u32>(2u * index, 0u), 0);
        let shape = textureLoad(terrain_horizon_records, vec2<u32>(2u * index + 1u, 0u), 0);
        let relative = world.xz - place.xy;
        let along = vec2<f32>(dot(relative, place.zw),
            dot(relative, vec2<f32>(-place.w, place.z)));
        let uv = along / shape.xy * 0.5 + 0.5;
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            continue;
        }
        let texel = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv, index, 0.0);
        let elevation = select(world.y - texel.z, 0.0, on_terrain);
        let edge = max(texel.y * shape.z, 0.5);
        visible = min(visible, smoothstep(-edge, edge, elevation + shape.w - texel.x));
        // Below one horizon, no other lets the sun back.
        if visible <= 0.0 {
            break;
        }
    }
    return visible;
}

// Visibility of the sun at the surface point `world`, whose
// position changes by `dx` and `dy` per pixel, looked up a little along the
// surface normal: what terrains' horizons leave of it, and of
// that what the shadow maps leave. `on_terrain`: the point is terrain's.
fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32, on_terrain: bool) -> f32 {
    if shadow.state.x == 0.0 {
        return cloud_passing(world);
    }
    let terrain = terrain_visibility(world, on_terrain);
    if terrain <= 0.0 {
        return 0.0;
    }
    // Clouds shade whatever else does: only what the terrain
    // and the maps leave.
    let visible = min(terrain, map_visibility(world, normal, dx, dy, distance));
    if visible == 0.0 {
        return 0.0;
    }
    return visible * cloud_passing(world);
}

// Clouds' shadows: CLOUD_SHADOW_CELLS by CLOUD_SHADOW_CELLS
// cells of the sunlight passing the cloud layer up from its base, two cells
// along x a word in half floats.
const CLOUD_SHADOW_CELLS: u32 = 128u;
@group(0) @binding(19) var<storage, read> cloud_shadows: array<u32>;

fn cloud_shadow_cell(x: u32, z: u32) -> f32 {
    let index = z * CLOUD_SHADOW_CELLS + x;
    return unpack2x16float(cloud_shadows[index / 2u])[index % 2u];
}

// The share of the sun's light the clouds let through to `world`: from the
// grid where the sun's ray from it meets the clouds' base, fading to all of
// it toward the grid's edges. The grid's first cell is
// `lighting.cloud_shadow.xy` (world cell indices); each cell keeps the slot
// of its index modulo the cells.
fn cloud_passing(world: vec3<f32>) -> f32 {
    let strength = lighting.cloud_shadow.w;
    if strength <= 0.0 || lighting.sun.y <= 0.01 {
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
    // The cells are a power of two: the wrap is the low bits.
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

// Visibility of the sun at `world` in the far maps, fading out at the
// shadow distance.
fn far_maps_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32) -> f32 {
    let fade = saturate((distance - shadow.state.w) / (shadow.offsets.x - shadow.state.w));
    // Faded out entirely, every visibility mixes to all of it.
    if fade >= 1.0 {
        return 1.0;
    }
    let visible = far_surface_visibility(world, normal, dx, dy);
    return mix(visible, 1.0, fade);
}

// Visibility of the sun at `world` in the shadow maps. Near cascades cover
// the first stretch of the view, each three times sharper than the next,
// and hand over to the next one across the last tenth of
// their reach, the last one to the far maps across `CASCADE_HANDOVER` of
// its reach; the far maps, the kept one for still casters
// and the one redrawn every frame, cover the rest and fade out at the
// shadow distance. Lookups compare four texels with the surface's plane.
fn map_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32) -> f32 {
    // The first cascade reaching the fragment. Reflection probe captures
    // look from elsewhere than the cascades cover and select
    // for: they read the far maps alone.
    let beyond = select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<f32>(distance) >= shadow.reach);
    let cascade = u32(dot(beyond, vec4<f32>(1.0)));
    let cascades = select(u32(shadow.state.z), 0u, camera.options.x != 0.0);
    if cascade < cascades {
        let visible = cascade_visibility(cascade, world, normal, dx, dy);
        if visible >= 0.0 {
            let reach = shadow.reach[cascade];
            let last = cascade + 1u == cascades;
            let band = select(0.1, CASCADE_HANDOVER, last);
            let handover = saturate((distance - reach * (1.0 - band)) / (reach * band));
            if handover <= 0.0 {
                return visible;
            }
            if last {
                return mix(visible, far_maps_visibility(world, normal, dx, dy, distance),
                    handover);
            }
            let next = cascade_visibility(cascade + 1u, world, normal, dx, dy);
            if next >= 0.0 {
                return mix(visible, next, handover);
            }
            return visible;
        }
    }
    return far_maps_visibility(world, normal, dx, dy, distance);
}

// Trowbridge-Reitz (GGX) distribution of microfacet normals.
fn distribution(n_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let d = n_h * n_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

// Smith-GGX visibility (Hammon's approximation of the correlated form).
fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    return 0.5 / mix(2.0 * n_l * n_v, n_l + n_v, alpha);
}

fn fresnel(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    let m = 1.0 - cosine;
    let m2 = m * m;
    return f0 + (vec3<f32>(1.0) - f0) * (m2 * m2 * m);
}

// Karis's analytic fit of the split-sum environment BRDF.
fn environment_brdf(f0: vec3<f32>, roughness: f32, n_v: f32) -> vec3<f32> {
    let r = roughness * vec4<f32>(-1.0, -0.0275, -0.572, 0.022)
        + vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let a004 = min(r.x * r.x, exp2(-9.28 * n_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// Interleaved gradient noise (Jimenez) at pixel `position`: in [0, 1),
// fixed per pixel.
fn pixel_noise(position: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))));
}

// Weighted blended order-independent transparency (McGuire and Bavoil):
// premultiplied color and coverage times a depth weight, added; and the product
// of what each surface lets through.
struct TransparentTargets {
    @location(0) accumulated: vec4<f32>,
    @location(1) revealed: f32,
}

// A transparent contribution of graded `color` covering `opacity` of what
// is behind it, `distance` from the eye: nearer surfaces weigh more.
fn weighted(color: vec3<f32>, opacity: f32, distance: f32) -> TransparentTargets {
    let near = distance / 5.0;
    let far = distance / 200.0;
    let far3 = far * far * far;
    let weight = opacity * clamp(10.0 / (1.0e-5 + near * near + far3 * far3), 0.01, 3000.0);
    var targets: TransparentTargets;
    targets.accumulated = vec4<f32>(color * opacity, opacity) * weight;
    targets.revealed = opacity;
    return targets;
}


// Optional view-only medium. The default module specializes this to false,
// including all buffer accesses below. No ninth storage binding is required.
const BASIN_FOG: bool = false;
const BASIN_PREFIX: u32 = CLOUD_SHADOW_CELLS * CLOUD_SHADOW_CELLS / 2u;

fn basin_float(word: u32) -> f32 { return bitcast<f32>(cloud_shadows[word]); }

// Signed plane distance divided by direction, clamped before division can
// overflow. Half differences also keep subtraction finite at large positions.
fn basin_plane(bound: f32, eye: f32, direction: f32, reach: f32) -> f32 {
    let half_delta = bound * 0.5 - eye * 0.5;
    if abs(half_delta) >= (reach * 0.5) * abs(direction) {
        return select(-reach, reach, (half_delta >= 0.0) == (direction >= 0.0));
    }
    return (half_delta / direction) * 2.0;
}

fn basin_interval(eye: vec3<f32>, ray: vec3<f32>, reach: f32,
        low: vec3<f32>, high: vec3<f32>) -> vec2<f32> {
    var interval = vec2<f32>(0.0, reach);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if ray[axis] == 0.0 {
            // Half-open horizontal cells avoid duplicate depth at an axial
            // edge. The vertical medium excludes its floor as well as its lid.
            if eye[axis] < low[axis] || eye[axis] >= high[axis] { return vec2<f32>(0.0); }
            if axis == 1u && eye.y == low.y { return vec2<f32>(0.0); }
        } else {
            let a = basin_plane(low[axis],eye[axis],ray[axis],reach);
            let b = basin_plane(high[axis],eye[axis],ray[axis],reach);
            interval = vec2<f32>(max(interval.x,min(a,b)),min(interval.y,max(a,b)));
        }
    }
    return interval;
}

// Density averaged over a segment already clipped to floor/top. Splitting
// at top-fade makes each segment constant or linear, integrated by a trapezoid.
fn basin_density(height: f32, top: f32, fade: f32) -> f32 {
    if height >= top { return 0.0; }
    // Compare half differences before division; even an extreme finite fade
    // or elevation cannot overflow the numerator or produce a ratio above 1.
    let half_depth = max(top*0.5-height*0.5,0.0);
    let half_fade = fade*0.5;
    if half_depth >= half_fade { return 1.0; }
    return half_depth/half_fade;
}
fn basin_average(a: f32, b: f32, top: f32, fade: f32) -> f32 {
    if fade == 0.0 { return 1.0; }
    return 0.5 * (basin_density(a,top,fade) + basin_density(b,top,fade));
}
fn basin_segment(length: f32, average: f32, sigma: f32) -> f32 {
    let weighted = length * average;
    if sigma == 0.0 || weighted == 0.0 { return 0.0; }
    if sigma > 2.351e-37 && weighted >= 80.0 / sigma { return 80.0; }
    return weighted * sigma;
}

fn basin_ray_passing(eye: vec3<f32>, ray: vec3<f32>, reach: f32) -> f32 {
    if !BASIN_FOG || camera.options.x != 0.0 || reach <= 0.0 { return 1.0; }
    let size = arrayLength(&cloud_shadows);
    if size < BASIN_PREFIX + 4u { return 1.0; }
    let end = cloud_shadows[BASIN_PREFIX];
    let sigma = basin_float(BASIN_PREFIX+2u);
    let fade = basin_float(BASIN_PREFIX+3u);
    if sigma == 0.0 || end > size { return 1.0; }
    var word = BASIN_PREFIX+4u;
    var depth = 0.0;
    loop {
        if word >= end || end-word < 12u { break; }
        let escape = cloud_shadows[word+3u];
        if escape <= word || escape > end { break; }
        let low = vec3<f32>(basin_float(word),basin_float(word+1u),basin_float(word+2u));
        let high = vec3<f32>(basin_float(word+4u),basin_float(word+5u),basin_float(word+6u));
        let interval = basin_interval(eye,ray,reach,low,high);
        if interval.y <= interval.x { word = escape; continue; }
        if cloud_shadows[word+7u] == 0u { word += 12u; continue; }
        let top = high.y;
        var split = interval.x;
        // A fade breakpoint below the floor cannot split this interval.
        // This guard also keeps top-fade within the finite cell bounds.
        if ray.y != 0.0 && fade > 0.0 && fade*0.5 < top*0.5-low.y*0.5 {
            split = clamp(basin_plane(top-fade,eye.y,ray.y,reach),interval.x,interval.y);
        }
        let y0 = clamp(eye.y+ray.y*interval.x,low.y,high.y);
        let ys = clamp(eye.y+ray.y*split,low.y,high.y);
        let y1 = clamp(eye.y+ray.y*interval.y,low.y,high.y);
        let first = basin_segment(split-interval.x,basin_average(y0,ys,top,fade),sigma);
        let last = basin_segment(interval.y-split,basin_average(ys,y1,top,fade),sigma);
        depth = min(depth+first+last,80.0);
        if depth >= 80.0 { break; }
        word = escape;
    }
    return exp(-depth);
}
fn basin_passing(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    if !BASIN_FOG { return 1.0; }
    let delta = point-eye;
    let distance = length(delta);
    if distance == 0.0 { return 1.0; }
    return basin_ray_passing(eye,delta/distance,distance);
}
fn basin_cover(eye: vec3<f32>, point: vec3<f32>) -> f32 {
    let old = fog(eye,point);
    if !BASIN_FOG { return old; }
    return 1.0-(1.0-old)*basin_passing(eye,point);
}
