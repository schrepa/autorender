// Shared scene declarations, prepended to the mesh, particle and sun shaft
// shaders: the first bind group's camera, atmosphere and sun
// shadows, and the lighting functions every path shades with.

// Uniform: three column-major mat4 and six vec4, 16-byte alignment,
// exactly 288 bytes. `view_projection` is the rasterized projection; temporally
// antialiased frames jitter it by `jitter.xy` clip units per w.
// Their motion uses the previous frame's unjittered `previous_view_projection`
// and `jitter.zw`, two over the render size in pixels. The unjittered
// inverse turns pixels into sky rays; `position.xyz` is the eye and
// `position.w` presentation time in seconds. `options.x` is 1
// while a reflection probe captures a cube face; `options.y`
// is the angle a display pixel spans at the view's center (radians), which
// `detail_across` measures pixels by; `options.w` the mip bias material
// textures sample with (`material_bias`). `wind` is
// the wind's velocity (x and z), gust strength and the previous frame's
// presentation time, which temporal frames' motion sways by. `region` is
// the frame's render size in pixels, which it draws at from the top-left
// corner of its screen-sized targets (the viewport), and that size's share
// of the display on each axis. `detail.xyz` is the detail eye every
// handover measures from (`detail_distance`, `detail_across`,
// `detail_span`): the eye, or the point `TerrainDebug::freeze` holds;
// `detail.w` is unused.
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
    // Contrast and vibrance; 1 with the physically based sky; 1 when 8-bit
    // color writes dither.
    grade: vec4<f32>,
    // Split toning: luminance-normalized gains of shadows and highlights.
    shadow_tint: vec4<f32>,
    highlight_tint: vec4<f32>,
    // Refraction through glass: the bend's strength (0 without
    // refraction) and the blur radius in pixels of fully rough glass.
    refraction: vec4<f32>,
    // Aerial perspective: start, density at height zero,
    // height falloff and the sun's glow; the air's own color, and 1 with
    // haze (w).
    aerial: vec4<f32>,
    aerial_color: vec4<f32>,
    // Occupancy: one over the finest map's side (meters), its
    // texels a side, the meters around the detail eye within which it holds
    // every point (each coarser map twice as far) and the strength covers
    // darken by (0 without maps); the maps' count and the reference height
    // their heights are above; each map's window's center (x and z), two
    // maps a vector.
    occupancy: vec4<f32>,
    occupancy_maps: vec4<f32>,
    occupancy_centers: array<vec4<f32>, 4>,
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
    // The night sky: the stars' brightness (0 without them), the moon's (0
    // without one), how much of the sky cirrus covers and the share of the
    // moon's face the sun lights; the quaternion turning world directions
    // into the galaxy's frame (x toward its centre, z its north pole); the
    // direction toward the moon.
    night: vec4<f32>,
    star_turn: vec4<f32>,
    moon: vec4<f32>,
    // Foliage: its transmission, its plants' variation, 1 where masked
    // leaves dissolve with hashed cutoffs this frame, and the meters its
    // bases blend into the ground over; all 0 off.
    foliage: vec4<f32>,
    // Highlights sized by their lights: the cosine and sine of the sun's
    // angular radius (1 and 0 for a point sun, both 0 while highlights are
    // neither sized nor capped), the exposed brightness local lights'
    // highlights ease toward (0 caps none), and the radius of local lights'
    // smallest sphere.
    highlights: vec4<f32>,
    // Windows lighting room by room: the sun's elevation, the
    // elevation rooms switch on around and how far either side they spread
    // (0 while windows are lit as their textures light them), all in
    // degrees, and the share of rooms staying dark.
    windows: vec4<f32>,
    night_shape: vec4<f32>,
}
// Sun shadows: a clipmap of levels around the detail eye, all in light
// space (`shadows/mod.rs`), the finest level's texels `level_state.y` wide
// and each next level's twice as wide; each level a window of tiles
// `level_state.z` texels a side, each tile drawn in a slot of one atlas.
struct Shadow {
    // World to light space: the sun's right, up and away-from-sun axes.
    light: array<vec4<f32>, 3>,
    // x: shadows on; y: meters into crowns the sun reaches
    // (`crown_softness`).
    state: vec4<f32>,
    // x: levels in use; y: the finest level's texel (world units); z:
    // texels along a tile's side; w: the terrains whose horizon records
    // lookups loop over.
    level_state: vec4<f32>,
    // x: tiles along a side of each level's window; y: slots along a row
    // of the atlas; zw: one over the atlas's width and height in texels.
    atlas: vec4<f32>,
    // Each level's window: light-space x and y of its first tile's corner,
    // its tiles' width and its texels' (world units).
    windows: array<vec4<f32>, SHADOW_LEVELS>,
    // Each level's depth band: its nearest depth from the sun and one over
    // its depth (z in [0, 1] over it), the normal offset of its lookups
    // (world units) and its steepest receiver plane (depth per texel).
    bands: array<vec4<f32>, SHADOW_LEVELS>,
}
// Levels of the clipmap the uniform holds, and the texels a tile's slot
// holds past its tile on each side, drawn with it.
const SHADOW_LEVELS: u32 = 20u;
const SHADOW_APRON: u32 = 1u;
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> lighting: Lighting;
@group(0) @binding(2) var<uniform> shadow: Shadow;
// The atlas of the clipmap's drawn tiles, each in a slot of its texels with
// SHADOW_APRON more on each side; slots run along rows of `shadow.atlas.y`,
// the rows downward, and texel rows run downward in each slot.
@group(0) @binding(3) var shadow_atlas: texture_depth_2d;
// The slots each level's tiles lie in: the tile of level `l` whose absolute
// tile is `(a, b)`, cell `(a, b)` modulo the window's side, at texel
// `(a mod side, l * side + b mod side)`. One more than its slot in the low
// half (0 where it is not drawn), and than the slot moving casters drew
// over it this frame in the high half (0 where they drew none).
@group(0) @binding(4) var shadow_tiles: texture_2d<u32>;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
// Unfiltered reads of four shadow texels, for receiver planes.
@group(0) @binding(7) var shadow_texels: sampler;
// Terrain's long sun shadows: each terrain's horizon map, a
// grid over the terrain cooked with its lighting, HORIZON_LAYERS layers per
// terrain. In each of HORIZON_DIRECTIONS directions around a texel
// (direction k a k-th of the directions' turn from +x toward +z), how high
// above its ground and how far away the crest of the highest ground ahead
// lies (meters): layer l holds directions 2l and 2l + 1, their heights in
// red and green and their distances in blue and alpha. The last layer holds
// the ground's height above the terrain's lowest (red). With bounce light,
// the bounce layers of each terrain it relights (by slot:
// surroundings hold none) follow every terrain's horizon layers, each of
// its levels' windows in a region of them (`bounce_light`).
@group(0) @binding(14) var terrain_horizons: texture_2d_array<f32>;
// HORIZON_RECORD_VECTORS vectors per terrain: its map's first corner (world
// x and z) and one over its extent (x and z, 0 without a map); its first
// layer, how fast its shadows soften per meter to their crests, the meters
// receivers rise before comparing and the lowest ground's world height; its
// bounce layers' first (-1 without bounce light), its finest level's probe
// columns along a side, its levels and its windows' columns along a side;
// one over the maps' side, the meters around the detail eye within which
// its finest level's window holds every point, the levels' regions a row
// of the bounce layers holds and its finest level relit; and each level's
// window's first column, x plus BOUNCE_WINDOW_ROW times z, four levels a
// vector.
const HORIZON_RECORDS: u32 = 384u;
@group(0) @binding(15) var<uniform> terrain_horizon_records: array<vec4<f32>, HORIZON_RECORDS>;
@group(0) @binding(16) var terrain_horizon_sampler: sampler;
// What placed objects cover around the detail eye, a layer per map, the
// finest first: how much covers hide over each texel, and their highest
// top and lowest bottom (negated), above the maps' reference height.
@group(0) @binding(17) var occupancy_coverage: texture_2d_array<f32>;
@group(0) @binding(18) var occupancy_heights: texture_2d_array<f32>;
// Environment maps: cubes of the light around, blurred down
// their levels by roughness; the first holds the sky.
@group(0) @binding(11) var environment_maps: texture_cube_array<f32>;
@group(0) @binding(12) var environment_sampler: sampler;

// The features of the scene shading looks up (`scene_shading::SceneShading`):
// sun shadows' clipmap, terrains' horizons, clouds' shadows, placed objects'
// covers, reflection probes, windows lighting room by room, and terrains'
// bounce light. One switched off compiles its lookups out of the pipelines
// given it, which then leave what its absent uniforms and bindings would;
// pipelines not given these keep every lookup.
override SHADOW_MAPS: bool = true;
override TERRAIN_HORIZONS: bool = true;
override CLOUD_SHADOWS: bool = true;
override OCCUPANCY: bool = true;
override PROBES: bool = true;
override WINDOWS: bool = true;
override BOUNCE: bool = true;

// How much the covers in occupancy map `map` hide the sky from `world`:
// their cover there, the more the deeper below their top; none above them;
// -1 outside the map's window. Each map is twice as wide as the one inside
// it, and wraps: a point's texels are where it falls in its map-wide page,
// those past one edge of the map at the other.
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

// The share of the sky light placed objects' covers leave at `world`:
// under a cover, its density's worth less; within it, less the deeper
// below its top; above it or outside the maps, all of it. The map the
// point's distance from the detail eye picks gives it (`detail_level` over
// half the finest map's reach), or the finest from there whose window holds
// the point, blended with the next coarser one as the levels hand over;
// past the coarsest a level holding no cover, so covers fade out over the
// outer half of its reach. A map's texels are the mean of the finer one's
// where they overlap, so a handover changes how sharp covers show, not how
// much.
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

// The world position of the point `relative` to the eye. Shaders turn the
// eye-relative positions their stages pass back into world positions by it
// alone, which `cargo xtask gpu`'s handover check knows to depend on where
// the camera stands no more.
fn eye_world(relative: vec3<f32>) -> vec3<f32> {
    return relative + camera.position.xyz;
}

// The facing cosine under which every measure of how a view meets a surface
// counts it as seen edge on: a pixel spans at most sixteen times as far
// along the view as across it.
const DETAIL_EDGE_ON: f32 = 0.0625;

// The meters from the detail eye (`camera.detail`) to `world`, which every
// handover by distance measures.
fn detail_distance(world: vec3<f32>) -> f32 {
    return length(world - camera.detail.xyz);
}

// The meters a pixel spans across the view at `world`, seen from the detail
// eye: its distance times the angle a pixel spans (`camera.options.y`),
// never the view's depth, so it stays where it stands however the view
// turns or moves, frozen or not. Every handover by size on screen measures
// it; texture filtering keeps the view's own derivatives.
fn detail_across(world: vec3<f32>) -> f32 {
    return detail_distance(world) * camera.options.y;
}

// The meters a pixel spans on a surface at `world` facing `normal`, seen
// from the detail eye: across the view (x, `detail_across`), and along it
// (y), as far again as the eye meets the surface obliquely, views more
// grazing than DETAIL_EDGE_ON counting as that. A feature wanting one size
// takes the mean of the two sides, `sqrt(x * y)`.
fn detail_span(world: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let across = detail_across(world);
    let toward = world - camera.detail.xyz;
    let facing = max(abs(dot(normal, toward)) / max(length(toward), 1.0e-6), DETAIL_EDGE_ON);
    return vec2<f32>(across, across / facing);
}

// The level of a pyramid a pixel reads, and the share of the next coarser
// level blended in, as brilinear filtering blends mipmaps: over the upper
// half of each level's octave alone, so the lower half reads one level.
struct DetailLevel {
    level: u32,
    blend: f32,
}

// The level of a pyramid of `levels` levels whose level 0 spaces its
// texels `base` meters apart that a pixel spanning `span` meters reads:
// log2 of `span` over `base`, within the pyramid; the next level's share
// rising from none halfway through its octave to all of it at its end;
// none past the coarsest.
fn detail_level(span: f32, base: f32, levels: u32) -> DetailLevel {
    let coarsest = f32(max(levels, 1u) - 1u);
    let at = clamp(log2(max(span / max(base, 1.0e-30), 1.0e-30)), 0.0, coarsest);
    var chosen: DetailLevel;
    chosen.level = u32(floor(at));
    chosen.blend = saturate(2.0 * (at - floor(at)) - 1.0);
    return chosen;
}

// The mip bias material textures sample with (`camera.options.w`): log2 of
// the render size's share of the display, 0 drawing the whole display and
// negative below it, so frames drawing smaller, which upscale, keep their
// textures as sharp as the display shows them.
fn material_bias() -> f32 {
    return camera.options.w;
}

// The render size's share of the display (`camera.region.z`): how much
// smaller than the screen derivatives they pick mips by shaders choosing
// their own scale them, as `material_bias` samples.
fn material_detail() -> f32 {
    return camera.region.z;
}

fn stable_normalize(value: vec3<f32>) -> vec3<f32> {
    let scale = maximum_abs(value);
    // Interpolation between opposing vertex normals can produce a zero vector.
    if scale == 0.0 { // optics: a zero vector's guard, whatever it measures
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(value / scale);
}

// Whether 8-bit color writes dither (`lighting.grade.w`).
fn dithering() -> bool {
    return lighting.grade.w != 0.0;
}

// Graded linear color as the canvas stores it at `pixel`: WebGPU canvases
// are UNORM, so shaders drawing into them encode sRGB themselves, and dither
// it when 8-bit writes do, by the noise temporally antialiased frames turn.
fn encoded(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let value = encode_srgb(color);
    if !dithering() {
        return value;
    }
    return dithered(value, pixel, camera.options.z, 0u);
}

// Graded linear color as the sRGB scene target stores it at `pixel`: as it
// is, or when 8-bit writes dither, dithered in the encoding the target
// stores.
fn stored(color: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    if !dithering() {
        return color;
    }
    return decode_srgb(dithered(encode_srgb(color), pixel, camera.options.z, 0u));
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
    return mix(sky_above(normal), lighting.ground.rgb, max(-normal.y, 0.0));
}

// The light of the sky alone a surface of `normal` sees: `sky_average`
// without the ground's color, for surfaces that take the ground's own light
// (`bounce_light`).
fn sky_above(normal: vec3<f32>) -> vec3<f32> {
    let side = mix(lighting.horizon.rgb, lighting.zenith.rgb, 0.5);
    return mix(side, lighting.zenith.rgb, max(normal.y, 0.0));
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

// Haze: the share of the sky's environment map's levels its light behind
// distant surfaces is blurred to, and the least elevation (its sine) it is
// looked up at.
const HAZE_BLUR: f32 = 0.3;
const HAZE_LIFT: f32 = 0.02;

// Whether fog and the air take the sky's own light behind what they cover
// (haze, `lighting.aerial_color.w`).
fn hazy() -> bool {
    return lighting.aerial_color.w != 0.0;
}

// The sky's light behind distant surfaces along `direction`: its
// environment map blurred, looked up no lower than just above the horizon,
// so the ground below fades into the lit haze there, as the sky behind it
// does.
fn sky_behind(direction: vec3<f32>) -> vec3<f32> {
    let lifted = normalize(vec3<f32>(direction.x, max(direction.y, HAZE_LIFT), direction.z));
    let level = HAZE_BLUR * f32(textureNumLevels(environment_maps) - 1u);
    return textureSampleLevel(environment_maps, environment_sampler, lifted, 0, level).rgb;
}

// What fog and air leave of light `lit` from `point` seen
// from `eye` looking along `direction`, and how much of it they cover
// together (w). With haze both take the sky's light behind the point.
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
    let behind = hazy();
    var sky_light = vec3<f32>(0.0);
    if behind {
        sky_light = sky_behind(direction);
    }
    let fogged = fog(eye, point);
    // Without fog, mixing in none of its haze leaves the light.
    var color = light;
    if fogged != 0.0 {
        color = mix(light, select(haze(direction), sky_light, behind), fogged);
    }
    var covered = fogged;
    if lighting.aerial.y > 0.0 {
        let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
            lighting.aerial.z);
        // Where the air is no depth, mixing in none of it leaves the color.
        if depth != 0.0 {
            let air = 1.0 - exp(-depth);
            color = mix(color, select(air_light(direction), sky_light, behind), air);
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
    let fogged = fog(eye, point);
    if lighting.aerial.y <= 0.0 {
        return fogged;
    }
    let depth = height_depth(eye, point, lighting.aerial.x, lighting.aerial.y,
        lighting.aerial.z);
    return fogged + (1.0 - fogged) * (1.0 - exp(-depth));
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

// Receiver planes: a surface's lookup compares each of the
// four texels around it with the surface's own plane at that texel's center.
// A tilted surface then matches its own depths, so lookups need no margin
// that grows with texel size, which would keep every caster's lowest part
// (all of a low one) from shadowing what it stands on. Coarser maps then
// only make shadow edges blockier. The margin left covers depth rounding.
const RECEIVER_MARGIN: f32 = 2.0 / 65535.0;

// A receiver in a shadow map: its position in texels, its depth, how its
// depth changes per texel along the map's axes, and, inside a crown, the
// depth past the first caster over which its light halves (0: shadowed
// whole at once).
struct MapPoint {
    texel: vec2<f32>,
    depth: f32,
    slope: vec2<f32>,
    soft: f32,
}

// How much of the fragment being shaded lies inside a crown (its surface's
// `crown`), set around its sun lookup; 0 elsewhere.
var<private> crown_light: f32;

// The depth over which light halves inside a crown in a map whose depth
// changes by `per_meter` per meter along the sun: the crown depth's worth,
// as light reaches past a crown's first leaves.
fn crown_softness(per_meter: f32) -> f32 {
    return crown_light * shadow.state.y * abs(per_meter);
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

// The four depths of the atlas around `point`, in gather order, gathered
// at their shared corner, so the hardware's limited subtexel precision
// cannot pick a neighboring block near texel edges.
fn atlas_depths(point: MapPoint) -> vec4<f32> {
    let corner = floor(point.texel - vec2<f32>(0.5)) + vec2<f32>(1.0);
    return textureGather(shadow_atlas, shadow_texels, corner * shadow.atlas.zw);
}

// Visibility of `point` from the four texel depths around it, gathered by
// `atlas_depths`, in gather order (least u and greatest v, both greatest,
// greatest u and least v, both least), each compared with the receiver's
// plane at the texel's center and weighted bilinearly. A soft point's
// light halves every softness behind a texel's caster.
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

// `world` in light space: across the sun's view (x and y) and away from
// the sun (z).
fn light_space(world: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(shadow.light[0].xyz, world), dot(shadow.light[1].xyz, world),
        dot(shadow.light[2].xyz, world));
}

// The level of the clipmap a pixel's size picks at `world`, seen from the
// detail eye, and the share of the next coarser level blended in: the one
// whose texels match what the pixel spans across the view.
fn shadow_level(world: vec3<f32>) -> DetailLevel {
    return detail_level(detail_across(world), shadow.level_state.y, u32(shadow.level_state.x));
}

// Where a light-space point lies in one level of the clipmap: its tile's
// cell (its absolute tile modulo the window's side, as `shadow_tiles`
// holds it), the share across that tile from its first corner and its
// depth over the level's band; `inside` while the window and the band hold
// it.
struct LevelPoint {
    cell: vec2<u32>,
    across: vec2<f32>,
    depth: f32,
    inside: bool,
}

// Where light-space `light` lies in level `level`.
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

// The slots of `point`'s tile in level `level`, each one more than the
// slot: the tile's own (x) and what moving casters drew over it this frame
// (y); 0 for none.
fn level_slots(level: u32, point: LevelPoint) -> vec2<u32> {
    let side = u32(shadow.atlas.x);
    let entry = textureLoad(shadow_tiles, vec2<u32>(point.cell.x, level * side + point.cell.y),
        0).r;
    return vec2<u32>(entry & 0xffffu, entry >> 16u);
}

// The atlas texel at share `across` of the tile in slot `slot`, past its
// apron; texel rows run downward.
fn slot_texel(slot: u32, across: vec2<f32>) -> vec2<f32> {
    let texels = shadow.level_state.z;
    let side = texels + 2.0 * f32(SHADOW_APRON);
    let per_row = u32(shadow.atlas.y);
    let corner = vec2<f32>(vec2<u32>(slot % per_row, slot / per_row)) * side;
    return corner + vec2<f32>(f32(SHADOW_APRON)) + vec2<f32>(across.x, 1.0 - across.y) * texels;
}

// The meters the ground of four levels of the clipmap, from `first`,
// stands above the point a ground fragment shades (`meshes/terrain.wgsl`'s
// `terrain_kept_lift`), which it looks those levels up from: a level holds
// the ground at a level of detail its texels choose, whatever level the
// fragment's ground draws at, and the ground's own fragments meet it on
// that ground, so it never shadows itself. Zero for every other surface.
struct KeptLift {
    first: u32,
    lifts: vec4<f32>,
}
var<private> kept_lift: KeptLift;

// The meters level `level`'s ground stands above the point shaded.
fn level_lift(level: u32) -> f32 {
    if level < kept_lift.first || level - kept_lift.first >= 4u {
        return 0.0;
    }
    return kept_lift.lifts[level - kept_lift.first];
}

// Taps around a receiver inside a crown: as many, on a spiral out to
// CROWN_REACH crown depths, each a golden angle round from the last.
const CROWN_TAPS: u32 = 4u;
const CROWN_REACH: f32 = 2.0;
const CROWN_SPIRAL_TURN: f32 = 2.39996323;

// The sun a receiver inside a crown takes in level `level`, at light-space
// `light` as `point` there (in the tile `center` holds, its still casters
// in `slot`, one more than the slot), of which `visible` reaches it along
// the sun:
// also what comes from the crown's sun-facing silhouette across the level,
// halving every crown depth of the way across to a tap and down past the
// caster there, the most any tap finds; so leaves under a crown's outer
// tiers on its sun-facing side light as its top does, and its far side and
// middle stay dim.
fn crown_spread(level: u32, light: vec3<f32>, point: MapPoint, center: LevelPoint, slot: u32,
        visible: f32) -> f32 {
    let depth_per_meter = max(shadow.bands[level].y, 1.0e-12);
    let crown = point.soft / depth_per_meter;
    // The spiral turns with the receiver's texel, and across temporally
    // antialiased frames; each tap a golden angle round from the last.
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
        // Taps within the receiver's tile read its slot; those crossing
        // into another look that one's up.
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

// Visibility of the sun at the surface point `world` in level `level`,
// looked up its normal offset along `normal` and from the level's ground's
// height over the point (`kept_lift`), the receiver's plane from how the
// point moves per pixel (`dx` and `dy`): what still casters leave of it,
// spread into crowns where `spread`, and what the moving casters drawn
// over the tile this frame leave, the less; -1 where the level holds no
// drawn tile there.
fn level_visibility(level: u32, world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>,
        dy: vec3<f32>, spread: bool) -> f32 {
    let window = shadow.windows[level];
    let band = shadow.bands[level];
    // A meter up, in light space.
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
    // Light-space units across the view, and depth, per pixel; texels a
    // world unit, rows running downward.
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

// The share of the sun terrains leave unblocked at `world`: where the sun's
// ray toward it passes under a terrain's crest there, none; over it, all;
// between, a soft edge growing with the crest's distance. The horizon toward
// the sun lies between those of the directions either side of it, as the
// slopes up to their crests and the crests' distances do; the ray passes
// under it by that distance times how much the slope rises above the sun's.
// A surface above the map's ground compares its height above it with that,
// and terrain's own surfaces (`on_terrain`) lie on that ground whatever
// detail they draw at.
fn terrain_visibility(world: vec3<f32>, on_terrain: bool) -> f32 {
    if !TERRAIN_HORIZONS {
        return 1.0;
    }
    let sun = lighting.sun.xyz;
    let level = length(sun.xz);
    // The sun overhead clears every horizon.
    if level < 1.0e-4 {
        return 1.0;
    }
    // How far its ray rises per meter toward it, and between which two of
    // the directions it stands.
    let rise = sun.y / level;
    let turn = fract(atan2(sun.z, sun.x) / (2.0 * PI) + 1.0) * f32(HORIZON_DIRECTIONS);
    let first = min(u32(turn), HORIZON_DIRECTIONS - 1u);
    let second = (first + 1u) % HORIZON_DIRECTIONS;
    let between = turn - f32(first);
    // Their count (`level_state.w`), within the records' uniform.
    let count = min(u32(shadow.level_state.w), HORIZON_RECORDS / HORIZON_RECORD_VECTORS);
    var visible = 1.0;
    for (var index = 0u; index < count; index++) {
        let at = HORIZON_RECORD_VECTORS * index;
        let place = terrain_horizon_records[at];
        let shape = terrain_horizon_records[at + 1u];
        let uv = (world.xz - place.xy) * place.zw;
        if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) {
            continue;
        }
        let base = u32(shape.x);
        // An even first direction shares its layer with the second; an odd
        // one's second starts the next layer.
        let near = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
            base + first / 2u, 0.0);
        var far = near;
        if (first & 1u) == 1u {
            far = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
                base + second / 2u, 0.0);
        }
        let odd = (first & 1u) == 1u;
        let heights = vec2<f32>(select(near.r, near.g, odd), select(far.g, far.r, odd));
        let distances = vec2<f32>(select(near.b, near.a, odd), select(far.a, far.b, odd));
        // The slopes up to the crests either side of the sun (no crest
        // looks down past any sun), and the meters above the ground the
        // sun's ray toward the crest between passes under it.
        let slopes = max(heights / max(distances, vec2<f32>(1.0e-3)), vec2<f32>(-1.0));
        let distance = mix(distances.x, distances.y, between);
        let clearance = distance * (mix(slopes.x, slopes.y, between) - rise);
        var elevation = 0.0;
        if !on_terrain {
            let ground = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
                base + HORIZON_LAYERS - 1u, 0.0).r;
            elevation = world.y - shape.w - ground;
        }
        let edge = max(distance * shape.y, 0.5);
        visible = min(visible, smoothstep(-edge, edge, elevation + shape.z - clearance));
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
// that what the shadow clipmap leaves. `on_terrain`: the point is
// terrain's.
fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32, on_terrain: bool) -> f32 {
    if !SHADOW_MAPS || shadow.state.x == 0.0 {
        return cloud_passing(world);
    }
    let terrain = terrain_visibility(world, on_terrain);
    if terrain <= 0.0 {
        return 0.0;
    }
    // Clouds shade whatever else does: only what the terrain
    // and the clipmap leave.
    let visible = min(terrain, map_visibility(world, normal, dx, dy));
    if visible == 0.0 {
        return 0.0;
    }
    return visible * cloud_passing(world);
}

// Terrains' bounce light (`terrain_bounce.wgsl`): its probes' heights above
// the ground (meters), and the columns a packed window's first row counts
// in.
const BOUNCE_LOW: f32 = 1.5;
const BOUNCE_HIGH: f32 = 24.0;
const BOUNCE_WINDOW_ROW: u32 = 4096u;

// Bounce light at a point: the light the ground around sends it, per unit
// of its diffuse color; the share of the sky the ground leaves it; and 1
// where a terrain's probes hold the point, 0 elsewhere (and everywhere with
// bounce light off), where terrain takes none and what stands on it the
// open sky.
struct BounceLight {
    light: vec3<f32>,
    sky: f32,
    weight: f32,
}

// A layer of probes' light at `texel` (the layers' coordinates) of the
// bounce layers from `first`, around `normal`: its mean and its turn along
// the normal, light in red, green and blue, the sky's share in alpha.
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

// Where level `level` of the probes of the terrain whose horizon record
// starts at `at` holds terrain-local `uv`, between its window's columns,
// each at its middle, the terrain's edges within its outer ones: a point of
// the bounce layers (their coordinates), in the level's region of them; x
// below 0 where the window does not hold the point.
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

// A level's probes' light at `texel` of the bounce layers from `first`,
// around `normal`: the lower layer's, the upper's mixed in by `above`.
fn bounce_held(texel: vec2<f32>, first: u32, normal: vec3<f32>, above: f32) -> vec4<f32> {
    let held = bounce_layer(texel, first, normal);
    if above <= 0.0 {
        return held;
    }
    return mix(held, bounce_layer(texel, first + 4u, normal), above);
}

// Bounce light at `world` on a surface of `normal`, from the first terrain
// whose probes relight it: from the level of them the point's distance
// from the detail eye picks (`detail_level` over half the finest level's
// reach), or the finest from there whose window holds the point, the coarsest
// holding the whole terrain; between that level's probe columns, blended
// with the next coarser level's as the levels hand over, each coarser
// probe the mean of the finer ones under it, so a handover changes how
// sharp the light is, never how much; and, for a surface above the ground
// (not `on_terrain`, whose surfaces lie on it), between the probes' two
// layers by its height above the ground.
fn bounce_light(world: vec3<f32>, normal: vec3<f32>, on_terrain: bool) -> BounceLight {
    var found: BounceLight;
    found.sky = 1.0;
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
        var above = 0.0;
        if !on_terrain {
            let shape = terrain_horizon_records[at + 1u];
            let ground = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, uv,
                u32(shape.x) + HORIZON_LAYERS - 1u, 0.0).r;
            above = saturate((world.y - shape.w - ground - BOUNCE_LOW)
                / (BOUNCE_HIGH - BOUNCE_LOW));
        }
        let first = u32(header.x);
        var held = bounce_held(texel, first, normal, above);
        if blend > 0.0 && level + 1u < levels {
            let next = bounce_texel(at, level + 1u, uv);
            if next.x >= 0.0 {
                held = mix(held, bounce_held(next, first, normal, above), blend);
            }
        }
        found.light = max(held.rgb, vec3<f32>(0.0));
        found.sky = saturate(held.a);
        found.weight = 1.0;
        return found;
    }
    return found;
}

// Clouds' shadows: CLOUD_SHADOW_CELLS by CLOUD_SHADOW_CELLS
// cells of the sunlight passing the cloud layer up from its base, two cells
// along x a word in half floats.
const CLOUD_SHADOW_CELLS: u32 = 128u;
@group(0) @binding(19) var<storage, read> cloud_shadows: array<u32, CLOUD_SHADOW_CELLS * CLOUD_SHADOW_CELLS / 2u>;

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

// A threshold in [0, 1) at pixel `position` that fades compare their
// progress with: interleaved gradient noise, turned a different way each
// temporally antialiased frame (`camera.options.z`) so the resolve blends
// what fades as it would a coverage.
fn fade_noise(position: vec2<f32>) -> f32 {
    return fract(fract(52.9829189 * fract(dot(position, vec2<f32>(0.06711056, 0.00583715))))
        + camera.options.z);
}

// The sun at the surface point `world` in the shadow clipmap (x) and the
// level that decided it (y; -1 where none holds a drawn tile there, which
// leaves all of it): from the level a pixel's size picks there, blended
// with the next coarser as `shadow_level` blends them, each looked up a
// little along `normal` with the receiver's plane from how the point moves
// per pixel (`dx` and `dy`). A level holding no drawn tile there gives way
// to the first coarser one that does, which then stands in for both.
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

// Visibility of the sun at the surface point `world` in the shadow
// clipmap (`map_lookup`): lookups compare four texels with the surface's
// plane, past the last level none shadows.
fn map_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> f32 {
    return map_lookup(world, normal, dx, dy).x;
}

// Visibility of the sun at `world`, a point of the air rather than a
// surface: a plain comparison in the first level holding a drawn tile
// there from the one a pixel's size picks there, of still casters and of
// those moving over the tile; 1 where none holds one.
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
// of what each surface lets through (x), and of what particles alone do (y),
// whose coverage temporal antialiasing's resolve answers faster.
struct TransparentTargets {
    @location(0) accumulated: vec4<f32>,
    @location(1) revealed: vec2<f32>,
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
    targets.revealed = vec2<f32>(opacity, 0.0);
    return targets;
}

// `weighted` for a particle, which also marks its coverage.
fn weighted_particle(color: vec3<f32>, opacity: f32, distance: f32) -> TransparentTargets {
    var targets = weighted(color, opacity, distance);
    targets.revealed.y = opacity;
    return targets;
}
