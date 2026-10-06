// Water (decision 067): terrains' lakes and rivers where their hydrology
// bakes found them (decision 066), drawn as chunks of a grid whose vertices
// read the water's surface, depth and flow at the bake's samples. Water
// covers the ground below by how little of its light comes back through
// the water between its surface and the ground behind it, which a copy of
// the opaque depth gives (decision 074), reflecting the sky, the sun and,
// through screen-space reflections, the scene; its ripples ride the wind on
// lakes and run downstream on rivers, and foam gathers where it runs
// shallow along shores and where rivers run fast.

// The water's surface (terrain-local meters), a texel a bake sample.
@group(1) @binding(33) var water_surface: texture_2d<f32>;
// Per sample: meters of water over the ground (x, a 32nd of 32 m), its
// flow along x and z (y and z, 0.5 still, a unit 16 m/s across).
@group(1) @binding(34) var water_info: texture_2d<f32>;

// One terrain's water, or the sea (decision 076): the camera's view and
// projection without its translation; the terrain's origin less the eye
// (xyz) and the bake's spacing (w); the bake's samples along x and z and
// quads along each side of a chunk; the color deep water scatters back
// (rgb) and the meters through which half the light passes (w); the
// ripples' strength, the meters across the largest, and the wind's
// velocity along x and z; and how much foam gathers and the meters of
// water shallower than which shores foam, with, for the sea, the
// compression below which crests foam and the waves' summed height. The
// sea's parts (`meshes/ocean.wgsl`): its still level less the eye, its
// grid's finest cell, cells around and rings a cell apart; the rings'
// growth, the meters it reaches, cells around a sector and its waves;
// each wave's vector (xy, radians per meter), height and phase at the eye;
// and how far each wave's points move along it, four to a vector. For the
// sea, `origin` holds its wind's heading (xy, a unit vector). For terrains'
// water, `ocean` holds the sea's level less the eye and whether it is on,
// below which their water gives way to it.
struct WaterUniform {
    relative_view_projection: mat4x4<f32>,
    origin: vec4<f32>,
    grid: vec4<u32>,
    color: vec4<f32>,
    ripples: vec4<f32>,
    foam: vec4<f32>,
    ocean: vec4<f32>,
    ocean_grid: vec4<f32>,
    ocean_waves: array<vec4<f32>, 16>,
    ocean_reach: array<vec4<f32>, 4>,
}
@group(1) @binding(35) var<uniform> water: WaterUniform;
// The opaque scene's depth, copied before water draws.
@group(2) @binding(36) var water_depth: texture_depth_2d;

// The opaque surface behind the water at `pixel` (its center), less the
// eye, without the jitter of temporally antialiased frames (decision 017).
fn water_ground(pixel: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(water_depth));
    let depth = textureLoad(water_depth, vec2<i32>(pixel), 0);
    let ndc = vec2<f32>(pixel.x / size.x * 2.0 - 1.0, 1.0 - pixel.y / size.y * 2.0)
        - camera.jitter.xy;
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    return point.xyz / point.w - camera.position.xyz;
}

// A chunk: its first sample (x and z) and the samples between its
// vertices.
struct WaterChunk {
    @builtin(vertex_index) vertex: u32,
    @location(0) first: vec2<u32>,
    @location(1) stride: u32,
}

struct WaterOutput {
    @builtin(position) clip_position: vec4<f32>,
    // The point less the eye.
    @location(0) relative: vec3<f32>,
    // Meters of water over the ground, and its flow along x and z.
    @location(1) water: vec3<f32>,
}

struct WaterTemporalOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) relative: vec3<f32>,
    @location(1) water: vec3<f32>,
    @location(2) previous: vec3<f32>,
}

// The corners of a quad's two triangles, split along the diagonal from its
// (1, 0) corner to its (0, 1) corner, or turned, along the other one.
const WATER_CORNERS = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(1u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 1u),
);
const WATER_CORNERS_TURNED = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 1u),
    vec2<u32>(0u, 0u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u),
);

// Meters of water over the ground at bake sample `sample`.
fn water_depth_at(sample: vec2<u32>) -> f32 {
    return textureLoad(water_info, min(sample, water.grid.xy - vec2<u32>(1u)), 0).x * 32.0;
}

fn water_output(input: WaterChunk) -> WaterOutput {
    let quads = water.grid.z / input.stride;
    let quad = input.vertex / 6u;
    let stride = input.stride;
    let first = input.first + vec2<u32>(quad % quads, quad / quads) * stride;
    // Each quad splits along the diagonal joining its wetter corners: a
    // river stepping diagonally from sample to sample then keeps its water
    // along the edge between them, where the other diagonal would join two
    // dry corners across its course.
    let straight = water_depth_at(first + vec2<u32>(stride, 0u))
        + water_depth_at(first + vec2<u32>(0u, stride));
    let turned = water_depth_at(first) + water_depth_at(first + vec2<u32>(stride));
    var corner = WATER_CORNERS[input.vertex % 6u];
    if turned > straight {
        corner = WATER_CORNERS_TURNED[input.vertex % 6u];
    }
    let sample = min(first + corner * stride, water.grid.xy - vec2<u32>(1u));
    let surface = textureLoad(water_surface, sample, 0).r;
    let info = textureLoad(water_info, sample, 0);
    let spacing = water.origin.w;
    let relative = water.origin.xyz
        + vec3<f32>(f32(sample.x) * spacing, surface, f32(sample.y) * spacing);
    var output: WaterOutput;
    output.clip_position = water.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = relative;
    output.water = vec3<f32>(info.x * 32.0, (info.yz - vec2<f32>(0.5)) * 16.0);
    return output;
}

@vertex
fn vs_water(input: WaterChunk) -> WaterOutput {
    return water_output(input);
}

// A point of a river's ribbon (decision 092): terrain-local, and the
// water's depth over the bed there and its flow along x and z.
struct RiverVertex {
    @location(0) position: vec3<f32>,
    @location(1) water: vec3<f32>,
}

fn river_output(input: RiverVertex) -> WaterOutput {
    let relative = water.origin.xyz + input.position;
    var output: WaterOutput;
    output.clip_position = water.relative_view_projection * vec4<f32>(relative, 1.0);
    output.relative = relative;
    output.water = input.water;
    return output;
}

@vertex
fn vs_river(input: RiverVertex) -> WaterOutput {
    return river_output(input);
}

@vertex
fn vs_river_temporal(input: RiverVertex) -> WaterTemporalOutput {
    let output = river_output(input);
    var temporal: WaterTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.water = output.water;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(output.relative + camera.position.xyz, 1.0)).xyw;
    return temporal;
}

@vertex
fn vs_water_temporal(input: WaterChunk) -> WaterTemporalOutput {
    let output = water_output(input);
    var temporal: WaterTemporalOutput;
    temporal.clip_position = output.clip_position;
    temporal.relative = output.relative;
    temporal.water = output.water;
    temporal.previous = (camera.previous_view_projection
        * vec4<f32>(output.relative + camera.position.xyz, 1.0)).xyw;
    return temporal;
}

// Value noise at `p` and its gradient (x: value, yz: d/dx and d/dz), from
// hashed lattice values blended by quintic steps.
fn water_noise(p: vec2<f32>) -> vec3<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let a = terrain_hash(cell);
    let b = terrain_hash(cell + vec2<f32>(1.0, 0.0));
    let c = terrain_hash(cell + vec2<f32>(0.0, 1.0));
    let d = terrain_hash(cell + vec2<f32>(1.0, 1.0));
    let k = vec4<f32>(a, b - a, c - a, a - b - c + d);
    return vec3<f32>(k.x + k.y * u.x + k.z * u.y + k.w * u.x * u.y,
        du * vec2<f32>(k.y + k.w * u.y, k.z + k.w * u.x));
}

// Ripple slopes at `p` (meters): four octaves of noise, the largest `size`
// meters across, each drifting along `drift` meters a second at its own
// pace and turned from the last, octaves finer than the pixel's
// `footprint` fading out so ripples never alias.
fn water_slopes(p: vec2<f32>, size: f32, drift: vec2<f32>, time: f32, footprint: f32)
        -> vec2<f32> {
    var slopes = vec2<f32>(0.0);
    var scale = 1.0 / size;
    var amplitude = 1.0;
    var turn = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    var point = p;
    for (var octave = 0u; octave < 4u; octave++) {
        let fade = saturate(1.0 / (4.0 * footprint * scale) - 0.5);
        let moving = (point - drift * time * (1.0 + 0.35 * f32(octave))) * scale;
        let noise = water_noise(moving + vec2<f32>(f32(octave) * 7.31));
        slopes += turn * noise.yz * (amplitude * scale * fade);
        point = turn * point;
        scale *= 2.03;
        amplitude *= 0.55;
        turn = turn * mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    }
    return slopes * size * 0.35;
}

// The water's shaded light at its point, and how much of what lies below
// it covers: the light coming back through its depth hides the ground.
struct WaterShade {
    shaded: Shaded,
    coverage: f32,
}

fn shade_water(input: WaterOutput) -> WaterShade {
    let world = input.relative + camera.position.xyz;
    let depth = max(input.water.x, 0.0);
    let flow = input.water.yz;
    let time = camera.position.w;
    // Rivers' ripples run downstream; lakes' drift with the wind. Two sets
    // half a cycle apart cross-fade, so drifting never stretches them.
    let wind = water.ripples.zw;
    let drift = select(wind * 0.12, flow, dot(flow, flow) > 1.0e-4);
    // Running water breaks into smaller, livelier ripples.
    let running = saturate(length(flow) * 0.5);
    let size = water.ripples.y * mix(1.0, 0.35, running);
    // How far the pixel spans on the water: finer ripples fade out.
    let footprint = max(length(dpdx(world.xz)), length(dpdy(world.xz)));
    let slopes = water_slopes(world.xz, size, drift, time, footprint);
    let distance = length(input.relative);
    let strength = water.ripples.x * mix(1.0, 1.3, running);
    let normal = normalize(vec3<f32>(-slopes.x * strength, 1.0, -slopes.y * strength));
    let view = stable_normalize(-input.relative);
    // The ground behind the water: the meters the view travels through it,
    // and how deep the water stands over the ground there. Behind the sky,
    // both run far.
    // Under the sea, the sea's water stands instead.
    if water.ocean.y > 0.5 && input.relative.y < water.ocean.x {
        discard;
    }
    let ground = water_ground(input.clip_position.xy);
    let through = max(length(ground - input.relative), 0.0);
    let below = max(input.relative.y - ground.y, 0.0);
    // Where the bake holds no water, its surfaces reach past the shore only
    // to meet the ground: over steep ground they stand off the slope, and
    // nothing of them draws there.
    if (depth < 0.05 && below > 1.0) {
        discard;
    }
    // What comes back from inside the water, lit as a diffuse surface:
    // the scattered color, for as much light as the water stops. The bake
    // bounds the path where its surfaces reach past the shore, which on
    // steep ground can stand meters over the slope: water is only where it
    // holds some.
    let path = min(through, depth * 2.0 + 0.3);
    let transmit = exp2(-path / water.color.w);
    let real = smoothstep(0.02, 0.25, depth);
    // Foam where the water runs shallow along shores and where rivers run
    // fast, broken by drifting noise and fading before it would alias.
    let shore = 1.0 - smoothstep(0.0, water.foam.y, below);
    let rapids = smoothstep(2.5, 6.0, length(flow));
    let grain = water_noise((world.xz - drift * time * 0.8) * 1.3).x * 0.65
        + water_noise(world.xz * 4.1 + vec2<f32>(time * 0.35, 0.0)).x * 0.35;
    let foam = water.foam.x * saturate((max(shore, rapids * 0.8) - grain * 0.7) * 2.5)
        * saturate(1.6 - footprint * 3.0) * real;
    var point: SurfacePoint;
    point.albedo = water.color.rgb;
    point.opacity = 1.0;
    point.emission = 0.0;
    point.emissive = vec3<f32>(0.0);
    point.occlusion = 1.0;
    point.bounce = vec3<f32>(0.0);
    point.roughness = mix(0.05, 0.7, foam);
    point.alpha = point.roughness * point.roughness;
    point.f0 = vec3<f32>(0.02);
    point.diffuse_color = mix(water.color.rgb * (1.0 - transmit), vec3<f32>(0.8), foam);
    point.geometric = vec3<f32>(0.0, 1.0, 0.0);
    point.normal = normal;
    point.distance = distance;
    point.view = view;
    point.n_v = max(dot(normal, view), 1.0e-4);
    point.world = world;
    point.world_dx = dpdx(world);
    point.world_dy = dpdy(world);
    point.shadow_offset = vec3<f32>(0.0);
    point.on_terrain = false;
    point.sun_shade = 0.0;
    var result: WaterShade;
    result.shaded = finish(point, vec3<f32>(0.0));
    // The surface reflects its Fresnel share; of the rest, the depth lets
    // the ground show through.
    let reflectance = saturate(luminance(environment_brdf(point.f0, point.roughness,
        point.n_v)));
    // Foam covers what lies below, and the water thins out to nothing where
    // it meets the shore.
    result.coverage = max(1.0 - (1.0 - reflectance) * transmit, foam)
        * smoothstep(0.0, 0.08, below);
    return result;
}

fn water_temporal_output(input: WaterTemporalOutput) -> WaterOutput {
    var output: WaterOutput;
    output.clip_position = input.clip_position;
    output.relative = input.relative;
    output.water = input.water;
    return output;
}

// Colors blend premultiplied over the ground: the scene target's alpha
// keeps the ground's ambient share, the other targets take the water's.
@fragment
fn fs_water(input: WaterOutput) -> @location(0) vec4<f32> {
    let shade = shade_water(input);
    return vec4<f32>(encoded(shade.shaded.color) * shade.coverage, shade.coverage);
}

@fragment
fn fs_water_post(input: WaterOutput) -> SceneTargets {
    let shade = shade_water(input);
    var targets = scene_targets(shade.shaded);
    targets.color = vec4<f32>(shade.shaded.color * shade.coverage, shade.coverage);
    return targets;
}

@fragment
fn fs_water_temporal(input: WaterTemporalOutput) -> TemporalTargets {
    let shade = shade_water(water_temporal_output(input));
    var targets = temporal_targets(shade.shaded, input.clip_position, input.previous);
    targets.color = vec4<f32>(shade.shaded.color * shade.coverage, shade.coverage);
    return targets;
}
