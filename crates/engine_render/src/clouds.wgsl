// Clouds: a layer of cumulus marched from the camera into a
// dome of the sky above it. The dome's texel (u, v) looks toward azimuth
// 2 pi u and up by the elevation mix(-asin(HORIZON), pi / 2, v), every row
// as many degrees (`scene.wgsl`'s `cloud_dome_uv`); each holds the light
// the clouds scatter toward the eye (linear, as the sky's) and the share of
// the sky behind them that passes. A tiling volume of noise, made once,
// shapes the clouds: Perlin-Worley in red, Worley at three frequencies in
// green, blue and alpha.

struct CloudUniform {
    // The camera (world).
    eye: vec4<f32>,
    // The layer's base and top (world meters), its coverage and density.
    layer: vec4<f32>,
    // The wind's drift so far (meters, x and z), the meters across the
    // noise volume's repeat, and the seed's offset (meters).
    drift: vec4<f32>,
    // The dome's texels along u and v, the first row this dispatch marches
    // and how many rows from it.
    dome: vec4<u32>,
    // Toward the sun, and its light.
    sun: vec4<f32>,
    sun_color: vec4<f32>,
    // The sky's light from above and at the horizon, and the haze far
    // clouds fade into.
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    // The shadows' grid: its first cell (world cell indices
    // along x and z), meters a cell and cells along a side, each cell kept
    // in the slot of its index modulo the cells; then the first row of
    // slots this dispatch marches and the rows between those it marches.
    shadow: vec4<f32>,
    shadow_rows: vec4<u32>,
}

@group(0) @binding(0) var<uniform> clouds: CloudUniform;
@group(0) @binding(1) var noise_volume: texture_3d<f32>;
@group(0) @binding(2) var noise_sampler: sampler;
@group(0) @binding(3) var dome: texture_storage_2d<rgba16float, write>;
// The shadows' grid: the sunlight passing the layer up from each cell of
// its base, two cells (along x) a word in half floats.
@group(0) @binding(5) var<storage, read_write> shadow_cells: array<u32>;

// Clouds in the view: its unjittered inverse
// view-projection; the marched blocks along x and y, the pixels along a
// block's side, and whether rays seeing the sky march (1) or leave it to
// the dome (0).
struct CloudView {
    inverse_view_projection: mat4x4<f32>,
    blocks: vec4<u32>,
}
@group(0) @binding(6) var<uniform> cloud_view: CloudView;
// The view's depth buffer (reversed: 0 at the far plane), and each block's
// light and passing share, and how far the clouds it met lie (meters).
@group(0) @binding(7) var view_depth: texture_depth_2d;
@group(0) @binding(8) var view_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var front_out: texture_storage_2d<r32float, write>;
// The composite's indirect draw: its vertices, set to its triangle's once
// any block meets a cloud, so views without any draw nothing.
@group(0) @binding(10) var<storage, read_write> composite_draw: array<atomic<u32>, 4>;

// Rays below the horizon by this much (sine of their elevation) see no
// clouds (`scene.wgsl`'s `CLOUD_HORIZON`).
const HORIZON: f32 = 0.05;
const TAU: f32 = 6.2831853;
// Steps along each ray through the layer (one per STEP_LENGTH meters,
// between the fewest and the most), toward the sun from each step, and the
// farthest clouds marched (meters).
const FEWEST_STEPS: f32 = 24.0;
const MOST_STEPS: f32 = 64.0;
const STEP_LENGTH: f32 = 150.0;
// Near the eye (inside the layer) steps start this many
// meters long and grow with the distance, by this share of it, up to the
// ray's even step; no ray takes more than this many.
const NEAREST_STEP: f32 = 25.0;
const STEP_SHARE: f32 = 0.1;
const MOST_STEPS_IN_ALL: u32 = 128u;
const LIGHT_STEPS: u32 = 5u;
const REACH: f32 = 60000.0;
// Beyond this many meters the clouds' fine detail, finer than the dome's
// texels there, no longer erodes them.
const DETAIL_REACH: f32 = 20000.0;
// Extinction per meter of fully dense cloud at density 1.
const EXTINCTION: f32 = 0.08;

fn hash3(p: vec3<u32>) -> vec3<f32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3<f32>(v) * (1.0 / 4294967296.0);
}

// Cellular noise repeating every `cells` cells across the unit cube: 1 at
// its feature points, falling to 0 a cell away.
fn worley(p: vec3<f32>, cells: i32) -> f32 {
    let scaled = p * f32(cells);
    let cell = vec3<i32>(floor(scaled));
    let f = scaled - vec3<f32>(cell);
    var nearest = 1.0e9;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let offset = vec3<i32>(x, y, z);
                let neighbor = ((cell + offset) % cells + cells) % cells;
                let point = vec3<f32>(offset) + hash3(vec3<u32>(neighbor) + vec3<u32>(u32(cells)));
                let away = point - f;
                nearest = min(nearest, dot(away, away));
            }
        }
    }
    return 1.0 - saturate(sqrt(nearest));
}

fn worley_fbm(p: vec3<f32>, cells: i32) -> f32 {
    return worley(p, cells) * 0.625 + worley(p, cells * 2) * 0.25 + worley(p, cells * 4) * 0.125;
}

// Gradient noise repeating every `period` cells across the unit cube, in
// about [-1, 1].
fn perlin(p: vec3<f32>, period: i32) -> f32 {
    let scaled = p * f32(period);
    let cell = vec3<i32>(floor(scaled));
    let f = scaled - vec3<f32>(cell);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var corners: array<f32, 8>;
    for (var index = 0; index < 8; index++) {
        let offset = vec3<i32>(index & 1, (index >> 1) & 1, index >> 2);
        let wrapped = ((cell + offset) % period + period) % period;
        let gradient = normalize(hash3(vec3<u32>(wrapped) + vec3<u32>(7u, 11u, 13u)) * 2.0 - 1.0);
        corners[index] = dot(gradient, f - vec3<f32>(offset));
    }
    let x0 = mix(corners[0], corners[1], u.x);
    let x1 = mix(corners[2], corners[3], u.x);
    let x2 = mix(corners[4], corners[5], u.x);
    let x3 = mix(corners[6], corners[7], u.x);
    return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}

fn perlin_fbm(p: vec3<f32>, period: i32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var scale = period;
    for (var octave = 0; octave < 4; octave++) {
        sum += perlin(p, scale) * amplitude;
        amplitude *= 0.5;
        scale *= 2;
    }
    return saturate(sum * 0.9 + 0.5);
}

fn remap(value: f32, low: f32, high: f32, to_low: f32, to_high: f32) -> f32 {
    return to_low + (value - low) * (to_high - to_low) / max(high - low, 1.0e-5);
}

@group(0) @binding(4) var noise_out: texture_storage_3d<rgba8unorm, write>;

// The noise volume, a texel per invocation.
@compute @workgroup_size(4, 4, 4)
fn make_noise(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(noise_out);
    if any(id >= size) {
        return;
    }
    let p = (vec3<f32>(id) + 0.5) / vec3<f32>(size);
    let low = worley_fbm(p, 4);
    // Perlin's billows carved by Worley's cells: rounded cumulus shapes.
    let shapes = saturate(remap(perlin_fbm(p, 4), low - 1.0, 1.0, 0.0, 1.0));
    textureStore(noise_out, id, vec4<f32>(shapes, low, worley_fbm(p, 8), worley_fbm(p, 16)));
}

// The direction the dome's texel at `uv` looks.
fn dome_direction(uv: vec2<f32>) -> vec3<f32> {
    let elevation = mix(-asin(HORIZON), 0.25 * TAU, uv.y);
    let up = sin(elevation);
    let around = cos(elevation);
    let angle = TAU * uv.x;
    return vec3<f32>(cos(angle) * around, up, sin(angle) * around);
}

// How dense the clouds are at world point `p` (0 to 1), with their fine
// detail eroding their edges by `detail` (0 to 1).
fn cloud_density(p: vec3<f32>, detail: f32) -> f32 {
    let height = (p.y - clouds.layer.x) / (clouds.layer.y - clouds.layer.x);
    if height <= 0.0 || height >= 1.0 {
        return 0.0;
    }
    // Rounded below and thinning toward the top, as cumulus are.
    let profile = saturate(height * 7.0) * saturate((1.0 - height) * 2.0);
    let moved = p + vec3<f32>(clouds.drift.x + clouds.drift.w, 0.0, clouds.drift.y - clouds.drift.w);
    let repeat = clouds.drift.z;
    // Where clouds gather and where the sky stays clear, broadly; coverage
    // sets the shapes' threshold around their average. Where no shape could
    // reach it at this height, the shapes are not read.
    let gather = textureSampleLevel(noise_volume, noise_sampler,
        vec3<f32>(moved.x / (repeat * 4.3), 0.37, moved.z / (repeat * 4.3)), 0.0).r;
    let local = saturate(clouds.layer.z * (0.4 + 1.2 * smoothstep(0.3, 0.7, gather)));
    let threshold = mix(0.95, 0.4, local);
    if profile <= threshold {
        return 0.0;
    }
    // Perlin-Worley shapes, about two thirds on average.
    let rounded = textureSampleLevel(noise_volume, noise_sampler, moved / repeat, 0.0).r;
    var cloud = saturate(remap(rounded * profile, threshold, 1.0, 0.0, 1.0));
    if detail > 0.0 && cloud > 0.0 {
        let fine = textureSampleLevel(noise_volume, noise_sampler, moved / (repeat * 0.13), 0.0);
        let wisps = fine.g * 0.625 + fine.b * 0.25 + fine.a * 0.125;
        // Wispy below, billowing above.
        let erode = mix(wisps, 1.0 - wisps, saturate(height * 3.0)) * 0.4 * detail;
        cloud = saturate(remap(cloud, erode, 1.0, 0.0, 1.0));
    }
    return cloud;
}

// The cloud along the sun's ray from `p`, in meters of full density.
fn sun_depth(p: vec3<f32>) -> f32 {
    var depth = 0.0;
    var along = 0.0;
    var step = 60.0;
    for (var index = 0u; index < LIGHT_STEPS; index++) {
        along += step * 0.5;
        depth += cloud_density(p + clouds.sun.xyz * along, 0.0) * step;
        along += step * 0.5;
        step *= 1.8;
    }
    return depth;
}

// Henyey-Greenstein's phase function.
fn henyey_greenstein(cosine: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (2.0 * TAU * pow(1.0 + g2 - 2.0 * g * cosine, 1.5));
}

// What a ray met of the clouds: the light they scatter toward the eye and
// the share of what lies behind that passes, and how far they lie: the
// distance of the light they scattered, on average.
struct Marched {
    cloud: vec4<f32>,
    front: f32,
}

// The clouds seen from the eye along `direction` up to `limit` meters.
fn march(direction: vec3<f32>, texel: vec2<u32>, limit: f32) -> Marched {
    let eye = clouds.eye.xyz;
    let base = clouds.layer.x;
    let top = clouds.layer.y;
    let reach = min(REACH, limit);
    var marched: Marched;
    marched.cloud = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    marched.front = reach;
    // Where the ray runs within the layer.
    var near = 0.0;
    var far = reach;
    if abs(direction.y) > 1.0e-5 {
        let to_base = (base - eye.y) / direction.y;
        let to_top = (top - eye.y) / direction.y;
        near = max(min(to_base, to_top), 0.0);
        far = min(max(to_base, to_top), reach);
    } else if eye.y < base || eye.y > top {
        return marched;
    }
    if far <= near {
        return marched;
    }
    // A step every few hundred meters, longer on the longest rays, and
    // near the eye a few tens of meters, growing with the distance as the
    // clouds' detail on screen shrinks; each ray starts its
    // steps at its own offset, which hides their banding.
    let even = (far - near) / clamp((far - near) / STEP_LENGTH, FEWEST_STEPS, MOST_STEPS);
    let jitter = hash3(vec3<u32>(texel, 17u)).x;
    let cosine = dot(direction, clouds.sun.xyz);
    let phase = mix(henyey_greenstein(cosine, 0.8), henyey_greenstein(cosine, -0.3), 0.3);
    let extinction = EXTINCTION * clouds.layer.w;
    var light = vec3<f32>(0.0);
    var passing = 1.0;
    var weighted = 0.0;
    var along = near + jitter * min(even, max(NEAREST_STEP, near * STEP_SHARE));
    for (var index = 0u; index < MOST_STEPS_IN_ALL && along < far; index++) {
        let step = min(even, max(NEAREST_STEP, along * STEP_SHARE));
        let here = along;
        along += step;
        let p = eye + direction * here;
        let density = cloud_density(p, 1.0 - smoothstep(0.5 * DETAIL_REACH, DETAIL_REACH, here));
        if density <= 0.0 {
            continue;
        }
        let sigma = density * extinction;
        let toward_sun = sun_depth(p) * extinction;
        // Beer's law over two octaves of light scattered more than once,
        // darkened at the edges facing the sun (powder); deep inside, light
        // scattered many times still lights the cloud dimly.
        let direct = exp(-toward_sun) + 0.5 * exp(-toward_sun * 0.5)
            + 0.25 * exp(-toward_sun * 0.25);
        let powder = 1.0 - 0.6 * exp(-2.0 * sigma * 120.0);
        let height = saturate((p.y - base) / (top - base));
        // The sky's light from above and the ground's from below, grayed as
        // clouds scatter every color alike.
        let sky = mix(clouds.horizon.rgb, clouds.zenith.rgb, 0.5 + 0.5 * height);
        let gray = vec3<f32>(dot(sky, vec3<f32>(0.2126, 0.7152, 0.0722)));
        let ambient = mix(sky, gray, 0.6) * (0.5 + 0.7 * height);
        let diffuse = clouds.sun_color.rgb * (0.06 * (0.4 + 0.6 * height));
        let source = clouds.sun_color.rgb * (direct * powder * phase * 4.0 * 3.14159) + ambient
            + diffuse;
        let through = exp(-sigma * step);
        light += passing * source * (1.0 - through);
        weighted += passing * (1.0 - through) * here;
        passing *= through;
        if passing < 0.01 {
            break;
        }
    }
    // Far clouds fade into the haze toward the horizon.
    let fade = exp(-near / 30000.0);
    light = mix(clouds.horizon.rgb * (1.0 - passing), light, fade);
    marched.cloud = vec4<f32>(light, passing);
    if passing < 1.0 {
        marched.front = weighted / (1.0 - passing);
    }
    return marched;
}

// The share of sunlight passing the layer up from `start` on its base.
fn sun_passing(start: vec3<f32>) -> f32 {
    let sun = clouds.sun.xyz;
    if sun.y <= 0.01 {
        return 1.0;
    }
    let thickness = clouds.layer.y - clouds.layer.x;
    let span = min(thickness / sun.y, 3.0 * thickness);
    let step = span / 12.0;
    var depth = 0.0;
    for (var index = 0u; index < 12u; index++) {
        depth += cloud_density(start + sun * ((f32(index) + 0.5) * step), 0.0) * step;
    }
    return exp(-depth * EXTINCTION * clouds.layer.w);
}

// Two cells of the shadows' grid, side by side along x, on a slice of its
// rows: from `clouds.shadow_rows.x`, every `clouds.shadow_rows.y`th.
@compute @workgroup_size(8, 8)
fn shadow_clouds(@builtin(global_invocation_id) id: vec3<u32>) {
    let cells = u32(clouds.shadow.w);
    let row = clouds.shadow_rows.x + id.y * clouds.shadow_rows.y;
    if id.x * 2u >= cells || row >= cells {
        return;
    }
    var pair = vec2<f32>(1.0);
    let size = i32(cells);
    let first = vec2<i32>(i32(clouds.shadow.x), i32(clouds.shadow.y));
    for (var half = 0u; half < 2u; half++) {
        // The world cell the slot holds: the one of the grid's cells whose
        // index modulo the cells is the slot's.
        let slot = vec2<i32>(i32(id.x * 2u + half), i32(row));
        let cell = first + (((slot - first) % size) + size) % size;
        let start = vec3<f32>((f32(cell.x) + 0.5) * clouds.shadow.z, clouds.layer.x,
            (f32(cell.y) + 0.5) * clouds.shadow.z);
        pair[half] = sun_passing(start);
    }
    shadow_cells[row * (cells / 2u) + id.x] = pack2x16float(pair);
}

// A band of the dome's rows: `clouds.dome.w` from `clouds.dome.z`.
@compute @workgroup_size(8, 8)
fn march_clouds(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = clouds.dome.z + id.y;
    if id.x >= clouds.dome.x || id.y >= clouds.dome.w || row >= clouds.dome.y {
        return;
    }
    let uv = (vec2<f32>(f32(id.x), f32(row)) + 0.5) / vec2<f32>(clouds.dome.xy);
    textureStore(dome, vec2<u32>(id.x, row),
        march(dome_direction(uv), vec2<u32>(id.x, row), REACH).cloud);
}

// The view's clouds: each block's ray, through its center,
// marched up to the farthest of its pixels the scene shows; blocks seeing
// the sky march to the clouds' reach, or leave them to the dome while the
// camera is under the layer.
@compute @workgroup_size(8, 8)
fn march_view(@builtin(global_invocation_id) id: vec3<u32>) {
    let blocks = cloud_view.blocks.xy;
    if any(id.xy >= blocks) {
        return;
    }
    let side = cloud_view.blocks.z;
    let pixels = textureDimensions(view_depth);
    let first = id.xy * side;
    var farthest = 1.0;
    for (var y = 0u; y < side; y++) {
        for (var x = 0u; x < side; x++) {
            let pixel = min(first + vec2<u32>(x, y), pixels - 1u);
            farthest = min(farthest, textureLoad(view_depth, pixel, 0));
        }
    }
    var marched: Marched;
    marched.cloud = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    marched.front = REACH;
    let sky = farthest <= 0.0;
    if !sky || cloud_view.blocks.w != 0u {
        let center = (vec2<f32>(first) + 0.5 * f32(side)) / vec2<f32>(pixels);
        let ndc = vec2<f32>(center.x * 2.0 - 1.0, 1.0 - center.y * 2.0);
        let far = cloud_view.inverse_view_projection * vec4<f32>(ndc, 0.0, 1.0);
        let direction = normalize(far.xyz / far.w - clouds.eye.xyz);
        var limit = REACH;
        if !sky {
            let point = cloud_view.inverse_view_projection * vec4<f32>(ndc, farthest, 1.0);
            limit = length(point.xyz / point.w - clouds.eye.xyz);
        }
        marched = march(direction, id.xy, limit);
    }
    textureStore(view_out, id.xy, marched.cloud);
    textureStore(front_out, id.xy, vec4<f32>(marched.front, 0.0, 0.0, 0.0));
    if marched.cloud.a < 0.999 {
        atomicStore(&composite_draw[0], 3u);
    }
}
