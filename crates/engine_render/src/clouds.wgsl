// Clouds: a layer of cumulus marched from the camera into a
// dome of the sky above it. The dome's texel (u, v) looks toward azimuth
// 2 pi u and up by the elevation mix(-asin(HORIZON), pi / 2, v), every row
// as many degrees (`scene.wgsl`'s `cloud_dome_uv`); each holds the light
// the clouds scatter toward the eye (linear, as the sky's) and the share of
// the sky behind them that passes. Two tiling volumes of noise, the
// scene's static tables (`engine_tables::clouds` makes them), shape the
// clouds: Perlin-Worley shapes, and the finer wisps of Worley's cells at
// five frequencies eroding their edges.

struct CloudUniform {
    // The camera (world); w: 1 with the physically based sky, whose sun
    // lights each cloud where it stands.
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
    // The light map: its first column's corner in the clouds' own frame
    // (meters, x and z), meters a column, and 1 while it holds this sun and
    // layer whole; then the first of its rows this dispatch builds and how
    // many.
    light: vec4<f32>,
    light_rows: vec4<u32>,
}

@group(0) @binding(0) var<uniform> clouds: CloudUniform;
@group(0) @binding(1) var noise_shape: texture_3d<f32>;
@group(0) @binding(2) var noise_sampler: sampler;
@group(0) @binding(4) var noise_detail: texture_3d<f32>;
@group(0) @binding(3) var dome: texture_storage_2d<rgba16float, write>;
// The shadows' grid: the sunlight passing the layer up from each cell of
// its base, two cells (along x) a word in half floats.
@group(0) @binding(5) var<storage, read_write> shadow_cells: array<u32>;

// Clouds in the view: its unjittered inverse
// view-projection; the marched blocks along x and y, the pixels along a
// block's side, and whether rays seeing the sky march (1) or leave it to
// the dome (0); and the pixels the frame draws, its render size, from the
// depth buffer's top-left corner (xy).
struct CloudView {
    inverse_view_projection: mat4x4<f32>,
    blocks: vec4<u32>,
    pixels: vec4<u32>,
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
// The physically based sky (`sky_common.wgsl`): its uniform, transmittance
// table and their sampler, through which its sun lights the clouds.
@group(0) @binding(11) var<uniform> cloud_air: SkyUniform;
@group(0) @binding(12) var cloud_transmittance: texture_2d<f32>;
@group(0) @binding(13) var cloud_table_sampler: sampler;
// The light map: the far part of each light march, marched once for
// LIGHT_TEXELS by LIGHT_TEXELS columns of the clouds' own frame, which the
// wind does not move, at LIGHT_HEIGHTS heights through the layer; each
// column holds its neighbouring heights in pairs of half floats, a word a
// pair, so one word gives both heights a point between them reads.
@group(0) @binding(14) var<storage, read_write> light_cells: array<u32>;

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
// Of the light march's steps, the first ones, which find the sunlit edges
// close to each point, march for every point; the rest the light map
// holds where it covers.
const LIGHT_NEAR_STEPS: u32 = 2u;
const LIGHT_TEXELS: u32 = 384u;
const LIGHT_HEIGHTS: u32 = 16u;
const LIGHT_WORDS: u32 = LIGHT_HEIGHTS - 1u;
// Columns over which the map hands over to marching toward its edge.
const LIGHT_EDGE: f32 = 16.0;
const REACH: f32 = 60000.0;
// Beyond this many meters the clouds' fine detail, finer than the dome's
// texels there, no longer erodes them.
const DETAIL_REACH: f32 = 20000.0;
// Extinction per meter of fully dense cloud at density 1.
const EXTINCTION: f32 = 0.08;

fn remap(value: f32, low: f32, high: f32, to_low: f32, to_high: f32) -> f32 {
    return to_low + (value - low) * (to_high - to_low) / max(high - low, 1.0e-5);
}

// The direction the dome's texel at `uv` looks.
fn dome_direction(uv: vec2<f32>) -> vec3<f32> {
    let elevation = mix(-asin(HORIZON), 0.25 * TAU, uv.y);
    let up = sin(elevation);
    let around = cos(elevation);
    let angle = TAU * uv.x;
    return vec3<f32>(cos(angle) * around, up, sin(angle) * around);
}

// Where the clouds' own frame (the noise's, which the wind and the seed
// shift) stands from the world's, along x and z.
fn cloud_offset() -> vec2<f32> {
    return vec2<f32>(clouds.drift.x + clouds.drift.w, clouds.drift.y - clouds.drift.w);
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
    let offset = cloud_offset();
    let moved = p + vec3<f32>(offset.x, 0.0, offset.y);
    let repeat = clouds.drift.z;
    // Where clouds gather and where the sky stays clear, broadly; coverage
    // sets the shapes' threshold around their average. Where no shape could
    // reach it at this height, the shapes are not read.
    let gather = textureSampleLevel(noise_shape, noise_sampler,
        vec3<f32>(moved.x / (repeat * 4.3), 0.37, moved.z / (repeat * 4.3)), 0.0).r;
    let local = saturate(clouds.layer.z * (0.4 + 1.2 * smoothstep(0.3, 0.7, gather)));
    let threshold = mix(0.95, 0.4, local);
    if profile <= threshold {
        return 0.0;
    }
    // Perlin-Worley shapes, about two thirds on average.
    let rounded = textureSampleLevel(noise_shape, noise_sampler, moved / repeat, 0.0).r;
    var cloud = saturate(remap(rounded * profile, threshold, 1.0, 0.0, 1.0));
    if detail > 0.0 && cloud > 0.0 {
        let wisps = textureSampleLevel(noise_detail, noise_sampler, moved / (repeat * 0.13), 0.0).r;
        // Wispy below, billowing above.
        let erode = mix(wisps, 1.0 - wisps, saturate(height * 3.0)) * 0.4 * detail;
        cloud = saturate(remap(cloud, erode, 1.0, 0.0, 1.0));
    }
    return cloud;
}

// The cloud along the sun's ray from `p` over the light march's steps
// from `first` to before `last` (60 m long, each next 1.8 times as long,
// sampled at their middles), in meters of full density.
fn sun_depth_steps(p: vec3<f32>, first: u32, last: u32) -> f32 {
    var depth = 0.0;
    var along = 0.0;
    var step = 60.0;
    for (var index = 0u; index < last; index++) {
        if index >= first {
            depth += cloud_density(p + clouds.sun.xyz * (along + step * 0.5), 0.0) * step;
        }
        along += step;
        step *= 1.8;
    }
    return depth;
}

// The far part of the light march from `p` (its steps from
// LIGHT_NEAR_STEPS) as the light map holds it: bilinear across its columns,
// linear between its heights; and how much to take it (0 outside the map
// or without one, rising over LIGHT_EDGE columns in from its edge).
fn mapped_far_depth(p: vec3<f32>) -> vec2<f32> {
    if clouds.light.w < 0.5 {
        return vec2<f32>(0.0);
    }
    let column = (p.xz + cloud_offset() - clouds.light.xy) / clouds.light.z - 0.5;
    let first = floor(column);
    let last = f32(LIGHT_TEXELS - 2u);
    let inside = min(min(first.x, first.y), min(last - first.x, last - first.y));
    if inside < 0.0 {
        return vec2<f32>(0.0);
    }
    let f = column - first;
    let height = (p.y - clouds.layer.x) / (clouds.layer.y - clouds.layer.x)
        * f32(LIGHT_HEIGHTS) - 0.5;
    let pair = u32(clamp(floor(height), 0.0, f32(LIGHT_WORDS - 1u)));
    let up = saturate(height - f32(pair));
    let row = LIGHT_TEXELS * LIGHT_WORDS;
    let at = (u32(first.y) * LIGHT_TEXELS + u32(first.x)) * LIGHT_WORDS + pair;
    let near_row = mix(unpack2x16float(light_cells[at]), unpack2x16float(light_cells[at + LIGHT_WORDS]),
        f.x);
    let far_row = mix(unpack2x16float(light_cells[at + row]),
        unpack2x16float(light_cells[at + row + LIGHT_WORDS]), f.x);
    let heights = mix(near_row, far_row, f.y);
    return vec2<f32>(mix(heights.x, heights.y, up), saturate(inside / LIGHT_EDGE));
}

// The cloud along the sun's ray from `p`, in meters of full density: the
// near steps marched, the rest from the light map where it covers.
fn sun_depth(p: vec3<f32>) -> f32 {
    let near = sun_depth_steps(p, 0u, LIGHT_NEAR_STEPS);
    let mapped = mapped_far_depth(p);
    var far = mapped.x;
    if mapped.y < 1.0 {
        far = mix(sun_depth_steps(p, LIGHT_NEAR_STEPS, LIGHT_STEPS), mapped.x, mapped.y);
    }
    return near + far;
}

// The sunlight at world point `p`: the eye's, or with the physically based
// sky what passes its air to the point's own place over the planet, so
// cloud tops keep the low sun's light after the ground has lost it and the
// planet's shadow rises through the layer.
fn sunlight_at(p: vec3<f32>) -> vec3<f32> {
    if clouds.eye.w < 0.5 {
        return clouds.sun_color.rgb;
    }
    let height = cloud_air.radii.w + (p.y - clouds.eye.y) * 0.001;
    return sunlight_over(cloud_transmittance, cloud_table_sampler, cloud_air,
        (p.xz - clouds.eye.xz) * 0.001, height, clouds.sun.xyz);
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
    // clouds' detail on screen shrinks; each ray starts its steps at its
    // own offset, which hides their banding: interleaved gradient noise
    // (Jimenez 2014) over the texels, whose offsets every 3x3 texels
    // spread evenly, so the sky's filtered lookups average them out.
    let even = (far - near) / clamp((far - near) / STEP_LENGTH, FEWEST_STEPS, MOST_STEPS);
    let jitter = fract(52.9829189 * fract(dot(vec2<f32>(texel), vec2<f32>(0.06711056, 0.00583715))));
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
        let sunlight = sunlight_at(p);
        let diffuse = sunlight * (0.06 * (0.4 + 0.6 * height));
        let source = sunlight * (direct * powder * phase * 4.0 * 3.14159) + ambient + diffuse;
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

// A slice of the light map's rows, `clouds.light_rows.y` from
// `clouds.light_rows.x`: each column's far light march from each of its
// heights, the column standing where the clouds' own frame now puts it.
@compute @workgroup_size(8, 8)
fn light_clouds(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = clouds.light_rows.x + id.y;
    if id.x >= LIGHT_TEXELS || id.y >= clouds.light_rows.y || row >= LIGHT_TEXELS {
        return;
    }
    let column = clouds.light.xy + (vec2<f32>(f32(id.x), f32(row)) + 0.5) * clouds.light.z
        - cloud_offset();
    let thickness = clouds.layer.y - clouds.layer.x;
    let at = (row * LIGHT_TEXELS + id.x) * LIGHT_WORDS;
    var below = 0.0;
    for (var level = 0u; level < LIGHT_HEIGHTS; level++) {
        let y = clouds.layer.x + thickness * (f32(level) + 0.5) / f32(LIGHT_HEIGHTS);
        let depth = sun_depth_steps(vec3<f32>(column.x, y, column.y), LIGHT_NEAR_STEPS,
            LIGHT_STEPS);
        if level > 0u {
            light_cells[at + level - 1u] = pack2x16float(vec2<f32>(below, depth));
        }
        below = depth;
    }
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
    let pixels = max(cloud_view.pixels.xy, vec2<u32>(1u));
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
