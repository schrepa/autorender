import package::sky::common::{SkyUniform, sunlight_over};

struct CloudUniform {
    eye: vec4<f32>,
    mirror: vec4<f32>,
    layer: vec4<f32>,
    drift: vec4<f32>,
    dome: vec4<u32>,
    sun: vec4<f32>,
    sun_color: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    shadow: vec4<f32>,
    shadow_rows: vec4<u32>,
    light: vec4<f32>,
    light_rows: vec4<u32>,
}

@group(0) @binding(0) var<uniform> clouds: CloudUniform;
@group(0) @binding(1) var noise_shape: texture_3d<f32>;
@group(0) @binding(2) var noise_sampler: sampler;
@group(0) @binding(4) var noise_detail: texture_3d<f32>;
@group(0) @binding(3) var dome: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(5) var<storage, read_write> shadow_cells: array<u32>;

struct CloudView {
    inverse_view_projection: mat4x4<f32>,
    pixels: vec4<u32>,
}
@group(0) @binding(6) var<uniform> cloud_view: CloudView;
@group(0) @binding(7) var view_depth: texture_depth_2d;
@group(0) @binding(8) var view_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var<storage, read_write> composite_draw: array<atomic<u32>, 4>;
@group(0) @binding(11) var<uniform> cloud_air: SkyUniform;
@group(0) @binding(12) var cloud_transmittance: texture_2d<f32>;
@group(0) @binding(13) var cloud_table_sampler: sampler;
@group(0) @binding(14) var<storage, read_write> light_cells: array<u32>;

const HORIZON: f32 = 0.05;
const MIRROR_LAYER: u32 = 1u;
const TAU: f32 = 6.2831853;
const FEWEST_STEPS: f32 = 24.0;
const MOST_STEPS: f32 = 64.0;
const STEP_LENGTH: f32 = 150.0;
const NEAREST_STEP: f32 = 25.0;
const STEP_SHARE: f32 = 0.1;
const MOST_STEPS_IN_ALL: u32 = 128u;
const LIGHT_STEPS: u32 = 5u;
const LIGHT_NEAR_STEPS: u32 = 2u;
const LIGHT_TEXELS: u32 = 384u;
const LIGHT_HEIGHTS: u32 = 16u;
const LIGHT_WORDS: u32 = LIGHT_HEIGHTS - 1u;
const LIGHT_EDGE: f32 = 16.0;
const REACH: f32 = 60000.0;
const DETAIL_REPEAT: f32 = 0.13;
const EXTINCTION: f32 = 0.08;

fn remap(value: f32, low: f32, high: f32, to_low: f32, to_high: f32) -> f32 {
    return to_low + (value - low) * (to_high - to_low) / max(high - low, 1.0e-5);
}

fn dome_direction(uv: vec2<f32>) -> vec3<f32> {
    let elevation = mix(-asin(HORIZON), 0.25 * TAU, uv.y);
    let up = sin(elevation);
    let around = cos(elevation);
    let angle = TAU * uv.x;
    return vec3<f32>(cos(angle) * around, up, sin(angle) * around);
}

fn cloud_offset() -> vec2<f32> {
    return vec2<f32>(clouds.drift.x + clouds.drift.w, clouds.drift.y - clouds.drift.w);
}

fn detail_lod(footprint: f32) -> f32 {
    let texel = clouds.drift.z * DETAIL_REPEAT / f32(textureDimensions(noise_detail).x);
    return log2(max(footprint / max(texel, 1.0e-6), 1.0));
}

fn cloud_density(p: vec3<f32>, eroded: bool, footprint: f32) -> f32 {
    let height = (p.y - clouds.layer.x) / (clouds.layer.y - clouds.layer.x);
    if height <= 0.0 || height >= 1.0 {
        return 0.0;
    }
    let profile = saturate(height * 7.0) * saturate((1.0 - height) * 2.0);
    let offset = cloud_offset();
    let moved = p + vec3<f32>(offset.x, 0.0, offset.y);
    let repeat = clouds.drift.z;
    let gather = textureSampleLevel(noise_shape, noise_sampler,
        vec3<f32>(moved.x / (repeat * 4.3), 0.37, moved.z / (repeat * 4.3)), 0.0).r;
    let local = saturate(clouds.layer.z * (0.4 + 1.2 * smoothstep(0.3, 0.7, gather)));
    let threshold = mix(0.95, 0.4, local);
    if profile <= threshold {
        return 0.0;
    }
    let rounded = textureSampleLevel(noise_shape, noise_sampler, moved / repeat, 0.0).r;
    var cloud = saturate(remap(rounded * profile, threshold, 1.0, 0.0, 1.0));
    if eroded && cloud > 0.0 {
        let wisps = textureSampleLevel(noise_detail, noise_sampler,
            moved / (repeat * DETAIL_REPEAT), detail_lod(footprint)).r;
        let erode = mix(wisps, 1.0 - wisps, saturate(height * 3.0)) * 0.4;
        cloud = saturate(remap(cloud, erode, 1.0, 0.0, 1.0));
    }
    return cloud;
}

fn sun_depth_steps(p: vec3<f32>, first: u32, last: u32) -> f32 {
    var depth = 0.0;
    var along = 0.0;
    var step = 60.0;
    for (var index = 0u; index < last; index++) {
        if index >= first {
            depth += cloud_density(p + clouds.sun.xyz * (along + step * 0.5), false, 0.0) * step;
        }
        along += step;
        step *= 1.8;
    }
    return depth;
}

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

fn sun_depth(p: vec3<f32>) -> f32 {
    let near = sun_depth_steps(p, 0u, LIGHT_NEAR_STEPS);
    let mapped = mapped_far_depth(p);
    var far = mapped.x;
    if mapped.y < 1.0 {
        far = mix(sun_depth_steps(p, LIGHT_NEAR_STEPS, LIGHT_STEPS), mapped.x, mapped.y);
    }
    return near + far;
}

fn sunlight_at(p: vec3<f32>) -> vec3<f32> {
    if clouds.eye.w < 0.5 {
        return clouds.sun_color.rgb;
    }
    let height = cloud_air.radii.w + (p.y - clouds.eye.y) * 0.001;
    return sunlight_over(cloud_transmittance, cloud_table_sampler, cloud_air,
        (p.xz - clouds.eye.xz) * 0.001, height, clouds.sun.xyz);
}

fn henyey_greenstein(cosine: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (2.0 * TAU * pow(1.0 + g2 - 2.0 * g * cosine, 1.5));
}

fn march(origin: vec3<f32>, direction: vec3<f32>, texel: vec2<u32>, limit: f32,
        cone: f32) -> vec4<f32> {
    let base = clouds.layer.x;
    let top = clouds.layer.y;
    let reach = min(REACH, limit);
    let clear = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var near = 0.0;
    var far = reach;
    if abs(direction.y) > 1.0e-5 {
        let to_base = (base - origin.y) / direction.y;
        let to_top = (top - origin.y) / direction.y;
        near = max(min(to_base, to_top), 0.0);
        far = min(max(to_base, to_top), reach);
    } else if origin.y < base || origin.y > top {
        return clear;
    }
    if far <= near {
        return clear;
    }
    let even = (far - near) / clamp((far - near) / STEP_LENGTH, FEWEST_STEPS, MOST_STEPS);
    let jitter = fract(52.9829189 * fract(dot(vec2<f32>(texel), vec2<f32>(0.06711056, 0.00583715))));
    let cosine = dot(direction, clouds.sun.xyz);
    let phase = mix(henyey_greenstein(cosine, 0.8), henyey_greenstein(cosine, -0.3), 0.3);
    let extinction = EXTINCTION * clouds.layer.w;
    var light = vec3<f32>(0.0);
    var passing = 1.0;
    var along = near + jitter * min(even, max(NEAREST_STEP, near * STEP_SHARE));
    for (var index = 0u; index < MOST_STEPS_IN_ALL && along < far; index++) {
        let step = min(even, max(NEAREST_STEP, along * STEP_SHARE));
        let here = along;
        along += step;
        let p = origin + direction * here;
        let density = cloud_density(p, true, here * cone);
        if density <= 0.0 {
            continue;
        }
        let sigma = density * extinction;
        let toward_sun = sun_depth(p) * extinction;
        let direct = exp(-toward_sun) + 0.5 * exp(-toward_sun * 0.5)
            + 0.25 * exp(-toward_sun * 0.25);
        let powder = 1.0 - 0.6 * exp(-2.0 * sigma * 120.0);
        let height = saturate((p.y - base) / (top - base));
        let sky = mix(clouds.horizon.rgb, clouds.zenith.rgb, 0.5 + 0.5 * height);
        let gray = vec3<f32>(dot(sky, vec3<f32>(0.2126, 0.7152, 0.0722)));
        let ambient = mix(sky, gray, 0.6) * (0.5 + 0.7 * height);
        let sunlight = sunlight_at(p);
        let diffuse = sunlight * (0.06 * (0.4 + 0.6 * height));
        let source = sunlight * (direct * powder * phase * 4.0 * 3.14159) + ambient + diffuse;
        let through = exp(-sigma * step);
        light += passing * source * (1.0 - through);
        passing *= through;
        if passing < 0.01 {
            break;
        }
    }
    let fade = exp(-near / 30000.0);
    light = mix(clouds.horizon.rgb * (1.0 - passing), light, fade);
    return vec4<f32>(light, passing);
}

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
        depth += cloud_density(start + sun * ((f32(index) + 0.5) * step), false, 0.0) * step;
    }
    return exp(-depth * EXTINCTION * clouds.layer.w);
}

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
        let slot = vec2<i32>(i32(id.x * 2u + half), i32(row));
        let cell = first + (((slot - first) % size) + size) % size;
        let start = vec3<f32>((f32(cell.x) + 0.5) * clouds.shadow.z, clouds.layer.x,
            (f32(cell.y) + 0.5) * clouds.shadow.z);
        pair[half] = sun_passing(start);
    }
    shadow_cells[row * (cells / 2u) + id.x] = pack2x16float(pair);
}

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

@compute @workgroup_size(8, 8)
fn march_clouds(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = clouds.dome.z + id.y;
    if id.x >= clouds.dome.x || id.y >= clouds.dome.w || row >= clouds.dome.y {
        return;
    }
    let uv = (vec2<f32>(f32(id.x), f32(row)) + 0.5) / vec2<f32>(clouds.dome.xy);
    let cone = max(TAU / f32(clouds.dome.x), (0.25 * TAU + asin(HORIZON)) / f32(clouds.dome.y));
    let origin = select(clouds.eye.xyz, clouds.mirror.xyz, id.z == MIRROR_LAYER);
    textureStore(dome, vec2<u32>(id.x, row), id.z,
        march(origin, dome_direction(uv), vec2<u32>(id.x, row), REACH, cone));
}

fn view_direction(ndc: vec2<f32>) -> vec3<f32> {
    let far = cloud_view.inverse_view_projection * vec4<f32>(ndc, 0.0, 1.0);
    return normalize(far.xyz / far.w - clouds.eye.xyz);
}

var<workgroup> view_clouded: atomic<u32>;

@compute @workgroup_size(8, 8)
fn march_view(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        atomicStore(&view_clouded, 0u);
    }
    workgroupBarrier();
    let pixels = max(cloud_view.pixels.xy, vec2<u32>(1u));
    if all(id.xy < pixels) {
        let depth = textureLoad(view_depth, id.xy, 0);
        var cloud = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        let sky = depth <= 0.0;
        if !sky || cloud_view.pixels.z != 0u {
            let center = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(pixels);
            let ndc = vec2<f32>(center.x * 2.0 - 1.0, 1.0 - center.y * 2.0);
            let direction = view_direction(ndc);
            let apart = 2.0 / vec2<f32>(pixels);
            let cone = max(length(view_direction(ndc + vec2<f32>(apart.x, 0.0)) - direction),
                length(view_direction(ndc - vec2<f32>(0.0, apart.y)) - direction));
            var limit = REACH;
            if !sky {
                let point = cloud_view.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
                limit = length(point.xyz / point.w - clouds.eye.xyz);
            }
            cloud = march(clouds.eye.xyz, direction, id.xy, limit, cone);
        }
        textureStore(view_out, id.xy, cloud);
        if cloud.a < 0.999 {
            atomicStore(&view_clouded, 1u);
        }
    }
    workgroupBarrier();
    if lane == 0u && atomicLoad(&view_clouded) != 0u {
        atomicStore(&composite_draw[0], 3u);
    }
}
