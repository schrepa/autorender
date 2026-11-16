import package::engine::color::luminance;
import package::engine::math::TAU;
import package::engine::view::camera;
import package::generated::sky::{
    CLOUD_GROUP, CLOUD_HORIZON, CLOUD_REACH, CloudUniform, LIGHT_HEIGHTS, LIGHT_NEAR_STEPS,
    LIGHT_STEPS, LIGHT_TEXELS, MIRROR_LAYER
};
import package::sky::common::sunlight_over;
import package::sky::piece::sky_air;

@group(1) @binding(0) var<uniform> clouds: CloudUniform;
@group(1) @binding(1) var noise_shape: texture_3d<f32>;
@group(1) @binding(2) var noise_sampler: sampler;
@group(1) @binding(4) var noise_detail: texture_3d<f32>;
@group(1) @binding(3) var dome: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(7) var view_depth: texture_depth_2d;
@group(1) @binding(8) var view_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(10) var<storage, read_write> composite_draw: array<atomic<u32>, 4>;
@group(1) @binding(12) var cloud_transmittance: texture_2d<f32>;
@group(1) @binding(13) var cloud_table_sampler: sampler;
@group(1) @binding(14) var<storage, read_write> light_cells: array<u32>;
@group(1) @binding(15) var<storage, read_write> view_tiles: array<u32>;

const LIGHT_WORDS: u32 = LIGHT_HEIGHTS - 1u;
const LIGHT_EDGE: f32 = 16.0;
const EXTINCTION: f32 = 0.08;
const OCTAVES: vec3<f32> = vec3<f32>(1.0, 0.5, 0.25);
const SECOND_OCTAVE: f32 = 0.5;
const THIRD_OCTAVE: f32 = 0.25;
const FULL_SPHERE: f32 = 12.566371;
const GOLDEN: f32 = 0.618034;
const GATHER_SLICE: f32 = 0.37;
const BANK_SLICE: f32 = 0.71;
const SHEET_SLICE: f32 = 0.13;
const COARSE_SLICE: f32 = 0.21;
const FINER_SLICE: f32 = 0.63;

struct Weather {
    cover: f32,
    warp: vec2<f32>,
}

fn dome_direction(uv: vec2<f32>) -> vec3<f32> {
    let elevation = mix(-asin(CLOUD_HORIZON), 0.25 * TAU, uv.y);
    let up = sin(elevation);
    let around = cos(elevation);
    let angle = TAU * uv.x;
    return vec3<f32>(cos(angle) * around, up, sin(angle) * around);
}

fn cloud_offset() -> vec2<f32> {
    return vec2<f32>(clouds.drift.x + clouds.drift.w, clouds.drift.y - clouds.drift.w);
}

fn shape_lod(footprint: f32) -> f32 {
    let texels = f32(textureDimensions(noise_shape).x) / clouds.drift.z;
    return log2(max(footprint * texels, 1.0));
}

fn detail_lod(footprint: f32) -> f32 {
    let texels = f32(textureDimensions(noise_detail).x) * clouds.profile.w;
    return log2(max(footprint * texels, 1.0));
}

fn weather_at(moved: vec2<f32>) -> Weather {
    let gather = textureSampleLevel(noise_shape, noise_sampler,
        vec3<f32>(moved.x * clouds.weather.x, GATHER_SLICE, moved.y * clouds.weather.x), 0.0).r;
    let banks = textureSampleLevel(noise_shape, noise_sampler,
        vec3<f32>(moved.x * clouds.weather.y, BANK_SLICE, moved.y * clouds.weather.y), 0.0).r;
    let bank = mix(clouds.bank.x, clouds.bank.y, smoothstep(clouds.bank.z, clouds.bank.w, banks));
    let group = mix(clouds.gather.x, clouds.gather.y,
        smoothstep(clouds.gather.z, clouds.gather.w, gather));
    return Weather(saturate(clouds.layer.z * bank * group),
        vec2<f32>(gather, banks) * clouds.weather.z);
}

fn cloud_shape(moved: vec3<f32>, height: f32, weather: Weather, footprint: f32) -> f32 {
    let topped = mix(clouds.top.x, clouds.top.y, weather.cover);
    let profile = saturate(height * clouds.profile.z) * saturate((1.0 - height) / (1.0 - topped));
    let threshold = mix(clouds.profile.x, clouds.profile.y, weather.cover);
    if profile <= threshold {
        return 0.0;
    }
    let at = moved / clouds.drift.z + vec3<f32>(weather.warp.x, 0.0, weather.warp.y);
    let rounded = textureSampleLevel(noise_shape, noise_sampler, at, shape_lod(footprint)).r;
    return saturate((rounded * profile - threshold) / max(1.0 - threshold, 1.0e-5));
}

fn eroded(cloud: f32, moved: vec3<f32>, height: f32, footprint: f32) -> f32 {
    let wisps = textureSampleLevel(noise_detail, noise_sampler, moved * clouds.profile.w,
        detail_lod(footprint)).r;
    let erode = mix(wisps, 1.0 - wisps, saturate(height * clouds.top.z)) * clouds.top.w;
    return saturate((cloud - erode) / max(1.0 - erode, 1.0e-5));
}

fn sun_depth_steps(p: vec3<f32>, weather: Weather, first: u32, last: u32) -> f32 {
    let offset = cloud_offset();
    var depth = 0.0;
    var along = 0.0;
    var step = clouds.sampling.y;
    for (var index = 0u; index < last; index++) {
        if index >= first {
            let at = p + clouds.sun.xyz * (along + step * 0.5);
            let height = (at.y - clouds.layer.x) / (clouds.layer.y - clouds.layer.x);
            if height > 0.0 && height < 1.0 {
                depth += cloud_shape(at + vec3<f32>(offset.x, 0.0, offset.y), height, weather,
                    step * clouds.sampling.x) * step;
            }
        }
        along += step;
        step *= clouds.sampling.z;
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

fn sun_depth(p: vec3<f32>, weather: Weather) -> f32 {
    let near = sun_depth_steps(p, weather, 0u, LIGHT_NEAR_STEPS);
    let mapped = mapped_far_depth(p);
    var far = mapped.x;
    if mapped.y < 1.0 {
        far = mix(sun_depth_steps(p, weather, LIGHT_NEAR_STEPS, LIGHT_STEPS), mapped.x, mapped.y);
    }
    return near + far;
}

fn sunlight_at(p: vec3<f32>) -> vec3<f32> {
    let height = sky_air.radii.w + (p.y - clouds.eye.y) * 0.001;
    return sunlight_over(cloud_transmittance, cloud_table_sampler,
        (p.xz - clouds.eye.xz) * 0.001, height, clouds.sun.xyz);
}

fn henyey_greenstein(cosine: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (FULL_SPHERE * pow(1.0 + g2 - 2.0 * g * cosine, 1.5));
}

fn cloud_phase(cosine: f32, narrowed: f32) -> f32 {
    return FULL_SPHERE * mix(henyey_greenstein(cosine, clouds.phase.x * narrowed),
        henyey_greenstein(cosine, clouds.phase.y * narrowed), clouds.phase.z);
}

fn march_jitter(texel: vec2<u32>) -> f32 {
    let spread = fract(52.9829189
        * fract(dot(vec2<f32>(texel), vec2<f32>(0.06711056, 0.00583715))));
    return fract(spread + f32(clouds.layers.y) * GOLDEN);
}

fn march_slab(origin: vec3<f32>, direction: vec3<f32>, texel: vec2<u32>, reach: f32,
        cone: f32) -> vec4<f32> {
    let base = clouds.layer.x;
    let top = clouds.layer.y;
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
    let even = (far - near)
        / clamp((far - near) / clouds.march.x, f32(clouds.counts.x), f32(clouds.counts.y));
    let cosine = dot(direction, clouds.sun.xyz);
    let phases = OCTAVES * vec3<f32>(cloud_phase(cosine, 1.0),
        cloud_phase(cosine, SECOND_OCTAVE), cloud_phase(cosine, THIRD_OCTAVE));
    let turned = 0.5 - 0.5 * cosine;
    let extinction = EXTINCTION * clouds.layer.w;
    let thickness = top - base;
    let offset = cloud_offset();
    let sun_slope = max(abs(clouds.sun.y), 1.0e-4);
    var light = vec3<f32>(0.0);
    var passing = 1.0;
    var along = near + march_jitter(texel) * min(even, max(clouds.march.y, near * clouds.march.z));
    var careful = 0.0;
    var resume = along;
    var striding = false;
    for (var index = 0u; index < clouds.counts.z && along < far; index++) {
        let fine = min(even, max(clouds.march.y, along * clouds.march.z));
        let here = along;
        let p = origin + direction * here;
        let height = (p.y - base) / thickness;
        let moved = p + vec3<f32>(offset.x, 0.0, offset.y);
        let weather = weather_at(moved.xz);
        let footprint = max(here * cone, fine * clouds.sampling.x);
        var density = 0.0;
        if height > 0.0 && height < 1.0 {
            density = cloud_shape(moved, height, weather, footprint);
        }
        if density <= 0.0 {
            striding = here >= careful;
            resume = here + fine;
            along = here + select(fine, fine * clouds.march.w, striding);
            continue;
        }
        if striding {
            striding = false;
            careful = here;
            along = resume;
            continue;
        }
        along = here + fine;
        density = eroded(density, moved, height, footprint);
        if density <= 0.0 {
            continue;
        }
        let sigma = density * extinction;
        let sunward = select(1.0 - height, height, clouds.sun.y < 0.0);
        let beyond = max(min(sunward * thickness / sun_slope, CLOUD_REACH) - clouds.sampling.w,
            0.0);
        let toward_sun = sun_depth(p, weather) * extinction
            + clouds.lighting.w * clouds.layer.w * weather.cover * sunward * beyond;
        let direct = dot(exp(-toward_sun * OCTAVES), phases);
        let powder = 1.0 - clouds.lighting.x * turned * exp(-2.0 * sigma * clouds.lighting.y);
        let sky = mix(clouds.horizon.rgb, clouds.zenith.rgb, 0.5 + 0.5 * height);
        let ambient = mix(sky, vec3<f32>(luminance(sky)), clouds.lighting.z)
            * mix(clouds.sky_light.x, clouds.sky_light.y, height);
        let source = sunlight_at(p) * (direct * powder) + ambient;
        let through = exp(-sigma * fine);
        light += passing * source * (1.0 - through);
        passing *= through;
        if passing < 0.01 {
            break;
        }
    }
    let fade = exp(-near * clouds.sky_light.z);
    light = mix(clouds.horizon.rgb * (1.0 - passing), light, fade);
    return vec4<f32>(light, passing);
}

fn high_cloud(origin: vec3<f32>, direction: vec3<f32>, reach: f32, cone: f32) -> vec4<f32> {
    let clear = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let rise = clouds.high.y - origin.y;
    if clouds.high.x <= 0.0 || rise <= 0.0 || direction.y <= 1.0e-5 {
        return clear;
    }
    let along = rise / direction.y;
    if along >= reach {
        return clear;
    }
    let p = origin + direction * along;
    let moved = p.xz + cloud_offset();
    let sheet = textureSampleLevel(noise_shape, noise_sampler,
        vec3<f32>(moved.x * clouds.weather.w, SHEET_SLICE, moved.y * clouds.weather.w), 0.0).r;
    let threshold = mix(clouds.high_turn.z, clouds.high_turn.w, clouds.high.x);
    let cover = smoothstep(threshold, threshold + clouds.high.w, sheet);
    if cover <= 0.0 {
        return clear;
    }
    let streak = vec2<f32>(moved.x * clouds.high_turn.x + moved.y * clouds.high_turn.y,
        moved.y * clouds.high_turn.x - moved.x * clouds.high_turn.y) * clouds.high_fibre.xy;
    let texels = f32(textureDimensions(noise_detail).x) * clouds.high_fibre.y;
    let lod = log2(max(along * cone * texels / sqrt(direction.y), 1.0));
    let coarse = textureSampleLevel(noise_detail, noise_sampler,
        vec3<f32>(streak.x, COARSE_SLICE, streak.y), lod).r;
    let finer = textureSampleLevel(noise_detail, noise_sampler,
        vec3<f32>(streak.x * 2.0, FINER_SLICE, streak.y * 2.0), lod + 1.0).r;
    let fibres = smoothstep(clouds.high_fibre.z, clouds.high_fibre.w,
        (coarse + 0.5 * finer) / 1.5);
    let fade = exp(-along * clouds.sky_light.z);
    let ended = exp(-CLOUD_REACH * clouds.sky_light.z);
    let covered = (1.0 - exp(-clouds.high.z * cover * fibres))
        * saturate((fade - ended) / (1.0 - ended));
    let sky = 0.5 * (clouds.zenith.rgb + clouds.horizon.rgb);
    let source = sunlight_at(p) * cloud_phase(dot(direction, clouds.sun.xyz), 1.0) + sky;
    return vec4<f32>(mix(clouds.horizon.rgb, source, fade) * covered, 1.0 - covered);
}

fn march(origin: vec3<f32>, direction: vec3<f32>, texel: vec2<u32>, limit: f32,
        cone: f32) -> vec4<f32> {
    let reach = min(CLOUD_REACH, limit);
    let slab = march_slab(origin, direction, texel, reach, cone);
    let high = high_cloud(origin, direction, reach, cone);
    return vec4<f32>(slab.rgb + slab.a * high.rgb, slab.a * high.a);
}

@compute @workgroup_size(CLOUD_GROUP, CLOUD_GROUP)
fn light_clouds(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = clouds.light_rows.x + id.y;
    if id.x >= LIGHT_TEXELS || id.y >= clouds.light_rows.y || row >= LIGHT_TEXELS {
        return;
    }
    let column = clouds.light.xy + (vec2<f32>(f32(id.x), f32(row)) + 0.5) * clouds.light.z
        - cloud_offset();
    let thickness = clouds.layer.y - clouds.layer.x;
    let at = (row * LIGHT_TEXELS + id.x) * LIGHT_WORDS;
    let weather = weather_at(column + cloud_offset());
    var below = 0.0;
    for (var level = 0u; level < LIGHT_HEIGHTS; level++) {
        let y = clouds.layer.x + thickness * (f32(level) + 0.5) / f32(LIGHT_HEIGHTS);
        let depth = sun_depth_steps(vec3<f32>(column.x, y, column.y), weather, LIGHT_NEAR_STEPS,
            LIGHT_STEPS);
        if level > 0u {
            light_cells[at + level - 1u] = pack2x16float(vec2<f32>(below, depth));
        }
        below = depth;
    }
}

@compute @workgroup_size(CLOUD_GROUP, CLOUD_GROUP)
fn march_clouds(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = clouds.dome.z + id.y;
    if id.x >= clouds.dome.x || id.y >= clouds.dome.w || row >= clouds.dome.y {
        return;
    }
    let uv = (vec2<f32>(f32(id.x), f32(row)) + 0.5) / vec2<f32>(clouds.dome.xy);
    let cone = max(TAU / f32(clouds.dome.x), (0.25 * TAU + asin(CLOUD_HORIZON)) / f32(clouds.dome.y));
    let origin = select(clouds.eye.xyz, clouds.mirror.xyz, id.z == MIRROR_LAYER);
    textureStore(dome, vec2<u32>(id.x, row), clouds.layers.x + id.z,
        march(origin, dome_direction(uv), vec2<u32>(id.x, row), CLOUD_REACH, cone));
}

fn view_direction(ndc: vec2<f32>) -> vec3<f32> {
    let far = camera.inverse_view_projection * vec4<f32>(ndc, 0.0, 1.0);
    return normalize(far.xyz / far.w - clouds.eye.xyz);
}

const VIEW_CLOUDED: u32 = 1u;
const VIEW_SHOWN: u32 = 2u;

var<workgroup> view_clouded: atomic<u32>;

@compute @workgroup_size(CLOUD_GROUP, CLOUD_GROUP)
fn march_view(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) tile: vec3<u32>) {
    if lane == 0u {
        atomicStore(&view_clouded, 0u);
    }
    workgroupBarrier();
    let pixels = max(vec2<u32>(camera.region.xy), vec2<u32>(1u));
    let inside = all(id.xy < pixels);
    var cloud = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if inside {
        let depth = textureLoad(view_depth, id.xy, 0);
        let sky = depth <= 0.0;
        if !sky || clouds.eye.w != 0.0 {
            let center = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(pixels);
            let ndc = vec2<f32>(center.x * 2.0 - 1.0, 1.0 - center.y * 2.0);
            let direction = view_direction(ndc);
            let apart = 2.0 / vec2<f32>(pixels);
            let cone = max(length(view_direction(ndc + vec2<f32>(apart.x, 0.0)) - direction),
                length(view_direction(ndc - vec2<f32>(0.0, apart.y)) - direction));
            var limit = CLOUD_REACH;
            if !sky {
                let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
                limit = length(point.xyz / point.w - clouds.eye.xyz);
            }
            cloud = march(clouds.eye.xyz, direction, id.xy, limit, cone);
        }
        if cloud.a < 0.999 {
            atomicOr(&view_clouded, VIEW_CLOUDED);
        }
        if !(cloud.a >= 1.0) {
            atomicOr(&view_clouded, VIEW_SHOWN);
        }
    }
    workgroupBarrier();
    let seen = atomicLoad(&view_clouded);
    if (seen & VIEW_SHOWN) == 0u {
        return;
    }
    if inside {
        textureStore(view_out, id.xy, cloud);
    }
    if lane == 0u {
        if (seen & VIEW_CLOUDED) != 0u {
            atomicStore(&composite_draw[0], 6u);
        }
        view_tiles[atomicAdd(&composite_draw[1], 1u)] = (tile.x << 16u) | tile.y;
    }
}
