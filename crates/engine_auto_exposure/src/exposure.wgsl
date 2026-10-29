import package::engine::color::{luminance, unfilmic};
import package::face::sky::{horizon, zenith};
import package::face::sun::{sun_color, sun_direction};
import package::frame::view::PI;

struct ExposureUniform {
    frame: vec4<f32>,
    reach: vec4<f32>,
    meter: vec4<f32>,
    day: vec4<f32>,
    drawn: vec4<f32>,
}

struct ExposureState {
    exposure: f32,
    stops: f32,
    metered: f32,
    held: f32,
    view: f32,
    day: f32,
}

@group(0) @binding(0) var<uniform> settings: ExposureUniform;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>, EXPOSURE_BINS>;
@group(0) @binding(4) var<storage, read_write> state: ExposureState;

const EXPOSURE_BINS: u32 = 128u;
const EXPOSURE_GRID: u32 = 128u;
const LIGHT_LOW: f32 = -20.0;
const LIGHT_HIGH: f32 = 12.0;
const WEIGHT_UNIT: f32 = 1024.0;
const VIEW_LIGHT: f32 = -3.9;
const VIEW_SLACK: f32 = 1.0;

var<workgroup> counted: array<atomic<u32>, EXPOSURE_BINS>;

@compute @workgroup_size(16, 16)
fn meter_exposure(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
) {
    if index < EXPOSURE_BINS {
        atomicStore(&counted[index], 0u);
    }
    workgroupBarrier();
    let size = max(vec2<u32>(settings.drawn.xy), vec2<u32>(1u));
    let place = (vec2<f32>(id.xy) + 0.5) / f32(EXPOSURE_GRID);
    let pixel = min(vec2<u32>(place * vec2<f32>(size)), size - 1u);
    let shown = clamp(luminance(textureLoad(scene_color, pixel, 0).rgb), 1.5e-4, 1.0);
    let light = unfilmic(shown) / max(state.exposure, 1.0e-6);
    let out = 2.0 * dot(place - 0.5, place - 0.5);
    var weight = mix(settings.meter.z, 1.0, out);
    if textureLoad(scene_depth, pixel, 0) <= 0.0 {
        weight *= settings.meter.w;
    }
    let at = clamp(
        (log2(max(light, 1.0e-30)) - LIGHT_LOW) / (LIGHT_HIGH - LIGHT_LOW) * f32(EXPOSURE_BINS)
            - 0.5,
        0.0,
        f32(EXPOSURE_BINS - 1u),
    );
    let low = min(u32(at), EXPOSURE_BINS - 2u);
    let near = at - f32(low);
    let units = weight * WEIGHT_UNIT;
    atomicAdd(&counted[low], u32(units * (1.0 - near) + 0.5));
    atomicAdd(&counted[low + 1u], u32(units * near + 0.5));
    workgroupBarrier();
    if index < EXPOSURE_BINS {
        let count = atomicLoad(&counted[index]);
        if count != 0u {
            atomicAdd(&bins[index], count);
        }
    }
}

@compute @workgroup_size(1)
fn adapt_exposure() {
    var total = 0.0;
    for (var bin = 0u; bin < EXPOSURE_BINS; bin++) {
        total += f32(atomicLoad(&bins[bin]));
    }
    let skip = total * settings.meter.x;
    let kept = total * max(1.0 - settings.meter.x - settings.meter.y, 0.0);
    var below = 0.0;
    var sum = 0.0;
    var weight = 0.0;
    for (var bin = 0u; bin < EXPOSURE_BINS; bin++) {
        let count = f32(atomicLoad(&bins[bin]));
        let part = max(min(below + count, skip + kept) - max(below, skip), 0.0);
        let stops = LIGHT_LOW + (f32(bin) + 0.5) * (LIGHT_HIGH - LIGHT_LOW) / f32(EXPOSURE_BINS);
        sum += part * stops;
        weight += part;
        below += count;
    }
    let held = state.held != 0.0;
    var metered = state.metered;
    if weight > 0.0 {
        metered = sum / weight;
    } else if !held {
        return;
    }
    let falling = luminance(sun_color()) * max(sun_direction().y, 0.0)
        + PI * luminance(0.5 * (zenith() + horizon()));
    let view = metered - log2(max(falling, 1.0e-12)) - VIEW_LIGHT;
    var day = 0.0;
    if settings.day.x > 0.0 {
        day = clamp(settings.day.y * log2(settings.day.x / max(falling, 1.0e-5)), 0.0,
            settings.day.z);
    }
    let beyond = sign(view) * max(abs(view) - VIEW_SLACK, 0.0);
    let wanted = day + clamp(-beyond * settings.frame.z, -settings.reach.y, settings.reach.x)
        + settings.frame.w;
    var stops = wanted;
    if held && settings.frame.y < 0.0 {
        stops = state.stops;
    } else if held && settings.frame.y > 0.0 {
        let seconds = select(settings.reach.w, settings.reach.z, wanted > state.stops);
        stops = mix(state.stops, wanted, 1.0 - exp(-settings.frame.y / max(seconds, 1.0e-3)));
    }
    let authored = max(settings.frame.x, 1.0e-6);
    state = ExposureState(authored * exp2(stops), stops, metered, 1.0, view, day);
}
