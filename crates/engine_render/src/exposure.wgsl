// Auto exposure: a meter of the frame the scene target holds, in a
// histogram of its light in stops, and the exposure later frames show,
// eased toward what the meter asks.

struct ExposureUniform {
    // x: the exposure the atmosphere authors; y: seconds since the last
    // metered frame, 0 taking what the meter asks at once (a repeated time
    // of the same view, as exact frames are), -1 keeping the exposure (a
    // repeated time of a moved view); z: how far exposure follows the view
    // (0 to 1); w: stops added after.
    frame: vec4<f32>,
    // x: the most stops the view raises exposure (none at night); y: the
    // most it lowers it;
    // z, w: seconds over which exposure eases toward a darker view and
    // toward a brighter one.
    reach: vec4<f32>,
    // x, y: the shares of the darkest and the brightest samples the
    // average leaves out; z: how many times what the corners count the
    // middle of the view counts; w: how much the sky counts.
    meter: vec4<f32>,
    // The physically based sky's light at noon (0 without it), the power
    // of its light's ratio to it that exposure rises by, and the most stops
    // it rises by (fewer at night).
    day: vec4<f32>,
    // The pixels of the scene the frame drew, from its top-left corner.
    drawn: vec4<f32>,
}

// The exposure the frame shows, which the lighting copies; its stops from
// the authored exposure; the meter's last average, log2 of the light
// before exposure; 1 once these hold a metered frame; how many stops
// brighter the view is than its light gives a view; and the stops the time
// of day raises exposure by.
struct ExposureState {
    exposure: f32,
    stops: f32,
    metered: f32,
    held: f32,
    view: f32,
    day: f32,
}

// The scene's lighting's leading fields (`scene.wgsl`).
struct Lighting {
    sun: vec4<f32>,
    sun_color: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
}

@group(0) @binding(0) var<uniform> settings: ExposureUniform;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var<storage, read_write> bins: array<atomic<u32>, EXPOSURE_BINS>;
@group(0) @binding(4) var<storage, read_write> state: ExposureState;
@group(0) @binding(5) var<uniform> lighting: Lighting;

// The histogram's bins, and the samples the meter reads across and down
// the view.
const EXPOSURE_BINS: u32 = 128u;
const EXPOSURE_GRID: u32 = 128u;
// The light the bins span, in stops: log2 of linear light before exposure.
const LIGHT_LOW: f32 = -20.0;
const LIGHT_HIGH: f32 = 12.0;
// A sample's weight of one, in the bins' fixed point.
const WEIGHT_UNIT: f32 = 1024.0;
// The meter's average of an open view in daylight, in stops over the light
// falling on level ground: what the authored exposure is made for.
const VIEW_LIGHT: f32 = -3.9;
// Views within this many stops of it keep the time of day's exposure: the
// light falling on level ground gives a view's only roughly.
const VIEW_SLACK: f32 = 1.0;

var<workgroup> counted: array<atomic<u32>, EXPOSURE_BINS>;

// One sample of a grid EXPOSURE_GRID across and down the view: its light
// before exposure (the luminance shown, the filmic curve undone, over the
// exposure shown), into the two bins around it in stops by how near it
// lies to each; weighted toward the middle of the view, and by the sky's
// weight where nothing was drawn.
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
    // Black reads as half the 8-bit target's first step: what lies below it
    // cannot be told apart.
    let shown = clamp(luminance(textureLoad(scene_color, pixel, 0).rgb), 1.5e-4, 1.0);
    let light = unfilmic(shown) / max(state.exposure, 1.0e-6);
    // 0 at the middle, 1 at the corners.
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

// The exposure later frames show, in stops over the authored one, from two
// parts. The time of day's: under the physically based sky, as the sky's
// own adaptation rises toward its light at noon, but within the night's
// limit, so dusk dims and night stays dark. The view's: the meter's average
// (the light between the shares of the darkest and the brightest samples
// left out, averaged in stops) against what the light falling on level
// ground gives an open view, beyond a slack, part of the way and within its
// limits, so a cave by day brightens and a view of the bright sky darkens.
// Eased from the last over the seconds since, slower toward a darker view
// than toward a brighter one.
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
    // The light falling on level ground, the sun's and the sky's, as the
    // physically based sky's own adaptation takes it.
    let falling = luminance(lighting.sun_color.rgb) * max(lighting.sun.y, 0.0)
        + SKY_PI * luminance(0.5 * (lighting.zenith.rgb + lighting.horizon.rgb));
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
