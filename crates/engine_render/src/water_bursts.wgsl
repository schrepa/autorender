// Turbulence reaching water's surface in bursts: the one rule every
// simulation of water draws the events of its turbulence by, the foam
// field's boils and patches (`meshes/water_foam_field.wgsl`) and the live
// solve's turbulent pushes alike. Below its grid a simulation knows how
// turbulent each place is: its turbulence, its eddies' speed, the energy it
// loses, the white water born in it. Turbulence reaches the surface
// intermittently, so its events are random in space and time, Poisson about
// the simulation's rate as their mean. Each is drawn from a hash of the
// cell of the world it stands at and the simulation's step index alone, so
// every host draws the same events however its frames fall. A caller names
// its sites (the cells of a grid anchored in the world, counted from the
// world's origin), how long its events live (`water_burst_slot`: each site
// runs through slots of steps about that long, each slot's events born as
// it begins) and how many it expects at a site over a slot, from the rate
// the simulation holds there (`water_burst_weight`).

// The most steps a slot spans: 2^WATER_BURST_LONGEST, minutes of the
// simulation's steps, longer than any eddy turns over.
const WATER_BURST_LONGEST: u32 = 16u;

// Four independent 32-bit draws for the events at site `site` (a cell of a
// grid anchored in the world, counted from the world's origin) of stream
// `stream` (what the caller draws them for and the level of its grid, each
// caller's own, so draws for different ends never correlate), over the slot
// `slot` (`water_burst_slot`'s): Jarzynski and Olano's pcg4d ("Hash
// Functions for GPU Rendering", JCGT 2020).
fn water_burst_bits(site: vec2<i32>, stream: u32, slot: vec2<u32>) -> vec4<u32> {
    var v = vec4<u32>(bitcast<vec2<u32>>(site), slot.x, (stream << 5u) | slot.y) * 1664525u
        + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> vec4<u32>(16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

// A draw's top 24 bits as a share in [0, 1).
fn water_burst_unit(bits: u32) -> f32 {
    return f32(bits >> 8u) * (1.0 / 16777216.0);
}

// The slot holding the simulation's step `step` (its index, each step
// `seconds` long) at site `site` of stream `stream` (`water_burst_bits`'),
// for events living `life` seconds: a run of 2^m steps, 2^m the power of two
// nearest life / seconds, at least `least` steps (a power of two: as many as
// the longest frame a simulation steps holds, so a frame meets no more than
// one slot's beginning) and at most 2^WATER_BURST_LONGEST, each site's runs
// shifted by a draw of its own, so neighbors' slots begin at steps of their
// own; its index and m. As what sets the life changes, the slots' length
// changes by whole octaves.
fn water_burst_slot(site: vec2<i32>, stream: u32, step: u32, life: f32, seconds: f32,
        least: u32) -> vec2<u32> {
    let steps = max(life / max(seconds, 1.0e-9), 1.0);
    let nearest = u32(clamp(round(log2(steps)), 0.0, f32(WATER_BURST_LONGEST)));
    let octaves = max(nearest, firstTrailingBit(max(least, 1u)));
    let shift = water_burst_bits(site, stream, vec2<u32>(0u, 31u)).x;
    return vec2<u32>((step + shift) >> octaves, octaves);
}

// The seconds slot `slot` (`water_burst_slot`'s) lasts, its steps each
// `seconds` long.
fn water_burst_life(slot: vec2<u32>, seconds: f32) -> f32 {
    return f32(1u << slot.y) * seconds;
}

// The weight of the event a site draws as `drawn` (`water_burst_unit`) over a
// slot where `mean` are expected (Poisson): it fires where one or more
// would, with the chance 1 − e^−mean, and stands for all that would,
// mean / (1 − e^−mean) of them, so the weights' mean is `mean` however many
// it holds; none where none fires.
fn water_burst_weight(mean: f32, drawn: f32) -> f32 {
    let fires = 1.0 - exp(-mean);
    if mean <= 0.0 || drawn >= fires {
        return 0.0;
    }
    return mean / fires;
}
