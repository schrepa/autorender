import package::face::heightfield::{WATER_HELD, terrain_uniform, terrain_view_uniform};
import package::frame::view::{PI, detail_across, eye_world};
import package::water::bursts::{water_burst_bits, water_burst_life, water_burst_slot,
    water_burst_weight};
import package::water::flow::{WATER_STEEPEST, water_stirring};
import package::water::foam::water_eddy_size;
import package::water::footprint::{WATER_SMOOTHEST, water_spread};
import package::water::surface::{WATER_BREAK_FULL, WATER_BREAK_ONSET, WATER_BREAK_THINNEST,
    WATER_GRAVITY, water_depth, water_eddies, water_eddy_froude, water_lift, water_pixel,
    water_relative};

const PARTICLE_GROUP: u32 = 256u;
const PARTICLE_SIDE: u32 = 8u;
const PARTICLE_LATTICE: u32 = 8u;
const PARTICLE_MARGIN: u32 = 2u;
const PARTICLE_FOAM_LIVE: u32 = 0u;
const PARTICLE_WAVES_LIVE: u32 = 1u;
const PARTICLE_FOAM_BORN: u32 = 2u;
const PARTICLE_WAVES_BORN: u32 = 3u;
const PARTICLE_FOAM_DROPPED: u32 = 4u;
const PARTICLE_WAVES_DROPPED: u32 = 5u;
const PARTICLE_FOAM_BEFORE: u32 = 6u;
const PARTICLE_WAVES_BEFORE: u32 = 7u;
const PARTICLE_ARGS: u32 = 8u;
const PARTICLE_DRAW: u32 = 14u;
const PARTICLE_ITEMS: u32 = 18u;
const WATER_FOAM_LIFE: f32 = 20.0;
const WATER_FOAM_WET: f32 = 0.005;
const WATER_FOAM_LOSS: f32 = 0.02;
const WATER_FOAM_POWER: f32 = 500.0;
const WATER_DENSITY: f32 = 1000.0;
const WATER_SURFACE_BEND: f32 = 7.0;
const WATER_SURFACE_BEND_MOST: f32 = 1.0;
const WATER_SURFACE_DRIFT: f32 = 0.03;
const PARTICLE_RISE: f32 = 0.25;
const WATER_FOAM_KEPT: f32 = 0.3;
const WATER_FOAM_FRESH: f32 = 2.0;
const PARTICLE_QUANTUM: f32 = 1.7777778;
const PARTICLE_THINNEST: f32 = 0.0625;
const PARTICLE_STRAIN: f32 = 0.5;
const PARTICLE_TEAR: f32 = 8.0;
const PARTICLE_NARROWEST: f32 = 0.01;
const PARTICLE_STRAINED_MOST: f32 = 0.5;
const PARTICLE_EVENT_MOST: u32 = 16u;
const PARTICLE_FIXED: f32 = 65536.0;
const PARTICLE_STRETCH: f32 = 2.0;
const PARTICLE_STRETCH_MOST: f32 = 16.0;
const PARTICLE_SMALLEST: f32 = 4.0;
const PARTICLE_STREAMS: u32 = 32u;
const PARTICLE_STREAM_SPAN: u32 = 64u;
const PARTICLE_BURSTS: u32 = 0u;
const PARTICLE_NO_TEXEL: u32 = 0xffffffffu;

struct ParticleFrame {
    window: vec4<i32>,
    lattice: vec4<i32>,
    step: vec4<f32>,
    flags: vec4<u32>,
    pool: vec4<u32>,
    eddies: vec4<f32>,
}

@group(2) @binding(32) var<uniform> particle_frame: ParticleFrame;
@group(2) @binding(33) var<storage, read_write> particle_lattice: array<vec4<u32>>;
@group(2) @binding(34) var<storage, read_write> particle_pool: array<vec4<u32>>;
@group(2) @binding(35) var<storage, read_write> particle_counts: array<atomic<u32>>;
@group(2) @binding(36) var<storage, read_write> particle_splat: array<atomic<i32>>;
@group(2) @binding(37) var particle_source: texture_2d<f32>;
@group(2) @binding(38) var particle_target: texture_storage_2d<rgba16float, write>;
@group(2) @binding(39) var particle_spread_source: texture_2d<f32>;
@group(2) @binding(40) var particle_spread_target: texture_storage_2d<rgba16float, write>;
@group(2) @binding(48) var<storage, read> foam_pool: array<vec4<u32>>;

struct ParticleWater {
    velocity: vec2<f32>,
    depth: f32,
    turbulence: f32,
    slope: vec2<f32>,
    surface: f32,
}

struct ParticleCarrier {
    current: vec2<f32>,
    churn: f32,
    depth: f32,
    stirring: f32,
    surface: f32,
    wet: f32,
}

struct Particle {
    at: vec2<f32>,
    wave: bool,
    amounts: vec2<f32>,
    axis: vec2<f32>,
    minor: f32,
    surface: f32,
    amplitude: f32,
    stretch: f32,
    heading: f32,
    radius: f32,
}

struct ParticleWords {
    first: vec4<u32>,
    second: vec4<u32>,
}

fn particle_remainder(value: vec2<i32>, cells: i32) -> vec2<i32> {
    return (value % cells + cells) % cells;
}

fn particle_spacing() -> f32 {
    return particle_frame.step.x * f32(PARTICLE_LATTICE);
}

fn particle_quantum(world: vec3<f32>) -> f32 {
    let size = max(particle_frame.step.x, detail_across(world));
    return size * size * PARTICLE_QUANTUM;
}

fn particle_eddy(depth: f32) -> f32 {
    return water_eddy_size(depth, particle_frame.step.x, particle_frame.eddies.y);
}

fn particle_draining(depth: f32) -> f32 {
    return max(depth, WATER_BREAK_THINNEST) / PARTICLE_RISE;
}

fn particle_spline(within: f32) -> vec3<f32> {
    let before = 0.5 - within;
    let after = 0.5 + within;
    return vec3<f32>(0.5 * before * before, 0.75 - within * within, 0.5 * after * after);
}

fn particle_spline_change(within: f32) -> vec3<f32> {
    return vec3<f32>(within - 0.5, -2.0 * within, within + 0.5);
}

fn particle_lattice_at(point: vec2<i32>, carrier: bool) -> u32 {
    let points = particle_frame.lattice.z;
    let at = clamp(point, vec2<i32>(0), vec2<i32>(points - 1));
    let index = u32(at.y * points + at.x);
    return select(index, index + u32(points * points), carrier);
}

fn particle_water(point: vec2<i32>) -> ParticleWater {
    let words = particle_lattice[particle_lattice_at(point, false)];
    let deep = unpack2x16float(words.y);
    var water: ParticleWater;
    water.velocity = unpack2x16float(words.x);
    water.depth = deep.x;
    water.turbulence = deep.y;
    water.slope = unpack2x16float(words.z);
    water.surface = bitcast<f32>(words.w);
    return water;
}

fn particle_beside(point: vec2<i32>, here: ParticleWater) -> ParticleWater {
    let water = particle_water(point);
    if water.depth <= WATER_FOAM_WET {
        return here;
    }
    return water;
}

fn particle_surface(here: ParticleWater, change: mat2x2<f32>) -> vec2<f32> {
    let u = here.velocity;
    let turning = change * u;
    let cubed = max(dot(u, u) * length(u), 1.0e-12);
    let tangent = clamp(WATER_SURFACE_BEND * here.depth * (u.x * turning.y - u.y * turning.x)
        / cubed, -WATER_SURFACE_BEND_MOST, WATER_SURFACE_BEND_MOST);
    let turned = vec2<f32>(u.x + tangent * u.y, u.y - tangent * u.x)
        * inverseSqrt(1.0 + tangent * tangent);
    return turned + WATER_SURFACE_DRIFT * particle_frame.step.zw;
}

fn particle_carrier_point(point: vec2<i32>) -> ParticleCarrier {
    let words = particle_lattice[particle_lattice_at(point, true)];
    let churning = unpack2x16float(words.y);
    var carrier: ParticleCarrier;
    carrier.current = unpack2x16float(words.x);
    carrier.churn = churning.x;
    carrier.depth = churning.y;
    carrier.stirring = unpack2x16float(words.z).x;
    carrier.surface = bitcast<f32>(words.w);
    carrier.wet = select(0.0, 1.0, carrier.depth > 0.0);
    return carrier;
}

fn particle_weighed(sum: ParticleCarrier, carrier: ParticleCarrier, weight: f32)
        -> ParticleCarrier {
    var added = sum;
    added.current += carrier.current * weight;
    added.churn += carrier.churn * weight;
    added.depth += carrier.depth * weight;
    added.stirring += carrier.stirring * weight;
    added.surface += carrier.surface * carrier.wet * weight;
    added.wet += carrier.wet * weight;
    return added;
}

fn particle_carrier(world: vec2<f32>) -> ParticleCarrier {
    let grid = world / particle_spacing() - vec2<f32>(particle_frame.lattice.xy);
    let nearest = floor(grid + 0.5);
    let along = particle_spline(grid.x - nearest.x);
    let across = particle_spline(grid.y - nearest.y);
    let first = vec2<i32>(nearest) - vec2<i32>(1);
    var carrier: ParticleCarrier;
    for (var row = 0; row < 3; row++) {
        for (var column = 0; column < 3; column++) {
            carrier = particle_weighed(carrier,
                particle_carrier_point(first + vec2<i32>(column, row)),
                along[column] * across[row]);
        }
    }
    if carrier.wet > 0.0 {
        carrier.surface /= carrier.wet;
    }
    return carrier;
}

fn particle_shear(world: vec2<f32>) -> mat2x2<f32> {
    let spacing = particle_spacing();
    let grid = world / spacing - vec2<f32>(particle_frame.lattice.xy);
    let nearest = floor(grid + 0.5);
    let along = particle_spline(grid.x - nearest.x);
    let across = particle_spline(grid.y - nearest.y);
    let along_change = particle_spline_change(grid.x - nearest.x) / spacing;
    let across_change = particle_spline_change(grid.y - nearest.y) / spacing;
    let first = vec2<i32>(nearest) - vec2<i32>(1);
    var x = vec2<f32>(0.0);
    var z = vec2<f32>(0.0);
    for (var row = 0; row < 3; row++) {
        for (var column = 0; column < 3; column++) {
            let current = particle_carrier_point(first + vec2<i32>(column, row)).current;
            x += current * (along_change[column] * across[row]);
            z += current * (along[column] * across_change[row]);
        }
    }
    return mat2x2<f32>(x, z);
}

struct ParticleBoils {
    current: vec2<f32>,
    shear: mat2x2<f32>,
}

fn particle_boils(world: vec2<f32>) -> ParticleBoils {
    let side = 2.0 * particle_frame.step.x;
    let grid = world / side - 0.5;
    let first = floor(grid);
    let within = grid - first;
    let cell = vec2<i32>(first);
    let a = particle_boil_texel(cell);
    let b = particle_boil_texel(cell + vec2<i32>(1, 0));
    let c = particle_boil_texel(cell + vec2<i32>(0, 1));
    let d = particle_boil_texel(cell + vec2<i32>(1, 1));
    var boils: ParticleBoils;
    boils.current = mix(mix(a, b, within.x), mix(c, d, within.x), within.y);
    boils.shear = mat2x2<f32>(mix(b - a, d - c, within.y), mix(c - a, d - b, within.x)) * (1.0 / side);
    return boils;
}

fn particle_boil_texel(cell: vec2<i32>) -> vec2<f32> {
    let finest = 2 * cell;
    let first = particle_frame.window.zw;
    if any(finest < first) || any(finest + 2 > first + particle_frame.lattice.w) {
        return vec2<f32>(0.0);
    }
    let at = particle_remainder(cell, particle_frame.lattice.w / 2);
    return textureLoad(particle_spread_source, at, 0).yz;
}

fn particle_read(first: vec4<u32>, second: vec4<u32>, wave: bool) -> Particle {
    var particle: Particle;
    particle.at = bitcast<vec2<f32>>(first.xy);
    particle.wave = wave;
    let held = bitcast<vec2<f32>>(first.zw);
    let shape = bitcast<vec3<f32>>(second.xyz);
    if !wave {
        particle.amounts = held;
        particle.axis = shape.xy;
        particle.minor = shape.z;
        particle.surface = bitcast<f32>(second.w);
        return particle;
    }
    particle.amplitude = held.x;
    particle.stretch = held.y;
    particle.heading = shape.x;
    particle.radius = shape.y;
    return particle;
}

fn particle_words(particle: Particle) -> ParticleWords {
    var words: ParticleWords;
    if !particle.wave {
        words.first = bitcast<vec4<u32>>(vec4<f32>(particle.at, particle.amounts));
        words.second = vec4<u32>(bitcast<vec3<u32>>(vec3<f32>(particle.axis, particle.minor)),
            bitcast<u32>(particle.surface));
        return words;
    }
    words.first = bitcast<vec4<u32>>(vec4<f32>(particle.at, particle.amplitude,
        particle.stretch));
    words.second = vec4<u32>(bitcast<vec2<u32>>(vec2<f32>(particle.heading, particle.radius)),
        0u, 0u);
    return words;
}

fn particle_at(index: u32, wave: bool) -> Particle {
    return particle_read(particle_pool[2u * index], particle_pool[2u * index + 1u], wave);
}

fn particle_put(index: u32, particle: Particle) {
    let words = particle_words(particle);
    particle_pool[2u * index] = words.first;
    particle_pool[2u * index + 1u] = words.second;
}

fn particle_raft(at: vec2<f32>, amounts: vec2<f32>, radius: f32) -> Particle {
    var particle: Particle;
    particle.at = at;
    particle.amounts = amounts;
    particle.axis = vec2<f32>(radius, 0.0);
    particle.minor = radius;
    return particle;
}

fn particle_thickness(particle: Particle) -> vec2<f32> {
    let area = PI * length(particle.axis) * particle.minor;
    return particle.amounts / max(area, 1.0e-12);
}

fn particle_square() -> vec4<f32> {
    let spacing = particle_spacing();
    let margin = i32(PARTICLE_MARGIN);
    let first = vec2<f32>(particle_frame.lattice.xy + margin) * spacing;
    let across = f32(particle_frame.lattice.z - 2 * margin - 1) * spacing;
    return vec4<f32>(first, first + across);
}

fn particle_window() -> vec4<f32> {
    let texel = particle_frame.step.x;
    let first = vec2<f32>(particle_frame.window.xy) * texel;
    return vec4<f32>(first, first + f32(particle_frame.lattice.w) * texel);
}

fn particle_inside(world: vec2<f32>, square: vec4<f32>) -> bool {
    return all(world >= square.xy) && all(world < square.zw);
}

fn particle_site_in(site: vec2<f32>, square: vec4<f32>) -> bool {
    let half = 0.5 * particle_spacing();
    return all(site - half >= square.xy) && all(site + half <= square.zw);
}

fn particle_stream(kind: u32, k: u32) -> u32 {
    return PARTICLE_STREAMS + kind * PARTICLE_STREAM_SPAN + k;
}

fn particle_draws(site: vec2<i32>, kind: u32, k: u32, slot: vec2<u32>) -> vec4<f32> {
    let bits = water_burst_bits(site, particle_stream(kind, k), slot);
    return vec4<f32>(bits >> vec4<u32>(8u)) * (1.0 / 16777216.0);
}

fn particle_slot(site: vec2<i32>, kind: u32, life: f32) -> vec3<u32> {
    let stream = particle_stream(kind, 0u);
    let seconds = particle_frame.eddies.x;
    let least = particle_frame.flags.w;
    let slot = water_burst_slot(site, stream, particle_frame.flags.z, life, seconds, least);
    let begun = water_burst_slot(site, stream, particle_frame.flags.y, life, seconds, least);
    return vec3<u32>(slot, u32(slot.x != begun.x));
}

fn particle_quanta(amount: f32, quantum: f32, drawn: f32) -> vec2<f32> {
    let quanta = max(amount, 0.0) / quantum;
    let most = f32(PARTICLE_EVENT_MOST);
    if quanta > most { // filter: rafts no smaller than a pixel laying the same white water, their mean the amount
        return vec2<f32>(most, amount / most);
    }
    return vec2<f32>(floor(quanta + drawn), quantum);
}

struct ParticleBirths {
    cell: vec2<i32>,
    site: vec2<f32>,
    slot: vec2<u32>,
    middle: vec2<f32>,
    laid: u32,
    each: f32,
    boils: u32,
    boil: f32,
    radius: f32,
    surface: f32,
}

fn particle_births_count(births: ParticleBirths) -> vec2<u32> {
    return vec2<u32>(births.laid, births.boils);
}

fn particle_births(point: vec2<i32>) -> ParticleBirths {
    var births: ParticleBirths;
    births.cell = particle_frame.lattice.xy + point;
    let spacing = particle_spacing();
    births.site = vec2<f32>(births.cell) * spacing;
    births.middle = births.site;
    let here = particle_carrier_point(point);
    if here.depth <= WATER_FOAM_WET || !particle_site_in(births.site, particle_square()) {
        return births;
    }
    births.surface = here.surface;
    let area = spacing * spacing;
    let size = particle_eddy(here.depth);
    births.radius = 0.5 * size;
    let number = PI / births.radius;
    let flattest = WATER_SMOOTHEST * WATER_SMOOTHEST;
    if here.stirring > 0.0 {
        let turnover = size / here.stirring;
        let slot = particle_slot(births.cell, PARTICLE_BURSTS, turnover);
        if slot.z != 0u {
            let life = water_burst_life(slot.xy, particle_frame.eddies.x);
            let mean = area / (size * size) * life / turnover;
            let draws = particle_draws(births.cell, PARTICLE_BURSTS, 0u, slot.xy);
            let weight = water_burst_weight(mean, draws.x);
            if weight > 0.0 {
                births.slot = slot.xy;
                births.middle = births.site + (draws.yz - 0.5) * spacing;
                let lasting = weight * size * size * turnover;
                let seen = vec3<f32>(births.middle.x, particle_water(point).surface,
                    births.middle.y);
                let white = particle_quanta(log(1.0 + max(here.churn, 0.0))
                    / particle_draining(here.depth) * lasting, particle_quantum(seen), draws.w);
                births.laid = u32(white.x);
                births.each = white.y;
                let boil = here.stirring * here.stirring / (2.0 * WATER_GRAVITY);
                if boil * number >= flattest && particle_site_in(births.site, particle_window()) {
                    births.boils = 1u;
                    births.boil = boil;
                }
            }
        }
    }
    return births;
}

fn particle_born_foam(births: ParticleBirths, k: u32) -> Particle {
    let draws = particle_draws(births.cell, PARTICLE_BURSTS, 2u + k, births.slot);
    let turn = 2.0 * PI * draws.y;
    var raft = particle_raft(births.middle
        + births.radius * sqrt(draws.x) * vec2<f32>(cos(turn), sin(turn)),
        vec2<f32>(births.each, 0.0), sqrt(births.each / (PI * WATER_FOAM_FRESH)));
    raft.surface = births.surface;
    return raft;
}

fn particle_born_wave(births: ParticleBirths, here: vec2<f32>) -> Particle {
    var particle: Particle;
    particle.wave = true;
    particle.radius = births.radius;
    particle.at = births.middle;
    particle.amplitude = births.boil;
    particle.stretch = 1.0;
    particle.heading = atan2(here.y, here.x);
    return particle;
}

fn particle_velocity(carrier: ParticleCarrier) -> vec2<f32> {
    return carrier.current;
}

struct ParticleAged {
    particle: Particle,
    leaves: u32,
}

fn particle_aged(before: Particle) -> ParticleAged {
    let seconds = particle_frame.step.y;
    var aged: ParticleAged;
    aged.particle = before;
    let here = particle_carrier(before.at);
    if !before.wave {
        let floating = before.amounts.y / max(before.amounts.x + before.amounts.y, 1.0e-12);
        let halfway = before.at + 0.5 * seconds
            * (particle_velocity(here) + floating * particle_boils(before.at).current);
        let middle = particle_carrier(halfway);
        let boils = particle_boils(halfway);
        aged.particle.at = before.at
            + seconds * (particle_velocity(middle) + floating * boils.current);
        if middle.depth <= WATER_FOAM_WET
            || !particle_inside(aged.particle.at, particle_square()) {
            return aged;
        }
        return particle_raft_aged(aged.particle, middle,
            particle_shear(halfway) + boils.shear * floating, seconds);
    }
    let halfway = before.at + 0.5 * seconds * particle_velocity(here);
    let middle = particle_carrier(halfway);
    aged.particle.at = before.at + seconds * particle_velocity(middle);
    if middle.depth <= WATER_FOAM_WET
        || !particle_inside(aged.particle.at, particle_window()) {
        return aged;
    }
    let turns = seconds * max(middle.stirring, 1.0e-3) / (2.0 * before.radius);
    let amplitude = before.amplitude * exp(-turns);
    aged.particle.amplitude = amplitude;
    aged.particle.stretch = min(before.stretch + PARTICLE_STRETCH * turns,
        PARTICLE_STRETCH_MOST);
    if length(middle.current) > 1.0e-3 {
        aged.particle.heading = atan2(middle.current.y, middle.current.x);
    }
    let flattest = WATER_SMOOTHEST * WATER_SMOOTHEST;
    aged.leaves = u32(amplitude * PI / (before.radius * aged.particle.stretch) >= flattest);
    return aged;
}

fn particle_raft_aged(carried: Particle, middle: ParticleCarrier, shear: mat2x2<f32>,
        seconds: f32) -> ParticleAged {
    var aged: ParticleAged;
    aged.particle = carried;
    if middle.wet > 0.0 {
        aged.particle.surface = middle.surface;
    }
    let white = carried.amounts.x * exp(-seconds / particle_draining(middle.depth));
    let foam = (carried.amounts.y + WATER_FOAM_KEPT * (carried.amounts.x - white))
        * exp(-seconds / WATER_FOAM_LIFE);
    aged.particle.amounts = vec2<f32>(white, foam);
    let kept = (white + foam) / max(carried.amounts.x + carried.amounts.y, 1.0e-12);
    var strain = shear;
    let speed = length(middle.current);
    if speed > 1.0e-3 && middle.stirring > 0.0 {
        let way = middle.current / speed;
        let rate = 0.5 * PARTICLE_STRAIN * middle.stirring / particle_eddy(middle.depth);
        let c = way.x * way.x - way.y * way.y;
        let s = 2.0 * way.x * way.y;
        strain += rate * mat2x2<f32>(vec2<f32>(c, s), vec2<f32>(s, -c));
    }
    let strained = seconds * sqrt(dot(strain[0], strain[0]) + dot(strain[1], strain[1]));
    let step = seconds * min(1.0, PARTICLE_STRAINED_MOST / max(strained, 1.0e-6));
    let axis = (carried.axis + step * (strain * carried.axis)) * sqrt(kept);
    let spread = clamp(step * (shear[0].x + shear[1].y), -PARTICLE_STRAINED_MOST,
        PARTICLE_STRAINED_MOST);
    let area = length(carried.axis) * carried.minor * exp(spread) * kept;
    var long = max(length(axis), 1.0e-6);
    var along = axis / long;
    var minor = area / long;
    if minor > long {
        along = vec2<f32>(-along.y, along.x);
        let short = long;
        long = minor;
        minor = short;
    }
    if minor < PARTICLE_NARROWEST {
        long *= minor / PARTICLE_NARROWEST;
        minor = PARTICLE_NARROWEST;
    }
    aged.particle.axis = along * long;
    aged.particle.minor = minor;
    let thick = particle_thickness(aged.particle);
    if thick.x + thick.y < PARTICLE_THINNEST || long < PARTICLE_NARROWEST {
        return aged;
    }
    aged.leaves = select(1u, 2u, long > PARTICLE_TEAR * minor);
    return aged;
}

fn particle_piece(particle: Particle, k: u32, pieces: u32) -> Particle {
    if pieces < 2u {
        return particle;
    }
    var piece = particle;
    piece.axis = 0.5 * particle.axis;
    piece.at = particle.at + select(-piece.axis, piece.axis, k == 1u);
    piece.amounts = 0.5 * particle.amounts;
    return piece;
}

fn particle_count_word(count: vec2<u32>) -> u32 {
    return count.x | (count.y << 16u);
}

fn particle_sites() -> u32 {
    return PARTICLE_ITEMS + particle_frame.pool.x + particle_frame.pool.y;
}

fn particle_groups() -> u32 {
    return particle_sites() + particle_frame.pool.w;
}

fn particle_index(index: u32, foam: u32, half: u32) -> u32 {
    let pools = particle_frame.pool.xy;
    let first = half * (pools.x + pools.y);
    return select(first + pools.x + (index - foam), first + index, index < foam);
}

fn particle_items() -> u32 {
    return atomicLoad(&particle_counts[PARTICLE_FOAM_BEFORE])
        + atomicLoad(&particle_counts[PARTICLE_WAVES_BEFORE]) + particle_frame.pool.w;
}

fn particle_item(item: u32) -> vec2<u32> {
    let pooled = atomicLoad(&particle_counts[PARTICLE_FOAM_BEFORE])
        + atomicLoad(&particle_counts[PARTICLE_WAVES_BEFORE]);
    let sites = particle_frame.pool.w;
    if item >= sites + pooled {
        return vec2<u32>(0u);
    }
    let at = select(PARTICLE_ITEMS + (item - sites), particle_sites() + item, item < sites);
    let word = atomicLoad(&particle_counts[at]);
    return vec2<u32>(word & 0xffffu, word >> 16u);
}

fn particle_group_sum(group: u32) -> vec2<u32> {
    let at = particle_groups() + 2u * group;
    return vec2<u32>(atomicLoad(&particle_counts[at]), atomicLoad(&particle_counts[at + 1u]));
}

fn particle_store_group(group: u32, sum: vec2<u32>) {
    let at = particle_groups() + 2u * group;
    atomicStore(&particle_counts[at], sum.x);
    atomicStore(&particle_counts[at + 1u], sum.y);
}

var<workgroup> particle_sums: array<vec2<u32>, PARTICLE_GROUP>;

fn particle_scan_group(value: vec2<u32>, local: u32) -> vec2<u32> {
    particle_sums[local] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < PARTICLE_GROUP; stride *= 2u) {
        var before = vec2<u32>(0u);
        if local >= stride {
            before = particle_sums[local - stride];
        }
        workgroupBarrier();
        particle_sums[local] += before;
        workgroupBarrier();
    }
    return particle_sums[local] - value;
}

fn particle_lay(particle: Particle, place: u32) {
    let pools = particle_frame.pool.xy;
    if place >= select(pools.x, pools.y, particle.wave) {
        return;
    }
    let first = (1u - particle_frame.pool.z) * (pools.x + pools.y);
    particle_put(first + select(place, pools.x + place, particle.wave), particle);
}

fn particle_texel(cell: vec2<i32>) -> u32 {
    let first = particle_frame.window.xy;
    let texels = particle_frame.lattice.w;
    if any(cell < first) || any(cell >= first + vec2<i32>(texels)) {
        return PARTICLE_NO_TEXEL;
    }
    let at = particle_remainder(cell, texels);
    return u32(at.y * texels + at.x) * 4u;
}

fn particle_splat_wave(particle: Particle) {
    let texel = particle_frame.step.x;
    let along = vec2<f32>(cos(particle.heading), sin(particle.heading));
    let across = vec2<f32>(-along.y, along.x);
    let long = particle.radius * particle.stretch;
    let reach = i32(ceil(long / texel));
    let holding = vec2<i32>(floor(particle.at / texel));
    let fall = PARTICLE_FIXED * 0.5 * particle.amplitude * PI;
    let spreads = PARTICLE_FIXED * sqrt(2.0 * WATER_GRAVITY * max(particle.amplitude, 0.0));
    for (var row = -reach; row <= reach; row++) {
        for (var column = -reach; column <= reach; column++) {
            let cell = holding + vec2<i32>(column, row);
            let away = (vec2<f32>(cell) + 0.5) * texel - particle.at;
            let a = dot(away, along) / long;
            let c = dot(away, across) / particle.radius;
            let d = sqrt(a * a + c * c);
            if d >= 1.0 || d <= 0.0 {
                continue;
            }
            let at = particle_texel(cell);
            if at == PARTICLE_NO_TEXEL {
                continue;
            }
            let gradient = (a / long * along + c / particle.radius * across) / d;
            let sums = vec2<i32>(round(fall * sin(PI * d) * gradient));
            let current = vec2<i32>(round(spreads * sin(PI * d) * normalize(gradient)));
            if sums.x != 0 {
                atomicAdd(&particle_splat[at], sums.x);
            }
            if sums.y != 0 {
                atomicAdd(&particle_splat[at + 1u], sums.y);
            }
            if current.x != 0 {
                atomicAdd(&particle_splat[at + 2u], current.x);
            }
            if current.y != 0 {
                atomicAdd(&particle_splat[at + 3u], current.y);
            }
        }
    }
}

@compute @workgroup_size(1)
fn particles_begin() {
    var foam = atomicLoad(&particle_counts[PARTICLE_FOAM_LIVE]);
    var waves = atomicLoad(&particle_counts[PARTICLE_WAVES_LIVE]);
    if particle_frame.flags.x != 0u {
        foam = 0u;
        waves = 0u;
    }
    atomicStore(&particle_counts[PARTICLE_FOAM_BEFORE], foam);
    atomicStore(&particle_counts[PARTICLE_WAVES_BEFORE], waves);
    atomicStore(&particle_counts[PARTICLE_FOAM_BORN], 0u);
    atomicStore(&particle_counts[PARTICLE_WAVES_BORN], 0u);
    atomicStore(&particle_counts[PARTICLE_FOAM_DROPPED], 0u);
    atomicStore(&particle_counts[PARTICLE_WAVES_DROPPED], 0u);
    let pooled = foam + waves;
    let items = pooled + particle_frame.pool.w;
    atomicStore(&particle_counts[PARTICLE_ARGS], (pooled + PARTICLE_GROUP - 1u) / PARTICLE_GROUP);
    atomicStore(&particle_counts[PARTICLE_ARGS + 1u], 1u);
    atomicStore(&particle_counts[PARTICLE_ARGS + 2u], 1u);
    atomicStore(&particle_counts[PARTICLE_ARGS + 3u],
        (items + PARTICLE_GROUP - 1u) / PARTICLE_GROUP);
    atomicStore(&particle_counts[PARTICLE_ARGS + 4u], 1u);
    atomicStore(&particle_counts[PARTICLE_ARGS + 5u], 1u);
}

@compute @workgroup_size(PARTICLE_SIDE, PARTICLE_SIDE)
fn particles_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let point = vec2<i32>(id.xy);
    if any(point >= vec2<i32>(particle_frame.lattice.z)) {
        return;
    }
    let spacing = particle_spacing();
    let world = vec2<f32>(particle_frame.lattice.xy + point) * spacing;
    let at = water_pixel(world - terrain_uniform().origin.xz, spacing);
    var words = vec4<u32>(0u);
    if at.held >= WATER_HELD && at.depth > WATER_FOAM_WET {
        words = vec4<u32>(pack2x16float(at.words.flow),
            pack2x16float(vec2<f32>(at.depth, at.words.state.x)), pack2x16float(at.slope),
            bitcast<u32>(terrain_uniform().origin.y + at.surface));
    }
    particle_lattice[particle_lattice_at(point, false)] = words;
}

@compute @workgroup_size(PARTICLE_SIDE, PARTICLE_SIDE)
fn particles_currents(@builtin(global_invocation_id) id: vec3<u32>) {
    let point = vec2<i32>(id.xy);
    if any(point >= vec2<i32>(particle_frame.lattice.z)) {
        return;
    }
    let here = particle_water(point);
    var words = vec4<u32>(0u);
    if here.depth > WATER_FOAM_WET {
        let span = 2.0 * particle_spacing();
        let west = particle_beside(point - vec2<i32>(1, 0), here);
        let east = particle_beside(point + vec2<i32>(1, 0), here);
        let south = particle_beside(point - vec2<i32>(0, 1), here);
        let north = particle_beside(point + vec2<i32>(0, 1), here);
        let change = mat2x2<f32>((east.velocity - west.velocity) / span,
            (north.velocity - south.velocity) / span);
        let u = here.velocity;
        let rising = 0.5 * vec2<f32>(dot(east.velocity, east.velocity)
            - dot(west.velocity, west.velocity), dot(north.velocity, north.velocity)
            - dot(south.velocity, south.velocity)) / span;
        let power = WATER_DENSITY * here.depth
            * (WATER_GRAVITY * dot(u, here.slope) - dot(u, rising));
        let breaking = smoothstep(WATER_BREAK_ONSET, WATER_BREAK_FULL,
            water_eddy_froude(length(u), length(here.slope), here.depth));
        var churn = max(power, 0.0) * breaking / WATER_FOAM_POWER;
        if WATER_GRAVITY * length(here.slope) * length(u) < WATER_FOAM_LOSS {
            churn = 0.0;
        }
        let speed = length(u);
        let stirring = water_stirring(speed, water_eddies(speed, length(here.slope), here.depth),
            here.turbulence);
        words = vec4<u32>(pack2x16float(particle_surface(here, change)),
            pack2x16float(vec2<f32>(churn, here.depth)),
            pack2x16float(vec2<f32>(stirring, 0.0)), bitcast<u32>(here.surface));
    }
    particle_lattice[particle_lattice_at(point, true)] = words;
}

@compute @workgroup_size(PARTICLE_GROUP)
fn particles_age(@builtin(global_invocation_id) id: vec3<u32>) {
    let foam = atomicLoad(&particle_counts[PARTICLE_FOAM_BEFORE]);
    let pooled = foam + atomicLoad(&particle_counts[PARTICLE_WAVES_BEFORE]);
    if id.x >= pooled {
        return;
    }
    let at = particle_index(id.x, foam, particle_frame.pool.z);
    let aged = particle_aged(particle_at(at, id.x >= foam));
    particle_put(at, aged.particle);
    let leaves = select(vec2<u32>(aged.leaves, 0u), vec2<u32>(0u, aged.leaves),
        aged.particle.wave);
    atomicStore(&particle_counts[PARTICLE_ITEMS + id.x], particle_count_word(leaves));
}

@compute @workgroup_size(PARTICLE_SIDE, PARTICLE_SIDE)
fn particles_births(@builtin(global_invocation_id) id: vec3<u32>) {
    let points = particle_frame.lattice.z;
    let point = vec2<i32>(id.xy);
    if any(point >= vec2<i32>(points)) {
        return;
    }
    let count = particle_births_count(particle_births(point));
    let site = u32(point.y * points + point.x);
    atomicStore(&particle_counts[particle_sites() + site], particle_count_word(count));
    if count.x > 0u { // filter: the tally of rafts no smaller than a pixel, laying the same white water
        atomicAdd(&particle_counts[PARTICLE_FOAM_BORN], count.x);
    }
    if count.y > 0u {
        atomicAdd(&particle_counts[PARTICLE_WAVES_BORN], count.y);
    }
}

@compute @workgroup_size(PARTICLE_GROUP)
fn particles_sum(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
    let count = particle_item(id.x);
    let before = particle_scan_group(count, local.x);
    if local.x == PARTICLE_GROUP - 1u {
        particle_store_group(group.x, before + count);
    }
}

@compute @workgroup_size(PARTICLE_GROUP)
fn particles_scan(@builtin(local_invocation_id) local: vec3<u32>) {
    let pool = particle_frame.pool;
    let most = (pool.x + pool.y + pool.w + PARTICLE_GROUP - 1u) / PARTICLE_GROUP;
    let share = (most + PARTICLE_GROUP - 1u) / PARTICLE_GROUP;
    let groups = (particle_items() + PARTICLE_GROUP - 1u) / PARTICLE_GROUP;
    let first = local.x * share;
    var sum = vec2<u32>(0u);
    for (var k = 0u; k < share; k++) {
        if first + k < groups {
            sum += particle_group_sum(first + k);
        }
    }
    var running = particle_scan_group(sum, local.x);
    let total = particle_sums[PARTICLE_GROUP - 1u];
    for (var k = 0u; k < share; k++) {
        let group = first + k;
        if group < groups {
            let own = particle_group_sum(group);
            particle_store_group(group, running);
            running += own;
        }
    }
    if local.x == 0u {
        let live = min(total, pool.xy);
        atomicStore(&particle_counts[PARTICLE_FOAM_LIVE], live.x);
        atomicStore(&particle_counts[PARTICLE_WAVES_LIVE], live.y);
        atomicStore(&particle_counts[PARTICLE_FOAM_DROPPED], total.x - live.x);
        atomicStore(&particle_counts[PARTICLE_WAVES_DROPPED], total.y - live.y);
        atomicStore(&particle_counts[PARTICLE_ARGS],
            (live.x + live.y + PARTICLE_GROUP - 1u) / PARTICLE_GROUP);
        atomicStore(&particle_counts[PARTICLE_ARGS + 1u], 1u);
        atomicStore(&particle_counts[PARTICLE_ARGS + 2u], 1u);
        atomicStore(&particle_counts[PARTICLE_DRAW], 6u);
        atomicStore(&particle_counts[PARTICLE_DRAW + 1u], live.x);
        atomicStore(&particle_counts[PARTICLE_DRAW + 2u], 0u);
        atomicStore(&particle_counts[PARTICLE_DRAW + 3u], 0u);
    }
}

@compute @workgroup_size(PARTICLE_GROUP)
fn particles_place(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
    let count = particle_item(id.x);
    let within = particle_scan_group(count, local.x);
    let foam = atomicLoad(&particle_counts[PARTICLE_FOAM_BEFORE]);
    let pooled = foam + atomicLoad(&particle_counts[PARTICLE_WAVES_BEFORE]);
    if all(count == vec2<u32>(0u)) || id.x >= pooled + particle_frame.pool.w {
        return;
    }
    let first = particle_group_sum(group.x) + within;
    let sites = particle_frame.pool.w;
    if id.x >= sites {
        let held = id.x - sites;
        let particle = particle_at(particle_index(held, foam, particle_frame.pool.z),
            held >= foam);
        if particle.wave {
            particle_lay(particle, first.y);
            return;
        }
        for (var k = 0u; k < count.x; k++) {
            particle_lay(particle_piece(particle, k, count.x), first.x + k);
        }
        return;
    }
    let site = id.x;
    let points = u32(particle_frame.lattice.z);
    let births = particle_births(vec2<i32>(vec2<u32>(site % points, site / points)));
    for (var k = 0u; k < count.x; k++) {
        particle_lay(particle_born_foam(births, k), first.x + k);
    }
    if count.y != 0u {
        let here = particle_carrier(births.middle);
        particle_lay(particle_born_wave(births, here.current), first.y);
    }
}

@compute @workgroup_size(PARTICLE_GROUP)
fn particles_splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let foam = atomicLoad(&particle_counts[PARTICLE_FOAM_LIVE]);
    if id.x < foam || id.x >= foam + atomicLoad(&particle_counts[PARTICLE_WAVES_LIVE]) {
        return;
    }
    let half = 1u - particle_frame.pool.z;
    particle_splat_wave(particle_at(particle_index(id.x, foam, half), true));
}

@compute @workgroup_size(PARTICLE_SIDE, PARTICLE_SIDE)
fn particles_look(@builtin(global_invocation_id) id: vec3<u32>) {
    let texels = particle_frame.lattice.w;
    let texel = vec2<i32>(id.xy);
    if any(texel >= vec2<i32>(texels)) {
        return;
    }
    let at = u32(texel.y * texels + texel.x) * 4u;
    var fall = vec2<f32>(f32(atomicExchange(&particle_splat[at], 0)),
        f32(atomicExchange(&particle_splat[at + 1u], 0))) / PARTICLE_FIXED;
    let current = vec2<f32>(f32(atomicExchange(&particle_splat[at + 2u], 0)),
        f32(atomicExchange(&particle_splat[at + 3u], 0))) / PARTICLE_FIXED;
    let steepness = length(fall);
    if steepness > WATER_STEEPEST {
        fall *= WATER_STEEPEST / steepness;
    }
    let spread = water_spread(fall);
    textureStore(particle_target, texel, vec4<f32>(fall, spread.xy));
    textureStore(particle_spread_target, texel, vec4<f32>(spread.z, current, 0.0));
}

@compute @workgroup_size(PARTICLE_SIDE, PARTICLE_SIDE)
fn particles_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(particle_target)) {
        return;
    }
    let texel = 2 * vec2<i32>(id.xy);
    let sum = textureLoad(particle_source, texel, 0) + textureLoad(particle_source,
        texel + vec2<i32>(1, 0), 0) + textureLoad(particle_source, texel + vec2<i32>(0, 1), 0)
        + textureLoad(particle_source, texel + vec2<i32>(1, 1), 0);
    textureStore(particle_target, vec2<i32>(id.xy), 0.25 * sum);
    let spread = textureLoad(particle_spread_source, texel, 0)
        + textureLoad(particle_spread_source, texel + vec2<i32>(1, 0), 0)
        + textureLoad(particle_spread_source, texel + vec2<i32>(0, 1), 0)
        + textureLoad(particle_spread_source, texel + vec2<i32>(1, 1), 0);
    textureStore(particle_spread_target, vec2<i32>(id.xy), 0.25 * spread);
}

struct FoamOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) within: vec2<f32>,
    @location(1) @interpolate(flat) layers: vec2<f32>,
}

fn foam_corner(vertex: u32) -> vec2<f32> {
    var corner = vertex;
    if vertex == 3u {
        corner = 0u;
    } else if vertex > 3u {
        corner = vertex - 2u;
    }
    return vec2<f32>(select(-1.0, 1.0, corner == 1u || corner == 2u),
        select(-1.0, 1.0, corner >= 2u));
}

struct FoamQuad {
    at: vec2<f32>,
    within: vec2<f32>,
    layers: vec2<f32>,
}

fn foam_quad(raft: Particle, corner: vec2<f32>) -> FoamQuad {
    let local = raft.at - terrain_uniform().origin.xz;
    let pixel = detail_across(eye_world(water_relative(local,
        raft.surface - terrain_uniform().origin.y)));
    let long = length(raft.axis);
    let along = select(vec2<f32>(1.0, 0.0), raft.axis / long, long > 0.0);
    let across = vec2<f32>(-along.y, along.x);
    let drawn = max(vec2<f32>(long, raft.minor), vec2<f32>(pixel));
    var quad: FoamQuad;
    quad.at = local + along * (corner.x * drawn.x) + across * (corner.y * drawn.y);
    quad.within = corner;
    quad.layers = raft.amounts
        / (0.5 * (1.0 + FOAM_FLAT * FOAM_FLAT) * PI * drawn.x * drawn.y);
    return quad;
}

const FOAM_FLAT: f32 = 0.85;

@vertex
fn vs_foam(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> FoamOutput {
    let pools = particle_frame.pool.xy;
    let index = (1u - particle_frame.pool.z) * (pools.x + pools.y) + instance;
    let raft = particle_read(foam_pool[2u * index], foam_pool[2u * index + 1u], false);
    let quad = foam_quad(raft, foam_corner(vertex));
    var relative = water_relative(quad.at, raft.surface - terrain_uniform().origin.y);
    relative.y += water_lift(eye_world(relative));
    var output: FoamOutput;
    output.clip_position = terrain_view_uniform().relative_view_projection
        * vec4<f32>(relative, 1.0);
    output.within = quad.within;
    output.layers = quad.layers;
    return output;
}

@fragment
fn fs_foam(input: FoamOutput) -> @location(0) vec2<f32> {
    if textureLoad(water_depth, vec2<i32>(input.clip_position.xy), 0) > input.clip_position.z {
        discard;
    }
    let heaped = saturate((1.0 - dot(input.within, input.within))
        / (1.0 - FOAM_FLAT * FOAM_FLAT));
    return input.layers * heaped;
}
