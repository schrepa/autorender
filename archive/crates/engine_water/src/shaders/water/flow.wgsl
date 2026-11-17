import package::engine::math::{PI, TAU, pcg, unit};
import package::generated::water::{FROUDE_GUARD_DEPTH, GRAVITY, WATER_BREAK_RANGE,
    WATER_NOISE_REACH, WATER_NOISE_WRAP, WATER_RIPPLE_LEVELS, WATER_RUSH_RANGE, WATER_STEEPEST};
import package::water::foam::water_foam_sampler;
import package::water::footprint::water_spread;
import package::water::uniforms::water;

@group(2) @binding(26) var water_ripple_texture: texture_2d<f32>;

const WATER_NOISE_GRADIENT: f32 = 1.2;
const WATER_FRAME_SCATTER: f32 = 61.0;
const WATER_RIPPLE_TAPS: f32 = 16.0;
const WATER_RIPPLE_SALT: u32 = 0x27d4eb2fu;
const WATER_FINE_SALT: u32 = 0x165667b1u;
const WATER_BOIL_SALT: u32 = 0xd3a2646cu;
const WATER_BUMP_SALT: u32 = 0xfd7046c5u;

fn water_broken(speed: f32, froude: f32, breaks: f32, dropping: f32) -> f32 {
    let tearing = smoothstep(water.foam.tearing.x, water.foam.tearing.y, speed);
    let shooting = smoothstep(water.foam.froude.x, water.foam.froude.y, froude);
    let falling = smoothstep(water.foam.drop.x, water.foam.drop.y, dropping);
    let breaking = smoothstep(water.foam.breaking.x, water.foam.breaking.y,
        breaks * WATER_BREAK_RANGE);
    return tearing * (1.0 - (1.0 - breaking) * (1.0 - shooting) * (1.0 - falling));
}

fn water_stirring(speed: f32, eddy: f32, turbulence: f32) -> f32 {
    return max(eddy, turbulence * max(speed, WATER_RUSH_RANGE));
}

fn water_turbulent_slopes(depth: f32, stirring: f32) -> f32 {
    let lifted = stirring * stirring / GRAVITY;
    let sloped = PI * lifted / max(depth, FROUDE_GUARD_DEPTH);
    return min(sloped * sloped, 0.5 * WATER_STEEPEST * WATER_STEEPEST);
}

fn water_noise(p: vec2<f32>, seed: u32) -> vec3<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let low = vec2<i32>(cell);
    let west = u32(low.x & (WATER_NOISE_WRAP - 1));
    let east = u32((low.x + 1) & (WATER_NOISE_WRAP - 1));
    let south = bitcast<u32>(low.y);
    let north = bitcast<u32>(low.y + 1);
    let ga = water_noise_gradient(vec2<u32>(west, south), seed);
    let gb = water_noise_gradient(vec2<u32>(east, south), seed);
    let gc = water_noise_gradient(vec2<u32>(west, north), seed);
    let gd = water_noise_gradient(vec2<u32>(east, north), seed);
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));
    let mixed = va - vb - vc + vd;
    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * mixed;
    let gradient = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
        + du * (u.yx * mixed + vec2<f32>(vb, vc) - va);
    return vec3<f32>(value, gradient);
}

fn water_noise_gradient(corner: vec2<u32>, seed: u32) -> vec2<f32> {
    let turn = TAU * unit(pcg(corner.x ^ pcg(corner.y ^ seed)));
    return vec2<f32>(cos(turn), sin(turn));
}

fn water_shown(wavelength: f32, across: f32) -> f32 {
    return smoothstep(water.flow.shown.x, water.flow.shown.y, wavelength / max(across, 1.0e-6)); // filter: an octave finer than a pixel draws averages out to roughness
}

fn water_tilt_spread(steep: vec2<f32>, stretch: f32) -> vec3<f32> {
    let gentle = 1.0 / (1.0 + stretch * stretch);
    return (1.0 - gentle) * water_spread(steep)
        + gentle * water_spread(vec2<f32>(-steep.y, steep.x));
}

struct WaterFrames {
    along: vec2<f32>,
    cell: vec2<i32>,
    within: vec2<f32>,
    weights: vec2<f32>,
}

fn water_frames(at: vec2<f32>, flow: vec2<f32>) -> WaterFrames {
    let grid = at / water.flow.frame - 0.5;
    let low = floor(grid);
    var frames: WaterFrames;
    frames.along = flow * inverseSqrt(max(dot(flow, flow), 1.0e-12));
    frames.cell = vec2<i32>(low);
    frames.within = grid - low;
    frames.weights = smoothstep(vec2<f32>(0.0), vec2<f32>(1.0), frames.within);
    return frames;
}

struct WaterCarry {
    rung: i32,
    toward: f32,
    speeds: vec2<f32>,
}

fn water_carry(speed: f32) -> WaterCarry {
    let steps = log2(max(speed, water.flow.carry_slowest) / water.flow.carry_slowest)
        / water.flow.carry_log;
    let rung = floor(steps);
    var carry: WaterCarry;
    carry.rung = i32(rung);
    carry.toward = steps - rung;
    carry.speeds = water.flow.carry_slowest
        * pow(vec2<f32>(water.flow.carry_step), vec2<f32>(rung, rung + 1.0));
    return carry;
}

fn water_carry_moved(carry: WaterCarry, wavelength: f32, time: f32) -> vec2<f32> {
    let wrap = f32(WATER_NOISE_WRAP);
    return fract(carry.speeds * (time / (wavelength * wrap))) * wrap;
}

struct WaterStream {
    at: vec2<f32>,
    flow: vec2<f32>,
    speed: f32,
    froude: f32,
    depth: f32,
    stirring: f32,
    turbulence: f32,
    across: f32,
    time: f32,
    dx: vec2<f32>,
    dy: vec2<f32>,
    frames: WaterFrames,
    carry: WaterCarry,
}

fn water_stream(at: vec2<f32>, flow: vec2<f32>, depth: f32, stirring: f32, turbulence: f32,
        across: f32, time: f32, dx: vec2<f32>, dy: vec2<f32>) -> WaterStream {
    var stream: WaterStream;
    stream.at = at;
    stream.flow = flow;
    stream.speed = length(flow);
    stream.froude = stream.speed / sqrt(GRAVITY * max(depth, FROUDE_GUARD_DEPTH));
    stream.depth = max(depth, 0.0);
    stream.stirring = stirring;
    stream.turbulence = turbulence;
    stream.across = across;
    stream.time = time;
    stream.dx = dx;
    stream.dy = dy;
    stream.frames = water_frames(at, flow);
    stream.carry = water_carry(stream.speed);
    return stream;
}

struct WaterLayer {
    lower: vec2<f32>,
    upper: vec2<f32>,
    moved: vec2<f32>,
    rung: i32,
    toward: f32,
    salt: u32,
}

fn water_frame_noise(q: vec2<f32>, size: vec2<f32>, moved: f32, scatter: vec2<f32>, seed: u32)
        -> vec3<f32> {
    let noise = water_noise(q / size + scatter - vec2<f32>(moved, 0.0), seed);
    return vec3<f32>(noise.x, noise.yz * (min(size.x, size.y) / size));
}

struct WaterCorner {
    weight: f32,
    at: vec2<f32>,
    scatter: vec2<f32>,
    higher: f32,
    kept: f32,
}

fn water_frame_corner(frames: WaterFrames, k: u32, toward: f32, salt: u32) -> WaterCorner {
    let lateral = vec2<f32>(-frames.along.y, frames.along.x);
    let corner = vec2<u32>(k & 1u, k >> 1u);
    let side = vec2<f32>(corner);
    let shares = mix(vec2<f32>(1.0) - frames.weights, frames.weights, side);
    let away = (frames.within - side) * water.flow.frame;
    let cell = bitcast<vec2<u32>>(frames.cell + vec2<i32>(corner));
    let draw = pcg(cell.x ^ pcg(cell.y ^ salt));
    let band = water.flow.rung_band;
    let chosen = mix(band, 1.0 - band, unit(pcg(draw ^ 0x9e3779b9u)));
    let higher = smoothstep(chosen - band, chosen + band, toward);
    var found: WaterCorner;
    found.weight = shares.x * shares.y;
    found.at = vec2<f32>(dot(away, frames.along), dot(away, lateral));
    found.scatter = vec2<f32>(unit(draw), unit(pcg(draw)));
    found.higher = higher;
    found.kept = found.weight * inverseSqrt(higher * higher + (1.0 - higher) * (1.0 - higher));
    return found;
}

fn water_framed(frames: WaterFrames, layer: WaterLayer) -> vec3<f32> {
    let lateral = vec2<f32>(-frames.along.y, frames.along.x);
    let seeds = vec2<u32>(pcg(bitcast<u32>(layer.rung) ^ layer.salt),
        pcg(bitcast<u32>(layer.rung + 1) ^ layer.salt));
    var sum = vec3<f32>(0.0);
    var squared = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let corner = water_frame_corner(frames, k, layer.toward, layer.salt);
        if corner.weight <= 0.0 {
            continue;
        }
        let scatter = corner.scatter * WATER_FRAME_SCATTER;
        if corner.higher < 1.0 {
            sum += corner.kept * (1.0 - corner.higher)
                * water_frame_noise(corner.at, layer.lower, layer.moved.x, scatter, seeds.x);
        }
        if corner.higher > 0.0 {
            sum += corner.kept * corner.higher
                * water_frame_noise(corner.at, layer.upper, layer.moved.y, scatter, seeds.y);
        }
        squared += corner.weight * corner.weight;
    }
    let blended = sum * inverseSqrt(max(squared, 1.0e-6));
    return vec3<f32>(blended.x, frames.along * blended.y + lateral * blended.z);
}

struct WaterDrift {
    at: vec2<f32>,
    scatter: vec2<f32>,
    share: f32,
    seed: u32,
}

fn water_drift(stream: WaterStream, velocity: vec2<f32>, seconds: f32, phase: u32, salt: u32)
        -> WaterDrift {
    let cycle = stream.time / seconds + 0.5 * f32(phase);
    let turn = floor(cycle);
    let within = cycle - turn;
    var drift: WaterDrift;
    drift.seed = pcg((u32(i32(turn)) * 2u + phase) ^ salt);
    drift.scatter = vec2<f32>(unit(drift.seed), unit(pcg(drift.seed)));
    drift.at = stream.at - velocity * ((within - 0.5) * seconds);
    drift.share = sin(PI * within);
    return drift;
}

fn water_drifting(stream: WaterStream, wavelength: f32, seconds: f32, salt: u32) -> f32 {
    var sum = 0.0;
    for (var phase = 0u; phase < 2u; phase++) {
        let drift = water_drift(stream, stream.flow, seconds, phase, salt);
        sum += drift.share * water_noise(drift.at / wavelength
            + drift.scatter * WATER_FRAME_SCATTER, drift.seed).x;
    }
    return sum;
}

fn water_ripple_texel(uv: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>) -> vec4<f32> {
    return water.ripples.reach
        * (textureSampleGrad(water_ripple_texture, water_foam_sampler, uv, dx, dy) - 0.5);
}

fn water_ripple_kept(span: f32, fine: bool) -> f32 {
    let coarsest = f32(textureNumLevels(water_ripple_texture)) - 1.0;
    let level = clamp(log2(span), 0.0, coarsest);
    let below = floor(level);
    let field = select(0u, WATER_RIPPLE_LEVELS, fine);
    let lower = field + u32(below);
    let upper = field + u32(ceil(level));
    return mix(water.ripples.kept[lower / 4u][lower % 4u],
        water.ripples.kept[upper / 4u][upper % 4u], level - below);
}

fn water_ripple_seen(dx: vec2<f32>, dy: vec2<f32>, fine: bool) -> f32 {
    let texels = vec2<f32>(textureDimensions(water_ripple_texture));
    let spans = vec2<f32>(length(dx * texels), length(dy * texels));
    let major = max(spans.x, spans.y);
    let minor = max(min(spans.x, spans.y), major / WATER_RIPPLE_TAPS);
    return 0.5 * (water_ripple_kept(major, fine) + water_ripple_kept(minor, fine)); // filter: waves finer than the sampler's footprint average out to roughness
}

fn water_ripple_tilt(steep: vec2<f32>, tile: vec2<f32>) -> vec2<f32> {
    return steep * tile.yx * inverseSqrt(dot(tile, tile));
}

fn water_ripple_framed(stream: WaterStream, tile: vec2<f32>, salt: u32) -> vec3<f32> {
    let frames = stream.frames;
    let lateral = vec2<f32>(-frames.along.y, frames.along.x);
    let dx = vec2<f32>(dot(stream.dx, frames.along), dot(stream.dx, lateral)) / tile;
    let dy = vec2<f32>(dot(stream.dy, frames.along), dot(stream.dy, lateral)) / tile;
    let seen = water_ripple_seen(dx, dy, false);
    if seen <= 0.0 {
        return vec3<f32>(0.0);
    }
    let moved = water_carry_moved(stream.carry, tile.x, stream.time);
    let even = (stream.carry.rung & 1) == 0;
    var sum = vec2<f32>(0.0);
    var squared = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let corner = water_frame_corner(frames, k, stream.carry.toward, salt);
        if corner.weight <= 0.0 {
            continue;
        }
        let q = corner.at / tile;
        let scatter = select(corner.scatter.yx, corner.scatter, even);
        if corner.higher < 1.0 {
            sum += corner.kept * (1.0 - corner.higher) * water_ripple_texel(
                q + scatter - vec2<f32>(moved.x, 0.0), dx, dy).xy;
        }
        if corner.higher > 0.0 {
            sum += corner.kept * corner.higher * water_ripple_texel(
                q + scatter.yx - vec2<f32>(moved.y, 0.0), dx, dy).xy;
        }
        squared += corner.weight * corner.weight;
    }
    let tilt = water_ripple_tilt(sum * inverseSqrt(max(squared, 1.0e-6)), tile);
    return vec3<f32>(frames.along * tilt.x + lateral * tilt.y, seen);
}

fn water_ripple_drifting(stream: WaterStream, velocity: vec2<f32>, tile: f32, seconds: f32,
        salt: u32) -> vec3<f32> {
    let dx = stream.dx / tile;
    let dy = stream.dy / tile;
    let seen = water_ripple_seen(dx, dy, true);
    if seen <= 0.0 {
        return vec3<f32>(0.0);
    }
    var sum = vec2<f32>(0.0);
    for (var phase = 0u; phase < 2u; phase++) {
        let drift = water_drift(stream, velocity, seconds, phase, salt);
        sum += drift.share * water_ripple_texel(drift.at / tile + drift.scatter, dx, dy).zw;
    }
    return vec3<f32>(sum, seen);
}

struct WaterRelief {
    slope: vec2<f32>,
    hidden: vec3<f32>,
    bumps: vec2<f32>,
}

fn water_flow_relief(stream: WaterStream, clear: f32) -> WaterRelief {
    var relief: WaterRelief;
    let moving = smoothstep(water.flow.moving.x, water.flow.moving.y, stream.speed);
    if moving <= 0.0 {
        return relief;
    }
    let along = stream.frames.along;
    let lateral = vec2<f32>(-along.y, along.x);
    let turbulent = water_turbulent_slopes(stream.depth, stream.stirring);
    let churning = smoothstep(water.ripples.boil_churning.x, water.ripples.boil_churning.y,
        stream.turbulence);
    let rippled = moving * moving
        * (water.ripples.least * water.ripples.least + water.ripples.share * turbulent);
    let fined = moving * moving * clear * churning * water.ripples.fine_share * turbulent;
    let boiled = moving * churning * water.ripples.boil_steep;
    let textured = rippled + fined + boiled * boiled;
    let coarse = water.ripples.coarse * textured;
    let carried = water_ripple_framed(stream, water.ripples.tile, WATER_RIPPLE_SALT);
    relief.slope -= sqrt(coarse) * carried.xy;
    relief.hidden += coarse * (1.0 - carried.z)
        * water_tilt_spread(lateral, water.ripples.stretch);
    let fine = textured - coarse;
    let skewed = stream.flow + water.ripples.skew * vec2<f32>(-stream.flow.y, stream.flow.x);
    let drifted = water_ripple_drifting(stream, skewed, water.ripples.fine_tile,
        water.flow.ripple_seconds, WATER_FINE_SALT);
    relief.slope -= sqrt(0.5 * fine) * drifted.xy;
    relief.hidden += fine * (1.0 - drifted.z) * vec3<f32>(0.5, 0.5, 0.0);
    let raised = moving
        * smoothstep(water.relief.bump_froude.x, water.relief.bump_froude.y, stream.froude);
    if raised > 0.0 {
        let wave = max(min(TAU * stream.speed * stream.speed / GRAVITY,
            water.relief.bump_depths * stream.depth), water.relief.bump_shortest);
        let steep = min(min(water.relief.bump_steep * raised, WATER_STEEPEST),
            TAU * water.relief.bump_deep * stream.depth / wave);
        let bumps_shown = water_shown(wave, stream.across);
        relief.hidden += steep * steep * (1.0 - bumps_shown * bumps_shown)
            * water_tilt_spread(along, water.relief.bump_crests);
        if steep * bumps_shown > 0.0 {
            let steps = log2(wave / water.relief.bump_shortest);
            let rung = floor(steps);
            let lower = water.relief.bump_shortest * exp2(rung)
                * vec2<f32>(1.0, water.relief.bump_crests);
            let bump = water_framed(stream.frames, WaterLayer(lower, 2.0 * lower,
                vec2<f32>(0.0), i32(rung), steps - rung, WATER_BUMP_SALT));
            let boils_shown = water_shown(water.relief.boil_across, stream.across);
            var churn = 0.0;
            if boils_shown > 0.0 {
                churn = water_drifting(stream, water.relief.boil_across,
                    water.flow.boil_seconds, WATER_BOIL_SALT);
            }
            let swell = clamp(1.0 + water.relief.bump_flicker * boils_shown * churn
                / WATER_NOISE_REACH, 0.0, 2.0);
            let standing = bumps_shown * swell;
            relief.slope -= steep * standing * (water.relief.bump_crest_gain / WATER_NOISE_GRADIENT)
                * bump.yz;
            relief.bumps = raised * standing * vec2<f32>(bump.x / WATER_NOISE_REACH,
                -dot(bump.yz, along) / WATER_NOISE_GRADIENT);
        }
    }
    return relief;
}

fn water_relief(stream: WaterStream, clear: f32, waved: vec2<f32>, unseen: vec3<f32>)
        -> WaterRelief {
    var relief = water_flow_relief(stream, clear);
    let slope = relief.slope + waved;
    relief.slope = slope * inverseSqrt(1.0
        + dot(slope, slope) / (water.flow.relief_steepest * water.flow.relief_steepest));
    relief.hidden += unseen;
    return relief;
}
