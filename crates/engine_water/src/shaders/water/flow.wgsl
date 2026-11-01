import package::engine::math::{GRAVITY, pcg, unit};
import package::engine::view::PI;
import package::generated::water::{BREAK_THINNEST, TURBULENT_SPEED};

const WATER_STEEPEST: f32 = 0.44879895;

fn water_stirring(speed: f32, eddy: f32, turbulence: f32) -> f32 {
    return max(eddy, turbulence * max(speed, TURBULENT_SPEED));
}

fn water_turbulent_slopes(depth: f32, stirring: f32) -> f32 {
    let lifted = stirring * stirring / GRAVITY;
    let sloped = PI * lifted / max(depth, BREAK_THINNEST);
    return min(sloped * sloped, 0.5 * WATER_STEEPEST * WATER_STEEPEST);
}

const WATER_DETAIL_SMALLEST: f32 = 0.017;
const WATER_DETAIL_OCTAVES: u32 = 7u;
const WATER_DETAIL_RIDE: f32 = 4.0;
const WATER_DETAIL_CURRENT: f32 = 2.0;
const WATER_DETAIL_PHASES: vec2<f32> = vec2<f32>(0.15, 2.0);
const WATER_DETAIL_GRADIENT: f32 = 1.2;
const WATER_DETAIL_SHOWN: vec2<f32> = vec2<f32>(8.0, 16.0);

fn water_noise(p: vec2<f32>, seed: u32) -> vec3<f32> {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let base = bitcast<vec2<u32>>(vec2<i32>(cell));
    let ga = water_noise_gradient(base, seed);
    let gb = water_noise_gradient(base + vec2<u32>(1u, 0u), seed);
    let gc = water_noise_gradient(base + vec2<u32>(0u, 1u), seed);
    let gd = water_noise_gradient(base + vec2<u32>(1u, 1u), seed);
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
    let turn = 6.2831853 * unit(pcg(corner.x ^ pcg(corner.y ^ seed)));
    return vec2<f32>(cos(turn), sin(turn));
}

fn water_detail_phase(wavelength: f32) -> f32 {
    return clamp(WATER_DETAIL_RIDE * wavelength / WATER_DETAIL_CURRENT,
        WATER_DETAIL_PHASES.x, WATER_DETAIL_PHASES.y);
}

fn water_detail_phases(time: f32, seconds: f32) -> vec2<f32> {
    let cycle = time / seconds;
    return 1.0 - abs(2.0 * fract(vec2<f32>(cycle, cycle + 0.5)) - 1.0);
}

struct WaterDetail {
    slope: vec2<f32>,
    hidden: f32,
    shown: f32,
}

fn water_detail(world: vec3<f32>, flow: vec2<f32>, depth: f32, stirring: f32, across: f32,
        time: f32) -> WaterDetail {
    var detail: WaterDetail;
    let largest = clamp(2.0 * depth, 2.0 * WATER_DETAIL_SMALLEST,
        WATER_DETAIL_SMALLEST * exp2(f32(WATER_DETAIL_OCTAVES - 1u)));
    let octaves = log2(largest / WATER_DETAIL_SMALLEST) + 1.0;
    let each = min(water_turbulent_slopes(depth, stirring),
        0.5 * WATER_STEEPEST * WATER_STEEPEST / octaves);
    let variance = each * octaves;
    let steep = sqrt(each) / WATER_DETAIL_GRADIENT;
    var slope = vec2<f32>(0.0);
    var drawn = 0.0;
    var hidden = 0.0;
    var wavelength = WATER_DETAIL_SMALLEST;
    for (var octave = 0u; octave < WATER_DETAIL_OCTAVES; octave++) {
        let raised = saturate(octaves - f32(octave));
        let shows = smoothstep(WATER_DETAIL_SHOWN.x, WATER_DETAIL_SHOWN.y, wavelength / across); // filter: an octave finer than a pixel draws averages out to roughness
        drawn += each * raised * shows * shows;
        hidden += each * raised * (1.0 - shows * shows);
        if shows * raised > 0.0 {
            let seconds = water_detail_phase(wavelength);
            let phases = water_detail_phases(time, seconds);
            let kept = inverseSqrt(max(dot(phases, phases), 1.0e-6));
            for (var phase = 0u; phase < 2u; phase++) {
                let cycle = time / seconds + 0.5 * f32(phase);
                let number = (u32(i32(floor(cycle))) * 2u + phase) * 8u + octave;
                let shift = vec2<f32>(unit(pcg(number)), unit(pcg(number ^ 0x9e3779b9u))) * 97.0;
                let at = world.xz - flow * (fract(cycle) * seconds);
                let noise = water_noise(at / wavelength + shift, pcg(number));
                let share = phases[phase] * kept * shows * raised;
                slope += share * steep * noise.yz;
            }
        }
        wavelength *= 2.0;
    }
    detail.slope = slope;
    detail.hidden = hidden;
    detail.shown = select(0.0, drawn / variance, variance > 0.0);
    return detail;
}
