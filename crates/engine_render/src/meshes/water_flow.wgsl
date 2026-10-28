// Terrains' water's surface between its samples (`shade_water`), from its
// simulation's state: the simulated surface itself, and over it the waves
// its turbulence raises at scales the simulation does not hold
// (`water_detail`), as the Photon Water System draws detail waves over its
// simulated flow (Mao and Wu, 2023): carried by the simulated current, as
// steep as the simulation's turbulence lifts its surface, as large as its
// eddies down to the ripples surface tension holds; those finer than a
// pixel draws roughen it instead, and so does how far its normal turns
// across a pixel (`water_turning`), so a pixel spanning its samples
// averages their slopes rather than catching one. Where nothing stirs it,
// it lies a calm mirror.

// The steepest slope a wave stands at before it breaks (Stokes's limit, a
// height a seventh of its length: a sine's slopes of π/7 at most, 0.32 in
// root mean square).
const WATER_STEEPEST: f32 = 0.44879895;

// The speed (meters a second) the eddies of water running `speed` meters a
// second turn at, those as large as it is deep turning at `eddy`
// (`water_eddies`), its turbulence `turbulence` (0 to 1, the simulation's
// swings of its speed against that speed, at least WATER_TURBULENT_SPEED,
// `engine_terrain`'s `turbulence`): the faster of the two, the eddies' speed
// and the swings the turbulence measures.
fn water_stirring(speed: f32, eddy: f32, turbulence: f32) -> f32 {
    return max(eddy, turbulence * max(speed, WATER_TURBULENT_SPEED));
}

// The variance of the slopes turbulence raises on water `depth` meters deep
// (at least WATER_BREAK_THINNEST, as `water_eddies` takes it) whose eddies
// turn at `stirring` meters a second (`water_stirring`): the eddies' dynamic
// pressure lifts and lowers the surface by u²/g in root mean square, each
// eddy as large as the water is deep, one lifting it and the next lowering
// it as a sine twice as long does: slopes of π·u²/(g·h) in root mean square,
// no steeper than waves stand before they break (WATER_STEEPEST, a sine's
// root mean square of it). So shallow fast riffles run rough, and deep slow
// pools and still lakes stay glassy.
fn water_turbulent_slopes(depth: f32, stirring: f32) -> f32 {
    let lifted = stirring * stirring / WATER_GRAVITY;
    let sloped = PI * lifted / max(depth, WATER_BREAK_THINNEST);
    return min(sloped * sloped, 0.5 * WATER_STEEPEST * WATER_STEEPEST);
}

// The detail waves' smallest wavelength (meters): where surface tension
// takes over from gravity, 1.7 centimeters, below which ripples die within
// their own length; the most octaves from the eddies' size down to it; the
// seconds the current carries a pattern before a fresh one takes over (a
// flow map's phase, Vlachos 2010), as long as a pattern stretches little in
// water whose speed changes over meters; the root mean square of the
// noise's gradient and of its value, by which its octaves are scaled to the
// slopes they carry; how many times as long along the current as across it
// its waves draw out for each meter a second the water runs, the current's
// shear straining them; and the pixels a wavelength spans on screen from
// which an octave draws, and from which it draws whole: the rest is
// roughness, as a pattern finer than a pixel averages out.
const WATER_DETAIL_SMALLEST: f32 = 0.017;
const WATER_DETAIL_OCTAVES: u32 = 7u;
const WATER_DETAIL_PHASE: f32 = 1.0;
const WATER_DETAIL_GRADIENT: f32 = 1.2;
const WATER_DETAIL_VALUE: f32 = 0.25;
const WATER_DETAIL_STRETCH: f32 = 0.6;
const WATER_DETAIL_SHOWN: vec2<f32> = vec2<f32>(4.0, 8.0);

// Gradient noise at `p` (in its cells) of seed `seed`: its value and its
// gradient (y, z), quintic between its cells' random unit gradients (Perlin
// 2002; its derivatives after Quilez).
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

// The unit gradient at noise corner `corner` of seed `seed`.
fn water_noise_gradient(corner: vec2<u32>, seed: u32) -> vec2<f32> {
    let turn = 6.2831853 * unit(pcg(corner.x ^ pcg(corner.y ^ seed)));
    return vec2<f32>(cos(turn), sin(turn));
}

// The share of its pattern each of a flow map's two phases shows at `time`
// seconds (`water_detail`): rising from none as it begins to all of it
// halfway through its phase and back, the second half a phase behind.
fn water_detail_phases(time: f32) -> vec2<f32> {
    let cycle = time / WATER_DETAIL_PHASE;
    return 1.0 - abs(2.0 * fract(vec2<f32>(cycle, cycle + 0.5)) - 1.0);
}

// The waves turbulence raises on water between its samples, as a pixel
// shows them: how far they tilt its surface (x and z), how near the top of
// their crests it stands, in their root mean square (of the share of them
// the pixel draws), the variance of the slopes it hides, and the share of
// their slopes' variance it draws.
struct WaterDetail {
    slope: vec2<f32>,
    crest: f32,
    hidden: f32,
    shown: f32,
}

// The detail waves on water at `world` `depth` meters deep running at
// `flow` (meters a second along x and z) whose eddies turn at `stirring`
// (`water_stirring`), a pixel there spanning `across` meters across the
// view (`detail_across`), at `time` seconds: octaves of gradient noise from
// the eddies' size, about twice the water's depth, down to
// WATER_DETAIL_SMALLEST, each as steep as the turbulence lifts the surface
// at its eddies' scale (`water_turbulent_slopes`), as a spectrum saturated
// by breaking holds the same slopes in every octave (Phillips 1985), so
// the finer octaves add to the eddies' own, all of them together no steeper
// than waves stand before they break (WATER_STEEPEST); drawn out along the
// current by
// WATER_DETAIL_STRETCH for each meter a second it runs; carried by the
// current over WATER_DETAIL_PHASE in two phases half a phase apart, each
// fresh pattern taking over as the other fades (`water_detail_phases`),
// blended keeping their variance (Heitz and Neyret 2018), so the waves ride
// the current and never stretch. An octave draws while its wavelength spans
// WATER_DETAIL_SHOWN pixels, its slopes hidden as it shrinks under them.
fn water_detail(world: vec3<f32>, flow: vec2<f32>, depth: f32, stirring: f32, across: f32,
        time: f32) -> WaterDetail {
    var detail: WaterDetail;
    let largest = clamp(2.0 * depth, 4.0 * WATER_DETAIL_SMALLEST, 2.0);
    let octaves = min(u32(floor(log2(largest / WATER_DETAIL_SMALLEST))) + 1u,
        WATER_DETAIL_OCTAVES);
    let each = min(water_turbulent_slopes(depth, stirring),
        0.5 * WATER_STEEPEST * WATER_STEEPEST / f32(octaves));
    let variance = each * f32(octaves);
    let steep = sqrt(each) / WATER_DETAIL_GRADIENT;
    let speed = length(flow);
    let along = select(vec2<f32>(1.0, 0.0), flow / max(speed, 1.0e-6), speed > 1.0e-4);
    let side = vec2<f32>(-along.y, along.x);
    let stretch = 1.0 + WATER_DETAIL_STRETCH * min(speed, 3.0);
    let phases = water_detail_phases(time);
    var slope = vec2<f32>(0.0);
    var crest = 0.0;
    var drawn = 0.0;
    var hidden = 0.0;
    var wavelength = largest;
    for (var octave = 0u; octave < octaves; octave++) {
        let shows = smoothstep(WATER_DETAIL_SHOWN.x, WATER_DETAIL_SHOWN.y, wavelength / across); // filter: an octave finer than a pixel draws averages out to roughness
        drawn += each * shows * shows;
        hidden += each * (1.0 - shows * shows);
        if shows > 0.0 {
            for (var phase = 0u; phase < 2u; phase++) {
                let cycle = time / WATER_DETAIL_PHASE + 0.5 * f32(phase);
                let number = u32(i32(floor(cycle))) * 2u + phase;
                let shift = vec2<f32>(unit(pcg(number)), unit(pcg(number ^ 0x9e3779b9u))) * 97.0;
                let at = world.xz - flow * (fract(cycle) * WATER_DETAIL_PHASE);
                let frame = vec2<f32>(dot(at, along) / stretch, dot(at, side));
                let noise = water_noise(frame / wavelength + shift, pcg(number * 8u + octave));
                let gradient = along * (noise.y / stretch) + side * noise.z;
                slope += phases[phase] * shows * steep * gradient;
                crest += phases[phase] * shows * noise.x / WATER_DETAIL_VALUE;
            }
        }
        wavelength *= 0.5;
    }
    let kept = inverseSqrt(max(dot(phases, phases), 1.0e-6));
    detail.slope = slope * kept;
    detail.crest = crest * kept * inverseSqrt(f32(octaves));
    detail.hidden = hidden;
    detail.shown = select(0.0, drawn / variance, variance > 0.0);
    return detail;
}
