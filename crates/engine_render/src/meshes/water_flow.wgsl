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
// wavelengths a pattern rides a current of WATER_DETAIL_CURRENT meters a
// second before a fresh one takes over (a flow map's phase, Vlachos 2010),
// each octave's phase as long as that takes it, so a pattern moves a few of
// its own lengths before it renews, never far enough for its own reset to
// show as a jump or for the current's shear to stretch it, and no shorter
// or longer than WATER_DETAIL_PHASES seconds, so it neither flickers from
// frame to frame nor lingers; the root mean square of the
// noise's gradient and of its value, by which its octaves are scaled to the
// slopes they carry; and the pixels a wavelength spans on screen from
// which an octave draws, and from which it draws whole: gradient noise's
// slopes turn over a quarter of its cells, so these keep them two pixels
// across at least, as every pattern drawn keeps two pixels a feature; the
// rest is roughness, as a pattern finer than that averages out.
const WATER_DETAIL_SMALLEST: f32 = 0.017;
const WATER_DETAIL_OCTAVES: u32 = 7u;
const WATER_DETAIL_RIDE: f32 = 4.0;
const WATER_DETAIL_CURRENT: f32 = 2.0;
const WATER_DETAIL_PHASES: vec2<f32> = vec2<f32>(0.15, 2.0);
const WATER_DETAIL_GRADIENT: f32 = 1.2;
const WATER_DETAIL_VALUE: f32 = 0.25;
const WATER_DETAIL_SHOWN: vec2<f32> = vec2<f32>(8.0, 16.0);

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

// The seconds a flow map's phase lasts for the detail waves' octave of
// wavelength `wavelength` (meters): as long as a current of
// WATER_DETAIL_CURRENT takes to carry it WATER_DETAIL_RIDE of its lengths,
// within WATER_DETAIL_PHASES. Fixed for each octave, as anything dividing
// the clock where each pixel stands would land elsewhere from pixel to
// pixel.
fn water_detail_phase(wavelength: f32) -> f32 {
    return clamp(WATER_DETAIL_RIDE * wavelength / WATER_DETAIL_CURRENT,
        WATER_DETAIL_PHASES.x, WATER_DETAIL_PHASES.y);
}

// The share of its pattern each of a flow map's two phases `seconds` long
// shows at `time` seconds (`water_detail`): rising from none as it begins
// to all of it halfway through its phase and back, the second half a phase
// behind.
fn water_detail_phases(time: f32, seconds: f32) -> vec2<f32> {
    let cycle = time / seconds;
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
// view (`detail_across`), at `time` seconds: octaves of gradient noise at
// wavelengths fixed in the world, WATER_DETAIL_SMALLEST doubled
// WATER_DETAIL_OCTAVES times, so a place's waves stand where they stand
// whatever the water does around them (a wavelength following the depth
// would rescale world coordinates kilometers out, landing elsewhere from
// pixel to pixel); those up to its eddies' size, about twice the water's
// depth, raised, fading out over the octave past it; each as steep as the
// turbulence lifts the surface
// at its eddies' scale (`water_turbulent_slopes`), as a spectrum saturated
// by breaking holds the same slopes in every octave (Phillips 1985), so
// the finer octaves add to the eddies' own, all of them together no steeper
// than waves stand before they break (WATER_STEEPEST); laid in the world's
// own axes, as anything turned by the current where each pixel stands would
// turn about the world's origin, kilometers off, and land elsewhere from
// pixel to pixel; carried by the current over its octave's phase
// (`water_detail_phase`) in two phases half a phase apart, each
// fresh pattern taking over as the other fades (`water_detail_phases`),
// blended keeping their variance (Heitz and Neyret 2018), so the waves ride
// the current and never stretch. An octave draws while its wavelength spans
// WATER_DETAIL_SHOWN pixels, its slopes hidden as it shrinks under them.
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
    var crest = 0.0;
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
                crest += share * noise.x / WATER_DETAIL_VALUE;
            }
        }
        wavelength *= 2.0;
    }
    detail.slope = slope;
    detail.crest = crest * inverseSqrt(max(octaves, 1.0));
    detail.hidden = hidden;
    detail.shown = select(0.0, drawn / variance, variance > 0.0);
    return detail;
}
