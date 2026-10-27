// Terrains' water's roughness (`shade_water`), from its simulation's state:
// its surface is the simulated surface itself, its normal that surface's own
// slope, and nothing adds waves to it. What the simulation cannot resolve of
// it, the surface its eddies lift and lower about its mean between its
// samples, roughens it instead (`water_unresolved`), by the simulation's own
// turbulence and eddies there, drawn as no pattern; and how far the
// simulated surface's own normal turns across a pixel roughens it too
// (`water_turning`), so a pixel spanning its samples averages their slopes
// rather than catching one. Where neither stirs it, it lies a calm mirror.

// The steepest slope a wave stands at before it breaks (Stokes's limit, a
// height a seventh of its length: a sine's slopes of π/7 at most, 0.32 in
// root mean square).
const WATER_STEEPEST: f32 = 0.44879895;

// The covariance of the slopes (`water_spread`) the simulation leaves
// unresolved in water `depth` meters deep (at least WATER_BREAK_THINNEST, as
// `water_eddies` takes it) running `speed` meters a second, its eddies
// turning at `eddy` meters a second (`water_eddies`) and its turbulence
// `turbulence` (0 to 1, the simulation's swings of its speed against that
// speed, at least WATER_TURBULENT_SPEED, `engine_terrain`'s `turbulence`):
// how much the surface it holds varies about its mean. Its eddies turn at
// the faster of the two, the eddies' speed and the swings the turbulence
// measures; their dynamic pressure lifts and lowers the surface by u²/g in
// root mean square, each eddy as large as the water is deep, one lifting it
// and the next lowering it as a sine twice as long does: slopes of
// π·u²/(g·h) in root mean square, no steeper than waves stand before they
// break (WATER_STEEPEST, a sine's root mean square of it), alike every way.
// So shallow fast riffles run rough, and deep slow pools and still lakes stay
// glassy.
fn water_unresolved(depth: f32, speed: f32, eddy: f32, turbulence: f32) -> vec3<f32> {
    let swings = turbulence * max(speed, WATER_TURBULENT_SPEED);
    let stirring = max(eddy, swings);
    let lifted = stirring * stirring / WATER_GRAVITY;
    let sloped = PI * lifted / max(depth, WATER_BREAK_THINNEST);
    let variance = min(sloped * sloped, 0.5 * WATER_STEEPEST * WATER_STEEPEST);
    return vec3<f32>(0.5 * variance, 0.5 * variance, 0.0);
}
