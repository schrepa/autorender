// Terrains' water's roughness (`shade_water`), from its simulation's state:
// its surface is the simulated surface itself, tilted by the disturbance the
// turbulence reaching it raises where the foam field runs
// (`meshes/water_foam_field.wgsl`), and nothing else adds waves to it. What
// neither resolves of it, the surface its eddies lift and lower about its
// mean between its samples, roughens it instead (`water_unresolved`), by the
// simulation's own turbulence and eddies there, drawn as no pattern; and how
// far its normal turns across a pixel roughens it too (`water_turning`), so
// a pixel spanning its samples averages their slopes rather than catching
// one. Where neither stirs it, it lies a calm mirror.

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

// The covariance of the slopes (`water_spread`) left unresolved in water
// `depth` meters deep (at least WATER_BREAK_THINNEST, as `water_eddies`
// takes it) whose eddies turn at `stirring` meters a second
// (`water_stirring`), by eddies smaller than the `resolved` meters across
// the foam field's bursts draw (`water_eddy_size`; none where none are
// drawn): how much the surface varies about its mean. The eddies' dynamic
// pressure lifts and lowers the surface by u²/g in root mean square, each
// eddy as large as the water is deep, one lifting it and the next lowering
// it as a sine twice as long does: slopes of π·u²/(g·h) in root mean square,
// no steeper than waves stand before they break (WATER_STEEPEST, a sine's
// root mean square of it), alike every way; less the variance of those the
// bursts draw, π·u²/(g·r) for eddies r across, which leaves none where they
// are as large as the water is deep. So shallow fast riffles run rough, and
// deep slow pools and still lakes stay glassy.
fn water_unresolved(depth: f32, stirring: f32, resolved: f32) -> vec3<f32> {
    let lifted = stirring * stirring / WATER_GRAVITY;
    let deep = max(depth, WATER_BREAK_THINNEST);
    let most = 0.5 * WATER_STEEPEST * WATER_STEEPEST;
    let sloped = PI * lifted / deep;
    var variance = min(sloped * sloped, most);
    if resolved > 0.0 {
        let drawn = PI * lifted / max(resolved, deep);
        variance -= min(drawn * drawn, most);
    }
    return vec3<f32>(0.5 * variance, 0.5 * variance, 0.0);
}
