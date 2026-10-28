// Foam on terrains' water and the sea: the layers of white water and foam
// over the surface, from the water's state and what it meets. Terrains'
// water's are what its particles lay (`water_particles.rs`): rafts of white
// water born where it breaks, as fast as it loses energy breaking, whose
// bubbles rise into the foam they float on, carried and strained by the
// current its surface runs at, so its shear draws them into streaks and
// they gather where it converges and against what stands in it, each raft
// drawn sharp where it lies, at every distance the particles reach
// (`fs_foam`, `water_foam_layers`), none past it. And white water where it
// runs against what stands in it behind the pixel fast enough to tear air
// in (`water_contact`), by how far it climbs what it meets
// (`water_climbing`). The sea's whitens where its crests fold and where its
// waves run up its shores (`meshes/ocean.wgsl`). White water and foam add
// into one coverage (`water_foam`), lit as foam is lit (`water_light`), the
// share of it drained foam setting how thin it lies. No grain and no
// texture draws any of it: what shape it takes, its flecks, streaks, rafts,
// lines along eddies and piles behind what stands in the water, comes from
// the particles and how the water carries them.

// What water's wave particles splat (`water_particles.rs`) around the
// detail eye, its texels anchored in the world: how fast the waves' surface
// falls along x and z and the squares of those falls, and their product
// (`water_spread`), each with its coarser levels; the sampler reading them,
// which repeats them, as their texels hold their world texels' remainders
// over its side; and the layers of white water and foam the foam
// particles' rafts lay over each pixel, drawn before water (`fs_foam`).
@group(2) @binding(24) var water_particles: texture_2d<f32>;
@group(2) @binding(25) var water_particles_sampler: sampler;
@group(2) @binding(26) var water_particles_spread: texture_2d<f32>;
@group(2) @binding(48) var water_foam_drawn: texture_2d<f32>;

// The meters a second over which water running against what stands in it
// tears air in, from the first to the second (Ervine, McKeogh and Elsawy
// 1980: about the meter a second past which water draws air in).
const WATER_FOAM_TEARING: vec2<f32> = vec2<f32>(0.7, 1.3);
// The compression over which a whitecap thins from all foam to none about
// the threshold the sea's wind sets (`water.foam.x`): from its fresh middle
// to its aging edge.
const WATER_FOAM_CAPPING: f32 = 0.12;
// How much of the light heaped foam's hollows lose to the bubbles around
// them.
const WATER_FOAM_HOLLOW: f32 = 0.35;
// The optical depth a layer of foam holds: a layer of bubbles a few
// millimeters across, each taking twice its cross section out of the light
// (the extinction paradox), so a single layer lets a third of the light
// through and heaped foam none.
const WATER_FOAM_OPTICAL: f32 = 4.0;
// The most of the surface foam covers as `water_layers` counts it: all but
// a step of a signed byte, 4.8 layers, past which it lies opaque however
// thick.
const WATER_DENSEST: f32 = 0.99212598;
// The share of water's depth an obstacle stands up through from which the
// water starts to climb it, and from which it climbs it whole: a stone half
// as tall as the water runs deep bulges the surface over it, and one
// reaching nine tenths of the way breaks it (Chanson 2004, flow over large
// roughness).
const WATER_FOAM_BLOCKING: vec2<f32> = vec2<f32>(0.5, 0.9);

// How strongly water running at `speed` meters a second foams against what
// it meets `under` meters under its surface (over it where negative),
// changing by `change` across a pixel, whose face turns the share `facing` of
// the water's velocity head up it (from none, flat or facing downstream, to
// all of it, a wall meeting the flow head on): water climbs what it runs
// against by its velocity head U²/2g (Bernoulli), whitest where it meets it,
// and tears air in there as it passes WATER_FOAM_TEARING, so calm water
// foams at nothing it touches. A band rising less than what it meets rises
// across a pixel of the display (`material_detail`) spreads its foam over
// that rise, keeping its share, so no line of foam draws finer than a pixel.
fn water_climbing(under: f32, change: f32, speed: f32, facing: f32) -> f32 {
    let tearing = smoothstep(WATER_FOAM_TEARING.x, WATER_FOAM_TEARING.y, speed);
    if tearing <= 0.0 {
        return 0.0;
    }
    let climbs = speed * speed / (2.0 * WATER_GRAVITY) * facing;
    let band = max(max(climbs, change * material_detail()), 1.0e-9);
    return tearing * saturate(1.0 - under / band) * (climbs / band); // filter: a band narrower than a pixel spreading its share over it
}

// How strongly water foams against what stands in it behind the pixel
// (Sea of Thieves' depth test), its surface `standing` meters over it,
// changing by `change` across a pixel whose footprint is `footprint`
// (`water_footprint`), where it runs at `speed` meters a second, turning
// the share `facing` of its velocity head up what it meets
// (`water_climbing`). Only solid ground counts: behind blades and twigs what
// the pixel sees jumps from pixel to pixel faster than any bank rises along
// the footprint's long side (WATER_GRADIENT_RANGE a meter).
fn water_contact(standing: f32, change: f32, footprint: WaterFootprint, speed: f32,
        facing: f32) -> f32 {
    let rise = change * material_detail() / max(footprint.along, 1.0e-6);
    let solid = 1.0 - smoothstep(WATER_GRADIENT_RANGE, 2.0 * WATER_GRADIENT_RANGE, rise);
    return solid * water_climbing(standing, change, speed, facing);
}

// The share of running water's velocity head an obstacle turns up its face
// where its top stands `standing` meters under the surface of water running
// over a bed `depth` meters under it: water climbs what blocks its way, as
// far as it stands up through the water's depth, from none for the bed it
// runs over and its stones standing a share of its depth, to all of it for
// a rock blocking WATER_FOAM_BLOCKING of the depth or more, so cobbles under
// running water raise no foam and a boulder breaking the surface rings with
// it.
fn water_blocking(standing: f32, depth: f32) -> f32 {
    let blocked = 1.0 - standing / max(depth, WATER_BREAK_THINNEST);
    return smoothstep(WATER_FOAM_BLOCKING.x, WATER_FOAM_BLOCKING.y, blocked);
}

// Foam over a pixel: the share of the light falling on it it sends back,
// and of the surface it hides (`cover`); how thin it lies, from heaped (0)
// to a drained lace (1); and how much its hollows shade it.
struct WaterFoam {
    cover: f32,
    thin: f32,
    shade: f32,
}

// Foam where `white` layers of fresh white water and `foam` layers of the
// foam the water carries lie: a slab of bubbles scattering light without
// taking it, of optical depth τ, WATER_FOAM_OPTICAL a layer, sends back
// τ/(τ + 2) of the light falling on it and lets the rest through
// (Kubelka and Munk; two-stream, conservative scattering), so heaped foam
// is bright and opaque and a thin layer grey, the water showing through it.
// Fresh white water is wet foam, round bubbles heaping as they churn
// (Weta's foam, 2022); the foam the water carries has drained, its bubbles
// burst into a lace of the walls between them. Its share of the two sets
// how thin the foam lies, and the hollows of what heaps shade it by
// WATER_FOAM_HOLLOW: age changes how it scatters light, never where it
// lies.
fn water_foam(white: f32, foam: f32) -> WaterFoam {
    var result: WaterFoam;
    let layers = max(white, 0.0) + max(foam, 0.0);
    let depth = WATER_FOAM_OPTICAL * layers;
    result.cover = depth / (depth + 2.0);
    result.thin = max(foam, 0.0) / max(layers, 1.0e-6);
    result.shade = WATER_FOAM_HOLLOW * (1.0 - result.thin);
    return result;
}

// The layers of foam covering the share `share` of the surface,
// −ln(1 − share), each hiding as often as not what the others leave; no
// more than WATER_DENSEST covers.
fn water_layers(share: f32) -> f32 {
    return -log(1.0 - clamp(share, 0.0, WATER_DENSEST));
}

// The layers of white water and foam over the pixel at `pixel`: those the
// foam particles' rafts drew there (`fs_foam`), at any distance; none where
// none lie.
fn water_foam_layers(pixel: vec2<f32>) -> vec2<f32> {
    return textureLoad(water_foam_drawn, vec2<i32>(pixel), 0).xy;
}

// The meters across the eddies of water `depth` meters deep that the waves
// water's particles raise stand for (`meshes/water_particles.wgsl`), their
// texels `texel` meters across, riding water whose cells are `largest`
// meters across: about its depth, at least PARTICLE_SMALLEST texels and no
// larger than those cells, whose own flow carries larger eddies; none where
// no particles run.
fn water_eddy_size(depth: f32, texel: f32, largest: f32) -> f32 {
    let smallest = PARTICLE_SMALLEST * texel;
    return clamp(depth, smallest, max(largest, smallest));
}

// What water's particles show under a pixel: how fast the waves' surface
// falls along x and z (`fall`, meters a meter), as much of it as the
// particles show; the covariance of the waves' slopes the pixel hides
// (`hidden`, `water_spread`); and how much of the particles shows (`shown`,
// 0 to 1), the rest the water's own state's.
struct WaterSplatted {
    fall: vec2<f32>,
    hidden: vec3<f32>,
    shown: f32,
}

// What water's particles show under a pixel at `world` whose footprint is
// `footprint` (`water_footprint`), their uniform `water.foam` (the meters
// across their texels, none where none run; the meters their textures span,
// which repeat; the pixels their texels span where they hand over; and the
// meters across the cells of the water they ride): filtered over the
// footprint by their levels, out to where their texels span water.foam.z
// pixels seen from the detail eye, handing over to the water's own state,
// as a pyramid's level hands over to the next coarser (`detail_level`); a
// surface they draw nothing on past it. The slopes the footprint averages
// hide in their covariance, the mean of their squares less their mean's
// (LEAN mapping, Olano and Baker 2010).
fn water_splatted(world: vec3<f32>, footprint: WaterFootprint) -> WaterSplatted {
    var held: WaterSplatted;
    let texel = water.foam.x;
    if texel <= 0.0 {
        return held;
    }
    let picked = detail_level(2.0 * water.foam.z * detail_across(world), texel, 2u);
    if picked.level > 0u {
        return held;
    }
    let side = water.foam.y;
    let at = world.xz / side;
    let dx = footprint.dx / side;
    let dy = footprint.dy / side;
    let splatted = textureSampleGrad(water_particles, water_particles_sampler, at, dx, dy);
    let product = textureSampleGrad(water_particles_spread, water_particles_sampler, at, dx,
        dy).x;
    let hidden = vec3<f32>(splatted.zw, product) - water_spread(splatted.xy);
    held.shown = 1.0 - picked.blend;
    held.fall = splatted.xy * held.shown;
    held.hidden = vec3<f32>(max(hidden.xy, vec2<f32>(0.0)), hidden.z);
    return held;
}
