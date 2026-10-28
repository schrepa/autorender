// Foam on terrains' water and the sea: the share of the surface foam hides,
// from the water's state and what it meets. Terrains' water's is what its
// particles lay (`water_particles.rs`, `water_splatted`): the white water
// turbulence's bursts lay where the water breaks (`engine_terrain`'s
// `breaking`), draining as its bubbles rise out, and the foam they churn up
// where it churns (`water_sim.wgsl`'s `sim_churn`: where it loses energy,
// plunges or jumps, along its eddy lines and where it converges), carried as
// particles by the current its surface runs at, drawn by it into streaks,
// gathering where it converges and against what stands in it, and fading
// as they age, with open water between them; out past where their texels
// span a pixel, the water's own state's, which their mean is. And white
// water where it runs against what stands in it behind the pixel fast
// enough to tear air in (`water_contact`), by how far it climbs what it
// meets (`water_climbing`). The sea's whitens where its crests fold and
// where its waves run up its shores (`meshes/ocean.wgsl`). Fresh white
// water and carried foam add into one coverage (`water_foam`), lit as foam
// is lit (`water_light`); age changes only how it scatters light. No grain
// and no texture draws any of it: what shape it takes, its flecks, streaks,
// rafts, lines along eddies and piles behind what stands in the water,
// comes from the particles and how the water carries them.

// What water's particles splat (`water_particles.rs`) around the detail
// eye, its texels anchored in the world: each the share of it white water
// covers and foam covers, and how fast the waves' surface falls along x and
// z; the squares of those falls and their product (`water_spread`); each
// with its coarser levels; and the sampler reading them, which repeats
// them, as their texels hold their world texels' remainders over its side.
@group(2) @binding(24) var water_particles: texture_2d<f32>;
@group(2) @binding(25) var water_particles_sampler: sampler;
@group(2) @binding(26) var water_particles_spread: texture_2d<f32>;

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
// The most of the surface water breaking white all over whitens at once:
// its breaking eddies reach its surface in patches, rollers and boils
// between which its water runs clear (Brocchini and Peregrine 2001; Chanson
// 1996, the surface of self-aerated flow), so that rapids show white crests
// and streaks over dark water.
const WATER_BREAKING_MOST: f32 = 0.6;
// The most of the surface the lace drained foam leaves covers: its walls'
// network over the open water between them, a third of it where it lies
// thickest (Craig and Stewart; lace below rapids).
const WATER_FOAM_LACE: f32 = 0.35;
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

// Foam over a pixel: the share of the surface it hides (`cover`), how thin
// it lies, from heaped (0) to a single drained layer (1), and how much its
// hollows shade it.
struct WaterFoam {
    cover: f32,
    thin: f32,
    shade: f32,
}

// The foam of fresh white water covering `fresh` of the surface and of foam
// the water carries holding `carried` of the most it holds (0 to 1 each):
// each lies where the other does as often as anywhere, so together they
// cover all but what both leave. Fresh white water is wet foam, round
// bubbles heaping as they churn (Weta's foam, 2022), as white as it is
// thick; the foam the water carries has drained, its bubbles burst into an
// open lace of the walls between them, which covers WATER_FOAM_LACE of the
// surface at the most. Its share of the two sets how thin the foam lies,
// and the hollows of what heaps shade it by WATER_FOAM_HOLLOW: age changes
// how it scatters light, never where it lies.
fn water_foam(fresh: f32, carried: f32) -> WaterFoam {
    var foam: WaterFoam;
    let lace = carried * WATER_FOAM_LACE;
    foam.cover = 1.0 - (1.0 - fresh) * (1.0 - lace);
    foam.thin = lace / max(fresh + lace, 1.0e-6);
    foam.shade = WATER_FOAM_HOLLOW * (1.0 - foam.thin);
    return foam;
}

// The scale of the logistic distribution as spread as a unit normal one,
// √3/π: the share of a crest field past a threshold, and the threshold
// leaving a share past it, in closed form.
const WATER_FOAM_LOGISTIC: f32 = 0.5513289;

// Where on the waves foam covering the share `mean` of the surface lies, as
// a pixel shows it, over the detail waves `detail` (`water_detail`): foam
// rides to their crests, so it covers what stands past the crest height
// leaving `mean` of the surface past it, the waves' crest heights taken as
// logistic, their edge as sharp as their crests change across the pixel; as
// much of that pattern as the pixel draws of the waves, the rest the mean
// itself, so the share stays `mean` wherever the waves draw or not.
fn water_foam_crests(mean: f32, detail: WaterDetail) -> f32 {
    if mean <= 1.0e-4 || mean >= 1.0 - 1.0e-4 {
        return mean;
    }
    let threshold = WATER_FOAM_LOGISTIC * log((1.0 - mean) / mean);
    let edge = max(fwidth(detail.crest), 0.05); // filter: the foam's edge spread over the pixel, keeping its share
    let crested = 1.0 / (1.0 + exp(-clamp((detail.crest - threshold) / edge, -30.0, 30.0)));
    return mix(mean, crested, saturate(detail.shown));
}

// The share of the surface water breaking white covers at once, where its
// white water lies as `white` (0 to 1) of what it holds: as much of
// WATER_BREAKING_MOST.
fn water_breaking_share(white: f32) -> f32 {
    return white * WATER_BREAKING_MOST;
}

// The share of the surface white water hides where `layers` of it lie: each
// layer hides as often as not what the others leave, so `layers` of them
// leave e^−layers of it open.
fn water_white(layers: f32) -> f32 {
    return 1.0 - exp(-max(layers, 0.0));
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

// What water's particles show under a pixel: the shares of it the white
// water and the foam the water carries cover (`foam`); how fast the waves'
// surface falls along x and z (`fall`, meters a meter), as much of it as
// the particles show; the covariance of the waves' slopes the pixel hides
// (`hidden`, `water_spread`); and how much of the particles shows (`shown`,
// 0 to 1), the rest the water's own state's.
struct WaterSplatted {
    foam: vec2<f32>,
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
// pixels seen from the detail eye, handing over to `settled`, the shares
// white water and foam cover as the water's own state holds them (the white
// water born there and the foam its flow churned up), the particles' mean,
// as a pyramid's level hands over to the next coarser (`detail_level`);
// `settled` and a surface they draw nothing on past it. The slopes the
// footprint averages hide in their covariance, the mean of their squares
// less their mean's (LEAN mapping, Olano and Baker 2010).
fn water_splatted(world: vec3<f32>, footprint: WaterFootprint, settled: vec2<f32>)
        -> WaterSplatted {
    var held: WaterSplatted;
    held.foam = settled;
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
    let squares = textureSampleGrad(water_particles_spread, water_particles_sampler, at, dx,
        dy).xyz;
    let hidden = squares - water_spread(splatted.zw);
    held.foam = mix(splatted.xy, settled, picked.blend);
    held.shown = 1.0 - picked.blend;
    held.fall = splatted.zw * held.shown;
    held.hidden = vec3<f32>(max(hidden.xy, vec2<f32>(0.0)), hidden.z);
    return held;
}
