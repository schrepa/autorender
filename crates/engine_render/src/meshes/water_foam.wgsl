// Foam on terrains' water and the sea: smooth coverage, the share of the
// surface foam hides, from the water's state and what it meets. Terrains'
// water's is the foam field the water carries (`water_foam_field.rs`,
// `water_foam_held`): the white water born where the water breaks
// (`engine_terrain`'s `breaking`), carried on as its bubbles drain, and the
// foam its flow churns up (`water_sim.wgsl`'s `sim_churn`: where it loses
// energy, plunges or jumps, along its eddy lines and where it converges),
// carried by the current its surface runs at, gathering where that
// converges and against what stands in it, and fading as it ages; out past
// where the field's cells span a few pixels, the water's own state's, which
// the field's mean is. And white water where it runs against what stands in
// it behind the pixel fast enough to tear air in (`water_contact`), by how
// far it climbs what it meets (`water_climbing`). The sea's whitens where
// its crests fold and where its waves run up its shores
// (`meshes/ocean.wgsl`). Fresh white water and carried foam add into one
// coverage (`water_foam`), lit as foam is lit (`water_light`); age changes
// only how it scatters light. No grain and no texture draws any of it: what
// shape it takes, its streaks, rafts, lines along eddies and piles behind
// what stands in the water, comes from how the water carries it.

// The foam field terrains' water carries (`water_foam_field.rs`), its
// texels its cells, each the white water covering it and the foam the water
// carries there as a share of WATER_FOAM_MOST, with its coarser levels; and
// the sampler reading it, which repeats it, as its texels hold their world
// cells' remainders over its side.
@group(2) @binding(24) var water_foam_field: texture_2d<f32>;
@group(2) @binding(25) var water_foam_sampler: sampler;

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
// (`water_footprint`), where it runs at `speed` meters a second, meeting it
// head on (`water_climbing`). Only solid ground counts: behind blades and
// twigs what the pixel sees jumps from pixel to pixel faster than any bank
// rises along the footprint's long side (WATER_GRADIENT_RANGE a meter).
fn water_contact(standing: f32, change: f32, footprint: WaterFootprint, speed: f32) -> f32 {
    let rise = change * material_detail() / max(footprint.along, 1.0e-6);
    let solid = 1.0 - smoothstep(WATER_GRADIENT_RANGE, 2.0 * WATER_GRADIENT_RANGE, rise);
    return solid * water_climbing(standing, change, speed, 1.0);
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
// the water carries covering `carried` (0 to 1 each), as shares of it: each
// lies where the other does as often as anywhere, so together they cover
// all but what both leave. Fresh white water is wet foam, round bubbles
// heaping as they churn (Weta's foam, 2022); the foam the water carries has
// aged, its bubbles drained into the walls of a single layer. Its share of
// the two sets how thin the foam lies, and the hollows of what heaps shade it
// by WATER_FOAM_HOLLOW: age changes how it scatters light, never where it
// lies.
fn water_foam(fresh: f32, carried: f32) -> WaterFoam {
    var foam: WaterFoam;
    foam.cover = 1.0 - (1.0 - fresh) * (1.0 - carried);
    foam.thin = carried / max(fresh + carried, 1.0e-6);
    foam.shade = WATER_FOAM_HOLLOW * (1.0 - foam.thin);
    return foam;
}

// The white water covering terrains' water at `world` and the foam it
// carries there (x and y, shares of the surface, 0 to 1 each) under a pixel
// whose footprint is `footprint` (`water_footprint`), as the foam field
// holds them (`water.foam`: the meters across its cells, none where none
// runs; the meters its texture spans, which repeats; and the pixels its
// cells span where it hands over): filtered over the footprint by its
// levels, out to where its cells span water.foam.z pixels seen from the
// detail eye, handing over to `settled`, the same as the water's own state
// holds them (the white water born there and the foam its flow churned up),
// the field's mean, as a pyramid's level hands over to the next coarser
// (`detail_level`); `settled` past it.
fn water_foam_held(world: vec3<f32>, footprint: WaterFootprint, settled: vec2<f32>)
        -> vec2<f32> {
    let cell = water.foam.x;
    if cell <= 0.0 {
        return settled;
    }
    let picked = detail_level(2.0 * water.foam.z * detail_across(world), cell, 2u);
    if picked.level > 0u {
        return settled;
    }
    let side = water.foam.y;
    let field = textureSampleGrad(water_foam_field, water_foam_sampler, world.xz / side,
        footprint.dx / side, footprint.dy / side).xy;
    return mix(field, settled, picked.blend);
}
