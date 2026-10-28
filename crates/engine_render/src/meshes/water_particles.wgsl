// Water's particles (`water_particles.rs`): what turbulence makes on
// terrains' water as it runs, its foam and the small waves on its surface,
// as particles the live flow carries around the detail eye. Foam particles
// are rafts of foam, each an ellipse holding white water and the foam the
// water carries: born where the water breaks, as fast as it loses energy
// breaking (`sim_churn`'s rule: where it plunges, jumps, runs against what
// stands in it or tumbles down a steep bed), in turbulence's bursts
// (`water_bursts.wgsl`), each drawn from the world point it is born at and
// the simulation's step alone, so every host draws the same; carried by the
// current its surface runs at and by the boils its eddies finer than the
// lattice raise, which well up and spread over the surface, so they gather
// where the current converges, against what stands in it and along the
// rims where boils meet, and strained by both, their long axes turning and
// stretching, until they tear in two; their white water's
// bubbles rising out over its depth into the foam they float on, which
// drains over the cook's foam life, so a raft goes from heaped white to a
// thin lace and then bursts. Wave particles are segments of
// wavefronts (Yuksel, House and Keyser, "Wave Particles", 2007), raised by
// the boils the bursts bring up and where the water runs against what
// stands in it: each runs at its waves' group speed along its heading plus
// the current, spreading, splitting in three as it spreads, and dying by
// the water's viscosity and its eddy viscosity. Rafts are born and ride
// over a square of their own around the detail eye (`particle_square`),
// each born no smaller than a pixel there; wave particles are born and run
// in a window of texels anchored in the world around the detail eye, inside
// that square, into which each frame they splat the slopes of the waves'
// surface, which water's shading reads (`meshes/water_foam.wgsl`'s
// `water_splatted`). Foam particles draw themselves, each its ellipse on
// the water's surface, sharp at every distance, into the layers of white
// water and foam over each pixel (`vs_foam`, `fs_foam`), which water's
// shading reads as foam (`water_foam_layers`). Nothing but the particles
// draws any of it: no texture, noise or pattern stands fixed to the world
// or the screen, and past the square no foam shows. Every frame, a fixed
// number of dispatches:
// - the particles the pools held before the frame, and the dispatches over
//   them (`particles_begin`);
// - the water read as shading reads it (`water_pixel`) at the points of a
//   lattice anchored in the world, PARTICLE_LATTICE texels apart, over the
//   square and PARTICLE_MARGIN points past it (`particles_gather`), and
//   what carries and churns foam there (`particles_currents`);
// - each particle's step: carried by the current, draining, strained or
//   spreading, and how many it leaves (`particles_age`);
// - what each of the lattice's sites gives birth to (`particles_births`);
// - a scan over the sites and then the particles, in their order, of what
//   each gives birth to or leaves (`particles_sum`, `particles_scan`), by
//   which each lays its own into the other half of the pools
//   (`particles_place`): integer sums, so every host lays them in the same
//   order, those past the pools' room dropped, the frame's births laid
//   first, so the pools drop their oldest;
// - every wave particle's splat, its slopes and the current its boil
//   spreads over the surface, by fixed-point atomic sums, which any order
//   adds alike (`particles_splat`);
// - and the textures shading reads, from those sums, and their coarser
//   levels (`particles_look`, `particles_down`);
// then the foam particles' draw, over the opaque scene's depth, before
// water draws.

// Particles and items a workgroup of the passes over them takes, and of the
// scan over their groups; and the side of a workgroup of the passes over
// the lattice's points and the window's texels.
const PARTICLE_GROUP: u32 = 256u;
const PARTICLE_SIDE: u32 = 8u;
// Texels along a step of the lattice, and the points it keeps past its
// square on each side.
const PARTICLE_LATTICE: u32 = 8u;
const PARTICLE_MARGIN: u32 = 2u;
// The words of `particle_counts`: the foam and the wave particles the pools
// hold after the frame, those born in it and those the pools had no room
// for, and those they held before it; the dispatch over the particles
// (three words) and over the scan's items (three more); the draw of the
// foam particles (four words: six vertices each, as many as live, from the
// first of each); and the first word of the items' counts.
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
// Foam as the cook churns it (`engine_terrain::shallow_water`): the seconds
// it takes to fade to 1/e; the meters of water under which a place holds
// none; the least rate (square meters a second cubed) water loses energy at,
// gravity times its surface's slope times its speed, for it to churn any;
// the power the water loses breaking (watts a square meter of bed) at which
// it holds half the most; and water's density (kilograms a cubic meter).
// Only breaking water draws air in (`water_sim.wgsl`'s `sim_churn`), and
// foam particles are born of it alone: the white water it churns up, whose
// bubbles rise into the foam they float on.
const WATER_FOAM_LIFE: f32 = 20.0;
const WATER_FOAM_WET: f32 = 0.005;
const WATER_FOAM_LOSS: f32 = 0.02;
const WATER_FOAM_POWER: f32 = 500.0;
const WATER_DENSITY: f32 = 1000.0;
// What floats rides the water's surface (`water_sim.wgsl`'s
// `sim_surface_current`): turned outward in a bend, the tangent of its turn
// WATER_SURFACE_BEND depths over the streamlines' radius of curvature, no
// more than WATER_SURFACE_BEND_MOST; and dragged by the wind at
// WATER_SURFACE_DRIFT of its speed.
const WATER_SURFACE_BEND: f32 = 7.0;
const WATER_SURFACE_BEND_MOST: f32 = 1.0;
const WATER_SURFACE_DRIFT: f32 = 0.03;
// The meters a second white water's bubbles rise out of it once it stops
// breaking: what the bubbles a few millimeters across that hold most of
// breaking water's air rise at (Clift, Grace and Weber 1978; Chanson 1996),
// so white water drains in as many seconds as it is deep in quarters of a
// meter.
const PARTICLE_RISE: f32 = 0.25;
// The share of the white water whose bubbles, once they rise, float on as
// foam (Callaghan, Deane and Stokes 2016: a whitecap's foam patch outlives
// its bubble plume, shrinking as it goes).
const WATER_FOAM_KEPT: f32 = 0.3;
// The layers of foam fresh white water lies in: wet foam, round bubbles
// heaped as they churn, which a raft is born as thick as.
const WATER_FOAM_FRESH: f32 = 2.0;
// The squares of a texel, or of a pixel where that spans more, over which
// a foam particle is born with one layer of foam at most
// (`particle_quantum`): sixteen ninths, so a raft born as thick as fresh
// white water lies is a little more than one of them across.
const PARTICLE_QUANTUM: f32 = 1.7777778;
// The layers of foam under which a raft goes, spread thin by the current: a
// sixteenth of a layer, covering 6 % of what it spans.
const PARTICLE_THINNEST: f32 = 0.0625;
// The e-folds a raft's length to its width grows along the current each
// turnover of the eddies finer than the lattice around it, as their shear
// strains it: near a channel's surface they stream along the flow (Nezu
// and Nakagawa 1993: the streaks turbulence draws on a channel's surface).
const PARTICLE_STRAIN: f32 = 0.5;
// The times as long as across a raft grows before it tears in two, end to
// end; and the meters across the narrowest raft, about a cluster of the
// bubbles a river's foam holds, a few millimeters each (Deane and Stokes
// 2002): strained thinner, a strand of foam breaks up instead, and a raft
// shrunk shorter is gone (`particle_raft_aged`).
const PARTICLE_TEAR: f32 = 8.0;
const PARTICLE_NARROWEST: f32 = 0.01;
// The most a raft's axes change by over a step, as a share of them: a step
// longer than its strain takes moves them that far, no farther.
const PARTICLE_STRAINED_MOST: f32 = 0.5;
// The most particles of each kind one event lays, each laying more where
// the event lays more than these hold.
const PARTICLE_EVENT_MOST: u32 = 16u;
// Steps of the splat's fixed-point sums a unit: 2^16, so a sum holds a
// slope 32,768 times as steep as a unit.
const PARTICLE_FIXED: f32 = 65536.0;
// How many times as long as across a boil draws out along the current
// each turnover of its eddy, as the shear its eddies stand for strains it:
// a boil wells up round and leaves as a scar streaked along the flow
// (Nezu and Nakagawa 1993, surface renewal in open channels); and the most
// it draws out to, as long as the half float its stretch is held in stays
// fine.
const PARTICLE_STRETCH: f32 = 2.0;
const PARTICLE_STRETCH_MOST: f32 = 16.0;
// The texels across the smallest eddies wave particles stand for: four, a
// kernel two texels in radius, the least whose slopes the texels resolve.
const PARTICLE_SMALLEST: f32 = 4.0;
// The first stream of the burst rule's draws (`water_burst_bits`) the
// particles take, past the live solve's (`WATER_BURST_LIVE`), and the
// streams each kind of event takes: its slots' and its own draws, then one
// for each particle it lays.
const PARTICLE_STREAMS: u32 = 32u;
const PARTICLE_STREAM_SPAN: u32 = 64u;
// The kind of events laying particles: turbulence's bursts, laying white
// water and raising boils, the only one.
const PARTICLE_BURSTS: u32 = 0u;
// What a texel outside the window splats into: nothing.
const PARTICLE_NO_TEXEL: u32 = 0xffffffffu;

// The particles' frame: the first texel of the window along x and z (world
// texels, `step.x` meters across, counted from the world's origin), then
// that of the window the frame before splatted (zw); the lattice's first
// point along x and z (world points, PARTICLE_LATTICE texels apart, counted
// from the world's origin), PARTICLE_MARGIN points before its square's, its
// points along a side and the window's texels along a side (w); the meters
// across a texel, the seconds this frame steps and the wind's velocity
// (meters a second along x and z); 1 where the pools hold none yet (x), the
// index of the simulation's step the frame begins in (y) and of the one it
// ends in (z), and the least steps a slot of events spans, a power of two
// (w, `water_burst_slot`); the most foam and wave particles the pools hold,
// the half of the pools holding the frame before's particles and the
// lattice's points, its sites (`pool`); and the seconds of a simulation
// step and the meters across the cells of the water the particles ride,
// past which wave particles stand for no eddies (`water_eddy_size`), then
// nothing (`eddies`).
struct ParticleFrame {
    window: vec4<i32>,
    lattice: vec4<i32>,
    step: vec4<f32>,
    flags: vec4<u32>,
    pool: vec4<u32>,
    eddies: vec4<f32>,
}

@group(2) @binding(32) var<uniform> particle_frame: ParticleFrame;
// The lattice's points, row by row, twice over: first the water there as
// `particles_gather` reads it (its velocity along x and z, meters a second;
// its depth, meters, none where it holds no foam, and how turbulent it
// runs, 0 to 1; how fast its surface falls along x and z, meters a meter;
// and the height its surface stands at, world meters, a float's bits); then
// what carries and churns the foam there as `particles_currents` finds it
// (the current its surface runs at along x and z, meters a second; how hard
// the water churns foam, `sim_churn`'s churn, and its depth; the speed its
// eddies turn at, meters a second; and the height its surface stands at, as
// gathered): half floats, a pair a word but the surfaces', none in the
// words left over.
@group(2) @binding(33) var<storage, read_write> particle_lattice: array<vec4<u32>>;
// The pools twice over, the frame before's particles and this frame's
// (`pool.z`'s half and the other), each the foam pool's `pool.x` particles,
// then the wave pool's `pool.y`, those living first, two words each
// (`particle_read`).
@group(2) @binding(34) var<storage, read_write> particle_pool: array<vec4<u32>>;
// The pools' counts (PARTICLE_FOAM_LIVE to PARTICLE_ARGS); from
// PARTICLE_ITEMS, the foam and the wave particles each living particle
// leaves after its step, and then those each site gives birth to, a word
// each (`particle_count_word`); and the scan's sums of them, a pair of words
// for each group of PARTICLE_GROUP items, the foam's then the waves'.
@group(2) @binding(35) var<storage, read_write> particle_counts: array<atomic<u32>>;
// What wave particles splat into each texel of the window, as its
// texture's texels lie (a world texel's remainder over the texels along a
// side): how fast the waves' surface falls along x and z, and the current
// their boils spread over the surface along x and z (meters a second), four
// fixed-point sums of PARTICLE_FIXED steps a unit, which `particles_look`
// takes and clears.
@group(2) @binding(36) var<storage, read_write> particle_splat: array<atomic<i32>>;
// A level of the textures shading reads (`particle_target`,
// `particle_spread_target`, which `particles_look` writes the finest level
// of and `particles_down` each coarser one), and the level finer than it
// (`particle_source`, `particle_spread_source`), which `particles_down`
// averages: each texel how fast the waves' surface falls along x and z and
// the squares of those falls (`particle_target`), and their product and the
// current the boils spread over the surface (`particle_spread_target`),
// the squares and product `water_spread`'s, whose mean less that of the
// falls' is the covariance of the slopes a texel hides.
@group(2) @binding(37) var particle_source: texture_2d<f32>;
@group(2) @binding(38) var particle_target: texture_storage_2d<rgba16float, write>;
@group(2) @binding(39) var particle_spread_source: texture_2d<f32>;
@group(2) @binding(40) var particle_spread_target: texture_storage_2d<rgba16float, write>;
// The pools as the foam particles' draw reads them (`particle_pool`'s), with
// their frame (`vs_foam`).
@group(2) @binding(48) var<storage, read> foam_pool: array<vec4<u32>>;

// The water at a lattice point as `particles_gather` reads it.
struct ParticleWater {
    velocity: vec2<f32>,
    depth: f32,
    turbulence: f32,
    slope: vec2<f32>,
    surface: f32,
}

// What carries and churns the foam at a point as `particles_currents`
// finds it, and the height (world meters) its water's surface stands at,
// summed over `wet` of the weights it was read with: the points holding
// water alone, so no shore pulls it down.
struct ParticleCarrier {
    current: vec2<f32>,
    churn: f32,
    depth: f32,
    stirring: f32,
    surface: f32,
    wet: f32,
}

// A particle: where it stands (world meters along x and z) and whether it
// is a wave's; a foam particle's white water and foam (layers times square
// meters), half its raft's long axis (meters along x and z) and half its
// short axis (meters), and the height (world meters) the surface of the
// water carrying it stands at, which it lies on; a wave particle's height
// (meters), how many times as long along the current as across it it has
// drawn out, its heading (the current's, radians from x toward z) and its
// radius across the current (meters).
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

// The two words the pools hold a particle in (`particle_read`).
struct ParticleWords {
    first: vec4<u32>,
    second: vec4<u32>,
}

// `value`'s remainder over `cells`, none or more.
fn particle_remainder(value: vec2<i32>, cells: i32) -> vec2<i32> {
    return (value % cells + cells) % cells;
}

// The meters between the lattice's points.
fn particle_spacing() -> f32 {
    return particle_frame.step.x * f32(PARTICLE_LATTICE);
}

// The most amount (layers times square meters) a foam particle born at
// world point `world` holds, where its event lays no more than
// PARTICLE_EVENT_MOST of them: a layer over PARTICLE_QUANTUM squares of a
// texel, or of the pixel there seen from the detail eye where that spans
// more (`detail_across`), so no raft is born smaller than either.
fn particle_quantum(world: vec3<f32>) -> f32 {
    let size = max(particle_frame.step.x, detail_across(world));
    return size * size * PARTICLE_QUANTUM;
}

// The meters across the eddies of water `depth` meters deep that wave
// particles stand for (`water_eddy_size`).
fn particle_eddy(depth: f32) -> f32 {
    return water_eddy_size(depth, particle_frame.step.x, particle_frame.eddies.y);
}

// The seconds white water `depth` meters deep drains in once it stops
// breaking, as its bubbles rise out over its depth at PARTICLE_RISE.
fn particle_draining(depth: f32) -> f32 {
    return max(depth, WATER_BREAK_THINNEST) / PARTICLE_RISE;
}

// The weights a quadratic B-spline gives three points a step apart along an
// axis at a point `within` (−1/2 to 1/2) of a step from the middle one.
fn particle_spline(within: f32) -> vec3<f32> {
    let before = 0.5 - within;
    let after = 0.5 + within;
    return vec3<f32>(0.5 * before * before, 0.75 - within * within, 0.5 * after * after);
}

// How fast those weights change a step along the axis.
fn particle_spline_change(within: f32) -> vec3<f32> {
    return vec3<f32>(within - 0.5, -2.0 * within, within + 0.5);
}

// Where the lattice keeps point `point` (counted from its first, the
// nearest of its points for one past them): as `particles_gather` reads the
// water there, or as `particles_currents` finds what carries its foam where
// `carrier`.
fn particle_lattice_at(point: vec2<i32>, carrier: bool) -> u32 {
    let points = particle_frame.lattice.z;
    let at = clamp(point, vec2<i32>(0), vec2<i32>(points - 1));
    let index = u32(at.y * points + at.x);
    return select(index, index + u32(points * points), carrier);
}

// The water at lattice point `point` as `particles_gather` read it.
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

// The water at lattice point `point` beside a point standing as `here`:
// `here` itself where it holds no foam, as the cook takes a dry neighbor
// (`sim_churn`), so shores turn and churn none of it.
fn particle_beside(point: vec2<i32>, here: ParticleWater) -> ParticleWater {
    let water = particle_water(point);
    if water.depth <= WATER_FOAM_WET {
        return here;
    }
    return water;
}

// The current the surface of water standing as `here` runs at, its
// velocity changing by `change` a meter along x and z (the columns):
// its velocity turned outward in a bend by δ, tan δ = WATER_SURFACE_BEND·h/R,
// no more than WATER_SURFACE_BEND_MOST, where 1/R = |u × (u·∇)u| / |u|³ is
// the streamlines' curvature and outward is away from the side u × (u·∇)u
// says they curve toward; dragged by WATER_SURFACE_DRIFT of the frame's
// wind.
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

// What carries and churns the foam at lattice point `point` as
// `particles_currents` found it.
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

// `sum` with `carrier` added at `weight`, its surface only where it holds
// water.
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

// What carries and churns the foam at world point `world` (meters along x
// and z): the quadratic B-spline through the lattice's 3 x 3 points around
// it, smooth across them, so no line of the lattice shows in the current
// and the particles it carries.
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

// How fast the current carrying foam at world point `world` changes a meter
// along x and along z (the columns), as the B-spline through the lattice's
// points around it (`particle_carrier`) changes: its shear, which strains
// the rafts it carries, and its spread, its trace.
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

// The current the boils spread over the surface at world point `world`
// (meters along x and z, `particle_splat_wave`), as the frame before
// splatted it, at the textures' level two of the window's texels across
// (`particle_spread_source`): bilinear between its texels around the point,
// none past the window it splatted; and how fast it changes a meter along x
// and along z (the columns), the bilinear's own.
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

// The current the boils spread over the surface over world texel `cell` of
// the textures' level two of the window's texels across, as the frame
// before splatted it; none where the window it splatted (`window.zw`)
// holds less than the whole of it, its texels another place's past it.
fn particle_boil_texel(cell: vec2<i32>) -> vec2<f32> {
    let finest = 2 * cell;
    let first = particle_frame.window.zw;
    if any(finest < first) || any(finest + 2 > first + particle_frame.lattice.w) {
        return vec2<f32>(0.0);
    }
    let at = particle_remainder(cell, particle_frame.lattice.w / 2);
    return textureLoad(particle_spread_source, at, 0).yz;
}

// The particle the pools' words `first` and `second` hold, floats' bits, a
// wave particle's where `wave`, as the pool holding it says: where it
// stands; then, for a foam particle, its white water and its foam, then
// half its raft's long axis along x and z and half its short axis; for a
// wave particle, its height and its stretch, then its heading and its
// radius.
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

// The words the pools hold `particle` in (`particle_read`).
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

// The particle the pools hold at `index` (particles from the first of the
// pools' first half), a wave particle's where `wave`; and laying `particle`
// there.
fn particle_at(index: u32, wave: bool) -> Particle {
    return particle_read(particle_pool[2u * index], particle_pool[2u * index + 1u], wave);
}

fn particle_put(index: u32, particle: Particle) {
    let words = particle_words(particle);
    particle_pool[2u * index] = words.first;
    particle_pool[2u * index + 1u] = words.second;
}

// A round raft of radius `radius` meters holding `amounts`, at `at`.
fn particle_raft(at: vec2<f32>, amounts: vec2<f32>, radius: f32) -> Particle {
    var particle: Particle;
    particle.at = at;
    particle.amounts = amounts;
    particle.axis = vec2<f32>(radius, 0.0);
    particle.minor = radius;
    return particle;
}

// The layers of white water and foam raft `particle` holds over its area.
fn particle_thickness(particle: Particle) -> vec2<f32> {
    let area = PI * length(particle.axis) * particle.minor;
    return particle.amounts / max(area, 1.0e-12);
}

// The lattice's square, over which rafts are born and ride: its first
// corner and its last (world meters along x and z, xy and zw),
// PARTICLE_MARGIN points in from the lattice's first point and its last,
// so what carries a raft in it reads the lattice's points alone.
fn particle_square() -> vec4<f32> {
    let spacing = particle_spacing();
    let margin = i32(PARTICLE_MARGIN);
    let first = vec2<f32>(particle_frame.lattice.xy + margin) * spacing;
    let across = f32(particle_frame.lattice.z - 2 * margin - 1) * spacing;
    return vec4<f32>(first, first + across);
}

// The window, over which wave particles are born and run: its first corner
// and its last (world meters along x and z, xy and zw).
fn particle_window() -> vec4<f32> {
    let texel = particle_frame.step.x;
    let first = vec2<f32>(particle_frame.window.xy) * texel;
    return vec4<f32>(first, first + f32(particle_frame.lattice.w) * texel);
}

// Whether world point `world` (meters along x and z) lies in `square`
// (`particle_square`'s or `particle_window`'s).
fn particle_inside(world: vec2<f32>, square: vec4<f32>) -> bool {
    return all(world >= square.xy) && all(world < square.zw);
}

// Whether the square a lattice spacing across around site `site` (world
// meters) lies in `square`.
fn particle_site_in(site: vec2<f32>, square: vec4<f32>) -> bool {
    let half = 0.5 * particle_spacing();
    return all(site - half >= square.xy) && all(site + half <= square.zw);
}

// The stream of the burst rule's draws the `k`th draw of an event of kind
// `kind` takes.
fn particle_stream(kind: u32, k: u32) -> u32 {
    return PARTICLE_STREAMS + kind * PARTICLE_STREAM_SPAN + k;
}

// Four draws, each uniform over [0, 1), the `k`th of the event of kind
// `kind` at site `site` (world lattice points) over slot `slot`
// (`water_burst_bits`), each its draw's top 24 bits.
fn particle_draws(site: vec2<i32>, kind: u32, k: u32, slot: vec2<u32>) -> vec4<f32> {
    let bits = water_burst_bits(site, particle_stream(kind, k), slot);
    return vec4<f32>(bits >> vec4<u32>(8u)) * (1.0 / 16777216.0);
}

// The slot of events of kind `kind` living `life` seconds at site `site`
// (world lattice points) the frame ends in (`water_burst_slot`), and 1
// where it begins this frame, after the step the frame begins in (z).
fn particle_slot(site: vec2<i32>, kind: u32, life: f32) -> vec3<u32> {
    let stream = particle_stream(kind, 0u);
    let seconds = particle_frame.eddies.x;
    let least = particle_frame.flags.w;
    let slot = water_burst_slot(site, stream, particle_frame.flags.z, life, seconds, least);
    let begun = water_burst_slot(site, stream, particle_frame.flags.y, life, seconds, least);
    return vec3<u32>(slot, u32(slot.x != begun.x));
}

// The foam particles an event laying `amount` (layers times square meters)
// lays, as the draw `drawn` (0 to 1) rounds it to whole particles of
// `quantum` each (`particle_quantum`'s, so their mean is the amount), and
// each one's amount: PARTICLE_EVENT_MOST each laying their share where it
// lays more.
fn particle_quanta(amount: f32, quantum: f32, drawn: f32) -> vec2<f32> {
    let quanta = max(amount, 0.0) / quantum;
    let most = f32(PARTICLE_EVENT_MOST);
    if quanta > most { // filter: rafts no smaller than a pixel laying the same white water, their mean the amount
        return vec2<f32>(most, amount / most);
    }
    return vec2<f32>(floor(quanta + drawn), quantum);
}

// What a site of the lattice gives birth to this frame (`particle_births`):
// the site (world lattice points) and its middle (world meters); its
// burst's slot and middle (world meters), the rafts of white water it lays
// and each one's amount, whether it raises a boil, the boil's height and
// the radius of the boils its eddies raise, its patch's; and the height
// (world meters) its water's surface stands at.
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

// The foam and the wave particles `births` holds.
fn particle_births_count(births: ParticleBirths) -> vec2<u32> {
    return vec2<u32>(births.laid, births.boils);
}

// What the site at lattice point `point` gives birth to this frame, where
// its square lies in the lattice's (`particle_square`) and its water holds
// foam, by the water there (`particle_carrier_point`): where a slot of its
// eddies' bursts begins (`particle_slot`), its eddies
//   `size` across turning over in size / u seconds (u their speed), each
//   reaching the surface over a size² of it once in that time, as
//   turbulence renews a surface (Danckwerts 1951): whether one or more is
//   born in the site's square over the slot (`water_burst_weight`), and
//   where; each lays over its patch, as wide as its eddy, the white water
//   the water churns up there over its eddy's turnover across its size²:
//   ln(1 + c)/τ layers a second, c the cook's churn, the power it loses
//   breaking (`particles_currents`), and τ the seconds white water drains in
//   (`particle_draining`), so the white water lying there at rest covers
//   c / (1 + c) of it, none where it does not break, in rafts no smaller
//   than a pixel seen from the detail eye at the burst's middle, on the
//   water's surface at the site (`particle_quantum`), so their count follows
//   the foam's area on the screen; and, where the site's square lies in the
//   window too, raises a boil, as high as its eddy's velocity head, u²/2g,
//   which the current carries off.
// Foam comes of nothing else: where the square newly covers water, its foam
// is born there as it runs on. A wave whose slopes fall under the smoothest
// water's lobe (`WATER_SMOOTHEST`'s square) turns no glint, and none is
// born.
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

// The `k`th foam particle `births` holds: a round raft of its burst's white
// water, as thick as fresh white water lies (WATER_FOAM_FRESH), at a point
// of the burst's patch drawn evenly over its disc, on its site's surface.
fn particle_born_foam(births: ParticleBirths, k: u32) -> Particle {
    let draws = particle_draws(births.cell, PARTICLE_BURSTS, 2u + k, births.slot);
    let turn = 2.0 * PI * draws.y;
    var raft = particle_raft(births.middle
        + births.radius * sqrt(draws.x) * vec2<f32>(cos(turn), sin(turn)),
        vec2<f32>(births.each, 0.0), sqrt(births.each / (PI * WATER_FOAM_FRESH)));
    raft.surface = births.surface;
    return raft;
}

// The wave particle `births` holds: its boil, at the burst's middle, round,
// headed along the current.
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

// The velocity (meters a second along x and z) a particle runs at where
// `carrier` carries it: the current its surface runs at, foam and boils
// alike.
fn particle_velocity(carrier: ParticleCarrier) -> vec2<f32> {
    return carrier.current;
}

// A particle after the frame's step, and whether it lives on.
struct ParticleAged {
    particle: Particle,
    leaves: u32,
}

// `before` after a step of the frame's seconds: carried by the current
// (`particle_velocity`) by the midpoint rule, the water at the midpoint its
// water over the step. It leaves none where the water there holds no foam,
// nor a raft where it leaves the lattice's square (`particle_square`) and a
// boil where it leaves the window (`particle_window`). A raft's foam,
// floating, rides the current the boils spread over the surface too
// (`particle_boils`, none past the window), where its white water's
// bubbles, under the surface, ride the water's own, so a raft rides it as
// far as it holds foam; and it ages as `particle_raft_aged` says, strained
// by the shear of what carries it at the midpoint (`particle_shear`). A
// boil sinks back by
// e each turnover of its eddy, 2r/u (u its eddies' speed where it rides),
// as turbulence renews the surface (Danckwerts 1951), and draws out along
// the current by PARTICLE_STRETCH each turnover as the shear strains it,
// headed along the current where it rides; it goes once its slopes fall
// under the smoothest water's lobe (`WATER_SMOOTHEST`'s square), turning no
// glint.
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

// Raft `carried`, carried over a step of `seconds`, the water at the step's
// midpoint `middle`, the current and the boils' changing by `shear` a meter
// there:
// - its white water drains as its bubbles rise out over the depth
//   (`particle_draining`), WATER_FOAM_KEPT of it floating on as foam, and
//   its foam fades over WATER_FOAM_LIFE, as the cook's: what it loses goes
//   from its edges, its area shrinking with it, so it keeps its layers,
//   bubbles bursting where it is thinnest;
// - its long axis, a line the water carries, turns and stretches with that
//   shear and with the strain of the eddies finer than the lattice, along
//   the current at PARTICLE_STRAIN e-folds of its length to its width each
//   turnover of them, no more than PARTICLE_STRAINED_MOST of it a step; its
//   area
//   spreads or gathers with the current, thinning or heaping it, its short
//   axis what keeps it, the two swapping where the short one grows the
//   longer;
// - it grows no narrower than PARTICLE_NARROWEST, past which it holds its
//   length, as foam strained thinner than its bubbles breaks up instead of
//   drawing out; it tears in two, end to end, once PARTICLE_TEAR times as
//   long as across (`particle_piece`); and it goes once shrunk shorter than
//   that, or spread thinner than PARTICLE_THINNEST.
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

// The `k`th of the `pieces` raft `particle` leaves: itself where one, and
// where two, its halves torn end to end, each half as long, holding half
// of it, at the middles of its two halves.
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

// The word an item's counts take (`particle_counts`): its foam particles
// (x) in the low sixteen bits, its wave particles (y) in the high.
fn particle_count_word(count: vec2<u32>) -> u32 {
    return count.x | (count.y << 16u);
}

// The first word of the sites' counts in `particle_counts`, and of the
// scan's groups' sums.
fn particle_sites() -> u32 {
    return PARTICLE_ITEMS + particle_frame.pool.x + particle_frame.pool.y;
}

fn particle_groups() -> u32 {
    return particle_sites() + particle_frame.pool.w;
}

// Where the pools hold the `index`th living particle of half `half`, the
// first `foam` of them foam particles and the rest waves.
fn particle_index(index: u32, foam: u32, half: u32) -> u32 {
    let pools = particle_frame.pool.xy;
    let first = half * (pools.x + pools.y);
    return select(first + pools.x + (index - foam), first + index, index < foam);
}

// The scan's items: the lattice's sites, then the particles the pools held
// before the frame.
fn particle_items() -> u32 {
    return atomicLoad(&particle_counts[PARTICLE_FOAM_BEFORE])
        + atomicLoad(&particle_counts[PARTICLE_WAVES_BEFORE]) + particle_frame.pool.w;
}

// The foam and the wave particles item `item` leaves or gives birth to
// (`particle_count_word`'s); none past the items.
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

// The scan's sum of group `group`'s items (`particles_sum`'s), or, once
// `particles_scan` ran, of the groups before it.
fn particle_group_sum(group: u32) -> vec2<u32> {
    let at = particle_groups() + 2u * group;
    return vec2<u32>(atomicLoad(&particle_counts[at]), atomicLoad(&particle_counts[at + 1u]));
}

fn particle_store_group(group: u32, sum: vec2<u32>) {
    let at = particle_groups() + 2u * group;
    atomicStore(&particle_counts[at], sum.x);
    atomicStore(&particle_counts[at + 1u], sum.y);
}

// A workgroup's items' counts as `particle_scan_group` sums them.
var<workgroup> particle_sums: array<vec2<u32>, PARTICLE_GROUP>;

// The sum of `value` over the workgroup's invocations before `local`
// (Hillis and Steele's scan), every invocation calling it; `particle_sums`
// then holds each one's sum through its own.
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

// Lays `particle` at `place` of its pool in this frame's half, where its
// pool has room for it.
fn particle_lay(particle: Particle, place: u32) {
    let pools = particle_frame.pool.xy;
    if place >= select(pools.x, pools.y, particle.wave) {
        return;
    }
    let first = (1u - particle_frame.pool.z) * (pools.x + pools.y);
    particle_put(first + select(place, pools.x + place, particle.wave), particle);
}

// Where `particle_splat` holds world texel `cell`'s sums: its remainder over
// the texels along a side, four words a texel; PARTICLE_NO_TEXEL outside the
// window.
fn particle_texel(cell: vec2<i32>) -> u32 {
    let first = particle_frame.window.xy;
    let texels = particle_frame.lattice.w;
    if any(cell < first) || any(cell >= first + vec2<i32>(texels)) {
        return PARTICLE_NO_TEXEL;
    }
    let at = particle_remainder(cell, texels);
    return u32(at.y * texels + at.x) * 4u;
}

// Splats boil `particle` into the texels its bump covers: a raised cosine
// A/2·(1 + cos(πd)) over the ellipse its radius r spans across the current
// and its stretch s times r along it, d = √((a/sr)² + (c/r)²) for a texel a
// meters along its heading and c across it; how fast that surface falls
// along x and z at each texel's middle, A·π/2·sin(πd)·∇d away from its
// middle; and the current it spreads over the surface there, away from its
// middle at √(2gA)·sin(πd), as fast as the eddy welling up in it turns at
// its height's velocity head, still at its middle and its rim, so the water
// spreading from it converges along its rim (Nezu and Nakagawa 1993: the
// boils of a river's surface).
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

// The pools' particles before the frame's step, as the frame before left
// them (none where they hold none yet); the frame's births and drops,
// none yet; and the dispatches over those particles and over the scan's
// items, a workgroup of PARTICLE_GROUP each.
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

// Each lattice point's water, read as shading reads it at the point
// (`water_pixel`, at the finest level the lattice resolves, a lattice
// spacing), over the terrain the particles ride, with the world height its
// surface stands at; none where it holds no foam: where the water ends or
// lies thinner than WATER_FOAM_WET.
@compute @workgroup_size(PARTICLE_SIDE, PARTICLE_SIDE)
fn particles_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let point = vec2<i32>(id.xy);
    if any(point >= vec2<i32>(particle_frame.lattice.z)) {
        return;
    }
    let spacing = particle_spacing();
    let world = vec2<f32>(particle_frame.lattice.xy + point) * spacing;
    let at = water_pixel(world - terrain.origin.xz, spacing);
    var words = vec4<u32>(0u);
    if at.held >= WATER_HELD && at.depth > WATER_FOAM_WET {
        words = vec4<u32>(pack2x16float(at.words.flow),
            pack2x16float(vec2<f32>(at.depth, at.words.state.x)), pack2x16float(at.slope),
            bitcast<u32>(terrain.origin.y + at.surface));
    }
    particle_lattice[particle_lattice_at(point, false)] = words;
}

// What carries and churns the foam at each lattice point: the current its
// surface runs at (`particle_surface`), none where it holds no foam, so the
// current carrying foam stops at whatever its water meets, dry ground or a
// solid standing up through it, and foam carried against it piles up
// there; how hard the water churns foam, by the cook's rule (`sim_churn`):
// the power it loses as it runs, ρ times its depth times its velocity
// against the fall of its energy head (gravity times its surface's slope
// less the rise of its velocity head along the flow), which takes in where
// it plunges, jumps or piles against what stands in it, and the drag of a
// rough bed's stones on shallow water, as far as that breaks its surface
// white (its eddies' turbulent Froude number from WATER_BREAK_ONSET to
// WATER_BREAK_FULL: water running smooth over its bed draws no air in), as
// a share of WATER_FOAM_POWER; none where it loses energy slower than
// WATER_FOAM_LOSS. And how fast its eddies turn, as its shading's roughness
// takes them (`water_stirring`). Its depth and its surface come along.
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

// Each particle the pools held before the frame, after the frame's step
// (`particle_aged`), in its place, with the foam or the wave particles it
// leaves as its item's counts.
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

// What each of the lattice's sites gives birth to this frame
// (`particle_births`), as its item's counts, and the frame's births.
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

// The sums of each group of PARTICLE_GROUP items' counts.
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

// The groups' sums turned into the sums of the groups before each, a run
// of the groups for each invocation (as many as the pools and the sites
// make at most); and the particles the pools hold after the frame, the
// sums of every item's counts within their room, those past it dropped,
// and the dispatch over them.
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

// Each item's particles laid into this frame's half of the pools, from the
// sum of the counts of the items before it, the groups' before its own
// (`particles_scan`'s) and its group's (`particle_scan_group`), within the
// pools' room (`particle_lay`): what a site gives birth to, its foam
// particles and then its boil (`particle_born_foam`, `particle_born_wave`);
// and a particle the pools held, as it left its step, a raft torn in two as
// its halves (`particle_piece`).
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

// Each wave particle the pools hold after the frame, splatted into the
// window's sums (`particle_splat_wave`); foam particles draw themselves
// (`vs_foam`).
@compute @workgroup_size(PARTICLE_GROUP)
fn particles_splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let foam = atomicLoad(&particle_counts[PARTICLE_FOAM_LIVE]);
    if id.x < foam || id.x >= foam + atomicLoad(&particle_counts[PARTICLE_WAVES_LIVE]) {
        return;
    }
    let half = 1u - particle_frame.pool.z;
    particle_splat_wave(particle_at(particle_index(id.x, foam, half), true));
}

// Each texel of the finest level of the textures shading reads, from what
// the wave particles splatted into it, which it clears for the next
// frame's: how fast the waves' surface falls along x and z, no steeper than
// a wave stands before it breaks (WATER_STEEPEST), past which the water
// breaks; the squares of those falls and their product (`water_spread`);
// and the current the boils spread over the surface, which the next frame's
// rafts ride (`particle_boils`).
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

// Each texel of a coarser level of the textures shading reads: the mean of
// the four of the finer level it covers. The window's texels along a side
// are a whole number of its coarsest level's texels, so every level wraps
// as the finest does.
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

// A raft as `vs_foam` draws it: where in its ellipse a pixel lies, in its
// half axes as drawn (1 on its edge), and the layers of white water and
// foam it lays there.
struct FoamOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) within: vec2<f32>,
    @location(1) @interpolate(flat) layers: vec2<f32>,
}

// The corner of a raft's quad (−1 or 1 along its long and short axes) that
// vertex `vertex` of its two triangles takes.
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

// A corner of the quad around a raft: where it stands (terrain-local
// meters along x and z), where in the raft's ellipse it lies, in its half
// axes as drawn, and the layers of white water and foam the raft lays at
// its middle (`fs_foam`).
struct FoamQuad {
    at: vec2<f32>,
    within: vec2<f32>,
    layers: vec2<f32>,
}

// Corner `corner` of the quad around raft `raft`: the raft no thinner along
// either axis than a pixel spans at its middle, seen from the detail eye
// (`detail_across`), its layers thinning as it widens, so it lays the foam
// it holds whatever its size on screen; its layers at its middle, which
// the ellipse holds (1 + FOAM_FLAT²)/2 of on average (`fs_foam`).
fn foam_quad(raft: Particle, corner: vec2<f32>) -> FoamQuad {
    let local = raft.at - terrain.origin.xz;
    let pixel = detail_across(eye_world(water_relative(local,
        raft.surface - terrain.origin.y)));
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

// How a raft heaps: as thick as at its middle out to FOAM_FLAT of its
// radius, then thinning to nothing at its edge as (1 − d²), as foam drains
// from its edges (Weaire and Hutzler, The Physics of Foams).
const FOAM_FLAT: f32 = 0.85;

// Vertex `vertex` of the foam particle at `instance` of this frame's half of
// the pools (the frame's `particles_place` laid them, those living first):
// its raft, an ellipse (`foam_quad`) lying on the surface of the water
// carrying it, as the lattice the raft rides read it (its `surface`), each
// corner lifted by `water_lift`: where water's own pixels draw that surface,
// which the mesh carrying them may stand under where it holds no water
// (`water_field`).
@vertex
fn vs_foam(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> FoamOutput {
    let pools = particle_frame.pool.xy;
    let index = (1u - particle_frame.pool.z) * (pools.x + pools.y) + instance;
    let raft = particle_read(foam_pool[2u * index], foam_pool[2u * index + 1u], false);
    let quad = foam_quad(raft, foam_corner(vertex));
    var relative = water_relative(quad.at, raft.surface - terrain.origin.y);
    relative.y += water_lift(eye_world(relative));
    var output: FoamOutput;
    output.clip_position = terrain_view.relative_view_projection * vec4<f32>(relative, 1.0);
    output.within = quad.within;
    output.layers = quad.layers;
    return output;
}

// The layers of white water and foam a raft lays over a pixel, as it heaps
// there (FOAM_FLAT); none where the opaque scene stands in front of it, as
// it hides the water there (depth runs reversed, nearer greater).
@fragment
fn fs_foam(input: FoamOutput) -> @location(0) vec2<f32> {
    if textureLoad(water_depth, vec2<i32>(input.clip_position.xy), 0) > input.clip_position.z {
        discard;
    }
    let heaped = saturate((1.0 - dot(input.within, input.within))
        / (1.0 - FOAM_FLAT * FOAM_FLAT));
    return input.layers * heaped;
}
