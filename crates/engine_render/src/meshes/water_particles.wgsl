// Water's particles (`water_particles.rs`): what turbulence makes on
// terrains' water as it runs, its foam and the small waves on its surface,
// as particles the live flow carries around the detail eye. Foam particles
// hold the white water and the foam the water carries: born where it breaks
// and churns, by the cook's rules, in turbulence's bursts
// (`water_bursts.wgsl`), each drawn from the world point it is born at and
// the simulation's step alone, so every host draws the same; carried by the
// current its surface runs at, so its shear draws them out into streaks and
// they gather where it converges and against what stands in it; and
// draining as white water's bubbles rise out over its depth and as foam
// fades over the cook's foam life. Wave particles are segments of
// wavefronts (Yuksel, House and Keyser, "Wave Particles", 2007), raised by
// the boils the bursts bring up and where the water runs against what
// stands in it: each runs at its waves' group speed along its heading plus
// the current, spreading, splitting in three as it spreads, and dying by
// the water's viscosity and its eddy viscosity. Each frame they splat into
// a window of texels anchored in the world around the detail eye, which
// water's shading reads (`meshes/water_foam.wgsl`'s `water_splatted`): the
// share of the surface white water and foam cover, 1 − e^−d of the layers d
// their particles lay, and the slopes of the waves' surface. Nothing but
// the particles draws any of it: no texture, noise or pattern stands fixed
// to the world or the screen. Every frame, a fixed number of dispatches:
// - the particles the pools held before the frame, and the dispatches over
//   them (`particles_begin`);
// - the water read as shading reads it (`water_pixel`) at the points of a
//   lattice anchored in the world, PARTICLE_LATTICE texels apart
//   (`particles_gather`), and what carries and churns foam there
//   (`particles_currents`);
// - each particle's step: carried by the current, draining or spreading,
//   and how many it leaves (`particles_age`);
// - what each of the lattice's sites gives birth to (`particles_births`);
// - a scan over the particles and then the sites, in their order, of what
//   each leaves or gives birth to (`particles_sum`, `particles_scan`), by
//   which each lays its own into the other half of the pools
//   (`particles_place`): integer sums, so every host lays them in the same
//   order, those past the pools' room dropped;
// - every particle's splat, by fixed-point atomic sums, which any order adds
//   alike (`particles_splat`);
// - and the textures shading reads, from those sums, and their coarser
//   levels (`particles_look`, `particles_down`).

// Particles and items a workgroup of the passes over them takes, and of the
// scan over their groups; and the side of a workgroup of the passes over
// the lattice's points and the window's texels.
const PARTICLE_GROUP: u32 = 256u;
const PARTICLE_SIDE: u32 = 8u;
// Texels along a step of the lattice.
const PARTICLE_LATTICE: u32 = 8u;
// The words of `particle_counts`: the foam and the wave particles the pools
// hold after the frame, those born in it and those the pools had no room
// for, and those they held before it; the dispatch over the particles
// (three words) and over the scan's items (three more); and the first word
// of the items' counts.
const PARTICLE_FOAM_LIVE: u32 = 0u;
const PARTICLE_WAVES_LIVE: u32 = 1u;
const PARTICLE_FOAM_BORN: u32 = 2u;
const PARTICLE_WAVES_BORN: u32 = 3u;
const PARTICLE_FOAM_DROPPED: u32 = 4u;
const PARTICLE_WAVES_DROPPED: u32 = 5u;
const PARTICLE_FOAM_BEFORE: u32 = 6u;
const PARTICLE_WAVES_BEFORE: u32 = 7u;
const PARTICLE_ARGS: u32 = 8u;
const PARTICLE_ITEMS: u32 = 16u;
// Foam as the cook churns it (`engine_terrain::shallow_water`): the seconds
// it takes to fade to 1/e; the meters of water under which a place holds
// none; the least rate (square meters a second cubed) water loses energy at,
// gravity times its surface's slope times its speed, for it to churn any;
// the power the water loses breaking (watts a square meter of bed) at which
// it holds half the most; and water's density (kilograms a cubic meter).
// Only breaking water draws air in (`water_sim.wgsl`'s `sim_churn`).
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
// breaking: about what bubbles a millimeter across rise at in natural
// water, whose surfactants slow them (Clift, Grace and Weber 1978), so white
// water drains in as many seconds as it is deep in tenths of a meter.
const PARTICLE_RISE: f32 = 0.1;
// The weight the quadratic B-spline a foam particle splats by gives the
// texel holding it at its middle, (3/4)²: a particle holding t²/PARTICLE_PEAK
// (`particle_quantum`, t a texel's side) lays one layer of foam there.
const PARTICLE_PEAK: f32 = 0.5625;
// The share of the amount a foam particle is born with that it drains to
// before it goes: a sixteenth of a layer at its middle. Births lay as much
// more as that leaves of what the cook's rules lay (`particle_births`).
const PARTICLE_DRAINED: f32 = 0.0625;
// The most particles of each kind one event lays, each laying more where
// the event lays more than these hold.
const PARTICLE_EVENT_MOST: u32 = 16u;
// Steps of the splat's fixed-point sums a unit: 2^16, so a sum holds 32,768
// layers or a slope as steep, and a sixteenth of a layer laid at a
// kernel's corner keeps a hundred steps.
const PARTICLE_FIXED: f32 = 65536.0;
// The most of the surface the water's own foam covers that particles lay
// foam for: all but a step of the signed byte its words hold it in
// (`water_words`), 4.8 layers.
const PARTICLE_DENSEST: f32 = 0.99212598;
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
// The kinds of events laying particles: turbulence's bursts, laying white
// water and foam and raising boils; the water's own foam seeded where the
// window newly covers a site; and its own foam carried in across the
// window's rim.
const PARTICLE_BURSTS: u32 = 0u;
const PARTICLE_SEEDS: u32 = 1u;
const PARTICLE_INFLOW: u32 = 2u;
// What a texel outside the window splats into: nothing.
const PARTICLE_NO_TEXEL: u32 = 0xffffffffu;

// The particles' frame: the first texel of the window along x and z (world
// texels, `step.x` meters across, counted from the world's origin), then
// that of the window held the frame before (zw); the lattice's first point
// along x and z (world points, PARTICLE_LATTICE texels apart, counted from
// the world's origin), its points along a side and the window's texels
// along a side (w); the meters across a texel, the seconds this frame steps
// and the wind's velocity (meters a second along x and z); 1 where the
// pools hold none yet (x), the index of the simulation's step the frame
// begins in (y) and of the one it ends in (z), and the least steps a slot
// of events spans, a power of two (w, `water_burst_slot`); the most foam
// and wave particles the pools hold, the half of the pools holding the
// frame before's particles and the lattice's points, its sites (`pool`);
// and the seconds of a simulation step and the meters across the cells of
// the water the particles ride, past which wave particles stand for no
// eddies (`water_eddy_size`), then nothing (`eddies`).
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
// its depth, meters, none where it holds no foam, and how strongly it
// breaks white, 0 to 1; how fast its surface falls along x and z, meters a
// meter; and the foam it carries, 0 to WATER_FOAM_MOST, and how turbulent it
// runs, 0 to 1); then what carries and churns the foam there as
// `particles_currents` finds it (the current its surface runs at along x and
// z, meters a second; how fast that current spreads, a second, negative
// where it converges, and how hard the water churns foam, `sim_churn`'s
// churn; its depth and how strongly it breaks white; and the foam it
// carries and the speed its eddies turn at, meters a second): half floats,
// a pair a word.
@group(2) @binding(33) var<storage, read_write> particle_lattice: array<vec4<u32>>;
// The pools twice over, the frame before's particles and this frame's
// (`pool.z`'s half and the other), each the foam pool's `pool.x` particles,
// then the wave pool's `pool.y`, those living first: where each stands
// (world meters along x and z, a float's bits each), and what it holds
// (`particle_read`).
@group(2) @binding(34) var<storage, read_write> particle_pool: array<vec4<u32>>;
// The pools' counts (PARTICLE_FOAM_LIVE to PARTICLE_ARGS); from
// PARTICLE_ITEMS, the foam and the wave particles each living particle
// leaves after its step, and then those each site gives birth to, a word
// each (`particle_count_word`); and the scan's sums of them, a pair of words
// for each group of PARTICLE_GROUP items, the foam's then the waves'.
@group(2) @binding(35) var<storage, read_write> particle_counts: array<atomic<u32>>;
// What the particles splat into each texel of the window, as its texture's
// texels lie (a world texel's remainder over the texels along a side): the
// layers of white water and of foam laid over it, and how fast the waves'
// surface falls along x and z, four fixed-point sums of PARTICLE_FIXED
// steps a unit, which `particles_look` takes and clears.
@group(2) @binding(36) var<storage, read_write> particle_splat: array<atomic<i32>>;
// A level of the textures shading reads (`particle_target`,
// `particle_spread_target`, which `particles_look` writes the finest level
// of and `particles_down` each coarser one), and the level finer than it
// (`particle_source`, `particle_spread_source`), which `particles_down`
// averages: each texel the share of it white water covers and foam covers,
// and how fast the waves' surface falls along x and z (`particle_target`);
// and the squares of those falls along x and along z and their product
// (`particle_spread_target`, `water_spread`), whose mean less that of the
// falls' is the covariance of the slopes a texel hides.
@group(2) @binding(37) var particle_source: texture_2d<f32>;
@group(2) @binding(38) var particle_target: texture_storage_2d<rgba16float, write>;
@group(2) @binding(39) var particle_spread_source: texture_2d<f32>;
@group(2) @binding(40) var particle_spread_target: texture_storage_2d<rgba16float, write>;

// The water at a lattice point as `particles_gather` reads it.
struct ParticleWater {
    velocity: vec2<f32>,
    depth: f32,
    born: f32,
    slope: vec2<f32>,
    foam: f32,
    turbulence: f32,
}

// What carries and churns the foam at a point as `particles_currents`
// finds it.
struct ParticleCarrier {
    current: vec2<f32>,
    spread: f32,
    churn: f32,
    depth: f32,
    born: f32,
    foam: f32,
    stirring: f32,
}

// A particle: where it stands (world meters along x and z) and whether it
// is a wave's; a foam particle's white water and foam (layers times square
// meters); a wave particle's height (meters), how many times as long along
// the current as across it it has drawn out, its heading (the current's,
// radians from x toward z) and its radius across the current (meters).
struct Particle {
    at: vec2<f32>,
    wave: bool,
    amounts: vec2<f32>,
    amplitude: f32,
    stretch: f32,
    heading: f32,
    radius: f32,
}

// `value`'s remainder over `cells`, none or more.
fn particle_remainder(value: vec2<i32>, cells: i32) -> vec2<i32> {
    return (value % cells + cells) % cells;
}

// The meters between the lattice's points.
fn particle_spacing() -> f32 {
    return particle_frame.step.x * f32(PARTICLE_LATTICE);
}

// The amount (layers times square meters) a foam particle is born with:
// one layer at the texel holding it at its middle (PARTICLE_PEAK).
fn particle_quantum() -> f32 {
    let texel = particle_frame.step.x;
    return texel * texel / PARTICLE_PEAK;
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

// The layers of foam covering the share `share` of the surface,
// −ln(1 − share), each hiding as often as not what the others leave; no
// more than PARTICLE_DENSEST covers.
fn particle_layers(share: f32) -> f32 {
    return -log(1.0 - clamp(share, 0.0, PARTICLE_DENSEST));
}

// The weights a quadratic B-spline gives three points a step apart along an
// axis at a point `within` (−1/2 to 1/2) of a step from the middle one.
fn particle_spline(within: f32) -> vec3<f32> {
    let before = 0.5 - within;
    let after = 0.5 + within;
    return vec3<f32>(0.5 * before * before, 0.75 - within * within, 0.5 * after * after);
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
    let carried = unpack2x16float(words.w);
    var water: ParticleWater;
    water.velocity = unpack2x16float(words.x);
    water.depth = deep.x;
    water.born = deep.y;
    water.slope = unpack2x16float(words.z);
    water.foam = carried.x;
    water.turbulence = carried.y;
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

// How fast the velocity of the water at lattice point `point`, standing as
// `here`, changes a meter along x and along z (the columns): between the
// points either side of it (`particle_beside`).
fn particle_change(point: vec2<i32>, here: ParticleWater) -> mat2x2<f32> {
    let span = 2.0 * particle_spacing();
    let west = particle_beside(point - vec2<i32>(1, 0), here).velocity;
    let east = particle_beside(point + vec2<i32>(1, 0), here).velocity;
    let south = particle_beside(point - vec2<i32>(0, 1), here).velocity;
    let north = particle_beside(point + vec2<i32>(0, 1), here).velocity;
    return mat2x2<f32>((east - west) / span, (north - south) / span);
}

// The current the surface of water standing as `here` runs at, its
// velocity changing by `change` a meter along x and z (`particle_change`):
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

// The current the surface of the water at lattice point `point` runs at
// (`particle_surface`); none where it holds no foam, so the current carrying
// foam stops at whatever its water meets.
fn particle_current(point: vec2<i32>) -> vec2<f32> {
    let here = particle_water(point);
    if here.depth <= WATER_FOAM_WET {
        return vec2<f32>(0.0);
    }
    return particle_surface(here, particle_change(point, here));
}

// What carries and churns the foam at lattice point `point` as
// `particles_currents` found it.
fn particle_carrier_point(point: vec2<i32>) -> ParticleCarrier {
    let words = particle_lattice[particle_lattice_at(point, true)];
    let spreading = unpack2x16float(words.y);
    let deep = unpack2x16float(words.z);
    let carried = unpack2x16float(words.w);
    var carrier: ParticleCarrier;
    carrier.current = unpack2x16float(words.x);
    carrier.spread = spreading.x;
    carrier.churn = spreading.y;
    carrier.depth = deep.x;
    carrier.born = deep.y;
    carrier.foam = carried.x;
    carrier.stirring = carried.y;
    return carrier;
}

// `sum` with `carrier` added at `weight`.
fn particle_weighed(sum: ParticleCarrier, carrier: ParticleCarrier, weight: f32)
        -> ParticleCarrier {
    var added = sum;
    added.current += carrier.current * weight;
    added.spread += carrier.spread * weight;
    added.churn += carrier.churn * weight;
    added.depth += carrier.depth * weight;
    added.born += carrier.born * weight;
    added.foam += carrier.foam * weight;
    added.stirring += carrier.stirring * weight;
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
    return carrier;
}

// The particle `words` (`particle_pool`'s) holds: where it stands, then, for
// a foam particle (the top bit of w clear), its white water and its foam,
// half floats; for a wave particle, its height and its stretch, half floats
// (z), and its heading as a share of a turn (w's fourteen low bits) and its
// radius in eighths of a texel, less one (eight bits from 18).
fn particle_read(words: vec4<u32>) -> Particle {
    var particle: Particle;
    particle.at = bitcast<vec2<f32>>(words.xy);
    particle.wave = (words.w >> 31u) != 0u;
    if !particle.wave {
        particle.amounts = unpack2x16float(words.z);
        return particle;
    }
    let shape = unpack2x16float(words.z);
    particle.amplitude = shape.x;
    particle.stretch = shape.y;
    particle.heading = f32(words.w & 0x3fffu) * (2.0 * PI / 16384.0);
    particle.radius = f32(((words.w >> 18u) & 255u) + 1u) * 0.125 * particle_frame.step.x;
    return particle;
}

// The words `particle_pool` holds `particle` in (`particle_read`).
fn particle_word(particle: Particle) -> vec4<u32> {
    let at = bitcast<vec2<u32>>(particle.at);
    if !particle.wave {
        return vec4<u32>(at, pack2x16float(particle.amounts), 0u);
    }
    let turn = u32(round(fract(particle.heading / (2.0 * PI)) * 16384.0)) & 0x3fffu;
    let eighths = clamp(round(particle.radius / particle_frame.step.x * 8.0), 1.0, 256.0);
    let radius = u32(eighths) - 1u;
    return vec4<u32>(at, pack2x16float(vec2<f32>(particle.amplitude, particle.stretch)),
        (1u << 31u) | (radius << 18u) | turn);
}

// Whether world point `world` (meters along x and z) lies in the window.
fn particle_inside(world: vec2<f32>) -> bool {
    let texel = particle_frame.step.x;
    let first = vec2<f32>(particle_frame.window.xy) * texel;
    let last = first + f32(particle_frame.lattice.w) * texel;
    return all(world >= first) && all(world < last);
}

// Whether the square a lattice spacing across around site `site` (world
// meters) lies in the window whose first texel is `first`.
fn particle_site_in(site: vec2<f32>, first: vec2<i32>) -> bool {
    let texel = particle_frame.step.x;
    let half = 0.5 * particle_spacing();
    let low = vec2<f32>(first) * texel;
    let high = low + f32(particle_frame.lattice.w) * texel;
    return all(site - half >= low) && all(site + half <= high);
}

// The way into the window from its nearest edge at site `site` (world
// meters), a unit vector along x or z, where the site's square lies on the
// window's rim, within a lattice spacing of its edge; none elsewhere.
fn particle_inward(site: vec2<f32>) -> vec2<f32> {
    let texel = particle_frame.step.x;
    let low = vec2<f32>(particle_frame.window.xy) * texel;
    let high = low + f32(particle_frame.lattice.w) * texel;
    let near = site - low;
    let far = high - site;
    let nearest = min(min(near.x, near.y), min(far.x, far.y));
    if nearest >= 1.5 * particle_spacing() {
        return vec2<f32>(0.0);
    }
    if nearest == near.x {
        return vec2<f32>(1.0, 0.0);
    }
    if nearest == far.x {
        return vec2<f32>(-1.0, 0.0);
    }
    if nearest == near.y {
        return vec2<f32>(0.0, 1.0);
    }
    return vec2<f32>(0.0, -1.0);
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
// lays, as the draw `drawn` (0 to 1) rounds it to whole particles
// (`particle_quantum`'s each, so their mean is the amount), and each one's
// amount: PARTICLE_EVENT_MOST each laying their share where it lays more.
fn particle_quanta(amount: f32, drawn: f32) -> vec2<f32> {
    let quantum = particle_quantum();
    let quanta = max(amount, 0.0) / quantum;
    let most = f32(PARTICLE_EVENT_MOST);
    if quanta > most {
        return vec2<f32>(most, amount / most);
    }
    return vec2<f32>(floor(quanta + drawn), quantum);
}

// What a site of the lattice gives birth to this frame (`particle_births`):
// the site (world lattice points) and its middle (world meters); its
// burst's slot and middle (world meters), the white water and the foam
// particles it lays and each one's amount, whether it raises a boil, the
// boil's height and the radius of the boils its eddies raise; the white
// water and the foam particles seeding the water's
// own foam and each one's amount; and the foam particles carried in across
// the window's rim, each one's amount and their slot.
struct ParticleBirths {
    cell: vec2<i32>,
    site: vec2<f32>,
    slot: vec2<u32>,
    middle: vec2<f32>,
    laid: vec2<u32>,
    each: vec2<f32>,
    boils: u32,
    boil: f32,
    radius: f32,
    seeds: vec2<u32>,
    seeded: vec2<f32>,
    inflow: u32,
    carried: f32,
    inflow_slot: vec2<u32>,
}

// The foam and the wave particles `births` holds.
fn particle_births_count(births: ParticleBirths) -> vec2<u32> {
    return vec2<u32>(births.laid.x + births.laid.y + births.seeds.x + births.seeds.y
        + births.inflow, births.boils);
}

// What the site at lattice point `point` gives birth to this frame, where
// its square lies in the window and its water holds foam, by the water
// there (`particle_carrier_point`):
// - where the window newly covers it, or the pools hold none yet, the
//   water's own foam over its square, the white water born there (its
//   layers) and the foam it carries (`particle_layers`), and nothing else;
// - where a slot of its eddies' bursts begins (`particle_slot`), its eddies
//   `size` across turning over in size / u seconds (u their speed), each
//   reaching the surface over a size² of it once in that time, as
//   turbulence renews a surface (Danckwerts 1951): whether one or more is
//   born in the site's square over the slot (`water_burst_weight`), and
//   where; each lays over its patch, as wide as its eddy, what the cook's
//   rules lay there over its eddy's turnover across its size²: white water
//   at b/τ layers a second, b how strongly the water breaks white there and
//   τ the seconds it drains in (`particle_draining`), so b layers lie there
//   at rest, and foam at ln(1 + c)/WATER_FOAM_LIFE, c the cook's churn, so
//   the water there holds the foam the cook's churning holds it toward,
//   c / (1 + c) of the most, both over 1 − PARTICLE_DRAINED, the share the
//   particles lay before they go; and raises a boil, as high as its eddy's
//   velocity head, u²/2g, which the current carries off;
// - and on the window's rim, where the current runs into the window, the
//   water's own foam over the site's square each time the current takes
//   across it.
// A wave whose slopes fall under the smoothest water's lobe
// (`WATER_SMOOTHEST`'s square) turns no glint, and none is born.
fn particle_births(point: vec2<i32>) -> ParticleBirths {
    var births: ParticleBirths;
    births.cell = particle_frame.lattice.xy + point;
    let spacing = particle_spacing();
    births.site = vec2<f32>(births.cell) * spacing;
    births.middle = births.site;
    let here = particle_carrier_point(point);
    if here.depth <= WATER_FOAM_WET || !particle_site_in(births.site, particle_frame.window.xy) {
        return births;
    }
    let area = spacing * spacing;
    let size = particle_eddy(here.depth);
    births.radius = 0.5 * size;
    let number = PI / births.radius;
    let flattest = WATER_SMOOTHEST * WATER_SMOOTHEST;
    if particle_frame.flags.x != 0u || !particle_site_in(births.site, particle_frame.window.zw) {
        let draws = particle_draws(births.cell, PARTICLE_SEEDS, 0u,
            vec2<u32>(particle_frame.flags.z, 0u));
        let white = particle_quanta(here.born * area, draws.x);
        let foam = particle_quanta(particle_layers(here.foam / WATER_FOAM_MOST) * area, draws.y);
        births.seeds = vec2<u32>(u32(white.x), u32(foam.x));
        births.seeded = vec2<f32>(white.y, foam.y);
        return births;
    }
    if here.stirring > 0.0 {
        let turnover = size / here.stirring;
        let slot = particle_slot(births.cell, PARTICLE_BURSTS, turnover);
        if slot.z != 0u {
            let life = water_burst_life(slot.xy, particle_frame.eddies.x);
            let mean = area / (size * size) * life / turnover;
            let draws = particle_draws(births.cell, PARTICLE_BURSTS, 0u, slot.xy);
            let weight = water_burst_weight(mean, draws.x);
            if weight > 0.0 {
                let more = particle_draws(births.cell, PARTICLE_BURSTS, 1u, slot.xy);
                births.slot = slot.xy;
                births.middle = births.site + (draws.yz - 0.5) * spacing;
                let lasting = weight * size * size * turnover / (1.0 - PARTICLE_DRAINED);
                let white = particle_quanta(here.born / particle_draining(here.depth) * lasting,
                    draws.w);
                let foam = particle_quanta(log(1.0 + max(here.churn, 0.0)) / WATER_FOAM_LIFE
                    * lasting, more.x);
                births.laid = vec2<u32>(u32(white.x), u32(foam.x));
                births.each = vec2<f32>(white.y, foam.y);
                let boil = here.stirring * here.stirring / (2.0 * WATER_GRAVITY);
                if boil * number >= flattest {
                    births.boils = 1u;
                    births.boil = boil;
                }
            }
        }
    }
    let flowing = dot(here.current, particle_inward(births.site));
    if flowing > 0.0 {
        let slot = particle_slot(births.cell, PARTICLE_INFLOW, spacing / flowing);
        if slot.z != 0u {
            let draws = particle_draws(births.cell, PARTICLE_INFLOW, 0u, slot.xy);
            let foam = particle_quanta(particle_layers(here.foam / WATER_FOAM_MOST) * area,
                draws.x);
            births.inflow = u32(foam.x);
            births.carried = foam.y;
            births.inflow_slot = slot.xy;
        }
    }
    return births;
}

// The `k`th foam particle `births` holds: its burst's white water, then its
// foam, each at a point of the burst's patch drawn evenly over its disc;
// then the water's own white water and foam seeded, and its foam carried in
// across the rim, each at a point of the site's square drawn evenly over it.
fn particle_born_foam(births: ParticleBirths, k: u32) -> Particle {
    var particle: Particle;
    let spacing = particle_spacing();
    let laid = births.laid.x + births.laid.y;
    if k < laid {
        let draws = particle_draws(births.cell, PARTICLE_BURSTS, 2u + k, births.slot);
        let turn = 2.0 * PI * draws.y;
        particle.at = births.middle
            + births.radius * sqrt(draws.x) * vec2<f32>(cos(turn), sin(turn));
        particle.amounts = select(vec2<f32>(0.0, births.each.y), vec2<f32>(births.each.x, 0.0),
            k < births.laid.x);
        return particle;
    }
    let seed = k - laid;
    let seeded = births.seeds.x + births.seeds.y;
    if seed < seeded {
        let draws = particle_draws(births.cell, PARTICLE_SEEDS, 1u + seed,
            vec2<u32>(particle_frame.flags.z, 0u));
        particle.at = births.site + (draws.xy - 0.5) * spacing;
        particle.amounts = select(vec2<f32>(0.0, births.seeded.y),
            vec2<f32>(births.seeded.x, 0.0), seed < births.seeds.x);
        return particle;
    }
    let carried = seed - seeded;
    let draws = particle_draws(births.cell, PARTICLE_INFLOW, 1u + carried, births.inflow_slot);
    particle.at = births.site + (draws.xy - 0.5) * spacing;
    particle.amounts = vec2<f32>(0.0, births.carried);
    return particle;
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
// water over the step. It leaves none where it leaves the window or the
// water there holds no foam. A foam particle's white water drains as its
// bubbles rise out over the depth (`particle_draining`), and its foam fades
// over WATER_FOAM_LIFE, as the cook's, the particle going once it holds
// less than PARTICLE_DRAINED of what it was born with. A boil sinks back by
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
    let middle = particle_carrier(before.at + 0.5 * seconds * particle_velocity(here));
    aged.particle.at = before.at + seconds * particle_velocity(middle);
    if middle.depth <= WATER_FOAM_WET || !particle_inside(aged.particle.at) {
        return aged;
    }
    if !before.wave {
        let white = before.amounts.x * exp(-seconds / particle_draining(middle.depth));
        let foam = before.amounts.y * exp(-seconds / WATER_FOAM_LIFE);
        aged.particle.amounts = vec2<f32>(white, foam);
        aged.leaves = u32(white + foam >= PARTICLE_DRAINED * particle_quantum());
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

// The scan's items before the frame's births: the particles the pools held
// before the frame, then the lattice's sites.
fn particle_items() -> u32 {
    return atomicLoad(&particle_counts[PARTICLE_FOAM_BEFORE])
        + atomicLoad(&particle_counts[PARTICLE_WAVES_BEFORE]) + particle_frame.pool.w;
}

// The foam and the wave particles item `item` leaves or gives birth to
// (`particle_count_word`'s); none past the items.
fn particle_item(item: u32) -> vec2<u32> {
    let pooled = atomicLoad(&particle_counts[PARTICLE_FOAM_BEFORE])
        + atomicLoad(&particle_counts[PARTICLE_WAVES_BEFORE]);
    if item >= pooled + particle_frame.pool.w {
        return vec2<u32>(0u);
    }
    let at = select(particle_sites() + (item - pooled), PARTICLE_ITEMS + item, item < pooled);
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
    particle_pool[first + select(place, pools.x + place, particle.wave)] = particle_word(particle);
}

// Where `particle_splat` holds world texel `cell`'s sums: its remainder over
// the texels along a side, four words a texel; PARTICLE_NO_TEXEL outside
// the window.
fn particle_texel(cell: vec2<i32>) -> u32 {
    let first = particle_frame.window.xy;
    let texels = particle_frame.lattice.w;
    if any(cell < first) || any(cell >= first + vec2<i32>(texels)) {
        return PARTICLE_NO_TEXEL;
    }
    let at = particle_remainder(cell, texels);
    return u32(at.y * texels + at.x) * 4u;
}

// Splats foam particle `particle` by the quadratic B-spline over the 3 x 3
// texels around it, its weights summing to one: its white water and its
// foam as layers, a texel's share of each over a texel's area.
fn particle_splat_foam(particle: Particle) {
    let texel = particle_frame.step.x;
    let grid = particle.at / texel;
    let holding = floor(grid);
    let along = particle_spline(grid.x - holding.x - 0.5);
    let across = particle_spline(grid.y - holding.y - 0.5);
    let first = vec2<i32>(holding) - vec2<i32>(1);
    let laid = particle.amounts * (PARTICLE_FIXED / (texel * texel));
    for (var row = 0; row < 3; row++) {
        for (var column = 0; column < 3; column++) {
            let at = particle_texel(first + vec2<i32>(column, row));
            if at == PARTICLE_NO_TEXEL {
                continue;
            }
            let sums = vec2<i32>(round(laid * (along[column] * across[row])));
            if sums.x != 0 {
                atomicAdd(&particle_splat[at], sums.x);
            }
            if sums.y != 0 {
                atomicAdd(&particle_splat[at + 1u], sums.y);
            }
        }
    }
}

// Splats boil `particle` into the texels its bump covers: a raised cosine
// A/2·(1 + cos(πd)) over the ellipse its radius r spans across the current
// and its stretch s times r along it, d = √((a/sr)² + (c/r)²) for a texel a
// meters along its heading and c across it; how fast that surface falls
// along x and z at each texel's middle, A·π/2·sin(πd)·∇d away from its
// middle.
fn particle_splat_wave(particle: Particle) {
    let texel = particle_frame.step.x;
    let along = vec2<f32>(cos(particle.heading), sin(particle.heading));
    let across = vec2<f32>(-along.y, along.x);
    let long = particle.radius * particle.stretch;
    let reach = i32(ceil(long / texel));
    let holding = vec2<i32>(floor(particle.at / texel));
    let fall = PARTICLE_FIXED * 0.5 * particle.amplitude * PI;
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
            if sums.x != 0 {
                atomicAdd(&particle_splat[at + 2u], sums.x);
            }
            if sums.y != 0 {
                atomicAdd(&particle_splat[at + 3u], sums.y);
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
// spacing), over the terrain the particles ride; none where it holds no
// foam: where the water ends or lies thinner than WATER_FOAM_WET.
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
            pack2x16float(vec2<f32>(at.depth, at.words.state.y)), pack2x16float(at.slope),
            pack2x16float(vec2<f32>(at.words.foam, at.words.state.x)));
    }
    particle_lattice[particle_lattice_at(point, false)] = words;
}

// What carries and churns the foam at each lattice point: the current its
// surface runs at (`particle_surface`) and how fast it spreads, between the
// currents at the points either side, where whatever the water meets, dry
// ground or a solid standing up through it, stands still
// (`particle_current`), so foam carried against it piles up there; how hard
// the water churns foam, by the cook's rule (`sim_churn`): the power it
// loses as it runs, ρ times its depth times its velocity against the fall
// of its energy head (gravity times its surface's slope less the rise of
// its velocity head along the flow), which takes in where it plunges or
// jumps, as far as it breaks white (its eddies' turbulent Froude number
// from WATER_BREAK_ONSET to WATER_BREAK_FULL), as a share of
// WATER_FOAM_POWER; none where it loses energy slower than WATER_FOAM_LOSS;
// and how fast its eddies turn, as its shading's roughness takes them
// (`water_stirring`). Its depth, how strongly it breaks white and the foam
// it carries come along.
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
        let spread = (particle_current(point + vec2<i32>(1, 0)).x
            - particle_current(point - vec2<i32>(1, 0)).x
            + particle_current(point + vec2<i32>(0, 1)).y
            - particle_current(point - vec2<i32>(0, 1)).y) / span;
        let speed = length(u);
        let stirring = water_stirring(speed, water_eddies(speed, length(here.slope), here.depth),
            here.turbulence);
        words = vec4<u32>(pack2x16float(particle_surface(here, change)),
            pack2x16float(vec2<f32>(spread, churn)),
            pack2x16float(vec2<f32>(here.depth, here.born)),
            pack2x16float(vec2<f32>(here.foam, stirring)));
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
    let aged = particle_aged(particle_read(particle_pool[at]));
    particle_pool[at] = particle_word(aged.particle);
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
    if count.x > 0u {
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
    }
}

// Each item's particles laid into this frame's half of the pools, from the
// sum of the counts of the items before it, the groups' before its own
// (`particles_scan`'s) and its group's (`particle_scan_group`), within the
// pools' room (`particle_lay`): a particle the pools held, as it left its
// step; and what a site gives birth to, its foam particles and then its
// boil (`particle_born_foam`, `particle_born_wave`).
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
    if id.x < pooled {
        let particle = particle_read(particle_pool[particle_index(id.x, foam,
            particle_frame.pool.z)]);
        if !particle.wave {
            particle_lay(particle, first.x);
        } else {
            particle_lay(particle, first.y);
        }
        return;
    }
    let site = id.x - pooled;
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

// Each particle the pools hold after the frame, splatted into the window's
// sums (`particle_splat_foam`, `particle_splat_wave`).
@compute @workgroup_size(PARTICLE_GROUP)
fn particles_splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let foam = atomicLoad(&particle_counts[PARTICLE_FOAM_LIVE]);
    if id.x >= foam + atomicLoad(&particle_counts[PARTICLE_WAVES_LIVE]) {
        return;
    }
    let half = 1u - particle_frame.pool.z;
    let particle = particle_read(particle_pool[particle_index(id.x, foam, half)]);
    if particle.wave {
        particle_splat_wave(particle);
    } else {
        particle_splat_foam(particle);
    }
}

// Each texel of the finest level of the textures shading reads, from what
// the particles splatted into it, which it clears for the next frame's: the
// share of it white water and foam cover, 1 − e^−d of the layers d laid
// over it, each hiding as often as not what the others leave, so the
// coarser levels average coverage; and how fast the waves' surface falls
// along x and z, no steeper than a wave stands before it breaks
// (WATER_STEEPEST), past which the water breaks; and the squares of those
// falls and their product (`water_spread`).
@compute @workgroup_size(PARTICLE_SIDE, PARTICLE_SIDE)
fn particles_look(@builtin(global_invocation_id) id: vec3<u32>) {
    let texels = particle_frame.lattice.w;
    let texel = vec2<i32>(id.xy);
    if any(texel >= vec2<i32>(texels)) {
        return;
    }
    let at = u32(texel.y * texels + texel.x) * 4u;
    let layers = vec2<f32>(f32(atomicExchange(&particle_splat[at], 0)),
        f32(atomicExchange(&particle_splat[at + 1u], 0))) / PARTICLE_FIXED;
    var fall = vec2<f32>(f32(atomicExchange(&particle_splat[at + 2u], 0)),
        f32(atomicExchange(&particle_splat[at + 3u], 0))) / PARTICLE_FIXED;
    let steepness = length(fall);
    if steepness > WATER_STEEPEST {
        fall *= WATER_STEEPEST / steepness;
    }
    let covered = 1.0 - exp(-max(layers, vec2<f32>(0.0)));
    textureStore(particle_target, texel, vec4<f32>(covered, fall));
    textureStore(particle_spread_target, texel, vec4<f32>(water_spread(fall), 0.0));
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
