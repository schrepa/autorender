// Foam's field (`water_foam_field.rs`): what turbulence makes on terrains'
// water as it runs, simulated live on a grid of its own, finer than the
// water's cells and anchored in the world around the detail eye. Each cell
// holds the white water covering it, the foam the water carries there, the
// turbulence reaching its surface, and the surface's own disturbance from
// it: its height and how fast that rises. Turbulence reaches the surface in
// bursts (`water_bursts.wgsl`), random in space and time about the
// simulation's rate: each an eddy reaching the surface, bringing it
// turbulence in a patch as wide as the eddy and raising a boil there. The
// turbulence reaching the surface lays fresh white water where the water
// breaks and churns foam where it churns, and dies as its eddies turn over;
// all of it rides the current. Every frame, a fixed number of dispatches:
// - the water is read as shading reads it (`water_pixel`), at the points of
//   a lattice standing at every other corner of the field's cells
//   (`foam_gather`);
// - at each lattice point, the current the water's surface runs at, how
//   fast that current spreads or converges, how hard the water churns foam,
//   by the cook's own rule, and how fast its eddies turn (`foam_currents`);
// - at each lattice point standing for a site of the bursts, the bursts
//   born there as the frame begins a slot of the site's (`foam_bursts`);
// - each cell takes what the surface's current brought it over the frame,
//   gathered where that current converges and against what stands in it;
//   its white water drains as its bubbles rise out and its foam fades as the
//   cook's does, the turbulence reaching it laying more; its disturbance
//   spreads as gravity waves do and dies by the water's friction and
//   viscosity; and the bursts born about it bring their turbulence and raise
//   their boils (`foam_step`);
// - and the textures shading reads (`meshes/water_foam.wgsl`'s
//   `water_foam_held`) take the cells' white water and foam, the
//   disturbance's slopes and their squares, and their coarser levels
//   (`foam_look`, `foam_down`).
// Between the patches lies open water. Nothing but the bursts makes any of
// it, nothing but the current moves it and nothing but physics loses it: no
// texture, noise or pattern stands fixed to the world or the screen. The
// cells the field's window newly covers, and what is carried in across its
// rim, start as the settled foam there on a still surface (`foam_still`).

// Cells along a side of the workgroups of the field's passes, and of its
// lattice's points.
const FOAM_GROUP: u32 = 8u;
// Foam as the cook churns it (`engine_terrain::shallow_water`): the seconds
// it takes to fade to 1/e; the meters of water under which a place holds
// none; the least rate (square meters a second cubed) water loses energy at,
// gravity times its surface's slope times its speed, for it to churn any;
// the vorticity (a second) past which eddy lines churn it, as much again
// past it holding half the most; the convergence (a second) and the power
// the water loses (watts a square meter of bed) at which each holds half
// the most; and water's density (kilograms a cubic meter).
const WATER_FOAM_LIFE: f32 = 20.0;
const WATER_FOAM_WET: f32 = 0.005;
const WATER_FOAM_LOSS: f32 = 0.02;
const WATER_FOAM_SWIRL: f32 = 1.0;
const WATER_FOAM_PILING: f32 = 0.5;
const WATER_FOAM_POWER: f32 = 500.0;
const WATER_DENSITY: f32 = 1000.0;
// Foam rides the water's surface (`water_sim.wgsl`'s
// `sim_surface_current`): turned outward in a bend, the tangent of its turn
// FOAM_BEND depths over the streamlines' radius of curvature, no more than
// FOAM_BEND_MOST; and dragged by the wind at FOAM_WIND_DRIFT of its speed.
const FOAM_BEND: f32 = 7.0;
const FOAM_BEND_MOST: f32 = 1.0;
const FOAM_WIND_DRIFT: f32 = 0.03;
// The meters a second white water's bubbles rise out of it once it stops
// breaking: about what bubbles a millimeter across rise at in natural
// water, whose surfactants slow them (Clift, Grace and Weber 1978), so white
// water drains in as many seconds as it is deep in tenths of a meter.
const FOAM_RISE: f32 = 0.1;
// Water's kinematic viscosity (square meters a second, at 15 °C), and von
// Kármán's constant, by which turbulent water's eddy viscosity is κ·u·h/6
// over its depth (Elder 1959), u the speed its eddies turn at, which near
// the surface is about the shear velocity (Nezu and Nakagawa 1993).
const FOAM_VISCOSITY: f32 = 1.1e-6;
const FOAM_KARMAN: f32 = 0.41;
// The fastest a wave runs, in cells a frame's step, that the field's
// explicit steps carry stably on its square grid (Courant, Friedrichs and
// Lewy: 1/√2 in two dimensions).
const FOAM_COURANT: f32 = 0.70710678;
// A burst's patch, its share at a share q of its radius from its middle
// (1 − q²)², holds over its disc π/12 of the square of its width: the mean
// of the turbulence reaching a surface its eddies renew, each bringing its
// patch, each patch dying as its eddy turns over.
const FOAM_PATCH: f32 = 0.26179939;

// The field's frame: the first cell of its window along x and z (world
// cells, `step.x` meters across, counted from the world's origin), then that
// of the window its state held the frame before (zw); its lattice's first
// point along x and z (world lattice points, two cells apart, counted from
// the world's origin), its points along a side and its cells along a side
// (w); the meters across a cell, the seconds this frame steps and the
// wind's velocity (meters a second along x and z); the half of `foam_state`
// and `foam_waves` holding the frame before's state (x), 1 where the field
// holds none yet (y), and the index of the simulation's step the frame
// begins in (z) and of the one it ends in (w); the least steps a slot of
// bursts spans, a power of two, and the levels of the bursts' sites
// (`foam_levels`), then nothing (`sites`); and the seconds of a simulation
// step and the meters across the largest eddies the bursts stand for, then
// nothing (`eddies`).
struct FoamFrame {
    window: vec4<i32>,
    lattice: vec4<i32>,
    step: vec4<f32>,
    flags: vec4<u32>,
    sites: vec4<u32>,
    eddies: vec4<f32>,
}

@group(2) @binding(32) var<uniform> foam_frame: FoamFrame;
// The lattice's points, row by row, twice over: first the water there as
// `foam_gather` reads it (its velocity along x and z, meters a second; its
// depth, meters, none where it holds no foam, and how strongly it breaks
// white, 0 to 1; how fast its surface falls along x and z, meters a meter;
// and the foam it carries, 0 to WATER_FOAM_MOST, and how turbulent it runs,
// 0 to 1), which `foam_bursts` overwrites, once `foam_currents` has read it,
// with the bursts born at the site the point stands for (`foam_site`); then
// what carries and churns the foam there as `foam_currents` finds it (the
// current its surface runs at along x and z, meters a second; how fast that
// current spreads, a second, negative where it converges, and how hard the
// water churns foam, `sim_churn`'s churn; its depth and how strongly it
// breaks white; and the foam it carries and the speed its eddies turn at,
// meters a second): half floats, a pair a word, but for a site's bursts.
@group(2) @binding(33) var<storage, read_write> foam_lattice: array<vec4<u32>>;
// The field's state twice over, the frame before's and this frame's
// (`flags.x`), each its cells row by row as their texels lie, a world
// cell's texel its remainder over the cells along a side: the white water
// covering it, in layers, each hiding as often as not what the others leave
// (none or more), and the foam the water carries there (0 to
// WATER_FOAM_MOST).
@group(2) @binding(34) var<storage, read_write> foam_state: array<vec2<f32>>;
// What else the field's state holds, laid out as `foam_state`: the height of
// the surface's disturbance over the water's simulated surface (meters) and
// how fast that rises (meters a second); and the turbulence reaching its
// surface (FOAM_PATCH on average where every eddy brings it, `foam_site`),
// then nothing: half floats, a pair a word.
@group(2) @binding(37) var<storage, read_write> foam_waves: array<vec2<u32>>;
// A level of the textures shading reads (`foam_target`,
// `foam_spread_target`, which `foam_look` writes the finest level of and
// `foam_down` each coarser one), and the level finer than it (`foam_source`,
// `foam_spread_source`), which `foam_down` averages: each texel the white
// water's layers, the foam as a share of WATER_FOAM_MOST and how fast the
// disturbance falls along x and z (`foam_target`); and the squares of those
// falls along x and along z and their product (`foam_spread_target`,
// `water_spread`), whose mean less that of the falls' is the covariance of
// the slopes a texel hides.
@group(2) @binding(35) var foam_source: texture_2d<f32>;
@group(2) @binding(36) var foam_target: texture_storage_2d<rgba16float, write>;
@group(2) @binding(38) var foam_spread_source: texture_2d<f32>;
@group(2) @binding(39) var foam_spread_target: texture_storage_2d<rgba16float, write>;

// The water at a lattice point as `foam_gather` reads it.
struct FoamWater {
    velocity: vec2<f32>,
    depth: f32,
    born: f32,
    slope: vec2<f32>,
    foam: f32,
    turbulence: f32,
}

// What carries and churns the foam at a point as `foam_currents` finds it.
struct FoamCarrier {
    current: vec2<f32>,
    spread: f32,
    churn: f32,
    depth: f32,
    born: f32,
    foam: f32,
    stirring: f32,
}

// A cell's state: its white water and the foam it carries, the turbulence
// reaching its surface and the height of its disturbance (`held`, x to w),
// and how fast that height rises (`rise`).
struct FoamCell {
    held: vec4<f32>,
    rise: f32,
}

// `value`'s remainder over `cells`, none or more.
fn foam_remainder(value: vec2<i32>, cells: i32) -> vec2<i32> {
    return (value % cells + cells) % cells;
}

// Where the lattice keeps point `point` (counted from its first, the
// nearest of its points for one past them): as `foam_gather` reads the
// water there (and `foam_bursts` writes the bursts of its site), or as
// `foam_currents` finds what carries its foam where `carrier`.
fn foam_lattice_at(point: vec2<i32>, carrier: bool) -> u32 {
    let points = foam_frame.lattice.z;
    let at = clamp(point, vec2<i32>(0), vec2<i32>(points - 1));
    let index = u32(at.y * points + at.x);
    return select(index, index + u32(points * points), carrier);
}

// The water at lattice point `point` as `foam_gather` read it.
fn foam_water(point: vec2<i32>) -> FoamWater {
    let words = foam_lattice[foam_lattice_at(point, false)];
    let deep = unpack2x16float(words.y);
    let carried = unpack2x16float(words.w);
    var water: FoamWater;
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
fn foam_beside(point: vec2<i32>, here: FoamWater) -> FoamWater {
    let water = foam_water(point);
    if water.depth <= WATER_FOAM_WET {
        return here;
    }
    return water;
}

// How fast the velocity of the water at lattice point `point`, standing as
// `here`, changes a meter along x and along z (the columns): between the
// points either side of it (`foam_beside`).
fn foam_change(point: vec2<i32>, here: FoamWater) -> mat2x2<f32> {
    let span = 4.0 * foam_frame.step.x;
    let west = foam_beside(point - vec2<i32>(1, 0), here).velocity;
    let east = foam_beside(point + vec2<i32>(1, 0), here).velocity;
    let south = foam_beside(point - vec2<i32>(0, 1), here).velocity;
    let north = foam_beside(point + vec2<i32>(0, 1), here).velocity;
    return mat2x2<f32>((east - west) / span, (north - south) / span);
}

// The current the surface of water standing as `here` runs at, its
// velocity changing by `change` a meter along x and z (`foam_change`): its
// velocity turned outward in a bend by δ, tan δ = FOAM_BEND·h/R, no more
// than FOAM_BEND_MOST, where 1/R = |u × (u·∇)u| / |u|³ is the streamlines'
// curvature and outward is away from the side u × (u·∇)u says they curve
// toward; dragged by FOAM_WIND_DRIFT of the frame's wind.
fn foam_surface(here: FoamWater, change: mat2x2<f32>) -> vec2<f32> {
    let u = here.velocity;
    let turning = change * u;
    let cubed = max(dot(u, u) * length(u), 1.0e-12);
    let tangent = clamp(FOAM_BEND * here.depth * (u.x * turning.y - u.y * turning.x) / cubed,
        -FOAM_BEND_MOST, FOAM_BEND_MOST);
    let turned = vec2<f32>(u.x + tangent * u.y, u.y - tangent * u.x)
        * inverseSqrt(1.0 + tangent * tangent);
    return turned + FOAM_WIND_DRIFT * foam_frame.step.zw;
}

// The current the surface of the water at lattice point `point` runs at
// (`foam_surface`); none where it holds no foam, so the current carrying
// foam stops at whatever its water meets.
fn foam_current(point: vec2<i32>) -> vec2<f32> {
    let here = foam_water(point);
    if here.depth <= WATER_FOAM_WET {
        return vec2<f32>(0.0);
    }
    return foam_surface(here, foam_change(point, here));
}

// What carries and churns the foam at lattice point `point` as
// `foam_currents` found it.
fn foam_carrier_point(point: vec2<i32>) -> FoamCarrier {
    let words = foam_lattice[foam_lattice_at(point, true)];
    let spreading = unpack2x16float(words.y);
    let deep = unpack2x16float(words.z);
    let carried = unpack2x16float(words.w);
    var carrier: FoamCarrier;
    carrier.current = unpack2x16float(words.x);
    carrier.spread = spreading.x;
    carrier.churn = spreading.y;
    carrier.depth = deep.x;
    carrier.born = deep.y;
    carrier.foam = carried.x;
    carrier.stirring = carried.y;
    return carrier;
}

// `first` blended `toward` `second`.
fn foam_blend(first: FoamCarrier, second: FoamCarrier, toward: f32) -> FoamCarrier {
    var carrier: FoamCarrier;
    carrier.current = mix(first.current, second.current, toward);
    carrier.spread = mix(first.spread, second.spread, toward);
    carrier.churn = mix(first.churn, second.churn, toward);
    carrier.depth = mix(first.depth, second.depth, toward);
    carrier.born = mix(first.born, second.born, toward);
    carrier.foam = mix(first.foam, second.foam, toward);
    carrier.stirring = mix(first.stirring, second.stirring, toward);
    return carrier;
}

// What carries and churns the foam at world point `world` (meters along x
// and z): its lattice's four points around it, blended bilinearly.
fn foam_carrier(world: vec2<f32>) -> FoamCarrier {
    let grid = world / (2.0 * foam_frame.step.x) - vec2<f32>(foam_frame.lattice.xy);
    let last = f32(foam_frame.lattice.z - 2);
    let corner = clamp(floor(grid), vec2<f32>(0.0), vec2<f32>(last));
    let within = clamp(grid - corner, vec2<f32>(0.0), vec2<f32>(1.0));
    let first = vec2<i32>(corner);
    let south = foam_blend(foam_carrier_point(first), foam_carrier_point(first + vec2<i32>(1, 0)),
        within.x);
    let north = foam_blend(foam_carrier_point(first + vec2<i32>(0, 1)),
        foam_carrier_point(first + vec2<i32>(1, 1)), within.x);
    return foam_blend(south, north, within.y);
}

// The meters across the eddies of water `depth` meters deep, which its
// bursts stand for (`water_eddy_size`): about its depth, at least a cell and
// no larger than the largest the bursts stand for.
fn foam_eddy(depth: f32) -> f32 {
    return water_eddy_size(depth, foam_frame.step.x, foam_frame.eddies.y);
}

// The seconds white water `depth` meters deep drains in once it stops
// breaking, as its bubbles rise out over its depth at FOAM_RISE.
fn foam_draining(depth: f32) -> f32 {
    return max(depth, WATER_BREAK_THINNEST) / FOAM_RISE;
}

// The speed (meters a second) the surface's disturbance runs at on the
// field's grid in water `depth` meters deep, as linear gravity waves run:
// waves as long as twice the eddies raising them (`foam_eddy`), of wave
// number k = π/size, at c² = (g/k)·tanh(k·h); no faster than the field's
// steps carry (FOAM_COURANT).
fn foam_wave_speed(depth: f32) -> f32 {
    let number = PI / foam_eddy(depth);
    let fastest = FOAM_COURANT * foam_frame.step.x / max(foam_frame.step.y, 1.0e-6);
    return min(sqrt(WATER_GRAVITY * tanh(number * depth) / number), fastest);
}

// The level of the bursts' sites holding eddies `size` meters across
// (`foam_eddy`): level L's sites are squares 2^L lattice points across,
// each standing at the point whose world lattice coordinates are whole
// numbers of 2^L, holding eddies up to twice as wide as they are. The
// finest holding them, so the patch of a burst born anywhere in its site
// reaches no farther than the sites beside it; the coarsest of the frame's
// levels (`sites.y`) at most, which holds the largest eddies the bursts
// stand for.
fn foam_level(size: f32) -> u32 {
    let wanted = ceil(log2(max(size / (4.0 * foam_frame.step.x), 1.0)));
    return min(u32(wanted), foam_frame.sites.y - 1u);
}

// The bursts born at the site lattice point `site` (world lattice points)
// stands for as the frame begins a slot of the site's (`water_burst_slot`),
// where the water there is as `here` holds it; none where the site's level
// (`foam_level`, of its eddies' size, `foam_eddy`) puts no site at the point.
// Its eddies, `size` across, turn over in size / u seconds (u their speed,
// `here.stirring`), each reaching the surface over a size² of it once in
// that time, as turbulence renews a surface (Danckwerts 1951), and each
// slot of the site lasts about as long. Each quarter of the site draws
// whether one or more is born in it over the slot (`water_burst_weight`),
// and where in the quarter (a sixteenth of it, two bits along x and along
// z); each brings the surface turbulence over its patch, as wide as its
// eddy, all of it at its middle (its weight), which dies as its eddies turn
// over, so its mean over the surface is FOAM_PATCH; and raises a boil, its
// surface rising at its middle so that, free, it would stand as high as its
// eddy's velocity head, u²/2g, lets it: u²/2g times the angular frequency
// c·k its waves run at (`foam_wave_speed`). The site's words: each quarter's
// five bits (whether one is born, then where), and its level plus one in
// the top four; the patches' radius (half the eddies' size, meters) and how
// fast a boil's middle rises (meters a second); and the turbulence a burst
// brings its middle (a float's bits); then nothing.
fn foam_site(site: vec2<i32>, here: FoamCarrier) -> vec4<u32> {
    let size = foam_eddy(here.depth);
    let level = foam_level(size);
    let stride = 1i << level;
    if any(foam_remainder(site, stride) != vec2<i32>(0)) {
        return vec4<u32>(0u);
    }
    let turnover = size / here.stirring;
    let seconds = foam_frame.eddies.x;
    let least = foam_frame.sites.x;
    let slot = water_burst_slot(site, level, foam_frame.flags.w, turnover, seconds, least);
    let begun = water_burst_slot(site, level, foam_frame.flags.z, turnover, seconds, least);
    if slot.x == begun.x {
        return vec4<u32>(0u);
    }
    let side = f32(stride) * 2.0 * foam_frame.step.x;
    let mean = 0.25 * side * side / (size * size) * water_burst_life(slot, seconds) / turnover;
    let places = water_burst_bits(site, level, slot);
    var fired = 0u;
    var weight = 0.0;
    for (var quarter = 0u; quarter < 4u; quarter++) {
        let drawn = water_burst_weight(mean, unit(places[quarter]));
        if drawn > 0.0 {
            weight = drawn;
            fired |= (1u | ((places[quarter] & 15u) << 1u)) << (5u * quarter);
        }
    }
    if fired == 0u {
        return vec4<u32>(0u);
    }
    let head = here.stirring * here.stirring / (2.0 * WATER_GRAVITY);
    let rise = head * foam_wave_speed(here.depth) * PI / size * weight;
    return vec4<u32>(fired | ((level + 1u) << 28u), pack2x16float(vec2<f32>(0.5 * size, rise)),
        bitcast<u32>(weight), 0u);
}

// What the bursts born this frame about world point `world` (meters along x
// and z) give it (`foam_site`'s): how fast their boils make the surface's
// disturbance rise there (meters a second), each at its middle and the
// other way about its rim, (1 − q²)²·(1 − 4q²) of its middle's at a share q
// of its radius out, which brings up no water over its disc, the water a
// boil brings up sinking about it; and the turbulence they bring its
// surface, each (1 − q²)² of its middle's. Every level's sites around the
// point take part, the three either way of the one it lies in.
struct FoamBursts {
    rise: f32,
    reaching: f32,
}

fn foam_bursts_at(world: vec2<f32>) -> FoamBursts {
    var bursts: FoamBursts;
    let spacing = 2.0 * foam_frame.step.x;
    let at = world / spacing;
    for (var level = 0u; level < foam_frame.sites.y; level++) {
        let stride = 1i << level;
        let side = f32(stride) * spacing;
        let home = vec2<i32>(floor(at / f32(stride)));
        for (var k = 0; k < 9; k++) {
            let site = (home + vec2<i32>(k % 3 - 1, k / 3 - 1)) * stride;
            let point = site - foam_frame.lattice.xy;
            if any(point < vec2<i32>(0)) || any(point >= vec2<i32>(foam_frame.lattice.z)) {
                continue;
            }
            let words = foam_lattice[foam_lattice_at(point, false)];
            if (words.x >> 28u) != level + 1u {
                continue;
            }
            let shape = unpack2x16float(words.y);
            let weight = bitcast<f32>(words.z);
            let origin = vec2<f32>(site) * spacing;
            for (var quarter = 0u; quarter < 4u; quarter++) {
                let bits = (words.x >> (5u * quarter)) & 31u;
                if (bits & 1u) == 0u {
                    continue;
                }
                let corner = vec2<f32>(f32(quarter & 1u), f32(quarter >> 1u));
                let place = vec2<f32>(f32((bits >> 1u) & 3u), f32((bits >> 3u) & 3u));
                let middle = origin + (corner + (place + 0.5) * 0.25) * (0.5 * side);
                let reach = length(world - middle) / shape.x;
                if reach >= 1.0 {
                    continue;
                }
                let rim = 1.0 - reach * reach;
                let bump = rim * rim;
                bursts.rise += shape.y * bump * (1.0 - 4.0 * reach * reach);
                bursts.reaching += weight * bump;
            }
        }
    }
    return bursts;
}

// Whether the field's state the frame before held world cell `cell`: it
// lies in the window it held.
fn foam_held(cell: vec2<i32>) -> bool {
    let first = foam_frame.window.zw;
    return all(cell >= first) && all(cell < first + vec2<i32>(foam_frame.lattice.w));
}

// The field's state where it holds none yet, the water there as `here`
// holds it: the settled foam, the water's own, which the field's mean is
// (the white water born there and the foam the water carries); the
// turbulence reaching its surface at its mean, FOAM_PATCH, where any eddy
// turns in it; and a still surface. None where the water holds no foam.
fn foam_still(here: FoamCarrier) -> FoamCell {
    var still: FoamCell;
    if here.depth > WATER_FOAM_WET {
        still.held = vec4<f32>(here.born, here.foam, select(0.0, FOAM_PATCH,
            here.stirring > 0.0), 0.0);
    }
    return still;
}

// Where `foam_state` and `foam_waves` hold texel `texel` of their copy
// `copy` (`flags.x`'s, or the other).
fn foam_cell_at(texel: vec2<i32>, copy: u32) -> u32 {
    let cells = foam_frame.lattice.w;
    return copy * u32(cells * cells) + u32(texel.y * cells + texel.x);
}

// The field's state the frame before at world cell `cell`, where it held
// it; else the field's state where it holds none (`foam_still`), as what is
// carried in across its rim comes.
fn foam_before(cell: vec2<i32>) -> FoamCell {
    if !foam_held(cell) {
        return foam_still(foam_carrier((vec2<f32>(cell) + 0.5) * foam_frame.step.x));
    }
    let at = foam_cell_at(foam_remainder(cell, foam_frame.lattice.w), foam_frame.flags.x);
    let waves = foam_waves[at];
    let disturbed = unpack2x16float(waves.x);
    var before: FoamCell;
    before.held = vec4<f32>(foam_state[at], unpack2x16float(waves.y).x, disturbed.x);
    before.rise = disturbed.y;
    return before;
}

// The weights Catmull and Rom's spline gives four cells a step apart along
// an axis at a point `t` (0 to 1) of the way from the second to the third.
fn foam_cubic(t: f32) -> vec4<f32> {
    return vec4<f32>(t * (-0.5 + t * (1.0 - 0.5 * t)), 1.0 + t * t * (-2.5 + 1.5 * t),
        t * (0.5 + t * (2.0 - 1.5 * t)), t * t * (-0.5 + 0.5 * t));
}

// The field's state the frame before at a world point (`foam_departed`), and
// the Laplacian of its disturbance's height there (a meter).
struct FoamDeparted {
    held: vec4<f32>,
    rise: f32,
    laplacian: f32,
}

// The field's state the frame before at world point `world` (meters along
// x and z, `foam_before`): Catmull and Rom's spline through the 4 x 4 cells
// around it, held within the least and the most of the four nearest, so it
// keeps lines a cell or two wide sharp as they travel and makes no new
// highs or lows; and the Laplacian of the disturbance's height at those
// four, each between its four neighbors, blended bilinearly.
fn foam_departed(world: vec2<f32>) -> FoamDeparted {
    let grid = world / foam_frame.step.x - 0.5;
    let corner = floor(grid);
    let t = grid - corner;
    let along = foam_cubic(t.x);
    let across = foam_cubic(t.y);
    let first = vec2<i32>(corner) - vec2<i32>(1);
    var heights: array<f32, 16>;
    var held = vec4<f32>(0.0);
    var rise = 0.0;
    var least = vec4<f32>(3.0e38);
    var most = vec4<f32>(-3.0e38);
    var slowest = 3.0e38;
    var fastest = -3.0e38;
    for (var row = 0; row < 4; row++) {
        for (var column = 0; column < 4; column++) {
            let value = foam_before(first + vec2<i32>(column, row));
            let weight = along[column] * across[row];
            heights[row * 4 + column] = value.held.w;
            held += value.held * weight;
            rise += value.rise * weight;
            if row == 1 || row == 2 {
                if column == 1 || column == 2 {
                    least = min(least, value.held);
                    most = max(most, value.held);
                    slowest = min(slowest, value.rise);
                    fastest = max(fastest, value.rise);
                }
            }
        }
    }
    let curving = vec4<f32>(
        heights[4] + heights[6] + heights[1] + heights[9] - 4.0 * heights[5],
        heights[5] + heights[7] + heights[2] + heights[10] - 4.0 * heights[6],
        heights[8] + heights[10] + heights[5] + heights[13] - 4.0 * heights[9],
        heights[9] + heights[11] + heights[6] + heights[14] - 4.0 * heights[10])
        / (foam_frame.step.x * foam_frame.step.x);
    var departed: FoamDeparted;
    departed.held = clamp(held, least, most);
    departed.rise = clamp(rise, slowest, fastest);
    departed.laplacian = mix(mix(curving.x, curving.y, t.x), mix(curving.z, curving.w, t.x), t.y);
    return departed;
}

// The surface's disturbance after a step of `seconds` where the water is as
// `here` holds it, carried there as `before` holds it, the boils born about
// it making it rise at `rise` meters a second (`foam_bursts_at`): its height
// and how fast that rises, by the damped wave equation on the field's grid,
// as linear gravity waves run (`foam_wave_speed`), its rise driven by its
// curvature times c²; lost to the water's friction and viscosity, dying by
// e at 2(ν + ν_t)·k² a second as a viscous fluid's gravity waves do (Lamb),
// ν water's viscosity and ν_t the turbulent water's eddy viscosity, κ·u·h/6
// (FOAM_KARMAN, u its eddies' speed, `here.stirring`, h their size). Its
// rise steps first, then its height by the rise, so its waves keep their
// energy but for what they lose.
fn foam_disturbed(before: FoamDeparted, here: FoamCarrier, rise: f32, seconds: f32)
        -> vec2<f32> {
    let size = foam_eddy(here.depth);
    let number = PI / size;
    let speed = foam_wave_speed(here.depth);
    let viscosity = FOAM_VISCOSITY + FOAM_KARMAN / 6.0 * here.stirring * size;
    let dying = 2.0 * viscosity * number * number;
    let rising = (before.rise + seconds * speed * speed * before.laplacian)
        * exp(-2.0 * dying * seconds) + rise;
    return vec2<f32>(before.held.w + seconds * rising, rising);
}

// Each lattice point's water, read as shading reads it at the point
// (`water_pixel`, at its finest level the lattice resolves, two of the
// field's cells), over the terrain the field runs over; none where it holds
// no foam: where the water ends or lies thinner than WATER_FOAM_WET.
@compute @workgroup_size(FOAM_GROUP, FOAM_GROUP)
fn foam_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let point = vec2<i32>(id.xy);
    if any(point >= vec2<i32>(foam_frame.lattice.z)) {
        return;
    }
    let spacing = 2.0 * foam_frame.step.x;
    let world = vec2<f32>(foam_frame.lattice.xy + point) * spacing;
    let at = water_pixel(world - terrain.origin.xz, spacing);
    var words = vec4<u32>(0u);
    if at.held >= WATER_HELD && at.depth > WATER_FOAM_WET {
        words = vec4<u32>(pack2x16float(at.words.flow),
            pack2x16float(vec2<f32>(at.depth, at.words.state.y)), pack2x16float(at.slope),
            pack2x16float(vec2<f32>(at.words.foam, at.words.state.x)));
    }
    foam_lattice[foam_lattice_at(point, false)] = words;
}

// What carries and churns the foam at each lattice point: the current its
// surface runs at (`foam_surface`) and how fast it spreads, between the
// currents at the points either side, where whatever the water meets, dry
// ground or a solid standing up through it, stands still (`foam_current`),
// so foam carried against it piles up there; how hard the water churns
// foam, by the cook's rule (`sim_churn`): the power it loses as it runs, ρ
// times its depth times its velocity against the fall of its energy head
// (gravity times its surface's slope less the rise of its velocity head
// along the flow), which takes in where it plunges or jumps, as a share of
// WATER_FOAM_POWER; the vorticity of its eddy lines past WATER_FOAM_SWIRL,
// as a share of it; and its convergence, as a share of WATER_FOAM_PILING;
// none where it loses energy slower than WATER_FOAM_LOSS; and how fast its
// eddies turn, as its shading's roughness takes them (`water_stirring`). Its
// depth, how strongly it breaks white and the foam it carries come along.
@compute @workgroup_size(FOAM_GROUP, FOAM_GROUP)
fn foam_currents(@builtin(global_invocation_id) id: vec3<u32>) {
    let point = vec2<i32>(id.xy);
    if any(point >= vec2<i32>(foam_frame.lattice.z)) {
        return;
    }
    let here = foam_water(point);
    var words = vec4<u32>(0u);
    if here.depth > WATER_FOAM_WET {
        let span = 4.0 * foam_frame.step.x;
        let west = foam_beside(point - vec2<i32>(1, 0), here);
        let east = foam_beside(point + vec2<i32>(1, 0), here);
        let south = foam_beside(point - vec2<i32>(0, 1), here);
        let north = foam_beside(point + vec2<i32>(0, 1), here);
        let change = mat2x2<f32>((east.velocity - west.velocity) / span,
            (north.velocity - south.velocity) / span);
        let u = here.velocity;
        let rising = 0.5 * vec2<f32>(dot(east.velocity, east.velocity)
            - dot(west.velocity, west.velocity), dot(north.velocity, north.velocity)
            - dot(south.velocity, south.velocity)) / span;
        let power = WATER_DENSITY * here.depth
            * (WATER_GRAVITY * dot(u, here.slope) - dot(u, rising));
        let swirl = abs(change[0].y - change[1].x);
        let converging = max(-(change[0].x + change[1].y), 0.0);
        var churn = max(power, 0.0) / WATER_FOAM_POWER
            + max(swirl - WATER_FOAM_SWIRL, 0.0) / WATER_FOAM_SWIRL
            + converging / WATER_FOAM_PILING;
        if WATER_GRAVITY * length(here.slope) * length(u) < WATER_FOAM_LOSS {
            churn = 0.0;
        }
        let spread = (foam_current(point + vec2<i32>(1, 0)).x
            - foam_current(point - vec2<i32>(1, 0)).x + foam_current(point + vec2<i32>(0, 1)).y
            - foam_current(point - vec2<i32>(0, 1)).y) / span;
        let speed = length(u);
        let stirring = water_stirring(speed, water_eddies(speed, length(here.slope), here.depth),
            here.turbulence);
        words = vec4<u32>(pack2x16float(foam_surface(here, change)),
            pack2x16float(vec2<f32>(spread, churn)),
            pack2x16float(vec2<f32>(here.depth, here.born)),
            pack2x16float(vec2<f32>(here.foam, stirring)));
    }
    foam_lattice[foam_lattice_at(point, true)] = words;
}

// The bursts born at each lattice point standing for a site of them as the
// frame begins a slot of its (`foam_site`), into the lattice's first half,
// which `foam_currents` has read: none where the water holds no foam or no
// eddy turns in it.
@compute @workgroup_size(FOAM_GROUP, FOAM_GROUP)
fn foam_bursts(@builtin(global_invocation_id) id: vec3<u32>) {
    let point = vec2<i32>(id.xy);
    if any(point >= vec2<i32>(foam_frame.lattice.z)) {
        return;
    }
    let here = foam_carrier_point(point);
    var words = vec4<u32>(0u);
    if here.depth > WATER_FOAM_WET && here.stirring > 0.0 {
        words = foam_site(foam_frame.lattice.xy + point, here);
    }
    foam_lattice[foam_lattice_at(point, false)] = words;
}

// Each cell of the field after the frame's step, into this frame's state. A
// cell where the water holds no foam holds none; one the field's window
// newly covers, or every cell where the field holds no state yet, the
// field's state where it holds none (`foam_still`). Every other takes what
// the surface's current carried to it over the step, semi-Lagrangian
// transport: the state at the point it left from, traced back along the
// current over the step by the midpoint rule (`foam_departed`), its white
// water, foam and turbulence in conservative form, taken by how much the
// surface around it shrank or grew over the step (exp(−∇·u Δt)), so they
// gather where the current converges and thin where it spreads, past the
// most foam heaps held at the most. The turbulence reaching its surface,
// `r`, lays white water at r·b/(FOAM_PATCH·τ) layers a second, b the white
// water the water's state holds born there, as its white water drains over
// τ as its bubbles rise out over its depth at FOAM_RISE (`foam_draining`);
// and churns foam into the room it leaves at r·c/(FOAM_PATCH·WATER_FOAM_LIFE)
// a second, c the cook's churn, as its foam fades over WATER_FOAM_LIFE (as
// the cook's, `sim_churn`), so on average, r being FOAM_PATCH, both are the
// cook's; and it dies as its eddies turn over, size / u, the bursts born
// about it bringing more (`foam_bursts_at`). Its disturbance spreads as
// gravity waves and dies, its boils raising it (`foam_disturbed`).
@compute @workgroup_size(FOAM_GROUP, FOAM_GROUP)
fn foam_step(@builtin(global_invocation_id) id: vec3<u32>) {
    let cells = foam_frame.lattice.w;
    let texel = vec2<i32>(id.xy);
    if any(texel >= vec2<i32>(cells)) {
        return;
    }
    let seconds = foam_frame.step.y;
    let first = foam_frame.window.xy;
    let cell = first + foam_remainder(texel - first, cells);
    let world = (vec2<f32>(cell) + 0.5) * foam_frame.step.x;
    let here = foam_carrier(world);
    var after: FoamCell;
    if foam_frame.flags.y != 0u || !foam_held(cell) {
        after = foam_still(here);
    } else if here.depth > WATER_FOAM_WET {
        let middle = foam_carrier(world - 0.5 * seconds * here.current).current;
        let before = foam_departed(world - seconds * middle);
        let gathered = exp(-seconds * here.spread);
        let reaching = before.held.z * gathered;
        let draining = foam_draining(here.depth);
        let laid = reaching * here.born / FOAM_PATCH;
        let white = laid + (before.held.x * gathered - laid) * exp(-seconds / draining);
        let carried = min(before.held.y * gathered, WATER_FOAM_MOST);
        let churning = reaching * here.churn / (FOAM_PATCH * WATER_FOAM_LIFE);
        let renewing = churning + 1.0 / WATER_FOAM_LIFE;
        let steady = WATER_FOAM_MOST * churning / renewing;
        let foam = steady + (carried - steady) * exp(-renewing * seconds);
        let bursts = foam_bursts_at(world);
        let turning = here.stirring / foam_eddy(here.depth);
        let disturbed = foam_disturbed(before, here, bursts.rise, seconds);
        after.held = vec4<f32>(white, foam, reaching * exp(-seconds * turning) + bursts.reaching,
            disturbed.x);
        after.rise = disturbed.y;
    }
    let at = foam_cell_at(texel, 1u - foam_frame.flags.x);
    foam_state[at] = after.held.xy;
    foam_waves[at] = vec2<u32>(pack2x16float(vec2<f32>(after.held.w, after.rise)),
        pack2x16float(vec2<f32>(after.held.z, 0.0)));
}

// The height of the surface's disturbance at world cell `cell` after this
// frame's step, where this frame's window holds it; `own` where it does not.
fn foam_height_after(cell: vec2<i32>, own: f32) -> f32 {
    let first = foam_frame.window.xy;
    let cells = foam_frame.lattice.w;
    if any(cell < first) || any(cell >= first + vec2<i32>(cells)) {
        return own;
    }
    let at = foam_cell_at(foam_remainder(cell, cells), 1u - foam_frame.flags.x);
    return unpack2x16float(foam_waves[at].x).x;
}

// Each texel of the finest level of the textures shading reads, from its
// cell's state after the frame's step: its white water, its foam as a share
// of WATER_FOAM_MOST and how fast its disturbance falls along x and z,
// between the cells either side (or toward the one inside the window, at its
// rim), no steeper than a wave stands before it breaks (WATER_STEEPEST),
// past which the water breaks; and the squares of those falls and their
// product (`water_spread`).
@compute @workgroup_size(FOAM_GROUP, FOAM_GROUP)
fn foam_look(@builtin(global_invocation_id) id: vec3<u32>) {
    let cells = foam_frame.lattice.w;
    let texel = vec2<i32>(id.xy);
    if any(texel >= vec2<i32>(cells)) {
        return;
    }
    let first = foam_frame.window.xy;
    let last = first + vec2<i32>(cells - 1);
    let cell = first + foam_remainder(texel - first, cells);
    let at = foam_cell_at(texel, 1u - foam_frame.flags.x);
    let state = foam_state[at];
    let own = unpack2x16float(foam_waves[at].x).x;
    let west = cell - vec2<i32>(1, 0);
    let east = cell + vec2<i32>(1, 0);
    let south = cell - vec2<i32>(0, 1);
    let north = cell + vec2<i32>(0, 1);
    let spans = foam_frame.step.x * vec2<f32>(f32(west.x >= first.x) + f32(east.x <= last.x),
        f32(south.y >= first.y) + f32(north.y <= last.y));
    var fall = -vec2<f32>(foam_height_after(east, own) - foam_height_after(west, own),
        foam_height_after(north, own) - foam_height_after(south, own))
        / max(spans, vec2<f32>(foam_frame.step.x));
    let steepness = length(fall);
    if steepness > WATER_STEEPEST {
        fall *= WATER_STEEPEST / steepness;
    }
    textureStore(foam_target, texel, vec4<f32>(state.x, state.y / WATER_FOAM_MOST, fall));
    textureStore(foam_spread_target, texel, vec4<f32>(water_spread(fall), 0.0));
}

// Each texel of a coarser level of the textures shading reads: the mean of
// the four of the finer level it covers. The field's cells along a side
// are a whole number of its coarsest level's texels, so every level wraps
// as the finest does.
@compute @workgroup_size(FOAM_GROUP, FOAM_GROUP)
fn foam_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(foam_target)) {
        return;
    }
    let texel = 2 * vec2<i32>(id.xy);
    let sum = textureLoad(foam_source, texel, 0) + textureLoad(foam_source,
        texel + vec2<i32>(1, 0), 0) + textureLoad(foam_source, texel + vec2<i32>(0, 1), 0)
        + textureLoad(foam_source, texel + vec2<i32>(1, 1), 0);
    textureStore(foam_target, vec2<i32>(id.xy), 0.25 * sum);
    let spread = textureLoad(foam_spread_source, texel, 0) + textureLoad(foam_spread_source,
        texel + vec2<i32>(1, 0), 0) + textureLoad(foam_spread_source, texel + vec2<i32>(0, 1), 0)
        + textureLoad(foam_spread_source, texel + vec2<i32>(1, 1), 0);
    textureStore(foam_spread_target, vec2<i32>(id.xy), 0.25 * spread);
}
