// Foam's field (`water_foam_field.rs`): the foam terrains' water makes,
// carries and loses as it runs, simulated live on a grid of its own,
// finer than the water's cells and anchored in the world around the detail
// eye: each cell holds the white water covering it and the foam the water
// carries there. Each frame, a fixed number of dispatches:
// - the water is read as shading reads it (`water_pixel`), at the points of
//   a lattice standing at every other corner of the field's cells
//   (`foam_gather`);
// - at each lattice point, the current the water's surface runs at, how
//   fast that current spreads or converges, and how hard the water churns
//   foam, by the cook's own rule (`foam_currents`);
// - each cell takes the foam the surface's current brought it over the
//   frame, gathered where that current converges and against what stands
//   in it, churned up and fading as the cook's foam is, and white where the
//   water breaks, draining as its bubbles rise out (`foam_step`);
// - and the texture shading reads (`meshes/water_foam.wgsl`'s
//   `water_foam_held`) takes the cells' state and its coarser levels
//   (`foam_down`).
// The cells the field's window newly covers, and foam carried in across its
// rim, start as the settled foam there.

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

// The field's frame: the first cell of its window along x and z (world
// cells, `step.x` meters across, counted from the world's origin), then that
// of the window its state held the frame before (zw); its lattice's first
// point along x and z (world lattice points, two cells apart, counted from
// the world's origin), its points along a side and its cells along a side
// (w); the meters across a cell, the seconds this frame steps and the
// wind's velocity (meters a second along x and z); and the half of
// `foam_state` holding the frame before's state (x), and 1 where the field
// holds none yet (y).
struct FoamFrame {
    window: vec4<i32>,
    lattice: vec4<i32>,
    step: vec4<f32>,
    flags: vec4<u32>,
}

@group(2) @binding(32) var<uniform> foam_frame: FoamFrame;
// The lattice's points, row by row, twice over: first the water there as
// `foam_gather` reads it (its velocity along x and z, meters a second; its
// depth, meters, none where it holds no foam, and how strongly it breaks
// white, 0 to 1; how fast its surface falls along x and z, meters a meter;
// and the foam it carries, 0 to WATER_FOAM_MOST); then what carries and
// churns the foam there as `foam_currents` finds it (the current its
// surface runs at along x and z, meters a second; how fast that current
// spreads, a second, negative where it converges, and how hard the water
// churns foam, `sim_churn`'s churn; its depth and how strongly it breaks
// white; and the foam it carries): half floats, a pair a word.
@group(2) @binding(33) var<storage, read_write> foam_lattice: array<vec4<u32>>;
// The field's state twice over, the frame before's and this frame's
// (`flags.x`), each its cells row by row as their texels lie, a world
// cell's texel its remainder over the cells along a side: the white water
// covering it (0 to 1) and the foam the water carries there (0 to
// WATER_FOAM_MOST).
@group(2) @binding(34) var<storage, read_write> foam_state: array<vec2<f32>>;
// A level of the texture shading reads (`foam_target`, which `foam_step`
// writes the finest level of and `foam_down` each coarser one), and the
// level finer than it (`foam_source`), which `foam_down` averages: each
// texel the white water's cover and the foam's as a share of
// WATER_FOAM_MOST.
@group(2) @binding(35) var foam_source: texture_2d<f32>;
@group(2) @binding(36) var foam_target: texture_storage_2d<rgba8unorm, write>;

// The water at a lattice point as `foam_gather` reads it.
struct FoamWater {
    velocity: vec2<f32>,
    depth: f32,
    born: f32,
    slope: vec2<f32>,
    foam: f32,
}

// What carries and churns the foam at a point as `foam_currents` finds it.
struct FoamCarrier {
    current: vec2<f32>,
    spread: f32,
    churn: f32,
    depth: f32,
    born: f32,
    foam: f32,
}

// `value`'s remainder over `cells`, none or more.
fn foam_remainder(value: vec2<i32>, cells: i32) -> vec2<i32> {
    return (value % cells + cells) % cells;
}

// Where the lattice keeps point `point` (counted from its first, the
// nearest of its points for one past them): as `foam_gather` reads the
// water there, or as `foam_currents` finds what carries its foam where
// `carrier`.
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
    var water: FoamWater;
    water.velocity = unpack2x16float(words.x);
    water.depth = deep.x;
    water.born = deep.y;
    water.slope = unpack2x16float(words.z);
    water.foam = unpack2x16float(words.w).x;
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
    var carrier: FoamCarrier;
    carrier.current = unpack2x16float(words.x);
    carrier.spread = spreading.x;
    carrier.churn = spreading.y;
    carrier.depth = deep.x;
    carrier.born = deep.y;
    carrier.foam = unpack2x16float(words.w).x;
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

// Whether the field's state the frame before held world cell `cell`: it
// lies in the window it held.
fn foam_held(cell: vec2<i32>) -> bool {
    let first = foam_frame.window.zw;
    return all(cell >= first) && all(cell < first + vec2<i32>(foam_frame.lattice.w));
}

// The settled foam at world cell `cell` (x: the white water born there, y:
// the foam the water carries), the water's own, which the field's mean is:
// none where it holds no foam.
fn foam_settled(cell: vec2<i32>) -> vec2<f32> {
    let water = foam_carrier((vec2<f32>(cell) + 0.5) * foam_frame.step.x);
    return select(vec2<f32>(0.0), vec2<f32>(water.born, water.foam),
        water.depth > WATER_FOAM_WET);
}

// The field's state the frame before at world cell `cell`, where it held
// it; else the settled foam there (`foam_settled`), as foam carried in
// across the field's rim comes.
fn foam_before(cell: vec2<i32>) -> vec2<f32> {
    if !foam_held(cell) {
        return foam_settled(cell);
    }
    let cells = foam_frame.lattice.w;
    let texel = foam_remainder(cell, cells);
    let before = foam_frame.flags.x * u32(cells * cells);
    return foam_state[before + u32(texel.y * cells + texel.x)];
}

// The weights Catmull and Rom's spline gives four cells a step apart along
// an axis at a point `t` (0 to 1) of the way from the second to the third.
fn foam_cubic(t: f32) -> vec4<f32> {
    return vec4<f32>(t * (-0.5 + t * (1.0 - 0.5 * t)), 1.0 + t * t * (-2.5 + 1.5 * t),
        t * (0.5 + t * (2.0 - 1.5 * t)), t * t * (-0.5 + 0.5 * t));
}

// The field's state the frame before at world point `world` (meters along
// x and z, `foam_before`): Catmull and Rom's spline through the 4 x 4 cells
// around it, held within the least and the most of the four nearest, so it
// keeps lines a cell or two wide sharp as they travel and makes no new
// highs or lows.
fn foam_departed(world: vec2<f32>) -> vec2<f32> {
    let grid = world / foam_frame.step.x - 0.5;
    let corner = floor(grid);
    let along = foam_cubic(grid.x - corner.x);
    let across = foam_cubic(grid.y - corner.y);
    let first = vec2<i32>(corner) - vec2<i32>(1);
    var sum = vec2<f32>(0.0);
    var least = vec2<f32>(WATER_FOAM_MOST);
    var most = vec2<f32>(0.0);
    for (var row = 0; row < 4; row++) {
        for (var column = 0; column < 4; column++) {
            let value = foam_before(first + vec2<i32>(column, row));
            sum += value * (along[column] * across[row]);
            if row == 1 || row == 2 {
                if column == 1 || column == 2 {
                    least = min(least, value);
                    most = max(most, value);
                }
            }
        }
    }
    return clamp(sum, least, most);
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
            pack2x16float(vec2<f32>(at.words.foam, 0.0)));
    }
    foam_lattice[foam_lattice_at(point, false)] = words;
}

// What carries and churns the foam at each lattice point: the current its
// surface runs at (`foam_surface`) and how fast it spreads, between the
// currents at the points either side, where whatever the water meets, dry
// ground or a solid standing up through it, stands still (`foam_current`),
// so foam carried against it piles up there; and how hard the water churns
// foam, by the cook's rule (`sim_churn`): the power it loses as it runs, ρ
// times its depth times its velocity against the fall of its energy head
// (gravity times its surface's slope less the rise of its velocity head
// along the flow), which takes in where it plunges or jumps, as a share of
// WATER_FOAM_POWER; the vorticity of its eddy lines past WATER_FOAM_SWIRL,
// as a share of it; and its convergence, as a share of WATER_FOAM_PILING;
// none where it loses energy slower than WATER_FOAM_LOSS. Its depth, how
// strongly it breaks white and the foam it carries come along.
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
        words = vec4<u32>(pack2x16float(foam_surface(here, change)),
            pack2x16float(vec2<f32>(spread, churn)),
            pack2x16float(vec2<f32>(here.depth, here.born)),
            pack2x16float(vec2<f32>(here.foam, 0.0)));
    }
    foam_lattice[foam_lattice_at(point, true)] = words;
}

// Each cell of the field after the frame's step, into this frame's state
// and the finest level of the texture shading reads. A cell where the water
// holds no foam holds none; one the field's window newly covers, or every
// cell where the field holds no state yet, the settled foam there, the
// water's own. Every other takes what the surface's current carried to it
// over the step, semi-Lagrangian transport in conservative form: the state
// at the point it left from, traced back along the current over the step
// by the midpoint rule (`foam_departed`), taken by how much the surface
// around it shrank or grew over the step (exp(−∇·u Δt)), so foam gathers
// where the current converges and thins where it spreads, past the most it
// heaps, its cover held at the most. Then its foam fades and the water
// churns more up, as the cook's does (`sim_churn`): churning at `c`, it
// nears c / (1 + c) of WATER_FOAM_MOST, 1 + c times as fast as foam fades
// over WATER_FOAM_LIFE; and the water is at least as white as it breaks
// (`engine_terrain`'s `breaking`), white water carried on draining as its
// bubbles rise out over its depth at FOAM_RISE.
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
    var held = vec2<f32>(0.0);
    if here.depth > WATER_FOAM_WET {
        if foam_frame.flags.y != 0u || !foam_held(cell) {
            held = vec2<f32>(here.born, here.foam);
        } else {
            let middle = foam_carrier(world - 0.5 * seconds * here.current).current;
            let carried = foam_departed(world - seconds * middle) * exp(-seconds * here.spread);
            let churn = here.churn;
            let steady = WATER_FOAM_MOST * churn / (1.0 + churn);
            let foam = steady + (min(carried.y, WATER_FOAM_MOST) - steady)
                * exp(-(1.0 + churn) * seconds / WATER_FOAM_LIFE);
            let draining = max(here.depth, WATER_BREAK_THINNEST) / FOAM_RISE;
            let white = max(here.born, min(carried.x, 1.0) * exp(-seconds / draining));
            held = vec2<f32>(white, foam);
        }
    }
    let written = (1u - foam_frame.flags.x) * u32(cells * cells);
    foam_state[written + u32(texel.y * cells + texel.x)] = held;
    textureStore(foam_target, texel, vec4<f32>(held.x, held.y / WATER_FOAM_MOST, 0.0, 1.0));
}

// Each texel of a coarser level of the texture shading reads: the mean of
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
}
