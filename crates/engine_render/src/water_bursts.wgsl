// Turbulence's bursts: what turbulence makes below a simulation's grid
// reaches what the grid resolves now and then, so each simulation drawing
// from it draws its events at random in space and in time, a Poisson
// process whose mean is the rate its own turbulence gives. One rule draws
// every one of them: each event from a hash of the world cell it starts in,
// on that simulation's own lattice of cells, and of that simulation's step
// index, integers alone, so every host draws the same events. The live
// solve's turbulent pushes (`water_sim.wgsl`'s `sim_burst`) draw their
// bursts by it, and foam's field (`meshes/water_foam_field.wgsl`) draws its
// own by it.

// A burst drawn for a cell at a step: whether one starts there then, where
// in the cell it starts (0 to 1 along x and z), and a draw of its own (0 to
// 1) for the burst to use, such as its heading.
struct WaterBurst {
    starts: bool,
    within: vec2<f32>,
    turn: f32,
}

// Four draws, each uniform over [0, 1), for world cell `cell` (on any
// lattice, counted from its origin) at step `tick`: Jarzynski and Olano's
// PCG4D hash (2020) of the cell, the step and a constant, each draw its
// output's top 24 bits, which a float holds exactly.
fn water_burst_draws(cell: vec2<i32>, tick: u32) -> vec4<f32> {
    var v = vec4<u32>(bitcast<vec2<u32>>(cell), tick, 0x9e3779b9u) * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> vec4<u32>(16u));
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return vec4<f32>(v >> vec4<u32>(8u)) * (1.0 / 16777216.0);
}

// The burst of world cell `cell` at step `tick`, where bursts start `mean`
// times a cell a step on average: one starts with the chance 1 − e^−mean
// that a Poisson process of that mean starts at least one, so at most one
// a cell a step, as a step far shorter than an eddy's life keeps the mean
// well under one.
fn water_burst(cell: vec2<i32>, tick: u32, mean: f32) -> WaterBurst {
    let draws = water_burst_draws(cell, tick);
    return WaterBurst(draws.x < 1.0 - exp(-max(mean, 0.0)), draws.yz, draws.w);
}
