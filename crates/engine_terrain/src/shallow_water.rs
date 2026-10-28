//! The shallow-water scheme's constants. The renderer runs the scheme on the
//! GPU (`water_sim.wgsl`, whose `SIM_` constants mirror these), around the
//! camera and to settle each tile's water for its record
//! ([`crate::SettleStart`]), one model in both: Kurganov and Petrova's
//! well-balanced, positivity-preserving central-upwind scheme (2007), second
//! order in space (each cell's water reconstructed linearly, its slopes
//! limited by a generalized minmod and kept from dipping below the ground at
//! its faces, a partially wet cell's lying flat against its waterline as
//! Bollermann et al. reconstruct it, 2013) and in time (Heun's two stages),
//! with Kurganov and Lin's anti-diffusion (2007) keeping shear from smearing,
//! friction from each cell's bed ([`resistance`]), gravity along the bed's
//! slope, each cell's outflow kept to the water it holds within a step, and
//! velocities desingularized where the water thins out. Each cell holds its
//! water's surface, its flow along x and z (square meters a second) and the
//! level a lake holds it at, if one does: lakes stand still, taking the
//! water that runs into them and giving none. Both run on the live cells
//! ([`live_split`] of them along each side of a settled cell), over the same
//! bed: the ground at the cells' corners on level 0's triangles, raised by
//! the relief of its materials at the cells' spacing, with each cell's bed's
//! roughness ([`roughness`]); and turbulence's bursts stir both. Rivers
//! bring their water at the cells their sources name.
//!
//! Settling, each cell first steps by as long as its own and its
//! neighbors' fastest waves allow ([`LOCAL_SHARE`]): water settling toward
//! its steady state gets there in far fewer steps, as each cell's water
//! stops changing when what flows in and out of it balances, whatever its
//! steps. Then it steps as the live water does, every cell by the same
//! seconds ([`step_seconds`]) and stirred by the same bursts, and sums each
//! cell's water over the steps it is wet, each weighted by its seconds, for
//! the means and spreads its record keeps ([`CELL_MEANS`]); the cells wet
//! most of the time hold its settled water ([`WET_MOST`]).

use crate::Bed;
use crate::hydraulics::GRAVITY;

/// The generalized minmod's theta.
pub const THETA: f32 = 1.3;
/// Meters of water below which velocities are desingularized.
pub const THIN: f32 = 0.01;
/// Meters of water under which a cell is dry.
pub const DRY: f32 = 1.0e-4;
/// The share of the time a cell must be wet for its settled water to stand
/// in it: settled water stands where water stands most of the time, at the
/// median of its waterline, as deep as it stands there while wet, so edges
/// water wets now and then, sloshing or spilling over, stand dry.
pub const WET_MOST: f32 = 0.5;
/// Meters a second the fastest water runs: a safety net the cook counts
/// the cells of where water reaches it.
pub const FASTEST: f32 = 20.0;
/// The level a cell no lake holds holds as its lake's.
pub const UNHELD: f32 = -1.0e30;
/// Kilograms a cubic meter of water, which the energy water loses weighs.
pub const DENSITY: f32 = 1000.0;
/// Ferguson's variable-power resistance law (2007) for a bed whose grains
/// 84 in 100 are finer than D84: U/u* = a1·a2·(h/D84) / √(a1² + a2²·
/// (h/D84)^(5/3)), so Darcy and Weisbach's friction factor is
/// f = 8/a2²·(D84/h)² + 8/a1²·(D84/h)^(1/3); its a1 and a2.
pub const FERGUSON_A1: f32 = 6.5;
pub const FERGUSON_A2: f32 = 2.5;
/// The least and the most a water's depth over its bed's D84 counts as,
/// water shallower over coarser grains rubbing as water this shallow does
/// and deeper water over finer ones as water this deep does: the span of
/// relative depths steep streams run at, where Ferguson's law holds.
pub const SHALLOWEST: f32 = 0.1;
pub const DEEPEST: f32 = 30.0;
/// Manning's n of a bed's grains alone, where steps or knobs carved into
/// it take up the rest of its resistance, is its D84 (meters) to the power
/// 1/6 over this (Strickler's grain roughness).
pub const STRICKLER: f32 = 20.4;
/// The D84 (meters) of ground no channel runs over: hillsides' soil, grass
/// and stones under sheets of water.
pub const GROUND_D84: f32 = 0.05;
/// The share of a cell the fastest water and waves may cross in a step:
/// Kurganov and Petrova keep the water's depth positive within a quarter,
/// in two dimensions, which keeps the steps stable; the depth stays
/// positive whatever the reconstruction and the anti-diffusion, as each
/// cell's outflow is kept to the water it holds (Bollermann et al.'s
/// draining time step, 2013).
pub const COURANT: f32 = 0.24;
/// Meters a second the fastest water and waves run as the live steps'
/// length counts them: the fastest water and the waves of water five meters
/// deep.
pub const BOUNDING_SPEED: f32 = 27.0;
/// Local steps take this share of the most their cells' waves allow, and
/// count no wave slower than [`SLOWEST`] meters a second.
pub const LOCAL_SHARE: f32 = 0.9;
pub const SLOWEST: f32 = 1.0;
/// Foam's rule, which water's particles lay it by
/// (`meshes/water_particles.wgsl`, whose `WATER_FOAM_` constants mirror
/// these): churned up where the water loses energy fast enough to break,
/// and fading as its bubbles break. Water churns it as far as it breaks
/// white (`tile_water::BREAK_ONSET` to `BREAK_FULL`), by its share of
/// [`FOAM_POWER`] of the power it loses; none where it loses energy slower
/// than [`FOAM_LOSS`].
///
/// The seconds foam takes to fade to 1/e: lace left below rapids floats
/// downstream for tens of seconds, 10 to 60, before its bubbles break.
pub const FOAM_LIFE: f32 = 20.0;
/// Meters of water under which foam lies on none: a film thinner than the
/// few millimeters across foam's bubbles are.
pub const FOAM_WET: f32 = 0.005;
/// The least rate water loses energy at for it to churn foam, in watts a
/// kilogram (square meters a second cubed): gravity times its surface's
/// slope times its speed, what water running 0.4 meters a second down a
/// riffle's half percent loses. Pools and gentle runs stay unbroken.
pub const FOAM_LOSS: f32 = 0.02;
/// The power water loses (watts a square meter of bed) at which its churn
/// is one: the power ρ·g·q·S of half a square meter a second running down a
/// tenth, a cascade's slope.
pub const FOAM_POWER: f32 = 500.0;
/// Meters a second the water rivers bring runs along them as it comes.
pub const SOURCE_SPEED: f32 = 1.5;
/// Entries of four floats a settling cell's means take, as settling reads
/// them back at its region's last look ([`crate::SettleStart::finish`]),
/// each over the time it was wet: its surface and flow along x and z, then
/// nothing; the energy it loses (watts a square meter), the variance of its
/// depth, the mean square of its vorticity and the variance of its
/// velocity; and the share of the time it was wet, the share of the time
/// wet its water left its bed, and the share of the time it ran at the
/// fastest water runs ([`FASTEST`]), then nothing.
pub const CELL_MEANS: usize = 3;

/// The seconds of a step of the live water, and of settling water once it
/// stirs, over cells `cell` meters across: the longest in which the fastest
/// water and waves cross [`COURANT`] of a cell.
pub fn step_seconds(cell: f32) -> f32 {
    COURANT * cell / BOUNDING_SPEED
}

/// The velocity (meters a second) of `flow` square meters a second of water
/// `depth` meters deep, desingularized as the water thins out, as the
/// scheme takes it.
pub fn velocity(depth: f32, flow: f32) -> f32 {
    let h4 = depth * depth * depth * depth;
    let thin = THIN * THIN * THIN * THIN;
    let floor = if h4 > thin { h4 } else { thin };
    std::f32::consts::SQRT_2 * depth * flow / (h4 + floor).sqrt()
}

/// The word a bed's roughness travels as, over `bed` (`None` outside
/// channels): the bed's D84 in meters, where friction follows Ferguson's
/// law; the same negated where steps or knobs are carved into it; and
/// [`GROUND_D84`] outside channels. The live water's setup and settling
/// read it at the sample nearest each cell's middle and turn it into the
/// roughness the scheme reads by one rule (`meshes/water_setup.wgsl`'s
/// `setup_rough`): where the cell's bed holds relief as its shape, carved
/// or raised by its materials, friction counts only its grains finer than a
/// cell.
pub fn roughness(bed: Option<&Bed>) -> f32 {
    match bed {
        Some(bed) if bed.carved => -bed.d84,
        Some(bed) => bed.d84,
        None => GROUND_D84,
    }
}

/// How hard a bed of `roughness` (the scheme's: a D84, negated where
/// friction counts its grains alone) holds back water `depth` meters deep
/// running over flat ground, as the scheme takes it: k of
/// dq/dt = −k·|q|·q (per square meter), Darcy and Weisbach's f/(8h²) of
/// Ferguson's f, or g·n²/h^(7/3) of the grains' Manning's n, the depth over
/// the D84 counting within [`SHALLOWEST`] and [`DEEPEST`]. Water running
/// down sloping ground is held back φ² = 1 + |∇b|² times as hard.
pub fn resistance(depth: f32, roughness: f32) -> f32 {
    let grain = engine_math::max(roughness.abs(), depth / DEEPEST);
    let relative = engine_math::clamp(depth / grain, SHALLOWEST, DEEPEST);
    if roughness >= 0.0 {
        let eighth = 1.0 / (FERGUSON_A2 * FERGUSON_A2 * relative * relative)
            + 1.0 / (FERGUSON_A1 * FERGUSON_A1 * relative.cbrt());
        eighth / (depth * depth)
    } else {
        let n = (depth / relative).powf(1.0 / 6.0) / STRICKLER;
        GRAVITY * n * n / (depth * depth * depth.cbrt())
    }
}

/// Live cells across the narrowest water whose flow the scheme turns: four,
/// the fewest over which its cells' water, linear across each of them
/// (`water_sim.wgsl`'s reconstruction), carries a velocity rising from a
/// bank or a rock's side to the stream's thread and falling back, two cells
/// each side, so its flow bends around what stands in it and sheds eddies in
/// its lee.
const LIVE_ACROSS: f32 = 4.0;

/// The live cells along each side of a settled cell `settled` meters across
/// on a terrain whose finest level's samples stand `finest` apart:
/// LIVE_ACROSS of them across the narrowest water its streams fill, the
/// narrowest channel its bake cuts ([`crate::NARROWEST`] of those samples,
/// the fewest any feature of its ground spans), each settled cell split
/// along each side into the fewest that does, a power of two no more than a
/// block's side ([`crate::BLOCK_SIDE`]), so the live cells tile the settled
/// cells and each block holds whole settled cells; one where the settled
/// cells are as fine. The live water and settling both run on them: for the
/// cook's meter cells over a meter's samples, two, half-meter cells, four
/// across a two-meter channel and two or more across a boulder a meter wide.
pub fn live_split(settled: f32, finest: f32) -> u32 {
    let narrowest = crate::NARROWEST * finest;
    let split = engine_math::max((LIVE_ACROSS * settled / narrowest).ceil(), 1.0) as u32;
    split.next_power_of_two().min(crate::BLOCK_SIDE)
}
