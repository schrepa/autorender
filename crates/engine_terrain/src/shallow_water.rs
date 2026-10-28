//! The shallow-water scheme's constants. The renderer runs the scheme on the
//! GPU (`water_sim.wgsl`, whose `SIM_` constants mirror these), around the
//! camera and to settle each tile's water for its record
//! ([`crate::SettleStart`]): Kurganov and Petrova's well-balanced,
//! positivity-preserving central-upwind scheme (2007), second order in space
//! (each cell's water reconstructed linearly, its slopes limited by a
//! generalized minmod and kept from dipping below the ground at its faces,
//! a partially wet cell's lying flat against its waterline as Bollermann et
//! al. reconstruct it, 2013) and in time (Heun's two stages), with Kurganov
//! and Lin's anti-diffusion (2007) keeping shear from smearing, friction
//! from each cell's bed ([`resistance`]), gravity along the bed's slope,
//! each cell's outflow kept to the water it holds within a step, and
//! velocities desingularized where the water thins out. Each cell holds its
//! water's surface, its flow along x and z (square meters a second) and the
//! level a lake holds it at, if one does: lakes stand still, taking the
//! water that runs into them and giving none. The ground stands at the
//! cells' corners, with each cell's bed's roughness ([`roughness`]).
//! Rivers bring their water at the cells their sources name. The water
//! carries foam, churned up where it breaks white as it runs and where it
//! plunges and jumps, gathered where its current converges, carried off by
//! the current and fading as its bubbles break.
//!
//! Steps run as the live simulation's do, every cell by the same seconds,
//! or, settling, each cell by as long as its own and its neighbors'
//! fastest waves allow ([`LOCAL_SHARE`]): water settling toward its steady
//! state gets there in far fewer steps, and the state it settles to is the
//! same, as each cell's water stops changing when what flows in and out of
//! it balances, whatever its steps. Settling sums each cell's water over
//! the steps it is wet, each weighted by its seconds, for the means and
//! spreads its record keeps ([`CELL_MEANS`]), and the cells wet most of the
//! time hold its settled water ([`WET_MOST`]).

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
/// Foam, the lace white water leaves on the water: churned up where the
/// water loses energy fast enough to break, and fading as its bubbles
/// break. A cell's churn is the share of what holds half the most foam
/// ([`FOAM_POWER`]) of the power its water loses as it runs, as far as it
/// breaks white (`tile_water::BREAK_ONSET` to `BREAK_FULL`), and that of the
/// water landing in it, plunging or jumping; none where its water loses
/// energy slower than [`FOAM_LOSS`]. Eddy lines and converging currents draw
/// no air in; they gather the foam the current carries. Churning at `c`, its
/// foam nears `c / (1 + c)` of [`FOAM_MOST`], `1 + c` times as fast as foam
/// fades, so it saturates smoothly.
///
/// The seconds foam takes to fade to 1/e: lace left below rapids floats
/// downstream for tens of seconds, 10 to 60, before its bubbles break.
pub const FOAM_LIFE: f32 = 20.0;
/// The most foam churning holds the water toward: white all over, and half
/// as thick again.
pub const FOAM_MOST: f32 = 1.5;
/// Meters of water under which a cell holds no foam: a film thinner than
/// the few millimeters across foam's bubbles are.
pub const FOAM_WET: f32 = 0.005;
/// The least rate water loses energy at for it to churn foam, in watts a
/// kilogram (square meters a second cubed): gravity times its surface's
/// slope times its speed, what water running 0.4 meters a second down a
/// riffle's half percent loses. Pools and gentle runs stay unbroken.
pub const FOAM_LOSS: f32 = 0.02;
/// The least meters further than it falls on past a cell water drops into
/// it to plunge (a step, not a slope's steady fall), and by more than its
/// own depth, so its jet falls free: falling that far it lands at the meter
/// a second (√(2·g·h)) past which plunging water draws air in (Ervine,
/// McKeogh and Elsawy 1980).
pub const FOAM_DROP: f32 = 0.05;
/// The Froude number at which water running into water slower than its
/// waves jumps, breaking into a roller; slower, its jump stands as undular
/// waves breaking no surface (Chow 1959).
pub const FOAM_JUMP: f32 = 1.7;
/// The power water loses (watts a square meter of bed) at which it holds
/// half the most foam: the power ρ·g·q·S of half a square meter a second
/// running down a tenth, a cascade's slope.
pub const FOAM_POWER: f32 = 500.0;
/// Meters a second the water rivers bring runs along them as it comes.
pub const SOURCE_SPEED: f32 = 1.5;
/// Entries of four floats a settling cell's means take, as settling reads
/// them back at its region's last look ([`crate::SettleStart::finish`]),
/// each over the time it was wet: its surface, flow along x and z and foam;
/// the energy it loses (watts a square meter), the variance of its depth,
/// the mean square of its vorticity and the variance of its velocity; and
/// the share of the time it was wet, the share of the time wet its water
/// left its bed, and the share of the time it ran at the fastest water runs
/// ([`FASTEST`]), then nothing.
pub const CELL_MEANS: usize = 3;

/// The seconds of a live step over cells `cell` meters across: the longest
/// in which the fastest water and waves cross [`COURANT`] of a cell.
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

/// The roughness the scheme reads of a cell over `bed` (`None` outside
/// channels): the bed's D84 in meters, where friction follows Ferguson's
/// law; the same negated where steps or knobs are carved into it, where
/// friction counts its grains alone; and [`GROUND_D84`] outside channels.
pub fn roughness(bed: Option<&Bed>) -> f32 {
    match bed {
        Some(bed) if bed.carved => -bed.d84,
        Some(bed) => bed.d84,
        None => GROUND_D84,
    }
}

/// How hard a bed of `roughness` ([`roughness`]'s) holds back water `depth`
/// meters deep running over flat ground, as the scheme takes it: k of
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
