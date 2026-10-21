//! The shallow-water scheme's constants. The renderer runs the scheme on the
//! GPU (`water_sim.wgsl`, whose `SIM_` constants mirror these), around the
//! camera and to settle each tile's water for its record
//! ([`crate::SettleStart`]): Kurganov and Petrova's well-balanced,
//! positivity-preserving central-upwind scheme (2007), second order in space
//! (each cell's water reconstructed linearly, its slopes limited by a
//! generalized minmod and kept from dipping below the ground at its faces)
//! and in time (Heun's two stages), with Manning's friction and velocities
//! desingularized where the water thins out. Each cell holds its water's
//! surface, its flow along x and z (square meters a second) and the level a
//! lake holds it at, if one does: lakes stand still, taking the water that
//! runs into them and giving none. The ground stands at the cells' corners.
//! Rivers bring their water at the cells their sources name. The water
//! carries foam, churned up where it runs faster than its waves and where
//! it slams into slower water, carried off by the current and fading as its
//! bubbles break.
//!
//! Steps run as the live simulation's do, every cell by the same seconds,
//! or, settling, each cell by as long as its own and its neighbors'
//! fastest waves allow ([`LOCAL_SHARE`]): water settling toward its steady
//! state gets there in far fewer steps, and the state it settles to is the
//! same, as each cell's water stops changing when what flows in and out of
//! it balances, whatever its steps.

/// The generalized minmod's theta.
pub const THETA: f32 = 1.3;
/// Meters of water below which velocities are desingularized.
pub const THIN: f32 = 0.01;
/// Meters of water under which a cell is dry.
pub const DRY: f32 = 1.0e-4;
/// Meters a second the fastest water runs.
pub const FASTEST: f32 = 20.0;
/// The level a cell no lake holds holds as its lake's.
pub const UNHELD: f32 = -1.0e30;
/// Manning's roughness of the ground the water runs over: a clean mountain
/// channel's.
pub const ROUGHNESS: f32 = 0.03;
/// The share of a cell the fastest water and waves may cross in a step:
/// Kurganov and Petrova keep the water's depth positive within a quarter,
/// in two dimensions.
pub const COURANT: f32 = 0.24;
/// Meters a second the fastest water and waves run as the live steps'
/// length counts them: the fastest water and the waves of water five meters
/// deep.
pub const BOUNDING_SPEED: f32 = 27.0;
/// Local steps take this share of the most their cells' waves allow, and
/// count no wave slower than [`SLOWEST`] meters a second.
pub const LOCAL_SHARE: f32 = 0.9;
pub const SLOWEST: f32 = 1.0;
/// Foam: the Froude number past which rapids churn it up and how fast (a
/// second, per unit past it); how fast water converging churns it (per
/// unit of convergence, a second); the seconds it takes to fade to 1/e; the
/// most it piles up to; and the meters of water under which a cell holds
/// none.
pub const FOAM_FROUDE: f32 = 2.0;
pub const FOAM_RAPIDS: f32 = 0.3;
pub const FOAM_PILING: f32 = 0.25;
pub const FOAM_LIFE: f32 = 6.0;
pub const FOAM_MOST: f32 = 1.5;
pub const FOAM_WET: f32 = 0.005;
/// Meters a second the water rivers bring runs along them as it comes.
pub const SOURCE_SPEED: f32 = 1.5;

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
