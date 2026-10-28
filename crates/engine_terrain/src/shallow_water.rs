use crate::Bed;
use crate::hydraulics::GRAVITY;

pub const THETA: f32 = 1.3;
pub const THIN: f32 = 0.01;
pub const DRY: f32 = 1.0e-4;
pub const WET_MOST: f32 = 0.5;
pub const FASTEST: f32 = 20.0;
pub const UNHELD: f32 = -1.0e30;
pub const DENSITY: f32 = 1000.0;
pub const FERGUSON_A1: f32 = 6.5;
pub const FERGUSON_A2: f32 = 2.5;
pub const SHALLOWEST: f32 = 0.1;
pub const DEEPEST: f32 = 30.0;
pub const STRICKLER: f32 = 20.4;
pub const GROUND_D84: f32 = 0.05;
pub const COURANT: f32 = 0.24;
pub const BOUNDING_SPEED: f32 = 27.0;
pub const LOCAL_SHARE: f32 = 0.9;
pub const SLOWEST: f32 = 1.0;
pub const FOAM_LIFE: f32 = 20.0;
pub const FOAM_WET: f32 = 0.005;
pub const FOAM_LOSS: f32 = 0.02;
pub const FOAM_POWER: f32 = 500.0;
pub const SOURCE_SPEED: f32 = 1.5;
pub const CELL_MEANS: usize = 3;

pub fn step_seconds(cell: f32) -> f32 {
    COURANT * cell / BOUNDING_SPEED
}

pub fn velocity(depth: f32, flow: f32) -> f32 {
    let h4 = depth * depth * depth * depth;
    let thin = THIN * THIN * THIN * THIN;
    let floor = if h4 > thin { h4 } else { thin };
    std::f32::consts::SQRT_2 * depth * flow / (h4 + floor).sqrt()
}

pub fn roughness(bed: Option<&Bed>) -> f32 {
    match bed {
        Some(bed) if bed.carved => -bed.d84,
        Some(bed) => bed.d84,
        None => GROUND_D84,
    }
}

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

const LIVE_ACROSS: f32 = 4.0;

pub fn live_split(settled: f32, finest: f32) -> u32 {
    let narrowest = crate::NARROWEST * finest;
    let split = engine_math::max((LIVE_ACROSS * settled / narrowest).ceil(), 1.0) as u32;
    split.next_power_of_two().min(crate::BLOCK_SIDE)
}
