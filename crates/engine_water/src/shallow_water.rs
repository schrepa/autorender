use engine_core::math::GRAVITY;
use engine_heightfield::Bed;

pub use crate::gpu::{
    COURANT, DEEPEST, DENSITY, DRY, FASTEST, FERGUSON_A1, FERGUSON_A2, FOAM_LIFE, FOAM_LOSS,
    FOAM_POWER, FOAM_WET, GROUND_D84, LOCAL_SHARE, SHALLOWEST, SLOWEST, SOURCE_SPEED, STRICKLER,
    THETA, THIN, UNHELD, WET_MOST,
};

pub const BOUNDING_SPEED: f32 = 27.0;
pub const CELL_MEANS: usize = crate::gpu::CELL_MEANS as usize;

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
