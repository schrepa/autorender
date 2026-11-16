use engine_heightfield::Bed;

pub use super::gpu::{COARSE_GRAVEL_D84, DENSITY};

pub fn roughness(bed: Option<&Bed>) -> f32 {
    match bed {
        Some(bed) if bed.carved => -bed.d84,
        Some(bed) => bed.d84,
        None => COARSE_GRAVEL_D84,
    }
}
