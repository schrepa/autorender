use engine_core::TargetName;

pub const HDR_COLOR: TargetName = TargetName("monolith_hdr");
pub const TAA_HDR: TargetName = TargetName("monolith_taa_hdr");
pub const BLOOM_A: TargetName = TargetName("monolith_bloom_a");
pub const BLOOM_B: TargetName = TargetName("monolith_bloom_b");
pub const BLOOM_C: TargetName = TargetName("monolith_bloom_c");
pub const BLOOM_UP_B: TargetName = TargetName("monolith_bloom_up_b");
pub const BLOOM_UP_A: TargetName = TargetName("monolith_bloom_up_a");

pub const NEAR_Z: f32 = 0.08;
pub const SUN_DIR: [f32; 3] = [0.18, 0.93, 0.32];
pub const DOUBLE_SIDED_MATERIALS: &[u8] = &[11, 14, 27, 28, 29, 30, 31];
