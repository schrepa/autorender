use engine_core::TargetName;

pub const HDR_COLOR: TargetName = TargetName("monolith_hdr");
pub const BLOOM_A: TargetName = TargetName("monolith_bloom_a");
pub const BLOOM_B: TargetName = TargetName("monolith_bloom_b");
pub const BLOOM_C: TargetName = TargetName("monolith_bloom_c");
pub const BLOOM_UP_B: TargetName = TargetName("monolith_bloom_up_b");
pub const BLOOM_UP_A: TargetName = TargetName("monolith_bloom_up_a");
pub const TONED_LDR: TargetName = TargetName("monolith_toned");

pub const NEAR_Z: f32 = 0.08;
pub const SUN_DIR: [f32; 3] = [0.22, 0.94, 0.20];
pub const DOUBLE_SIDED_MASK: u32 =
    (1 << 11) | (1 << 14) | (1 << 27) | (1 << 28) | (1 << 29) | (1 << 30) | (1 << 31);
