use engine_core::{DrawKind, TargetName};

pub const HDR_COLOR: TargetName = TargetName("monolith_hdr");
pub const BLOOM_A: TargetName = TargetName("monolith_bloom_a");
pub const BLOOM_B: TargetName = TargetName("monolith_bloom_b");
pub const BLOOM_C: TargetName = TargetName("monolith_bloom_c");
pub const BLOOM_UP_B: TargetName = TargetName("monolith_bloom_up_b");
pub const BLOOM_UP_A: TargetName = TargetName("monolith_bloom_up_a");

pub const GROUND_KIND: DrawKind = DrawKind("monolith_ground");

pub const NEAR_Z: f32 = 0.1;
pub const SUN_DIR: [f32; 3] = [-0.42029, 0.24016, -0.87560];
