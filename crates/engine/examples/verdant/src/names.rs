use engine_core::{DrawKind, TargetName};

pub const COLOR: TargetName = TargetName("verdant_color");
pub const DEPTH: TargetName = TargetName("verdant_depth");
pub const BLOOM_A: TargetName = TargetName("verdant_bloom_a");
pub const BLOOM_B: TargetName = TargetName("verdant_bloom_b");
pub const BLOOM_C: TargetName = TargetName("verdant_bloom_c");
pub const BLOOM_UP_B: TargetName = TargetName("verdant_bloom_up_b");
pub const BLOOM_UP_A: TargetName = TargetName("verdant_bloom_up_a");

pub const COLOR_PASS: &str = "verdant_color_pass";
pub const WORLD: DrawKind = DrawKind("verdant_world");

pub const NEAR: f32 = 0.12;
pub const FAR: f32 = 850.0;
pub const DEFAULT_FOV_DEGREES: f32 = 46.0;
