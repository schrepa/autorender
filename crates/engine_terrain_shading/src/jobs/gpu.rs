use engine_core::{gpu_bits, gpu_consts};

gpu_consts! {
    pub const BOUNCE_LOW: f32 = 1.5;
    pub const BOUNCE_HIGH: f32 = 24.0;
    pub const BOUNCE_NEAREST: f32 = 0.5;
    pub const BOUNCE_STEPS: f32 = 16.0;
    pub const TILE_STRIDE: u32 = 259;
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BounceBlockWord {
    pub detailed: bool,
    #[bits(31)]
    pub first: u32,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BounceBin {
    #[bits(8)]
    pub filled: u8,
    #[bits(8)]
    pub distance: u8,
}
