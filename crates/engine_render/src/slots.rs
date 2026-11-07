use engine_core::{GpuBits, gpu_bits};

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawnSlot {
    #[bits(31)]
    pub slot: u32,
    pub moved: bool,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FadedSlot {
    #[bits(24)]
    pub slot: u32,
    #[bits(7)]
    pub level: u32,
    pub out: bool,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FadeWord {
    #[bits(7)]
    pub level: u32,
    pub out: bool,
    pub on: bool,
}

pub const DRAWN_SLOTS: usize = <DrawnSlot as GpuBits>::FIELDS[0].mask() as usize + 1;
pub const FADE_LEVEL_MAX: u32 = <FadeWord as GpuBits>::FIELDS[0].mask();
