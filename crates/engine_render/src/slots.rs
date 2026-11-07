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

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardPhase {
    Steady = 0,
    Fading = 1,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SteadyCard {
    #[bits(31)]
    pub index: u32,
    #[bits(1)]
    pub phase: CardPhase,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FadingCard {
    #[bits(15)]
    pub index: u32,
    pub over: bool,
    #[bits(14)]
    pub fade: u32,
    pub out: bool,
    #[bits(1)]
    pub phase: CardPhase,
}

pub const DRAWN_SLOTS: usize = <DrawnSlot as GpuBits>::FIELDS[0].mask() as usize + 1;
pub const FADE_LEVEL_MAX: u32 = <FadeWord as GpuBits>::FIELDS[0].mask();
pub const FADING_CARD_INDEX_MAX: u32 = <FadingCard as GpuBits>::FIELDS[0].mask();
pub const FADING_CARD_FADE_MAX: u32 = <FadingCard as GpuBits>::FIELDS[2].mask();
