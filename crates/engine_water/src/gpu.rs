use bytemuck::{Pod, Zeroable};
use engine_core::{gpu_bits, gpu_consts, gpu_struct};
use engine_heightfield::tile::PATCHES_PER_TILE;

use crate::jobs::gpu::{WaterInfoWord, WaterStateWord};

gpu_consts! {
    pub const NO_WATER_BLOCK: u32 = 0xffffffff;
    pub const WATER_WAITING: u32 = 0xfffffffe;
    pub const WATER_GRADIENT_RANGE: f32 = 4.0;
    pub const WATER_BLOCKS_SIDE: u32 = 16;
    pub const WATER_TILE_PATCHES: u32 = 16;
    pub const WATER_TILE_WORDS: u32 = WATER_TILE_PATCHES * WATER_TILE_PATCHES;
    pub const WATER_LEAF_FIRST: u32 = 5;
    pub const DEBUG_SPAN_STEPS: f32 = 254.0;
    pub const WATER_RIPPLE_LEVELS: u32 = 16;
    pub const WATER_NOISE_WRAP: i32 = 256;
    pub const WATER_NOISE_REACH: f32 = 0.5;
    pub const WATER_STEEPEST: f32 = 0.44879895;
    pub const WATER_GUST_SALT: u32 = 0x51ed270b;
    pub const WATER_GUST_FINE_SALT: u32 = 0x2545f491;
    pub const WATER_PUSH_SALT: u32 = 0x9e3779b1;
    pub const WATER_PUSH_FINE_SALT: u32 = 0x7f4a7c15;
}

pub(crate) type RippleKept = [[f32; 4]; 8];

const _: () = assert!(
    WATER_TILE_PATCHES == PATCHES_PER_TILE,
    "WATER_TILE_PATCHES differs from PATCHES_PER_TILE"
);

const _: () = assert!(
    size_of::<RippleKept>() == 2 * WATER_RIPPLE_LEVELS as usize * size_of::<f32>(),
    "RippleKept does not hold WATER_RIPPLE_LEVELS levels of both ripple fields"
);

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(
    clippy::enum_variant_names,
    reason = "the variants name the debug views water_depth to water_roughness"
)]
pub(crate) enum Shown {
    WaterDepth = 0,
    WaterSpeed = 1,
    WaterFroude = 2,
    WaterTurbulence = 3,
    WaterWhite = 4,
    WaterReach = 5,
    WaterSpan = 6,
    WaterEddies = 7,
    WaterRoughness = 8,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WaterProbeDetail {
    #[bits(8)]
    pub span: u8,
    #[bits(8)]
    pub roughness: u8,
    pub falls: bool,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WaterProbeState {
    #[bits(8)]
    pub turbulence: u8,
    #[bits(8)]
    pub white: u8,
    #[bits(8, at = 24)]
    pub reach: u8,
}

gpu_struct! {
    pub(crate) struct WaterWavesLook {
        pub gusts: [[f32; 4]; 2],
        pub pushes: [[f32; 4]; 2],
        pub rides: [[f32; 4]; 2],
        pub anchor: [f32; 2],
        pub downwind: [f32; 2],
        pub still: [f32; 2],
        pub shown: [f32; 2],
        pub fan: [f32; 2],
        pub count: u32,
        pub chop: f32,
        pub breaking: f32,
        pub calmest: f32,
        pub mesh_error: f32,
        pub pad: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterWave {
        pub heading: [f32; 2],
        pub group: [f32; 2],
        pub number: f32,
        pub steep: f32,
        pub gusted: f32,
        pub wavelength: f32,
        pub phase: f32,
        pub speed: f32,
        pub left: f32,
        pub pad: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterWaveRow {
        pub wave: WaterWave,
        pub rests: [[f32; 4]; 10],
    }
}

gpu_struct! {
    pub(crate) struct WaterFlowLook {
        pub moving: [f32; 2],
        pub shown: [f32; 2],
        pub frame: f32,
        pub carry_slowest: f32,
        pub carry_step: f32,
        pub carry_log: f32,
        pub rung_band: f32,
        pub relief_steepest: f32,
        pub ripple_seconds: f32,
        pub boil_seconds: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterRippleLook {
        pub kept: [[f32; 4]; 8],
        pub reach: [f32; 4],
        pub boil_churning: [f32; 2],
        pub tile: [f32; 2],
        pub fine_tile: f32,
        pub stretch: f32,
        pub coarse: f32,
        pub skew: f32,
        pub least: f32,
        pub share: f32,
        pub fine_share: f32,
        pub boil_steep: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterReliefLook {
        pub bump_froude: [f32; 2],
        pub roughness: [f32; 2],
        pub bump_steep: f32,
        pub bump_depths: f32,
        pub bump_deep: f32,
        pub bump_shortest: f32,
        pub bump_crests: f32,
        pub bump_crest_gain: f32,
        pub bump_flicker: f32,
        pub boil_across: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterFoamLook {
        pub fades: [f32; 2],
        pub covers: [f32; 2],
        pub tearing: [f32; 2],
        pub froude: [f32; 2],
        pub breaking: [f32; 2],
        pub drop: [f32; 2],
        pub blocking: [f32; 2],
        pub crested: [f32; 2],
        pub around: [f32; 2],
        pub rough: [f32; 2],
        pub drift_amount: [f32; 2],
        pub drift_gathers: [f32; 2],
        pub tile: f32,
        pub steep: f32,
        pub soft: f32,
        pub halo: f32,
        pub body: f32,
        pub filmy: f32,
        pub optical: f32,
        pub hollow: f32,
        pub albedo: f32,
        pub sky: f32,
        pub mirrors: f32,
        pub open: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterShoreLook {
        pub racing: [f32; 2],
        pub lace_depth: [f32; 2],
        pub lace_speed: [f32; 2],
        pub fade: f32,
        pub edge: f32,
        pub edge_pixels: f32,
        pub wet: f32,
        pub soaked: f32,
        pub lace_amount: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterOpticsLook {
        pub phase_forward: f32,
        pub phase_mean: f32,
        pub bubbles: f32,
        pub bubble_albedo: f32,
        pub refraction_reach: f32,
        pub caustic_contrast: f32,
        pub caustic_most: f32,
        pub caustic_depth: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterFallLook {
        pub scale_shares: [f32; 4],
        pub wander_scale: [f32; 2],
        pub curtain_scale: [f32; 2],
        pub rope_scale: [f32; 2],
        pub strand_scale: [f32; 2],
        pub plunge_drowns: [f32; 2],
        pub pad: [f32; 2],
        pub wander_reach: f32,
        pub gathering: f32,
        pub glassy_gathering: f32,
        pub tear_thickness: f32,
        pub rock_roughness: f32,
        pub broken_stands: f32,
        pub tear_softness: f32,
        pub edge_thinning: f32,
        pub relief_steepest: f32,
        pub furrow_shade: f32,
        pub matte_roughness: f32,
        pub albedo: f32,
        pub thin_tint: f32,
        pub scattering: f32,
        pub forward: f32,
        pub faintest: f32,
    }
}

gpu_struct! {
    pub(crate) struct WaterUniform {
        pub color: [f32; 4],
        pub absorption: [f32; 4],
        pub sediment: [f32; 4],
        pub terrains: [u32; 4],
        pub reach: [f32; 4],
        pub waves: WaterWavesLook,
        pub flow: WaterFlowLook,
        pub ripples: WaterRippleLook,
        pub relief: WaterReliefLook,
        pub foam: WaterFoamLook,
        pub shore: WaterShoreLook,
        pub optics: WaterOpticsLook,
        pub falls: WaterFallLook,
    }
}

gpu_struct! {
    #[derive(bevy_ecs::resource::Resource)]
    pub(crate) struct WaterPiece {
        pub debug: [u32; 4],
        pub ranges: [f32; 4],
        pub breaking: [f32; 4],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct FallVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub sheet: [f32; 4],
    pub state: [f32; 2],
    pub sky: f32,
    pub terrain: u32,
    pub flight: [f32; 4],
}

pub(crate) const GENERATED: &[&str] = &[
    WGSL,
    crate::jobs::gpu::WGSL,
    <Shown as engine_core::BitValue>::WGSL,
    <WaterProbeDetail as engine_core::GpuBits>::WGSL,
    <WaterProbeState as engine_core::GpuBits>::WGSL,
    <WaterInfoWord as engine_core::GpuBits>::WGSL,
    <WaterStateWord as engine_core::GpuBits>::WGSL,
    <WaterWavesLook as engine_core::GpuStruct>::WGSL,
    <WaterWave as engine_core::GpuStruct>::WGSL,
    <WaterWaveRow as engine_core::GpuStruct>::WGSL,
    <WaterFlowLook as engine_core::GpuStruct>::WGSL,
    <WaterRippleLook as engine_core::GpuStruct>::WGSL,
    <WaterReliefLook as engine_core::GpuStruct>::WGSL,
    <WaterFoamLook as engine_core::GpuStruct>::WGSL,
    <WaterShoreLook as engine_core::GpuStruct>::WGSL,
    <WaterOpticsLook as engine_core::GpuStruct>::WGSL,
    <WaterFallLook as engine_core::GpuStruct>::WGSL,
    <WaterUniform as engine_core::GpuStruct>::WGSL,
];
