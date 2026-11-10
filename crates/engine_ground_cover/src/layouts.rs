use engine_core::{GpuBits, GpuStruct, gpu_bits};

engine_core::gpu_struct! {
    pub(crate) struct Grow {
        pub(crate) origin: [f32; 4],
        pub(crate) eye: [f32; 4],
        pub(crate) grid: [i32; 4],
        pub(crate) counts: [u32; 4],
        pub(crate) scale: [f32; 4],
        pub(crate) tier: [f32; 4],
        pub(crate) foliage: [f32; 4],
        pub(crate) view: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverFrame {
        pub(crate) wind: [f32; 4],
        pub(crate) ground: [f32; 4],
        pub(crate) blades: [u32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverBlade {
        pub(crate) root: [f32; 3],
        pub(crate) size: u32,
        pub(crate) facing: u32,
        pub(crate) leaves: u32,
        pub(crate) root_color: u32,
        pub(crate) tip_color: u32,
    }
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CoverFacing {
    #[bits(16)]
    pub(crate) angle: u32,
    #[bits(8)]
    pub(crate) tilt_x: u32,
    #[bits(8)]
    pub(crate) tilt_y: u32,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CoverLeaves {
    #[bits(8)]
    pub(crate) steps: u32,
    #[bits(8)]
    pub(crate) lean_x: u32,
    #[bits(8)]
    pub(crate) lean_y: u32,
}

engine_core::gpu_struct! {
    pub(crate) struct CoverMaterial {
        pub(crate) shape: [f32; 4],
        pub(crate) color: [f32; 4],
        pub(crate) canopy: [f32; 4],
    }
}

engine_core::gpu_struct! {
    pub(crate) struct CoverTerrain {
        pub(crate) base: u32,
        pub(crate) densest: f32,
        pub(crate) width: f32,
        pub(crate) pad: u32,
    }
}

engine_core::gpu_consts! {
    pub(crate) const COVER_LEAVES: u32 = 5;
    pub(crate) const COVER_LUSH_RATIO: f32 = 2.0;
    pub(crate) const COVER_MATERIAL_ROWS: u32 = 3;
}

const _: () = assert!(
    size_of::<CoverMaterial>() == COVER_MATERIAL_ROWS as usize * size_of::<[u32; 4]>(),
    "material_fills_its_rows"
);

const _: () = assert!(size_of::<CoverTerrain>() == size_of::<[u32; 4]>(), "terrain_fills_a_row");

pub(crate) const GENERATED: &[&str] = &[
    Grow::WGSL,
    CoverFrame::WGSL,
    CoverBlade::WGSL,
    <CoverFacing as GpuBits>::WGSL,
    <CoverLeaves as GpuBits>::WGSL,
    CoverMaterial::WGSL,
    CoverTerrain::WGSL,
    WGSL,
];
