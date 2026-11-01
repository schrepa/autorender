engine_core::gpu_struct! {
    pub(crate) struct Grow {
        pub(crate) origin: [f32; 4],
        pub(crate) eye: [f32; 4],
        pub(crate) grid: [i32; 4],
        pub(crate) counts: [u32; 4],
        pub(crate) scale: [f32; 4],
        pub(crate) tier: [f32; 4],
        pub(crate) planes: [[f32; 4]; 6],
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
    pub(crate) const COVER_ENTRIES: u32 = 113;
}
