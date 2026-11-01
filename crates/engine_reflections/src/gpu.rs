use bevy_ecs::prelude::*;
use engine_core::{GpuStruct, gpu_consts, gpu_struct};

pub(crate) const GENERATED: &[&str] = &[
    WGSL,
    <TraceTerrain as GpuStruct>::WGSL,
    <PyramidExtent as GpuStruct>::WGSL,
];

gpu_struct! {
    #[derive(Resource)]
    pub(crate) struct ReflectionsPiece {
        pub surface: [f32; 4],
        pub trace: [f32; 4],
        pub applied: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct TraceTerrain {
        pub cutout: [f32; 4],
        pub maps: [u32; 4],
        pub reach: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct PyramidExtent {
        pub size: [u32; 2],
        pub last: [u32; 2],
    }
}

gpu_consts! {
    pub(crate) const NO_PLACED: u32 = u32::MAX;
    pub(crate) const PYRAMID_GROUP: u32 = 8;
}
