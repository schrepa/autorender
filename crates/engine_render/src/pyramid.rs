use engine_core::{GpuStruct, gpu_consts, gpu_struct};

gpu_struct! {
    pub struct PyramidExtent {
        pub size: [u32; 2],
        pub last: [u32; 2],
    }
}

gpu_consts! {
    pub const PYRAMID_GROUP: u32 = 8;
}

pub(crate) fn source() -> String {
    let mut source = String::from(include_str!("depth_pyramid.wgsl"));
    source.push_str(<PyramidExtent as GpuStruct>::WGSL);
    source.push_str(WGSL);
    source
}
