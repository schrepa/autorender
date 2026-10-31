use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ClusterUniform as ["Clusters" in "lights::clusters"] {
        pub view_projection: [[f32; 4]; 4],
        pub depth: [f32; 4],
        pub projection: [f32; 4],
        pub grid: [f32; 4],
        pub counts: [u32; 4],
        pub lights: [u32; 4],
        pub offsets: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct LightInput as ["Input" in "lights::binning"] {
        pub position: [f32; 3],
        pub range: f32,
        pub color: [f32; 3],
        pub slot: u32,
        pub direction: [f32; 3],
        pub cos_inner: f32,
        pub cos_outer: f32,
        pub radius: f32,
        pub length: f32,
        pub region: u32,
    }
}

engine_render::gpu_consts! {
    pub(crate) const NO_SLOT: u32 = u32::MAX => ["NO_SLOT" in "lights::binning"];
    pub(crate) const GROUP_LIGHTS: u32 = 8 => ["LIGHTS" in "lights::binning"];
    pub(crate) const LIGHT_BYTES: u64 = 64 => ["size_of Light"];
    pub(crate) const SHAPED_LIGHTS: u32 = 1 => ["SHAPED_LIGHTS" in "lights::clusters"];
    pub(crate) const KEPT_REGIONS: u32 = 2 => ["KEPT_REGIONS" in "lights::clusters"];
    pub(crate) const LIGHT_INTERIOR: u32 = 1 => ["LIGHT_INTERIOR" in "lights::faces::lights"];
    pub(crate) const LIGHT_EXTERIOR: u32 = 2 => ["LIGHT_EXTERIOR" in "lights::faces::lights"];
}
