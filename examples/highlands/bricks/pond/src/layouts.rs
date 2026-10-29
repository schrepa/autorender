use bytemuck::{Pod, Zeroable};
use engine_render::bindings::{Sample, Stages, Table, View, read, texture, uniform_of};

engine_render::gpu_consts! {
    pub const POND_MOST: usize = 16 => ["POND_MOST" in "pond::surface"];
}

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct PondShape as ["PondShape" in "pond::surface"] {
        pub area: [f32; 4],
        pub surface: [f32; 4],
        pub terrain: [f32; 4],
        pub ground: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct PondUniform as ["PondUniform" in "pond::surface"] {
        pub count: [u32; 4],
        pub shapes: [PondShape; POND_MOST],
    }
}

pub const POND: Table = Table {
    label: "pond/pond-layout",
    bindings: &[
        uniform_of("ponds", size_of::<PondUniform>(), Stages::VERTEX_FRAGMENT),
        texture(
            "heights",
            Sample::Unfilterable,
            View::D2Array,
            Stages::FRAGMENT,
        ),
        read("directory", Stages::FRAGMENT),
    ],
};

engine_render::pipeline_layouts! {
    POND_PIPELINES = [(frame without [refraction]), POND];
}
