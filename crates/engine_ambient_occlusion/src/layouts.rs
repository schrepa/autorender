use engine_render::bindings::{Sample, Stages, Table, View, texture, uniform};

pub const AMBIENT: &str = "ambient";

const F: Stages = Stages::FRAGMENT;

pub const AMBIENT_OCCLUSION: Table = Table {
    label: "engine/ambient-occlusion-layout",
    bindings: &[
        uniform("post", F),
        uniform("canopy", F),
        texture("depth_texture", Sample::Depth, View::D2, F),
    ],
};

pub const OCCLUSION_FILTER: Table = Table {
    label: "engine/ambient-occlusion-filter-layout",
    bindings: &[
        uniform("post", F),
        texture("raw_occlusion", Sample::Unfilterable, View::D2, F),
    ],
};

engine_render::pipeline_layouts! {
    AMBIENT_OCCLUSION_PIPELINES = [AMBIENT_OCCLUSION, (ambient without [])];
    OCCLUSION_FILTER_PIPELINES = [OCCLUSION_FILTER];
}
