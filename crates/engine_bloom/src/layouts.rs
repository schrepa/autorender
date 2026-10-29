use engine_render::bindings::{
    Contribution, Sample, Sampling, Stages, Table, View, sampler, texture, uniform,
};

const F: Stages = Stages::FRAGMENT;

pub const BLOOM_FIRST: Table = Table {
    label: "engine/bloom-first-layout",
    bindings: &[
        uniform("post", F),
        texture("scene", Sample::Float, View::D2, F),
        texture("glow", Sample::Float, View::D2, F),
        sampler("linear", Sampling::Filtering, F),
    ],
};

pub const BLOOM_LEVEL: Table = Table {
    label: "engine/bloom-level-layout",
    bindings: &[
        uniform("post", F),
        texture("source", Sample::Float, View::D2, F),
        sampler("linear", Sampling::Filtering, F),
    ],
};

pub const COMPOSITE_BLOOM: Contribution = Contribution {
    group: "composite",
    set: "bloom",
    bindings: &[
        texture("bloom", Sample::Float, View::D2, F),
        sampler("bloom_sampler", Sampling::Filtering, F),
    ],
};

engine_render::pipeline_layouts! {
    BLOOM_FIRST_PIPELINES = [BLOOM_FIRST];
    BLOOM_LEVEL_PIPELINES = [BLOOM_LEVEL];
}
