use engine_render::bindings::Sample::{Depth, Float, Unfilterable};
use engine_render::bindings::Sampling::Filtering;
use engine_render::bindings::View::D2;
use engine_render::bindings::{Stages, Table, sampler, texture, uniform, uniform_of};
use engine_render::gpu_layouts::scene::PostUniform;

use crate::gpu_layouts::ResolveUniform;

const F: Stages = Stages::FRAGMENT;

pub(crate) const TAA_RESOLVE: Table = Table {
    label: "engine/taa-resolve-layout",
    bindings: &[
        uniform_of("taa", size_of::<ResolveUniform>(), F),
        texture("scene", Unfilterable, D2, F),
        texture("motion", Unfilterable, D2, F),
        texture("history", Float, D2, F),
        sampler("history_sampler", Filtering, F),
        texture("occlusion", Float, D2, F),
        uniform_of("post", size_of::<PostUniform>(), F),
        texture("depth_texture", Depth, D2, F),
        texture("revealed", Unfilterable, D2, F),
        uniform("pieces", F),
    ],
};
