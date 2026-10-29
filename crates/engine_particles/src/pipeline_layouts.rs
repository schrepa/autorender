use engine_render::bindings::{Sample, Stages, Table, View, read, texture, uniform};

pub const PARTICLES: Table = Table {
    label: "engine/particle-layout",
    bindings: &[
        uniform("particles", Stages::VERTEX_FRAGMENT),
        read("emitters", Stages::VERTEX_FRAGMENT),
        read("owners", Stages::VERTEX_FRAGMENT),
        texture("depth_texture", Sample::Depth, View::D2, Stages::FRAGMENT),
    ],
};

engine_render::pipeline_layouts! {
    PARTICLE_PIPELINES = [(frame without [refraction]), PARTICLES];
}
