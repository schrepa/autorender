use engine_render::bindings::Sample::{Depth, Float};
use engine_render::bindings::View::D2;
use engine_render::bindings::{Stages, Table, texture, uniform, write};

const C: Stages = Stages::COMPUTE;

pub(crate) const EXPOSURE: Table = Table {
    label: "engine/exposure-layout",
    bindings: &[
        uniform("settings", C),
        texture("scene_color", Float, D2, C),
        texture("scene_depth", Depth, D2, C),
        write("bins", C),
        write("state", C),
        uniform("pieces", C),
    ],
};
