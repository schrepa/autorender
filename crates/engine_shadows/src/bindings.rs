use engine_render::bindings::Sample::{Depth, Uint};
use engine_render::bindings::View::D2;
use engine_render::bindings::{Stages, Table, read, texture, uniform_of, write};
use engine_render::gpu_data::CameraUniform;

use crate::gpu_layouts::{ShadowCullUniform, ShadowUniform};

const C: Stages = Stages::COMPUTE;

pub(crate) const SHADOW_SELECT: Table = Table {
    label: "engine/shadow-select-layout",
    bindings: &[
        uniform_of("cull", size_of::<ShadowCullUniform>(), C),
        read("candidates", C),
        read("batches", C),
        read("poses", C),
        read("flags", C),
        write("stillness", C),
        write("ids", C),
        write("args", C),
        write("counters", C),
        texture("tile_slots", Uint, D2, C),
    ],
};

pub(crate) const SHADOW_MARKS: Table = Table {
    label: "engine/shadow-marks-layout",
    bindings: &[
        uniform_of("camera", size_of::<CameraUniform>(), C),
        uniform_of("shadow", size_of::<ShadowUniform>(), C),
        texture("shadow_depth", Depth, D2, C),
        write("shadow_marks", C),
    ],
};
