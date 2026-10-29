use engine_render::bindings::{
    Sample, Stages, Table, View, dynamic_uniform, read, texture, uniform_of, write,
};

use crate::gpu_layouts::{FrameUniform, GrowUniform, PALETTE_BYTES};

use Sample::{Float, Uint, Unfilterable};
use View::D2Array;

const C: Stages = Stages::COMPUTE;
const V: Stages = Stages::VERTEX;

pub(crate) const COVER_GROW: Table = Table {
    label: "engine/cover-grow-layout",
    bindings: &[
        texture("heights", Unfilterable, D2Array, C),
        texture("control", Uint, D2Array, C),
        read("directory", C),
        uniform_of("palette", PALETTE_BYTES, C),
        read("covers", C),
        dynamic_uniform("grow", size_of::<GrowUniform>(), C),
        write("blades", C),
        write("counts", C),
        write("args", C),
        texture("macro_maps", Float, D2Array, C),
        texture("normals", Uint, D2Array, C),
    ],
};

pub(crate) const COVER_DRAW: Table = Table {
    label: "engine/cover-layout",
    bindings: &[
        read("cover_blades", V),
        uniform_of("cover_frame", size_of::<FrameUniform>(), V),
    ],
};
