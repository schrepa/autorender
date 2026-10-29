use engine_render::bindings::{
    Sample, Stages, Table, View, read, texture, uniform, uniform_of, write,
};

use crate::gpu_layouts::{FrameUniform, SnapUniform, SplatUniform};

use Sample::{Float, Uint, Unfilterable};
use View::{D2, D2Array};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VC: Stages = Stages::VERTEX_COMPUTE;
const VFC: Stages = Stages::VERTEX_FRAGMENT_COMPUTE;

pub(crate) const SCATTER_CULL: Table = Table {
    label: "engine/scatter-cull-layout",
    bindings: &[
        uniform_of("frame", size_of::<FrameUniform>(), C),
        write("records", C),
        read("groups", C),
        write("ids", C),
        write("draws", C),
        read("views", C),
        write("cards", C),
        write("card_args", C),
    ],
};

pub(crate) const SCATTER_SNAP: Table = Table {
    label: "engine/scatter-snap-layout",
    bindings: &[
        uniform_of("snap", size_of::<SnapUniform>(), C),
        write("records", C),
        write("poses", C),
        read("terrains", C),
        read("directory", C),
        texture("heights", Unfilterable, D2Array, C),
        write("grounds", C),
        texture("control", Uint, D2Array, C),
        uniform("palette", C),
        texture("macro_maps", Float, D2Array, C),
    ],
};

pub(crate) const OCCUPANCY: Table = Table {
    label: "engine/occupancy-layout",
    bindings: &[
        uniform_of("splat", size_of::<SplatUniform>(), VFC),
        read("records", VC),
        read("poses", VC),
        read("covers", VC),
        read("listed", V),
        write("listing", C),
        write("listed_draw", C),
        texture("finer_coverage", Unfilterable, D2, F),
        texture("finer_heights", Unfilterable, D2, F),
    ],
};
