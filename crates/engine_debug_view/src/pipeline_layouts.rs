use engine_render::bindings::{Sample, Stages, Table, View, texture, uniform_of};

use crate::gpu_layouts::DebugViewUniform;

pub const DEBUG_VIEW: Table = Table {
    label: "engine/debug-view-layout",
    bindings: &[
        uniform_of("debug_view", size_of::<DebugViewUniform>(), Stages::FRAGMENT),
        texture("debug_ids", Sample::Uint, View::D2, Stages::FRAGMENT),
        texture("debug_count", Sample::Unfilterable, View::D2, Stages::FRAGMENT),
        texture("debug_terrain", Sample::Uint, View::D2, Stages::FRAGMENT),
    ],
};

engine_render::pipeline_layouts! {
    DEBUG_VIEW_PIPELINES = [DEBUG_VIEW];
}
