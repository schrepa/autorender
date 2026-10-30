use engine_core::PoseRows;
use engine_faces::heightfield::PALETTE_BYTES;
use engine_faces::placed::PlacedRecord;
use engine_render::bindings::Sample::{Depth, Float, Uint, Unfilterable};
use engine_render::bindings::Sampling::Filtering;
use engine_render::bindings::View::{D2, D2Array};
use engine_render::bindings::{Stages, Table, read, read_of, sampler, texture, uniform_of};
use engine_render::gpu_layouts::scene::PostUniform;

use crate::gpu_layouts::{PlacedGroup, TraceTerrain};

const F: Stages = Stages::FRAGMENT;

pub(crate) const REFLECTIONS: Table = Table {
    label: "engine/post-reflections-layout",
    bindings: &[
        uniform_of("post", size_of::<PostUniform>(), F),
        texture("depth_texture", Depth, D2, F),
        texture("glow", Float, D2, F),
        sampler("linear", Filtering, F),
        texture("nearest_depth", Unfilterable, D2, F),
        texture("surface_texture", Uint, D2, F),
        texture("scene_levels", Float, D2, F),
        texture("heights", Unfilterable, D2Array, F),
        read("directory", F),
        texture("control", Uint, D2Array, F),
        uniform_of("palette", PALETTE_BYTES, F),
        texture("macro_maps", Float, D2Array, F),
        texture("terrain_sky", Float, D2Array, F),
        read_of("world_terrains", size_of::<TraceTerrain>(), F),
        read("world_placed", F),
        read_of("placed_records", size_of::<PlacedRecord>(), F),
        read_of("placed_poses", size_of::<PoseRows>(), F),
        read_of("placed_groups", size_of::<PlacedGroup>(), F),
        texture("placed_frames", Float, D2Array, F),
    ],
};

pub(crate) const REFLECTION_LEVELS: Table = Table {
    label: "engine/post-reflection-levels-layout",
    bindings: &[
        uniform_of("post", size_of::<PostUniform>(), F),
        texture("source", Float, D2, F),
        sampler("linear", Filtering, F),
    ],
};

engine_render::pipeline_layouts! {
    REFLECTION_PIPELINES = [(frame without [occupancy, sky_tables, probes, refraction]), REFLECTIONS];
    REFLECTION_LEVEL_PIPELINES = [REFLECTION_LEVELS];
}
