use engine_faces::heightfield::{PALETTE_BYTES, SubdivisionUniform};
use engine_render::bindings::Format::{R32Uint, Rgba16Float};
use engine_render::bindings::Sample::{Float, Uint, Unfilterable};
use engine_render::bindings::Sampling::Filtering;
use engine_render::bindings::View::{D2, D2Array};
use engine_render::bindings::{
    Kind, Stages, Table, bind, dynamic_uniform, read, read_of, sampler, texture, uniform, uniform_of,
    write,
};
use engine_render::gpu_layouts::scene::CameraUniform;

use crate::gpu_layouts::bounce::{BounceFrame, BounceTerrain};
use crate::gpu_layouts::constants::HORIZON_RECORD_SLOTS;

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;

pub(crate) const HORIZON_RECORDS_BYTES: usize = HORIZON_RECORD_SLOTS * size_of::<[f32; 4]>();

const SHADOW_ROUTE_BYTES: usize =
    engine_core::MAX_MOVING_TILES as usize * size_of::<[[f32; 4]; 5]>() + size_of::<[f32; 4]>();

pub(crate) const TERRAIN_RESOLVE: Table = Table {
    label: "engine/terrain-resolve-layout",
    bindings: &[texture("terrain_ids_texture", Uint, D2, F)],
};

pub(crate) const TERRAIN_SUBDIVISION_CAMERA: Table = Table {
    label: "engine/terrain-subdivision-camera-layout",
    bindings: &[uniform_of("camera", size_of::<CameraUniform>(), C)],
};

pub(crate) const TERRAIN_SUBDIVISION: Table = Table {
    label: "engine/terrain-subdivision-layout",
    bindings: &[
        dynamic_uniform("subdivision", size_of::<SubdivisionUniform>(), C),
        write("subdivision_tree", C),
        write("subdivision_nodes", C),
        write("subdivision_leaves", C),
        write("subdivision_args", C),
    ],
};

pub(crate) const TERRAIN_NORMALS: Table = Table {
    label: "engine/terrain-normals-layout",
    bindings: &[
        texture("heights", Unfilterable, D2Array, C),
        uniform_of("job", 16, C),
        bind("normals", Kind::Storage(R32Uint), C),
    ],
};

pub(crate) const TERRAIN_CONTROL: Table = Table {
    label: "engine/terrain-control-layout",
    bindings: &[
        texture("heights", Unfilterable, D2Array, C),
        uniform_of("job", 32, C),
        bind("data", Kind::Read { min_size: 4 }, C),
        bind("control", Kind::Storage(R32Uint), C),
        texture("rivers", Uint, D2Array, C),
    ],
};

pub(crate) const BOUNCE: Table = Table {
    label: "engine/terrain-bounce-layout",
    bindings: &[
        uniform("pieces", C),
        texture("terrain_horizons", Float, D2Array, C),
        uniform_of("terrain_horizon_records", HORIZON_RECORDS_BYTES, C),
        sampler("terrain_horizon_sampler", Filtering, C),
        read("cloud_shadows", C),
        uniform_of("bounce_frame", size_of::<BounceFrame>(), C),
        read_of("bounce_terrains", size_of::<BounceTerrain>(), C),
        texture("bounce_sky", Float, D2Array, C),
        texture("bounce_surface", Uint, D2Array, C),
        read_of("bounce_blocks", 4, C),
        read_of("bounce_records", 4, C),
        texture("bounce_light_in", Float, D2Array, C),
        bind("bounce_light_out", Kind::StorageArray(Rgba16Float), C),
        bind("bounce_probes", Kind::StorageArray(Rgba16Float), C),
        sampler("bounce_sampler", Filtering, C),
        uniform_of("bounce_palette", PALETTE_BYTES, C),
        texture("bounce_macro", Float, D2Array, C),
        write("bounce_scratch", C),
    ],
};

pub(crate) const TERRAIN_SHADOW_VIEW: Table = Table {
    label: "engine/terrain-shadow-view-layout",
    bindings: &[
        dynamic_uniform("view", 96, V),
        read_of("routes", SHADOW_ROUTE_BYTES, VF),
    ],
};
