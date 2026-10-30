use engine_render::bindings::{
    Contribution, Format, Kind, Sample, Sampling, Stages, Table, View, bind, read_of, sampler,
    texture, uniform, uniform_of, write,
};
use engine_render::gpu_layouts::scene::PostUniform;

use crate::gpu_layouts::{CLOUD_SHADOW_BYTES, CloudUniform, ViewUniform};

use Format::Rgba16Float;
use Sample::{Depth, Float, Unfilterable};
use Sampling::Filtering;
use View::{Cube, CubeArray, D2, D2Array, D3};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;

pub(crate) const CLOUD_MARCH: Table = Table {
    label: "engine/cloud-march-layout",
    bindings: &[
        uniform_of("clouds", size_of::<CloudUniform>(), C),
        texture("noise_shape", Float, D3, C),
        sampler("noise_sampler", Filtering, C),
        bind("dome", Kind::StorageArray(Rgba16Float), C),
        texture("noise_detail", Float, D3, C),
        write("shadow_cells", C),
        uniform("pieces", C),
        texture("cloud_transmittance", Float, D2, C),
        sampler("cloud_table_sampler", Filtering, C),
        write("light_cells", C),
    ],
};

pub(crate) const CLOUD_VIEW: Table = Table {
    label: "engine/cloud-view-layout",
    bindings: &[
        uniform_of("clouds", size_of::<CloudUniform>(), C),
        texture("noise_shape", Float, D3, C),
        sampler("noise_sampler", Filtering, C),
        texture("noise_detail", Float, D3, C),
        uniform_of("cloud_view", size_of::<ViewUniform>(), C),
        texture("view_depth", Depth, D2, C),
        bind("view_out", Kind::Storage(Rgba16Float), C),
        write("composite_draw", C),
        uniform("pieces", C),
        texture("cloud_transmittance", Float, D2, C),
        sampler("cloud_table_sampler", Filtering, C),
        write("light_cells", C),
    ],
};

pub(crate) const CLOUD_COMPOSITE: Table = Table {
    label: "engine/cloud-composite-layout",
    bindings: &[texture("view_clouds", Float, D2, F)],
};

pub(crate) const SKY_TRANSMITTANCE: Table = Table {
    label: "engine/sky-transmittance-layout",
    bindings: &[texture("sky_transmittance_table", Float, D2, F)],
};

pub(crate) const SKY_TABLES: Table = Table {
    label: "engine/sky-tables-layout",
    bindings: &[
        uniform("pieces", C),
        texture("transmittance_table", Float, D2, C),
        texture("multiple_table", Float, D2, C),
        texture("view_table", Float, D2, C),
        sampler("table_sampler", Filtering, C),
        bind("table_out", Kind::Storage(Rgba16Float), C),
        write("derived", C),
    ],
};

pub(crate) const HAZE: Table = Table {
    label: "engine/post-haze-layout",
    bindings: &[
        uniform_of("post", size_of::<PostUniform>(), F),
        texture("depth_texture", Depth, D2, F),
    ],
};

pub(crate) const HAZE_FILTER: Table = Table {
    label: "engine/haze-filter-layout",
    bindings: &[
        uniform_of("post", size_of::<PostUniform>(), F),
        texture("depth_texture", Depth, D2, F),
        texture("marched", Unfilterable, D2, F),
        texture("haze", Float, D2, F),
        uniform("pieces", F),
    ],
};

pub(crate) const ENVIRONMENT_SKY: Table = Table {
    label: "engine/environment-sky-layout",
    bindings: &[
        uniform("pieces", F),
        texture("sky_view_table", Float, D2, F),
        sampler("sky_table_sampler", Filtering, F),
        texture("sky_clouds", Float, D2Array, F),
        texture("sky_clouds_old", Float, D2Array, F),
        sampler("sky_cloud_sampler", Filtering, F),
    ],
};

pub(crate) const ENVIRONMENT_DOWN: Table = Table {
    label: "engine/environment-down-layout",
    bindings: &[
        texture("level_source", Float, Cube, F),
        sampler("cube_sampler", Filtering, F),
    ],
};

pub(crate) const ENVIRONMENT_PREFILTER: Table = Table {
    label: "engine/environment-prefilter-layout",
    bindings: &[
        texture("radiance", Float, Cube, F),
        sampler("cube_sampler", Filtering, F),
    ],
};

pub(crate) const FRAME_CLOUDS: Contribution = Contribution {
    group: "frame",
    set: "clouds",
    bindings: &[read_of("cloud_shadows", CLOUD_SHADOW_BYTES, F)],
};

pub(crate) const FRAME_SKY_TABLES: Contribution = Contribution {
    group: "frame",
    set: "sky_tables",
    bindings: &[
        texture("sky_view_table", Float, D2, F),
        sampler("sky_table_sampler", Filtering, F),
        texture("sky_clouds", Float, D2Array, F),
        texture("sky_clouds_old", Float, D2Array, F),
        sampler("sky_cloud_sampler", Filtering, F),
    ],
};

pub(crate) const FRAME_ENVIRONMENT: Contribution = Contribution {
    group: "frame",
    set: "environment",
    bindings: &[
        texture("environment_maps", Float, CubeArray, F),
        sampler("environment_sampler", Filtering, F),
    ],
};

pub(crate) const COMPOSITE_HAZE: Contribution = Contribution {
    group: "composite",
    set: "haze",
    bindings: &[texture("haze", Float, D2, F)],
};
