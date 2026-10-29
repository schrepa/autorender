use engine_faces::heightfield::{SubdivisionUniform, TerrainUniform, ViewUniform};
use engine_render::bindings::{
    Format, Kind, Sample, Sampling, Stages, Table, View, bind, dynamic_uniform, read, sampler,
    texture, uniform_of, write,
};

use crate::gpu_layouts::water::{
    ParticleFrame, SimFrame, SimSetup, SimSlots, SimTouches, WaterUniform,
};

use Format::Rgba16Float;
use Sample::{Depth, Float, Uint, Unfilterable};
use Sampling::Filtering;
use View::{D2, D2Array};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;
const VFC: Stages = Stages::VERTEX_FRAGMENT_COMPUTE;

pub(crate) const WATER: Table = Table {
    label: "engine/water-layout",
    bindings: &[
        texture("terrain_heights", Unfilterable, D2Array, V),
        dynamic_uniform("terrain_view", size_of::<ViewUniform>(), VF),
        dynamic_uniform("terrain", size_of::<TerrainUniform>(), VFC),
        read("terrain_directory", VFC),
        read("terrain_surface_leaves", V),
        texture("water_surface", Unfilterable, D2Array, VFC),
        texture("water_info", Uint, D2Array, VFC),
        uniform_of("water", size_of::<WaterUniform>(), VFC),
        read("water_live", VFC),
        dynamic_uniform("water_tree", size_of::<SubdivisionUniform>(), V),
    ],
};

pub(crate) const WATER_BEHIND: Table = Table {
    label: "engine/water-behind-layout",
    bindings: &[
        texture("water_depth", Depth, D2, F),
        texture("water_scene", Float, D2, F),
        texture("water_particles", Float, D2Array, F),
        sampler("water_particles_sampler", Filtering, F),
        uniform_of("particle_frame", size_of::<ParticleFrame>(), V),
        read("foam_pool", V),
        texture("water_foam_drawn", Float, D2, F),
    ],
};

pub(crate) const WATER_PARTICLES: Table = Table {
    label: "engine/water-particles-layout",
    bindings: &[
        uniform_of("particle_frame", size_of::<ParticleFrame>(), C),
        write("particle_lattice", C),
        write("particle_pool", C),
        write("particle_counts", C),
        write("particle_splat", C),
        texture("particle_source", Float, D2, C),
        bind("particle_target", Kind::Storage(Rgba16Float), C),
        texture("particle_spread_source", Float, D2, C),
        bind("particle_spread_target", Kind::Storage(Rgba16Float), C),
    ],
};

pub(crate) const WATER_SIM: Table = Table {
    label: "engine/water-sim-layout",
    bindings: &[
        uniform_of("frame", size_of::<SimFrame>(), C),
        read("sim_ground", C),
        write("sim_cells", C),
        write("sim_depths", C),
        write("sim_fluxes", C),
        read("sim_inputs", C),
        read("sim_blocks", C),
        write("sim_steps", C),
        uniform_of("sim_slots", size_of::<SimSlots>(), C),
        write("sim_averages", C),
        uniform_of("sim_touches", size_of::<SimTouches>(), C),
    ],
};

pub(crate) const WATER_SETUP: Table = Table {
    label: "engine/water-setup-layout",
    bindings: &[
        uniform_of("setup_frame", size_of::<SimSetup>(), C),
        read("setup_list", C),
        write("setup_ground", C),
        write("setup_cells", C),
        write("setup_looks", C),
        write("setup_edges", C),
        read("setup_live", C),
    ],
};
