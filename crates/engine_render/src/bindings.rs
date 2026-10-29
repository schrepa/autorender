pub(crate) use crate::gpu_layouts::constants::PYRAMID_EXTENT_BYTES;
use crate::gpu_layouts::impostors::{ImpostorBakeUniform, ImpostorRecord};
use crate::gpu_layouts::scatter::Record as ScatterRecord;
use crate::gpu_layouts::scene::{
    CameraUniform, ColorData, GroundData, LightingUniform, MaterialUniform, PoseData, PostUniform,
    ResolveUniform, RouteData, ShadowCullUniform, ShadowUniform, SkyUniform,
};
use crate::gpu_layouts::world_trace::{PlacedGroup, TraceTerrain};
use crate::gpu_layouts::{
    bounce, clouds, debug_view, ground_cover, occupancy, scatter, sky, terrain, water,
};

mod groups;
mod table;

pub use groups::{
    BindingDecl, Contribution, GroupRef, ResolvedBinding, ResolvedLayout, SharedBinding,
    SharedGroups, StageCounts,
};
pub use table::{
    Binding, Format, Kind, Sample, Sampling, Stages, Table, View, bind, dynamic_uniform, read,
    read_of, sampler, texture, uniform, uniform_of, write,
};

use Format::{R32Float, R32Uint, Rgba16Float};
use Sample::{Depth, Float, Uint, Unfilterable};
use Sampling::{Comparison, Filtering, NonFiltering};
use View::{Cube, CubeArray, D2, D2Array, D3};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;
const VC: Stages = Stages::VERTEX_COMPUTE;
const FC: Stages = Stages::FRAGMENT_COMPUTE;
const VFC: Stages = Stages::VERTEX_FRAGMENT_COMPUTE;
const POSE: usize = size_of::<PoseData>();
const COLOR: usize = size_of::<ColorData>();
const GROUND: usize = size_of::<GroundData>();
pub(crate) const HORIZON_RECORDS_BYTES: usize =
    crate::gpu_layouts::constants::HORIZON_RECORDS * size_of::<[f32; 4]>();

macro_rules! tables {
    ($(
        $(#[$meta:meta])*
        const $name:ident: Table = Table {
            label: $label:literal,
            bindings: &[$($binding:expr),* $(,)?] $(,)?
        };
    )*) => {
        $(
            $(#[$meta])*
            pub(crate) const $name: Table = Table {
                label: $label,
                bindings: &[$($binding),*],
            };
        )*

        pub const TABLES: &[(&str, &Table)] = &[$((stringify!($name), &$name)),*];
    };
}

tables! {
    const EMPTY: Table = Table {
        label: "engine/empty-layout",
        bindings: &[],
    };

    const MATERIAL: Table = Table {
        label: "engine/material-layout",
        bindings: &[
            texture("base_color_texture", Float, D2, F),
            sampler("base_color_sampler", Filtering, F),
            texture("metallic_roughness_texture", Float, D2, F),
            sampler("metallic_roughness_sampler", Filtering, F),
            texture("normal_texture", Float, D2, F),
            sampler("normal_sampler", Filtering, F),
            texture("occlusion_texture", Float, D2, F),
            sampler("occlusion_sampler", Filtering, F),
            texture("emissive_texture", Float, D2, F),
            sampler("emissive_sampler", Filtering, F),
            uniform_of("material_maps", size_of::<MaterialUniform>(), F),
        ],
    };

    const RETAINED_POSES: Table = Table {
        label: "engine/retained-pose-layout",
        bindings: &[
            read_of("poses", POSE, VF),
            read_of("colors", COLOR, V),
            read_of("grounds", GROUND, F),
        ],
    };

    const TEMPORAL_POSES: Table = Table {
        label: "engine/temporal-pose-layout",
        bindings: &[
            read_of("poses", POSE, VF),
            read_of("colors", COLOR, V),
            read_of("previous_poses", POSE, V),
            read_of("grounds", GROUND, F),
        ],
    };

    const IMPOSTOR_DRAW: Table = Table {
        label: "engine/impostor-layout",
        serves: [SHADER @ 1, SHADOW_SHADER @ 2],
        bindings: &[
            read_of(11, "impostor_records", size_of::<ImpostorRecord>(), VF),
            texture(12, "impostor_albedo", Float, D2Array, F),
            texture(13, "impostor_emissive", Float, D2Array, F),
            texture(14, "impostor_geometry", Float, D2Array, F),
            texture(15, "impostor_surface", Float, D2Array, F),
            sampler(16, "impostor_sampler", Filtering, F),
        ],
    };

    const IMPOSTOR_BAKE: Table = Table {
        label: "engine/impostor-bake-layout",
        serves: [SHADER @ 2],
        bindings: &[
            uniform_of(15, "impostor_bake", size_of::<ImpostorBakeUniform>(), VF),
            texture(16, "impostor_shade", Depth, D2, F),
        ],
    };

    const PRESENTATION_COMPOSE: Table = Table {
        label: "engine/presentation-compose-layout",
        bindings: &[
            uniform("frame", C),
            read("locals", C),
            read("info", C),
            read("authority", C),
            write("composed", C),
            read("parents", C),
        ],
    };

    const PRESENTATION_LEVEL: Table = Table {
        label: "engine/presentation-level-layout",
        bindings: &[uniform("level", C)],
    };

    const PRESENTATION_CULL: Table = Table {
        label: "engine/presentation-cull-layout",
        bindings: &[
            uniform("frame", C),
            read("locals", C),
            read("info", C),
            read("authority", C),
            write("composed", C),
            read("batches", C),
            write("poses", C),
            write("scan", C),
            read("slot_candidates", C),
        ],
    };

    const PRESENTATION_TESTS: Table = Table {
        label: "engine/presentation-test-layout",
        bindings: &[
            uniform("frame", C),
            read("info", C),
            read("batches", C),
            write("poses", C),
            write("scan", C),
            write("locals_shift", C),
            read("slot_candidates", C),
            texture("depth_pyramid", Unfilterable, D2, C),
            write("occlusion_counts", C),
            write("still_frames", C),
        ],
    };

    const PRESENTATION_COMPACT: Table = Table {
        label: "engine/presentation-compact-layout",
        bindings: &[
            uniform("frame", C),
            read("candidates", C),
            read("batches", C),
            write("scan", C),
            write("batch_bases", C),
            write("visible_ids", C),
            write("draws", C),
            write("occlusion_counts", C),
            write("impostor_ids", C),
        ],
    };

    const PRESENTATION_FADES: Table = Table {
        label: "engine/presentation-fades-layout",
        bindings: &[
            uniform("frame", C),
            read("batches", C),
            write("poses", C),
            write("scan", C),
            read("slot_candidates", C),
            write("lod_states", C),
            write("fade_cards", C),
        ],
    };

    const SHADOW_SELECT: Table = Table {
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

    const SHADOW_MARKS: Table = Table {
        label: "engine/shadow-marks-layout",
        bindings: &[
            uniform_of("camera", size_of::<CameraUniform>(), C),
            uniform_of("shadow", size_of::<ShadowUniform>(), C),
            texture("shadow_depth", Depth, D2, C),
            write("shadow_marks", C),
        ],
    };

    const TERRAIN: Table = Table {
        label: "engine/terrain-layout",
        bindings: &[
            texture("terrain_heights", Unfilterable, D2Array, VFC),
            dynamic_uniform("terrain_view", size_of::<terrain::ViewUniform>(), VF),
            dynamic_uniform("terrain", size_of::<terrain::TerrainUniform>(), VFC),
            read("terrain_holes", VFC),
            texture("terrain_normals", Uint, D2Array, VFC),
            sampler("terrain_sky_sampler", Filtering, F),
            read("terrain_directory", VFC),
            texture("terrain_control", Uint, D2Array, VFC),
            uniform_of("terrain_palette", terrain::PALETTE_BYTES, VFC),
            texture("terrain_albedo", Float, D2Array, F),
            texture("terrain_material_data", Float, D2Array, VFC),
            sampler("terrain_material_sampler", Filtering, VFC),
            texture("terrain_sky", Float, D2Array, F),
            sampler("terrain_control_sampler", NonFiltering, VFC),
            texture("terrain_macro", Float, D2Array, F),
            read("terrain_surface_leaves", V),
            texture("water_surface", Unfilterable, D2Array, FC),
            texture("water_info", Uint, D2Array, C),
            texture("terrain_errors", Unfilterable, D2Array, C),
            texture("water_errors", Unfilterable, D2Array, C),
        ],
    };

    const TERRAIN_RESOLVE: Table = Table {
        label: "engine/terrain-resolve-layout",
        bindings: &[texture("terrain_ids_texture", Uint, D2, F)],
    };

    const TERRAIN_SUBDIVISION_CAMERA: Table = Table {
        label: "engine/terrain-subdivision-camera-layout",
        bindings: &[uniform_of("camera", size_of::<CameraUniform>(), C)],
    };

    const TERRAIN_SUBDIVISION: Table = Table {
        label: "engine/terrain-subdivision-layout",
        bindings: &[
            dynamic_uniform("subdivision", size_of::<terrain::SubdivisionUniform>(), C),
            write("subdivision_tree", C),
            write("subdivision_nodes", C),
            write("subdivision_leaves", C),
            write("subdivision_args", C),
        ],
    };

    const CLOUD_MARCH: Table = Table {
        label: "engine/cloud-march-layout",
        bindings: &[
            uniform_of("clouds", size_of::<clouds::CloudUniform>(), C),
            texture("noise_shape", Float, D3, C),
            sampler("noise_sampler", Filtering, C),
            bind("dome", Kind::StorageArray(Rgba16Float), C),
            texture("noise_detail", Float, D3, C),
            write("shadow_cells", C),
            uniform_of("cloud_air", size_of::<sky::SkyUniform>(), C),
            texture("cloud_transmittance", Float, D2, C),
            sampler("cloud_table_sampler", Filtering, C),
            write("light_cells", C),
        ],
    };

    const CLOUD_VIEW: Table = Table {
        label: "engine/cloud-view-layout",
        bindings: &[
            uniform_of("clouds", size_of::<clouds::CloudUniform>(), C),
            texture("noise_shape", Float, D3, C),
            sampler("noise_sampler", Filtering, C),
            texture("noise_detail", Float, D3, C),
            uniform_of("cloud_view", size_of::<clouds::ViewUniform>(), C),
            texture("view_depth", Depth, D2, C),
            bind("view_out", Kind::Storage(Rgba16Float), C),
            write("composite_draw", C),
            uniform_of("cloud_air", size_of::<sky::SkyUniform>(), C),
            texture("cloud_transmittance", Float, D2, C),
            sampler("cloud_table_sampler", Filtering, C),
            write("light_cells", C),
        ],
    };

    const CLOUD_COMPOSITE: Table = Table {
        label: "engine/cloud-composite-layout",
        bindings: &[texture("view_clouds", Float, D2, F)],
    };

    const SKY_TRANSMITTANCE: Table = Table {
        label: "engine/sky-transmittance-layout",
        bindings: &[texture("sky_transmittance_table", Float, D2, F)],
    };

    const SKY_TABLES: Table = Table {
        label: "engine/sky-tables-layout",
        bindings: &[
            uniform_of("air", size_of::<sky::SkyUniform>(), C),
            texture("transmittance_table", Float, D2, C),
            texture("multiple_table", Float, D2, C),
            texture("view_table", Float, D2, C),
            sampler("table_sampler", Filtering, C),
            bind("table_out", Kind::Storage(Rgba16Float), C),
            write("derived", C),
        ],
    };

    const TERRAIN_NORMALS: Table = Table {
        label: "engine/terrain-normals-layout",
        bindings: &[
            texture("heights", Unfilterable, D2Array, C),
            uniform_of("job", 16, C),
            bind("normals", Kind::Storage(R32Uint), C),
        ],
    };

    const TERRAIN_CONTROL: Table = Table {
        label: "engine/terrain-control-layout",
        bindings: &[
            texture("heights", Unfilterable, D2Array, C),
            uniform_of("job", 32, C),
            bind("data", Kind::Read { min_size: 4 }, C),
            bind("control", Kind::Storage(R32Uint), C),
            texture("rivers", Uint, D2Array, C),
        ],
    };

    const BOUNCE: Table = Table {
        label: "engine/terrain-bounce-layout",
        bindings: &[
            uniform_of("lighting", size_of::<LightingUniform>(), C),
            uniform_of("shadow", size_of::<ShadowUniform>(), C),
            texture("terrain_horizons", Float, D2Array, C),
            uniform_of("terrain_horizon_records", HORIZON_RECORDS_BYTES, C),
            sampler("terrain_horizon_sampler", Filtering, C),
            read_of("cloud_shadows", clouds::CLOUD_SHADOW_BYTES, C),
            uniform_of("bounce_frame", size_of::<bounce::BounceFrame>(), C),
            read_of("bounce_terrains", size_of::<bounce::BounceTerrain>(), C),
            texture("bounce_sky", Float, D2Array, C),
            texture("bounce_surface", Uint, D2Array, C),
            read_of("bounce_blocks", 4, C),
            read_of("bounce_records", 4, C),
            texture("bounce_light_in", Float, D2Array, C),
            bind("bounce_light_out", Kind::StorageArray(Rgba16Float), C),
            bind("bounce_probes", Kind::StorageArray(Rgba16Float), C),
            sampler("bounce_sampler", Filtering, C),
            uniform_of("bounce_palette", terrain::PALETTE_BYTES, C),
            texture("bounce_macro", Float, D2Array, C),
            write("bounce_scratch", C),
        ],
    };

    const COVER_GROW: Table = Table {
        label: "engine/cover-grow-layout",
        bindings: &[
            texture("heights", Unfilterable, D2Array, C),
            texture("control", Uint, D2Array, C),
            read("directory", C),
            uniform_of("palette", terrain::PALETTE_BYTES, C),
            read("covers", C),
            dynamic_uniform("grow", size_of::<ground_cover::GrowUniform>(), C),
            write("blades", C),
            write("counts", C),
            write("args", C),
            texture("macro_maps", Float, D2Array, C),
            texture("normals", Uint, D2Array, C),
        ],
    };

    const WATER: Table = Table {
        label: "engine/water-layout",
        serves: [SHADER @ 1],
        bindings: &[
            texture(17, "terrain_heights", Unfilterable, D2Array, V),
            dynamic_uniform(18, "terrain_view", size_of::<terrain::ViewUniform>(), VF),
            dynamic_uniform(19, "terrain", size_of::<terrain::TerrainUniform>(), VFC),
            read(23, "terrain_directory", VFC),
            read(36, "terrain_surface_leaves", V),
            texture(50, "water_surface", Unfilterable, D2Array, VFC),
            texture(51, "water_info", Uint, D2Array, VFC),
            uniform_of(52, "water", size_of::<water::WaterUniform>(), VFC),
            read(56, "water_live", VFC),
            dynamic_uniform(
                60,
                "water_tree",
                size_of::<terrain::SubdivisionUniform>(),
                V,
            ),
        ],
    };

    const WATER_BEHIND: Table = Table {
        label: "engine/water-behind-layout",
        serves: [SHADER @ 2],
        bindings: &[
            texture(22, "water_depth", Depth, D2, F),
            texture(23, "water_scene", Float, D2, F),
            texture(24, "water_particles", Float, D2, F),
            sampler(25, "water_particles_sampler", Filtering, F),
            texture(26, "water_particles_spread", Float, D2, F),
            uniform_of(32, "particle_frame", size_of::<water::ParticleFrame>(), V),
            read(48, "foam_pool", V),
            texture(49, "water_foam_drawn", Float, D2, F),
        ],
    };

    const WATER_PARTICLES: Table = Table {
        label: "engine/water-particles-layout",
        serves: [SHADER @ 2],
        bindings: &[
            uniform_of(32, "particle_frame", size_of::<water::ParticleFrame>(), C),
            write(33, "particle_lattice", C),
            write(34, "particle_pool", C),
            write(35, "particle_counts", C),
            write(36, "particle_splat", C),
            texture(37, "particle_source", Float, D2, C),
            bind(38, "particle_target", Kind::Storage(Rgba16Float), C),
            texture(39, "particle_spread_source", Float, D2, C),
            bind(40, "particle_spread_target", Kind::Storage(Rgba16Float), C),
        ],
    };

    const WATER_SIM: Table = Table {
        label: "engine/water-sim-layout",
        bindings: &[
            uniform_of("frame", size_of::<water::SimFrame>(), C),
            read("sim_ground", C),
            write("sim_cells", C),
            write("sim_depths", C),
            write("sim_fluxes", C),
            read("sim_inputs", C),
            read("sim_blocks", C),
            write("sim_steps", C),
            uniform_of("sim_slots", size_of::<water::SimSlots>(), C),
            write("sim_averages", C),
            uniform_of("sim_touches", size_of::<water::SimTouches>(), C),
        ],
    };

    const WATER_SETUP: Table = Table {
        label: "engine/water-setup-layout",
        bindings: &[
            uniform_of("setup_frame", size_of::<water::SimSetup>(), C),
            read("setup_list", C),
            write("setup_ground", C),
            write("setup_cells", C),
            write("setup_looks", C),
            write("setup_edges", C),
            read("setup_live", C),
        ],
    };

    const COVER_DRAW: Table = Table {
        label: "engine/cover-layout",
        bindings: &[
            read("cover_blades", V),
            uniform_of("cover_frame", size_of::<ground_cover::FrameUniform>(), V),
        ],
    };

    const SCATTER_CULL: Table = Table {
        label: "engine/scatter-cull-layout",
        bindings: &[
            uniform_of("frame", size_of::<scatter::FrameUniform>(), C),
            write("records", C),
            read("groups", C),
            write("ids", C),
            write("draws", C),
            read("views", C),
            write("cards", C),
            write("card_args", C),
        ],
    };

    const SCATTER_SNAP: Table = Table {
        label: "engine/scatter-snap-layout",
        bindings: &[
            uniform_of("snap", size_of::<scatter::SnapUniform>(), C),
            write("records", C),
            write("poses", C),
            read("terrains", C),
            read("directory", C),
            texture("heights", Unfilterable, D2Array, C),
            write("grounds", C),
            texture("control", Uint, D2Array, C),
            uniform_of("palette", terrain::PALETTE_BYTES, C),
            texture("macro_maps", Float, D2Array, C),
        ],
    };

    const OCCUPANCY: Table = Table {
        label: "engine/occupancy-layout",
        bindings: &[
            uniform_of("splat", size_of::<occupancy::SplatUniform>(), VFC),
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

    const SHADOW_VIEW: Table = Table {
        label: "engine/shadow-view-layout",
        bindings: &[
            dynamic_uniform("view", 96, V),
            read_of("routes", size_of::<RouteData>(), VF),
        ],
    };

    const SHADOW_POSES: Table = Table {
        label: "engine/shadow-pose-layout",
        bindings: &[
            read_of("slot_poses", POSE, V),
            read_of("slot_colors", COLOR, V),
            texture("shadow_regions", Depth, D2Array, F),
        ],
    };

    const LIGHT_BINNING: Table = Table {
        label: "engine/light-binning-layout",
        bindings: &[
            uniform("clusters", C),
            read("inputs", C),
            read("poses", C),
            write("lights", C),
            write("counts", C),
            write("indices", C),
            write("totals", C),
            write("ranges", C),
            read("still_frames", C),
        ],
    };

    const POST: Table = Table {
        label: "engine/post-layout",
        serves: [POST_SHADER @ 0],
        bindings: &[
            uniform(0, "post", F),
            texture(1, "depth_texture", Depth, D2, F),
            texture(2, "scene", Float, D2, F),
            texture(3, "glow", Float, D2, F),
            texture(4, "source", Float, D2, F),
            texture(5, "occlusion", Float, D2, F),
            texture(6, "bloom", Float, D2, F),
            sampler(7, "linear", Filtering, F),
            texture(8, "reflections", Float, D2, F),
            texture(9, "marched", Unfilterable, D2, F),
            texture(12, "raw_occlusion", Unfilterable, D2, F),
            texture(15, "bloom_source", Float, D2, F),
            texture(16, "sky_maps", Depth, D2Array, F),
            sampler(17, "sky_sampler", Comparison, F),
            uniform_of(18, "sky", size_of::<SkyUniform>(), F),
            texture(19, "haze", Float, D2, F),
        ],
    };

    const REFLECTIONS: Table = Table {
        label: "engine/post-reflections-layout",
        serves: [REFLECTIONS_SHADER @ 1],
        bindings: &[
            uniform_of(0, "post", size_of::<PostUniform>(), F),
            texture(1, "depth_texture", Depth, D2, F),
            texture(3, "glow", Float, D2, F),
            sampler(7, "linear", Filtering, F),
            texture(10, "nearest_depth", Unfilterable, D2, F),
            texture(13, "surface_texture", Uint, D2, F),
            texture(14, "scene_levels", Float, D2, F),
            texture(20, "heights", Unfilterable, D2Array, F),
            read(21, "directory", F),
            texture(23, "control", Uint, D2Array, F),
            uniform_of(24, "palette", terrain::PALETTE_BYTES, F),
            texture(25, "macro_maps", Float, D2Array, F),
            texture(26, "terrain_sky", Float, D2Array, F),
            read_of(27, "world_terrains", size_of::<TraceTerrain>(), F),
            read(28, "world_placed", F),
            read_of(29, "placed_records", size_of::<ScatterRecord>(), F),
            read_of(30, "placed_poses", POSE, F),
            read_of(31, "placed_groups", size_of::<PlacedGroup>(), F),
            texture(32, "placed_frames", Float, D2Array, F),
        ],
    };

    const HAZE: Table = Table {
        label: "engine/post-haze-layout",
        serves: [HAZE_SHADER @ 1],
        bindings: &[
            uniform_of(0, "post", size_of::<PostUniform>(), F),
            texture(1, "depth_texture", Depth, D2, F),
        ],
    };

    const TAA_RESOLVE: Table = Table {
        label: "engine/taa-resolve-layout",
        serves: [TEMPORAL_SHADER @ 0],
        bindings: &[
            uniform_of(0, "resolve", size_of::<ResolveUniform>(), F),
            texture(1, "scene", Unfilterable, D2, F),
            texture(2, "motion", Unfilterable, D2, F),
            texture(3, "history", Float, D2, F),
            sampler(4, "history_sampler", Filtering, F),
            texture(5, "occlusion", Float, D2, F),
            texture(6, "bloom", Float, D2, F),
            uniform_of(7, "post", size_of::<PostUniform>(), F),
            texture(8, "reflections", Float, D2, F),
            texture(9, "depth_texture", Depth, D2, F),
            texture(10, "revealed", Unfilterable, D2, F),
            texture(11, "haze", Float, D2, F),
        ],
    };

    const TRANSPARENT_COMPOSITE: Table = Table {
        label: "engine/transparent-composite-layout",
        serves: [TRANSPARENT_SHADER @ 0],
        bindings: &[
            texture(0, "accumulated", Unfilterable, D2, F),
            texture(1, "revealed", Unfilterable, D2, F),
        ],
    };

    const EXPOSURE: Table = Table {
        label: "engine/exposure-layout",
        serves: [EXPOSURE_SHADER @ 0],
        bindings: &[
            uniform(0, "settings", C),
            texture(1, "scene_color", Float, D2, C),
            texture(2, "scene_depth", Depth, D2, C),
            write(3, "bins", C),
            write(4, "state", C),
            uniform(5, "lighting", C),
        ],
    };

    const DEBUG_VIEW: Table = Table {
        label: "engine/debug-view-layout",
        serves: [DEBUG_VIEW_SHADER @ 0],
        bindings: &[
            uniform_of(0, "debug_view", size_of::<debug_view::DebugViewUniform>(), F),
            texture(1, "debug_ids", Uint, D2, F),
            texture(2, "debug_count", Unfilterable, D2, F),
            texture(3, "debug_terrain", Uint, D2, F),
        ],
    };

    const PARTICLES: Table = Table {
        label: "engine/particle-layout",
        serves: [PARTICLE_SHADER @ 1],
        bindings: &[
            uniform(0, "particles", VF),
            read(1, "emitters", VF),
            read(2, "owners", VF),
            texture(3, "depth_texture", Depth, D2, F),
        ],
    };

    const ENVIRONMENT_SKY: Table = Table {
        label: "engine/environment-sky-layout",
        bindings: &[
            uniform("lighting", F),
            uniform_of("sky_air", size_of::<sky::SkyUniform>(), F),
            texture("sky_view_table", Float, D2, F),
            sampler("sky_table_sampler", Filtering, F),
            texture("sky_clouds", Float, D2Array, F),
            texture("sky_clouds_old", Float, D2Array, F),
            sampler("sky_cloud_sampler", Filtering, F),
            uniform_of("sky_domes", size_of::<clouds::DomeUniform>(), F),
        ],
    };

    const ENVIRONMENT_DOWN: Table = Table {
        label: "engine/environment-down-layout",
        bindings: &[
            texture("level_source", Float, Cube, F),
            sampler("cube_sampler", Filtering, F),
        ],
    };

    const ENVIRONMENT_PREFILTER: Table = Table {
        label: "engine/environment-prefilter-layout",
        bindings: &[
            texture("radiance", Float, Cube, F),
            sampler("cube_sampler", Filtering, F),
        ],
    };

    const MIPMAPS: Table = Table {
        label: "engine/texture-mip-layout",
        bindings: &[
            texture("source", Float, D2, F),
            sampler("source_sampler", Filtering, F),
        ],
    };

    const PYRAMID_FIRST: Table = Table {
        label: "engine/depth-pyramid-first-layout",
        bindings: &[
            texture("source_depth", Depth, D2, C),
            bind("target_level", Kind::Storage(R32Float), C),
            uniform_of("extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };

    const PYRAMID_DOWN: Table = Table {
        label: "engine/depth-pyramid-down-layout",
        bindings: &[
            texture("source_level", Unfilterable, D2, C),
            bind("target_level", Kind::Storage(R32Float), C),
            uniform_of("extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };
}
