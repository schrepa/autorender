pub(crate) use crate::gpu_layouts::constants::PYRAMID_EXTENT_BYTES;
use crate::gpu_layouts::impostors::{ImpostorBakeUniform, ImpostorRecord};
use crate::gpu_layouts::scatter::Record as ScatterRecord;
use crate::gpu_layouts::scene::{
    CameraUniform, ColorData, GroundData, LightingUniform, MaterialUniform, PoseData, PostUniform,
    ResolveUniform, RouteData, ShadowCullUniform, ShadowUniform, SkyUniform,
};
use crate::gpu_layouts::world_trace::{PlacedGroup, TraceTerrain};
use crate::gpu_layouts::{
    bounce, clouds, debug_view, deformation, ground_cover, occupancy, scatter, sky, terrain, water,
};

mod shared;
mod table;

pub use shared::{FeatureRange, SHARED_GROUPS, SharedGroup};
pub use table::{Binding, Format, Kind, Sample, Sampling, Serves, Stages, Table, View};
use table::{bind, dynamic_uniform, read, read_of, sampler, texture, uniform, uniform_of, write};

use Format::{R32Float, R32Uint, Rgba16Float};
use Sample::{Depth, Float, Uint, Unfilterable};
use Sampling::{Comparison, Filtering, NonFiltering};
use View::{Cube, CubeArray, D2, D2Array, D3};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;
const VC: Stages = Stages(V.0 | C.0);
const FC: Stages = Stages(F.0 | C.0);
const VFC: Stages = Stages(VF.0 | C.0);
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
            serves: [$($module:ident @ $group:literal),* $(,)?],
            bindings: &[$($binding:expr),* $(,)?] $(,)?
        };
    )*) => {
        $(
            $(#[$meta])*
            pub(crate) const $name: Table = Table {
                label: $label,
                serves: &[$(Serves {
                    module: {
                        let _: &str = crate::shaders::$module;
                        stringify!($module)
                    },
                    group: $group,
                }),*],
                bindings: &[$($binding),*],
            };
        )*

        pub const TABLES: &[(&str, &Table)] = &[$((stringify!($name), &$name)),*];
    };
}

tables! {
    const EMPTY: Table = Table {
        label: "engine/empty-layout",
        serves: [],
        bindings: &[],
    };

    const CAMERA: Table = Table {
        label: "engine/camera-layout",
        serves: [SHADER @ 0, PARTICLE_SHADER @ 0, HAZE_SHADER @ 0],
        bindings: &[
            uniform_of(0, "camera", size_of::<CameraUniform>(), VF),
            uniform_of(1, "lighting", size_of::<LightingUniform>(), VF),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), VF),
            texture(3, "shadow_atlas", Depth, D2, VF),
            texture(4, "shadow_tiles", Uint, D2, VF),
            sampler(6, "shadow_sampler", Comparison, F),
            sampler(7, "shadow_texels", NonFiltering, VF),
            texture(9, "refraction_source", Float, D2, F),
            sampler(10, "refraction_sampler", Filtering, F),
            texture(11, "environment_maps", Float, CubeArray, F),
            sampler(12, "environment_sampler", Filtering, F),
            read(13, "probe_words", F),
            texture(14, "terrain_horizons", Float, D2Array, F),
            uniform_of(15, "terrain_horizon_records", HORIZON_RECORDS_BYTES, F),
            sampler(16, "terrain_horizon_sampler", Filtering, F),
            texture(17, "occupancy_coverage", Float, D2Array, F),
            texture(18, "occupancy_heights", Unfilterable, D2Array, F),
            read_of(19, "cloud_shadows", clouds::CLOUD_SHADOW_BYTES, F),
            uniform_of(23, "sky_air", size_of::<sky::SkyUniform>(), F),
            texture(24, "sky_view_table", Float, D2, F),
            sampler(25, "sky_table_sampler", Filtering, F),
            texture(26, "sky_clouds", Float, D2Array, F),
            texture(27, "sky_clouds_old", Float, D2Array, F),
            sampler(28, "sky_cloud_sampler", Filtering, F),
            uniform_of(29, "sky_domes", size_of::<clouds::DomeUniform>(), F),
        ],
    };

    const CAMERA_TERRAIN: Table = Table {
        label: "engine/camera-terrain-layout",
        serves: [SHADER @ 0, REFLECTIONS_SHADER @ 0],
        bindings: &[
            uniform_of(0, "camera", size_of::<CameraUniform>(), VF),
            uniform_of(1, "lighting", size_of::<LightingUniform>(), F),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), F),
            texture(3, "shadow_atlas", Depth, D2, F),
            texture(4, "shadow_tiles", Uint, D2, F),
            sampler(6, "shadow_sampler", Comparison, F),
            sampler(7, "shadow_texels", NonFiltering, F),
            texture(11, "environment_maps", Float, CubeArray, F),
            sampler(12, "environment_sampler", Filtering, F),
            read(13, "probe_words", F),
            texture(14, "terrain_horizons", Float, D2Array, F),
            uniform_of(15, "terrain_horizon_records", HORIZON_RECORDS_BYTES, F),
            sampler(16, "terrain_horizon_sampler", Filtering, F),
            texture(17, "occupancy_coverage", Float, D2Array, F),
            texture(18, "occupancy_heights", Unfilterable, D2Array, F),
            read_of(19, "cloud_shadows", clouds::CLOUD_SHADOW_BYTES, F),
        ],
    };

    const CAMERA_WATER: Table = Table {
        label: "engine/camera-water-layout",
        serves: [SHADER @ 0],
        bindings: &[
            uniform_of(0, "camera", size_of::<CameraUniform>(), VFC),
            uniform_of(1, "lighting", size_of::<LightingUniform>(), VF),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), VF),
            texture(3, "shadow_atlas", Depth, D2, VF),
            texture(4, "shadow_tiles", Uint, D2, VF),
            sampler(6, "shadow_sampler", Comparison, F),
            sampler(7, "shadow_texels", NonFiltering, VF),
            sampler(10, "refraction_sampler", Filtering, F),
            texture(11, "environment_maps", Float, CubeArray, F),
            sampler(12, "environment_sampler", Filtering, F),
            read(13, "probe_words", F),
            texture(14, "terrain_horizons", Float, D2Array, F),
            uniform_of(15, "terrain_horizon_records", HORIZON_RECORDS_BYTES, F),
            sampler(16, "terrain_horizon_sampler", Filtering, F),
            texture(17, "occupancy_coverage", Float, D2Array, F),
            texture(18, "occupancy_heights", Unfilterable, D2Array, F),
            read_of(19, "cloud_shadows", clouds::CLOUD_SHADOW_BYTES, F),
            uniform_of(23, "sky_air", size_of::<sky::SkyUniform>(), F),
            texture(24, "sky_view_table", Float, D2, F),
            sampler(25, "sky_table_sampler", Filtering, F),
            texture(26, "sky_clouds", Float, D2Array, F),
            texture(27, "sky_clouds_old", Float, D2Array, F),
            sampler(28, "sky_cloud_sampler", Filtering, F),
            uniform_of(29, "sky_domes", size_of::<clouds::DomeUniform>(), F),
        ],
    };

    const MATERIAL: Table = Table {
        label: "engine/material-layout",
        serves: [SHADER @ 1, SHADOW_SHADER @ 2],
        bindings: &[
            texture(0, "base_color_texture", Float, D2, F),
            sampler(1, "base_color_sampler", Filtering, F),
            texture(2, "metallic_roughness_texture", Float, D2, F),
            sampler(3, "metallic_roughness_sampler", Filtering, F),
            texture(4, "normal_texture", Float, D2, F),
            sampler(5, "normal_sampler", Filtering, F),
            texture(6, "occlusion_texture", Float, D2, F),
            sampler(7, "occlusion_sampler", Filtering, F),
            texture(8, "emissive_texture", Float, D2, F),
            sampler(9, "emissive_sampler", Filtering, F),
            uniform_of(10, "material_maps", size_of::<MaterialUniform>(), F),
        ],
    };

    const RETAINED_POSES: Table = Table {
        label: "engine/retained-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, VF),
            read_of(1, "colors", COLOR, V),
            read_of(5, "grounds", GROUND, F),
        ],
    };

    const TEMPORAL_POSES: Table = Table {
        label: "engine/temporal-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, VF),
            read_of(1, "colors", COLOR, V),
            read_of(2, "previous_poses", POSE, V),
            read_of(5, "grounds", GROUND, F),
        ],
    };

    const LIGHT_SHADING: Table = Table {
        label: "engine/light-shading-layout",
        serves: [SHADER @ 3],
        bindings: &[
            uniform(0, "clusters", F),
            read(1, "lights", F),
            read(2, "cluster_ranges", F),
            read(3, "cluster_lights", F),
        ],
    };

    const DECALS: Table = Table {
        label: "engine/decal-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read(3, "palette", V),
            read(13, "decal_records", VF),
            texture(14, "decal_depth", Depth, D2, F),
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
        serves: [PRESENTATION_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(1, "locals", C),
            read(2, "info", C),
            read(3, "authority", C),
            write(4, "composed", C),
            read(5, "parents", C),
        ],
    };

    const PRESENTATION_LEVEL: Table = Table {
        label: "engine/presentation-level-layout",
        serves: [PRESENTATION_SHADER @ 1],
        bindings: &[uniform(0, "level", C)],
    };

    const PRESENTATION_CULL: Table = Table {
        label: "engine/presentation-cull-layout",
        serves: [PRESENTATION_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(1, "locals", C),
            read(2, "info", C),
            read(3, "authority", C),
            write(4, "composed", C),
            read(7, "batches", C),
            write(8, "poses", C),
            write(9, "scan", C),
            read(14, "slot_candidates", C),
        ],
    };

    const PRESENTATION_TESTS: Table = Table {
        label: "engine/presentation-test-layout",
        serves: [PRESENTATION_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(2, "info", C),
            read(7, "batches", C),
            write(8, "poses", C),
            write(9, "scan", C),
            write(13, "locals_shift", C),
            read(14, "slot_candidates", C),
            texture(18, "depth_pyramid", Unfilterable, D2, C),
            write(19, "occlusion_counts", C),
            write(23, "still_frames", C),
        ],
    };

    const PRESENTATION_COMPACT: Table = Table {
        label: "engine/presentation-compact-layout",
        serves: [PRESENTATION_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(6, "candidates", C),
            read(7, "batches", C),
            write(9, "scan", C),
            write(10, "batch_bases", C),
            write(11, "visible_ids", C),
            write(12, "draws", C),
            write(19, "occlusion_counts", C),
            write(24, "impostor_ids", C),
        ],
    };

    const PRESENTATION_FADES: Table = Table {
        label: "engine/presentation-fades-layout",
        serves: [PRESENTATION_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(7, "batches", C),
            write(8, "poses", C),
            write(9, "scan", C),
            read(14, "slot_candidates", C),
            write(25, "lod_states", C),
            write(26, "fade_cards", C),
        ],
    };

    const SHADOW_SELECT: Table = Table {
        label: "engine/shadow-select-layout",
        serves: [SHADOW_SHADER @ 0],
        bindings: &[
            uniform_of(0, "cull", size_of::<ShadowCullUniform>(), C),
            read(1, "candidates", C),
            read(2, "batches", C),
            read(3, "poses", C),
            read(4, "flags", C),
            write(5, "stillness", C),
            write(6, "ids", C),
            write(7, "args", C),
            write(8, "counters", C),
            texture(12, "tile_slots", Uint, D2, C),
        ],
    };

    const SHADOW_MARKS: Table = Table {
        label: "engine/shadow-marks-layout",
        serves: [SHADOW_MARKS_SHADER @ 0],
        bindings: &[
            uniform_of(0, "camera", size_of::<CameraUniform>(), C),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), C),
            texture(24, "shadow_depth", Depth, D2, C),
            write(25, "shadow_marks", C),
        ],
    };

    const TERRAIN: Table = Table {
        label: "engine/terrain-layout",
        serves: [SHADER @ 1, TERRAIN_SHADOW_SHADER @ 1],
        bindings: &[
            texture(17, "terrain_heights", Unfilterable, D2Array, VFC),
            dynamic_uniform(18, "terrain_view", size_of::<terrain::ViewUniform>(), VF),
            dynamic_uniform(
                19,
                "terrain",
                size_of::<terrain::TerrainUniform>(),
                VFC,
            ),
            read(20, "terrain_holes", VFC),
            texture(21, "terrain_normals", Uint, D2Array, VFC),
            sampler(22, "terrain_sky_sampler", Filtering, F),
            read(23, "terrain_directory", VFC),
            texture(24, "terrain_control", Uint, D2Array, VFC),
            uniform_of(25, "terrain_palette", terrain::PALETTE_BYTES, VFC),
            texture(26, "terrain_albedo", Float, D2Array, F),
            texture(27, "terrain_material_data", Float, D2Array, VFC),
            sampler(29, "terrain_material_sampler", Filtering, VFC),
            texture(30, "terrain_sky", Float, D2Array, F),
            read(31, "terrain_deformation", F),
            sampler(32, "terrain_control_sampler", NonFiltering, VFC),
            texture(35, "terrain_macro", Float, D2Array, F),
            read(36, "terrain_surface_leaves", V),
            texture(50, "water_surface", Unfilterable, D2Array, FC),
            texture(51, "water_info", Uint, D2Array, C),
            texture(62, "terrain_errors", Unfilterable, D2Array, C),
            texture(63, "water_errors", Unfilterable, D2Array, C),
        ],
    };

    const TERRAIN_RESOLVE: Table = Table {
        label: "engine/terrain-resolve-layout",
        serves: [SHADER @ 2],
        bindings: &[texture(18, "terrain_ids_texture", Uint, D2, F)],
    };

    const TERRAIN_SUBDIVISION_CAMERA: Table = Table {
        label: "engine/terrain-subdivision-camera-layout",
        serves: [SHADER @ 0],
        bindings: &[uniform_of(0, "camera", size_of::<CameraUniform>(), C)],
    };

    const TERRAIN_SUBDIVISION: Table = Table {
        label: "engine/terrain-subdivision-layout",
        serves: [SHADER @ 2],
        bindings: &[
            dynamic_uniform(
                27,
                "subdivision",
                size_of::<terrain::SubdivisionUniform>(),
                C,
            ),
            write(28, "subdivision_tree", C),
            write(29, "subdivision_nodes", C),
            write(30, "subdivision_leaves", C),
            write(31, "subdivision_args", C),
        ],
    };

    const CLOUD_MARCH: Table = Table {
        label: "engine/cloud-march-layout",
        serves: [CLOUDS_SHADER @ 0],
        bindings: &[
            uniform_of(0, "clouds", size_of::<clouds::CloudUniform>(), C),
            texture(1, "noise_shape", Float, D3, C),
            sampler(2, "noise_sampler", Filtering, C),
            bind(3, "dome", Kind::StorageArray(Rgba16Float), C),
            texture(4, "noise_detail", Float, D3, C),
            write(5, "shadow_cells", C),
            uniform_of(11, "cloud_air", size_of::<sky::SkyUniform>(), C),
            texture(12, "cloud_transmittance", Float, D2, C),
            sampler(13, "cloud_table_sampler", Filtering, C),
            write(14, "light_cells", C),
        ],
    };

    const CLOUD_VIEW: Table = Table {
        label: "engine/cloud-view-layout",
        serves: [CLOUDS_SHADER @ 0],
        bindings: &[
            uniform_of(0, "clouds", size_of::<clouds::CloudUniform>(), C),
            texture(1, "noise_shape", Float, D3, C),
            sampler(2, "noise_sampler", Filtering, C),
            texture(4, "noise_detail", Float, D3, C),
            uniform_of(6, "cloud_view", size_of::<clouds::ViewUniform>(), C),
            texture(7, "view_depth", Depth, D2, C),
            bind(8, "view_out", Kind::Storage(Rgba16Float), C),
            write(10, "composite_draw", C),
            uniform_of(11, "cloud_air", size_of::<sky::SkyUniform>(), C),
            texture(12, "cloud_transmittance", Float, D2, C),
            sampler(13, "cloud_table_sampler", Filtering, C),
            write(14, "light_cells", C),
        ],
    };

    const CLOUD_COMPOSITE: Table = Table {
        label: "engine/cloud-composite-layout",
        serves: [SHADER @ 1],
        bindings: &[texture(46, "view_clouds", Float, D2, F)],
    };

    const SKY_TRANSMITTANCE: Table = Table {
        label: "engine/sky-transmittance-layout",
        serves: [SHADER @ 1],
        bindings: &[texture(42, "sky_transmittance_table", Float, D2, F)],
    };

    const SKY_TABLES: Table = Table {
        label: "engine/sky-tables-layout",
        serves: [SKY_SHADER @ 0],
        bindings: &[
            uniform_of(0, "air", size_of::<sky::SkyUniform>(), C),
            texture(1, "transmittance_table", Float, D2, C),
            texture(2, "multiple_table", Float, D2, C),
            texture(3, "view_table", Float, D2, C),
            sampler(4, "table_sampler", Filtering, C),
            bind(5, "table_out", Kind::Storage(Rgba16Float), C),
            write(6, "derived", C),
        ],
    };

    const DEFORMATION: Table = Table {
        label: "engine/deformation-layout",
        serves: [DEFORMATION_SHADER @ 0],
        bindings: &[
            dynamic_uniform(0, "job", size_of::<deformation::JobUniform>(), C),
            read(1, "stamps", C),
            write(2, "map", C),
        ],
    };

    const TERRAIN_NORMALS: Table = Table {
        label: "engine/terrain-normals-layout",
        serves: [TERRAIN_NORMALS_SHADER @ 0],
        bindings: &[
            texture(0, "heights", Unfilterable, D2Array, C),
            uniform_of(1, "job", 16, C),
            bind(2, "normals", Kind::Storage(R32Uint), C),
        ],
    };

    const TERRAIN_CONTROL: Table = Table {
        label: "engine/terrain-control-layout",
        serves: [TERRAIN_CONTROL_SHADER @ 0],
        bindings: &[
            texture(0, "heights", Unfilterable, D2Array, C),
            uniform_of(1, "job", 32, C),
            bind(2, "data", Kind::Read { min_size: 4 }, C),
            bind(3, "control", Kind::Storage(R32Uint), C),
            texture(4, "rivers", Uint, D2Array, C),
        ],
    };

    const BOUNCE: Table = Table {
        label: "engine/terrain-bounce-layout",
        serves: [BOUNCE_SHADER @ 0],
        bindings: &[
            uniform_of(1, "lighting", size_of::<LightingUniform>(), C),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), C),
            texture(14, "terrain_horizons", Float, D2Array, C),
            uniform_of(15, "terrain_horizon_records", HORIZON_RECORDS_BYTES, C),
            sampler(16, "terrain_horizon_sampler", Filtering, C),
            read_of(19, "cloud_shadows", clouds::CLOUD_SHADOW_BYTES, C),
            uniform_of(60, "bounce_frame", size_of::<bounce::BounceFrame>(), C),
            read_of(61, "bounce_terrains", size_of::<bounce::BounceTerrain>(), C),
            texture(62, "bounce_sky", Float, D2Array, C),
            texture(63, "bounce_surface", Uint, D2Array, C),
            read_of(64, "bounce_blocks", 4, C),
            read_of(65, "bounce_records", 4, C),
            texture(66, "bounce_light_in", Float, D2Array, C),
            bind(67, "bounce_light_out", Kind::StorageArray(Rgba16Float), C),
            bind(68, "bounce_probes", Kind::StorageArray(Rgba16Float), C),
            sampler(69, "bounce_sampler", Filtering, C),
            uniform_of(70, "bounce_palette", terrain::PALETTE_BYTES, C),
            texture(71, "bounce_macro", Float, D2Array, C),
            write(72, "bounce_scratch", C),
        ],
    };

    const COVER_GROW: Table = Table {
        label: "engine/cover-grow-layout",
        serves: [COVER_GROW_SHADER @ 0],
        bindings: &[
            texture(0, "heights", Unfilterable, D2Array, C),
            texture(1, "control", Uint, D2Array, C),
            read(2, "directory", C),
            uniform_of(3, "palette", terrain::PALETTE_BYTES, C),
            read(4, "covers", C),
            dynamic_uniform(5, "grow", size_of::<ground_cover::GrowUniform>(), C),
            write(6, "blades", C),
            write(7, "counts", C),
            write(8, "args", C),
            read(9, "deformation", C),
            texture(10, "macro_maps", Float, D2Array, C),
            texture(11, "normals", Uint, D2Array, C),
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
        serves: [WATER_SIM_SHADER @ 0],
        bindings: &[
            uniform_of(0, "frame", size_of::<water::SimFrame>(), C),
            read(1, "sim_ground", C),
            write(2, "sim_cells", C),
            write(3, "sim_depths", C),
            write(4, "sim_fluxes", C),
            read(5, "sim_inputs", C),
            read(6, "sim_blocks", C),
            write(7, "sim_steps", C),
            uniform_of(8, "sim_slots", size_of::<water::SimSlots>(), C),
            write(9, "sim_averages", C),
            uniform_of(10, "sim_touches", size_of::<water::SimTouches>(), C),
        ],
    };

    const WATER_SETUP: Table = Table {
        label: "engine/water-setup-layout",
        serves: [SHADER @ 2],
        bindings: &[
            uniform_of(41, "setup_frame", size_of::<water::SimSetup>(), C),
            read(42, "setup_list", C),
            write(43, "setup_ground", C),
            write(44, "setup_cells", C),
            write(45, "setup_looks", C),
            write(46, "setup_edges", C),
            read(47, "setup_live", C),
        ],
    };

    const COVER_DRAW: Table = Table {
        label: "engine/cover-layout",
        serves: [SHADER @ 1],
        bindings: &[
            read(33, "cover_blades", V),
            uniform_of(34, "cover_frame", size_of::<ground_cover::FrameUniform>(), V),
        ],
    };

    const SCATTER_CULL: Table = Table {
        label: "engine/scatter-cull-layout",
        serves: [SCATTER_CULL_SHADER @ 0],
        bindings: &[
            uniform_of(0, "frame", size_of::<scatter::FrameUniform>(), C),
            write(1, "records", C),
            read(2, "groups", C),
            write(3, "ids", C),
            write(4, "draws", C),
            read(5, "views", C),
            write(6, "cards", C),
            write(7, "card_args", C),
        ],
    };

    const SCATTER_SNAP: Table = Table {
        label: "engine/scatter-snap-layout",
        serves: [SCATTER_SNAP_SHADER @ 0],
        bindings: &[
            uniform_of(0, "snap", size_of::<scatter::SnapUniform>(), C),
            write(1, "records", C),
            write(2, "poses", C),
            read(3, "terrains", C),
            read(4, "directory", C),
            texture(5, "heights", Unfilterable, D2Array, C),
            write(6, "grounds", C),
            texture(7, "control", Uint, D2Array, C),
            uniform_of(8, "palette", terrain::PALETTE_BYTES, C),
            texture(9, "macro_maps", Float, D2Array, C),
        ],
    };

    const OCCUPANCY: Table = Table {
        label: "engine/occupancy-layout",
        serves: [OCCUPANCY_SHADER @ 0],
        bindings: &[
            uniform_of(0, "splat", size_of::<occupancy::SplatUniform>(), VFC),
            read(1, "records", VC),
            read(2, "poses", VC),
            read(3, "covers", VC),
            read(4, "listed", V),
            write(5, "listing", C),
            write(6, "listed_draw", C),
            texture(7, "finer_coverage", Unfilterable, D2, F),
            texture(8, "finer_heights", Unfilterable, D2, F),
        ],
    };

    const SHADOW_VIEW: Table = Table {
        label: "engine/shadow-view-layout",
        serves: [SHADOW_SHADER @ 0, TERRAIN_SHADOW_SHADER @ 0],
        bindings: &[
            dynamic_uniform(9, "view", 96, V),
            read_of(13, "routes", size_of::<RouteData>(), VF),
        ],
    };

    const SHADOW_POSES: Table = Table {
        label: "engine/shadow-pose-layout",
        serves: [SHADOW_SHADER @ 1],
        bindings: &[
            read_of(0, "slot_poses", POSE, V),
            read_of(1, "slot_colors", COLOR, V),
            texture(3, "shadow_regions", Depth, D2Array, F),
        ],
    };

    const LIGHT_BINNING: Table = Table {
        label: "engine/light-binning-layout",
        serves: [LIGHT_SHADER @ 0],
        bindings: &[
            uniform(0, "clusters", C),
            read(1, "inputs", C),
            read(2, "poses", C),
            write(3, "lights", C),
            write(4, "counts", C),
            write(5, "indices", C),
            write(6, "totals", C),
            write(7, "ranges", C),
            read(8, "still_frames", C),
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
        serves: [ENVIRONMENT_SHADER @ 0],
        bindings: &[
            uniform(1, "lighting", F),
            uniform_of(23, "sky_air", size_of::<sky::SkyUniform>(), F),
            texture(24, "sky_view_table", Float, D2, F),
            sampler(25, "sky_table_sampler", Filtering, F),
            texture(26, "sky_clouds", Float, D2Array, F),
            texture(27, "sky_clouds_old", Float, D2Array, F),
            sampler(28, "sky_cloud_sampler", Filtering, F),
            uniform_of(29, "sky_domes", size_of::<clouds::DomeUniform>(), F),
        ],
    };

    const ENVIRONMENT_DOWN: Table = Table {
        label: "engine/environment-down-layout",
        serves: [ENVIRONMENT_SHADER @ 0],
        bindings: &[
            texture(20, "level_source", Float, Cube, F),
            sampler(22, "cube_sampler", Filtering, F),
        ],
    };

    const ENVIRONMENT_PREFILTER: Table = Table {
        label: "engine/environment-prefilter-layout",
        serves: [ENVIRONMENT_SHADER @ 0],
        bindings: &[
            texture(21, "radiance", Float, Cube, F),
            sampler(22, "cube_sampler", Filtering, F),
        ],
    };

    const MIPMAPS: Table = Table {
        label: "engine/texture-mip-layout",
        serves: [MIPMAP_SHADER @ 0],
        bindings: &[
            texture(0, "source", Float, D2, F),
            sampler(1, "source_sampler", Filtering, F),
        ],
    };

    const PYRAMID_FIRST: Table = Table {
        label: "engine/depth-pyramid-first-layout",
        serves: [DEPTH_PYRAMID_SHADER @ 0],
        bindings: &[
            texture(0, "source_depth", Depth, D2, C),
            bind(2, "target_level", Kind::Storage(R32Float), C),
            uniform_of(3, "extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };

    const PYRAMID_DOWN: Table = Table {
        label: "engine/depth-pyramid-down-layout",
        serves: [DEPTH_PYRAMID_SHADER @ 0],
        bindings: &[
            texture(1, "source_level", Unfilterable, D2, C),
            bind(2, "target_level", Kind::Storage(R32Float), C),
            uniform_of(3, "extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };
}
