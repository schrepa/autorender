//! Bind group layouts as data: every layout the renderer
//! builds is a table of the bindings its shaders read, each with its
//! number, WGSL name, kind and the stages reading it. Layouts and bind
//! groups are built from the tables (`crate::pipelines::layout` and
//! `group`), so a binding's number is written once in WGSL and once here,
//! and groups bind resources by name.

// Native builds read only the tables; Wasm builds build layouts from them.

pub(crate) use crate::gpu_layouts::constants::PYRAMID_EXTENT_BYTES;
use crate::gpu_layouts::impostors::{ImpostorBakeUniform, ImpostorRecord};
use crate::gpu_layouts::scatter::Record as ScatterRecord;
use crate::gpu_layouts::scene::{
    CameraUniform, ColorData, GroundData, LightingUniform, MaterialUniform, PoseData, PostUniform,
    ResolveUniform, RouteData, ShadowCullUniform, ShadowUniform, SkyUniform,
};
use crate::gpu_layouts::virtual_geometry::MeshRecord;
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
// Tables a family's compute and render passes share.
const VC: Stages = Stages(V.0 | C.0);
const VFC: Stages = Stages(VF.0 | C.0);
// Bytes of a pose, a slot color, a placed object's ground,
// and of virtual geometry's mesh, cluster and page pool word.
const POSE: usize = size_of::<PoseData>();
const COLOR: usize = size_of::<ColorData>();
const GROUND: usize = size_of::<GroundData>();
const VIRTUAL_MESH: usize = size_of::<MeshRecord>();
const VIRTUAL_CLUSTER: usize = engine_core::CLUSTER_WORDS * 4;
const VIRTUAL_PAGE: usize = 4;
/// Bytes of the terrains' horizon records.
pub(crate) const HORIZON_RECORDS_BYTES: usize =
    crate::gpu_layouts::constants::HORIZON_RECORDS * size_of::<[f32; 4]>();

/// Declares bind group tables and [`TABLES`] from the same list:
/// `const NAME: Table = Table { label, serves: [MODULE @ group, ..], bindings }`,
/// each `MODULE` a constant of `crate::shaders`.
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

        /// Every table, by name.
        pub const TABLES: &[(&str, &Table)] = &[$((stringify!($name), &$name)),*];
    };
}

tables! {
    /// A group binding nothing, standing in for groups a pipeline skips.
    const EMPTY: Table = Table {
        label: "engine/empty-layout",
        serves: [],
        bindings: &[],
    };

    // The mesh shader (`gpu_data::SHADER`).

    /// Group 0 of mesh, sky and effect pipelines: the camera, lighting and
    /// shadow lookups, what glass refracts,
    /// the environment maps and reflection probes, and the sky's tables and
    /// the clouds' shown domes, which the sky draws and smooth surfaces
    /// mirror (`sky_light.wgsl`).
    const CAMERA: Table = Table {
        label: "engine/camera-layout",
        serves: [SHADER @ 0, PARTICLE_SHADER @ 0, HAZE_SHADER @ 0],
        bindings: &[
            // Fragments read the eye, sky rays and the jitter.
            uniform_of(0, "camera", size_of::<CameraUniform>(), VF),
            // Particles' vertices look up the sun's shadow maps at their
            // middles.
            uniform_of(1, "lighting", size_of::<LightingUniform>(), VF),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), VF),
            // The shadow clipmap's tiles, each in a slot of one atlas, and
            // the slot each level's tiles lie in.
            texture(3, "shadow_atlas", Depth, D2, VF),
            texture(4, "shadow_tiles", Uint, D2, VF),
            sampler(6, "shadow_sampler", Comparison, F),
            // Receiver-plane lookups read four texels.
            sampler(7, "shadow_texels", NonFiltering, VF),
            texture(9, "refraction_source", Float, D2, F),
            sampler(10, "refraction_sampler", Filtering, F),
            texture(11, "environment_maps", Float, CubeArray, F),
            sampler(12, "environment_sampler", Filtering, F),
            read(13, "probe_words", F),
            // Terrain's long sun shadows: each terrain's horizon
            // map, every terrain's record (a uniform the lookups loop over),
            // and the maps' sampler.
            texture(14, "terrain_horizons", Float, D2Array, F),
            uniform_of(15, "terrain_horizon_records", HORIZON_RECORDS_BYTES, F),
            sampler(16, "terrain_horizon_sampler", Filtering, F),
            // What placed objects cover around the detail eye: how much
            // they hide, filtered, and the covers' tops and bottoms; a layer
            // per map, the finest first.
            texture(17, "occupancy_coverage", Float, D2Array, F),
            texture(18, "occupancy_heights", Unfilterable, D2Array, F),
            // Clouds' shadows: the eighth storage buffer the
            // terrain's lit fragments read (a uniform array read at each
            // fragment's own index is many times slower).
            read_of(19, "cloud_shadows", clouds::CLOUD_SHADOW_BYTES, F),
            // The physically based sky's uniform, its sky-view table and
            // their sampler, and the clouds' newer and older shown domes,
            // each a layer marched from the eye and one from its mirror
            // point, their sampler and their uniform: the sky's light along
            // a view and what smooth surfaces mirror of it.
            uniform_of(23, "sky_air", size_of::<sky::SkyUniform>(), F),
            texture(24, "sky_view_table", Float, D2, F),
            sampler(25, "sky_table_sampler", Filtering, F),
            texture(26, "sky_clouds", Float, D2Array, F),
            texture(27, "sky_clouds_old", Float, D2Array, F),
            sampler(28, "sky_cloud_sampler", Filtering, F),
            uniform_of(29, "sky_domes", size_of::<clouds::DomeUniform>(), F),
        ],
    };

    /// Group 0 of terrain's pipelines: the main passes' without what glass
    /// refracts and without the sky's tables and the clouds' domes, which
    /// keeps their fragment stages within the 16 sampled textures one may
    /// bind; terrain mirrors the sky's environment map alone
    /// (`meshes/shading.wgsl`'s `finish_ground`). Reflections light what
    /// their world trace meets through it too (`reflections.wgsl`).
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

    /// Group 0 of water's pipelines: the main passes' without what glass
    /// refracts, which water reads from its own copy of what lies behind it
    /// (`WATER_BEHIND`), keeping the sampler it reads that copy through.
    const CAMERA_WATER: Table = Table {
        label: "engine/camera-water-layout",
        serves: [SHADER @ 0],
        bindings: &[
            uniform_of(0, "camera", size_of::<CameraUniform>(), VF),
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

    /// Group 1 of mesh pipelines: a material's maps and their samplers and its
    /// map parameters; shadow casters bind it as group 2.
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

    /// Group 2 of retained mesh pipelines: slot poses, colors, the joint
    /// palettes of skinned slots and placed objects' grounds, which the
    /// fragments of those blending into the ground read with their poses.
    const RETAINED_POSES: Table = Table {
        label: "engine/retained-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, VF),
            read_of(1, "colors", COLOR, V),
            read_of(3, "palette", POSE, V),
            read_of(5, "grounds", GROUND, F),
        ],
    };

    /// Group 2 of temporal mesh pipelines: the retained poses
    /// and the previous frame's poses and palettes.
    const TEMPORAL_POSES: Table = Table {
        label: "engine/temporal-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, VF),
            read_of(1, "colors", COLOR, V),
            read_of(2, "previous_poses", POSE, V),
            read_of(3, "palette", POSE, V),
            read_of(4, "previous_palette", POSE, V),
            read_of(5, "grounds", GROUND, F),
        ],
    };

    /// Group 2 of virtual geometry's pipelines: poses, colors,
    /// previous poses and the arena's meshes, clusters and pages; and
    /// grounds none of its slots stand on, as the forward fragments they
    /// share with retained meshes read.
    const VIRTUAL_POSES: Table = Table {
        label: "engine/virtual-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, VF),
            read_of(1, "colors", COLOR, V),
            read_of(2, "previous_poses", POSE, V),
            read_of(5, "grounds", GROUND, F),
            read_of(6, "virtual_meshes", VIRTUAL_MESH, V),
            read_of(7, "virtual_clusters", VIRTUAL_CLUSTER, V),
            read_of(8, "virtual_pages", VIRTUAL_PAGE, V),
        ],
    };

    /// Group 2 of the visibility pass: poses, the arena and each
    /// draw's bin through a dynamic offset.
    const VISIBILITY_PASS: Table = Table {
        label: "engine/visibility-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, V),
            read_of(6, "virtual_meshes", VIRTUAL_MESH, V),
            read_of(7, "virtual_clusters", VIRTUAL_CLUSTER, V),
            read_of(8, "virtual_pages", VIRTUAL_PAGE, V),
            dynamic_uniform(9, "visibility_draw", 16, V),
        ],
    };

    /// Group 2 of the visibility buffer's resolves: the poses,
    /// arena and draw, the selected clusters, the ID target and the tile lists.
    const VISIBILITY_RESOLVE: Table = Table {
        label: "engine/visibility-resolve-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, F),
            read_of(1, "colors", COLOR, F),
            read_of(2, "previous_poses", POSE, F),
            read_of(6, "virtual_meshes", VIRTUAL_MESH, F),
            read_of(7, "virtual_clusters", VIRTUAL_CLUSTER, F),
            read_of(8, "virtual_pages", VIRTUAL_PAGE, F),
            dynamic_uniform(9, "visibility_draw", 16, VF),
            read_of(10, "visible_clusters", 8, F),
            texture(11, "visibility_ids", Uint, D2, VF),
            read_of(12, "resolve_tiles", 4, V),
        ],
    };

    /// Group 3 of lit pipelines: the clustered local lights.
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

    /// Group 2 of decals beside the mesh poses' bindings: the
    /// joint palette holds retained frames' decal poses.
    const DECALS: Table = Table {
        label: "engine/decal-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read(3, "palette", V),
            read(13, "decal_records", VF),
            texture(14, "decal_depth", Depth, D2, F),
        ],
    };

    /// Group 1 of impostor cards: the records and the four
    /// arrays of baked frames.
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

    /// Group 2 of impostor bakes: the impostor being baked, and the nearest
    /// depths of its frames.
    const IMPOSTOR_BAKE: Table = Table {
        label: "engine/impostor-bake-layout",
        serves: [SHADER @ 2],
        bindings: &[
            uniform_of(15, "impostor_bake", size_of::<ImpostorBakeUniform>(), VF),
            texture(16, "impostor_shade", Depth, D2, F),
        ],
    };

    // GPU presentation's families (`presentation.wgsl`).

    /// Parent composition, and skinned slots' palettes from the composed
    /// parents.
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
            read(15, "joint_nodes", C),
            read(16, "inverse_binds", C),
            write(17, "palette", C),
        ],
    };

    /// Each depth level's parent range, group 1 of composition.
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

    /// The occlusion test, reflection probe culling and still frame count, and
    /// the previous-pose shift.
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

    /// Group sums, their scan, draw arguments and the scatter of visible slots:
    /// one group per compaction binds all four.
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

    /// Levels of detail kept and faded after culling: each
    /// slot's level as last drawn, and the cards fading over their meshes.
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

    /// Skinned slots' bounds from their palettes, culled.
    const PRESENTATION_SKIN_BOUNDS: Table = Table {
        label: "engine/presentation-skin-bounds-layout",
        serves: [PRESENTATION_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(7, "batches", C),
            write(8, "poses", C),
            write(9, "scan", C),
            write(17, "palette", C),
            read(20, "slot_colors", C),
            read(21, "skinned", C),
            write(22, "skin_bounds", C),
        ],
    };

    // Virtual geometry's selection (`virtual_geometry.wgsl`).
    // Passes dispatching from the dispatch arguments (15) never bind them.

    /// Seeding, and the dispatch arguments of each level, the work and the
    /// retest.
    const VIRTUAL_SEED: Table = Table {
        label: "engine/virtual-seed-layout",
        serves: [VIRTUAL_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(1, "batches", C),
            read(2, "meshes", C),
            read(7, "ids", C),
            read(8, "draws", C),
            write(9, "queue_a", C),
            write(10, "queue_b", C),
            write(12, "counters", C),
            write(15, "dispatch", C),
        ],
    };

    const VIRTUAL_TRAVERSE: Table = Table {
        label: "engine/virtual-traverse-layout",
        serves: [VIRTUAL_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(3, "nodes", C),
            read(6, "poses", C),
            write(9, "queue_a", C),
            write(10, "queue_b", C),
            write(11, "work", C),
            write(12, "counters", C),
        ],
    };

    /// Selection, and the retest of the clusters it left waiting.
    const VIRTUAL_SELECT: Table = Table {
        label: "engine/virtual-select-layout",
        serves: [VIRTUAL_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            read(4, "groups", C),
            read(5, "clusters", C),
            read(6, "poses", C),
            write(11, "work", C),
            write(12, "counters", C),
            write(13, "visible", C),
            read(14, "bins", C),
            texture(19, "depth_pyramid", Unfilterable, D2, C),
            write(20, "retest", C),
        ],
    };

    const VIRTUAL_BIN: Table = Table {
        label: "engine/virtual-bin-layout",
        serves: [VIRTUAL_SHADER @ 0],
        bindings: &[
            uniform(0, "frame", C),
            write(12, "counters", C),
            read(14, "bins", C),
            write(16, "draw_args", C),
        ],
    };

    /// Depth-only drawing of shadow views' bins: the arena, the frame's poses
    /// and the shadow maps' views at dynamic offsets.
    const VIRTUAL_SHADOW: Table = Table {
        label: "engine/virtual-shadow-layout",
        serves: [VIRTUAL_SHADER @ 0],
        bindings: &[
            read(2, "meshes", V),
            read(5, "clusters", V),
            read(6, "poses", V),
            read(17, "pages", V),
            dynamic_uniform(18, "shadow_view", 64, V),
        ],
    };

    // Shadows (`shadows.wgsl`).

    /// Caster selection and counting, with the slot each level's tiles lie
    /// in, which moving casters draw over.
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

    /// Skinned casters by their palette bounds: within eight
    /// storage buffers.
    const SHADOW_SKINNED: Table = Table {
        label: "engine/shadow-skinned-layout",
        serves: [SHADOW_SHADER @ 0],
        bindings: &[
            uniform_of(0, "cull", size_of::<ShadowCullUniform>(), C),
            read(2, "batches", C),
            read(4, "flags", C),
            write(6, "ids", C),
            write(7, "args", C),
            write(8, "counters", C),
            read(10, "skinned", C),
            read(11, "skin_bounds", C),
            texture(12, "tile_slots", Uint, D2, C),
        ],
    };

    /// The shadow clipmap's marks (`shadow_marks.wgsl`): the camera, the
    /// shadow lookups' uniform, the depth buffer and the marks of the tiles
    /// its pixels read.
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

    /// Group 1 of terrain pipelines (`meshes/terrain_vertex.wgsl`):
    /// the height array, each view's eye-relative projection and each
    /// terrain's anchor and material, the uniforms at dynamic offsets.
    /// Fragments read the heights too, where the shadow clipmap's levels
    /// hold the ground at a level of their own, and the tiles' water's
    /// surfaces, where the ground wets at its waterline. Vertices read the
    /// frame's leaves of the terrains' subdivisions and the true surface
    /// under them, and where their materials' mesh band moves it their
    /// blend, normals and heights, which the subdivision's compute passes
    /// measure leaves by alike, with the error pyramids of every tile's
    /// ground and water, which they alone read.
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
            // Each tile's normal maps, gathered through the control
            // values' sampler; and the sky maps' sampler.
            texture(21, "terrain_normals", Uint, D2Array, VFC),
            sampler(22, "terrain_sky_sampler", Filtering, F),
            read(23, "terrain_directory", VFC),
            texture(24, "terrain_control", Uint, D2Array, VFC),
            uniform_of(25, "terrain_palette", terrain::PALETTE_BYTES, VFC),
            // Surface materials' maps, base color and data (each
            // material's normal and packed maps), and their sampler.
            texture(26, "terrain_albedo", Float, D2Array, F),
            texture(27, "terrain_material_data", Float, D2Array, VFC),
            sampler(29, "terrain_material_sampler", Filtering, VFC),
            // Each terrain's sky visibility map.
            texture(30, "terrain_sky", Float, D2Array, F),
            // Runtime deformation's map.
            read(31, "terrain_deformation", F),
            // Gathers of the control values (nearest).
            sampler(32, "terrain_control_sampler", NonFiltering, VFC),
            // Each terrain's macro colors with their mips, read through
            // the materials' sampler: the slot the horizon records' uniform
            // left of the 16 sampled textures a fragment stage binds.
            texture(35, "terrain_macro", Float, D2Array, F),
            // The frame's leaves of every terrain's subdivision.
            read(36, "terrain_surface_leaves", V),
            // Tiles' water's surfaces beside their ground, as water's pass
            // binds them, read through water's directory in the terrain
            // directory: the ground's fragment stages bind fifteen of the
            // sixteen sampled textures a stage may with it, and their lit
            // ones the eight storage buffers.
            texture(50, "water_surface", Unfilterable, D2Array, F),
            // Each tile's error pyramids, its ground's and its water's
            // surface's, a layer each beside its heights' (`atlas.rs`).
            texture(62, "terrain_errors", Unfilterable, D2Array, C),
            texture(63, "water_errors", Unfilterable, D2Array, C),
        ],
    };

    /// Group 2 of terrain's resolves: each pixel's terrain and
    /// depth, as its visibility pass wrote them.
    const TERRAIN_RESOLVE: Table = Table {
        label: "engine/terrain-resolve-layout",
        serves: [SHADER @ 2],
        bindings: &[texture(18, "terrain_ids_texture", Uint, D2, F)],
    };

    /// Group 0 of terrain's subdivision, compute passes over each terrain's
    /// tree (`meshes/terrain_subdivision.wgsl`): the camera.
    const TERRAIN_SUBDIVISION_CAMERA: Table = Table {
        label: "engine/terrain-subdivision-camera-layout",
        serves: [SHADER @ 0],
        bindings: &[uniform_of(0, "camera", size_of::<CameraUniform>(), C)],
    };

    /// Group 2 of terrain's subdivision, compute passes after the camera
    /// (group 0, `TERRAIN_SUBDIVISION_CAMERA`) and the terrain
    /// group (`meshes/terrain_subdivision.wgsl`): each terrain's subdivision
    /// at a dynamic offset, its tree, the frame's leaves' nodes and the
    /// leaves the frame draws, and every terrain's counters, which its
    /// indirect arguments are copied from.
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

    /// The clouds' march into their dome (`clouds.wgsl`): the
    /// layer, sun and sky, the noise's volumes and their sampler, the dome
    /// in both its layers, the physically based sky's uniform,
    /// transmittance table and sampler, and the light map.
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

    /// The view's march of the clouds (`clouds.wgsl`'s
    /// `march_view`): the march from the camera, the noise's volumes and
    /// their sampler, the view, its depth buffer, the marched pixels'
    /// clouds, the composite's draw, the physically based sky's uniform,
    /// transmittance table and sampler, and the light map.
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

    /// Group 1 of the view's clouds' composite (
    /// `meshes/sky.wgsl`): the marched pixels' clouds.
    const CLOUD_COMPOSITE: Table = Table {
        label: "engine/cloud-composite-layout",
        serves: [SHADER @ 1],
        bindings: &[texture(46, "view_clouds", Float, D2, F)],
    };

    /// Group 1 of the sky's pipelines (`meshes/sky.wgsl`): the physically
    /// based sky's transmittance table, through which the sun's disc and
    /// the night sky shine; the sky's other tables and the clouds' domes are
    /// the camera group's.
    const SKY_TRANSMITTANCE: Table = Table {
        label: "engine/sky-transmittance-layout",
        serves: [SHADER @ 1],
        bindings: &[texture(42, "sky_transmittance_table", Float, D2, F)],
    };

    /// The physically based sky's tables and light (
    /// `sky.wgsl`): the sky's uniform, the tables a pass reads, the one it
    /// writes and the light it derives.
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

    /// Runtime deformation's map (`deformation.wgsl`): each job's
    /// uniform at a dynamic offset, the stamps it presses and the map.
    const DEFORMATION: Table = Table {
        label: "engine/deformation-layout",
        serves: [DEFORMATION_SHADER @ 0],
        bindings: &[
            dynamic_uniform(0, "job", size_of::<deformation::JobUniform>(), C),
            read(1, "stamps", C),
            write(2, "map", C),
        ],
    };

    /// A tile's normal map (`terrain_normals.wgsl`): the height
    /// array, the job's layer and spacing, and the layer of the normal array
    /// written.
    const TERRAIN_NORMALS: Table = Table {
        label: "engine/terrain-normals-layout",
        serves: [TERRAIN_NORMALS_SHADER @ 0],
        bindings: &[
            texture(0, "heights", Unfilterable, D2Array, C),
            uniform_of(1, "job", 16, C),
            bind(2, "normals", Kind::Storage(R32Uint), C),
        ],
    };

    /// A tile's control values from its terrain's surface (
    /// `terrain_control.wgsl`): the height array, the job's uniform, the
    /// surface's data, the control layer it writes and the array of tiles'
    /// rivers: their samples' meters to rivers' edges and the rivers' beds.
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

    /// Terrain bounce light (`terrain_bounce.wgsl`, after the scene's
    /// declarations, whose numbers the bindings it shares keep): the scene's
    /// lighting, the shadow lookups' uniform (the horizon records' count),
    /// the horizon maps' own layers with their records and sampler and the
    /// clouds' shadows, as the sun's lookups read them; the bins' moments
    /// and each terrain's bounce at a level of its probes this frame, the
    /// terrains' sky maps, their cooked ground and probes, the ground's
    /// light read (a level, or all of them) and written (a level), the
    /// bounce layers of the horizon maps written, the light's sampler, the
    /// terrain palette, the terrains' macro colors, and the probes each
    /// level relights, which the next coarser one averages.
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

    /// Ground cover's growth (`cover_grow.wgsl`): terrain's
    /// height, control and normal arrays, directory, palette and covers, each
    /// terrain's growth at a dynamic offset, the blades and their counts, and
    /// the indirect draw's arguments.
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
            // Runtime deformation's map: blades lie down in
            // tracks.
            read(9, "deformation", C),
            // The terrains' macro colors: they tint blades as they do the
            // ground.
            texture(10, "macro_maps", Float, D2Array, C),
            // The normal maps: blades stand along the ground's smooth
            // surface, as it shades.
            texture(11, "normals", Uint, D2Array, C),
        ],
    };

    /// Group 1 of water's pipelines (`meshes/water.wgsl`): terrain's height
    /// array and the views' and terrains' uniforms at dynamic offsets, its
    /// directory, with water's after it, and this frame's leaves of its
    /// subdivisions, as the terrain group binds them; tiles' water blocks,
    /// their surfaces beside their ground and their words (packed flow,
    /// whitewater and kind, their state, and the level the water's mesh
    /// stands at, which its vertices read), which pixels read as the water's
    /// smooth surface and where it lies; water's uniform; the live water's
    /// cells' looks, its water as water's pass draws it in the layout of
    /// tiles' samples, which vertices and pixels read where it runs; and at
    /// a dynamic offset, the subdivision of each terrain's water, whose
    /// leaves its vertices take. With water's first group (`CAMERA_WATER`)
    /// and what lies behind the water, within a fragment stage's 16 sampled
    /// textures and 8 storage buffers. The sea's group has placeholders for
    /// what only tiles' water reads. Water's particles' passes
    /// (`WATER_PARTICLES`) read the water through it as shading reads it
    /// (`water_pixel`).
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

    /// Group 2 of water's pipelines: what lies behind the water, copied
    /// from the opaque scene before it draws: the depth, and the color in
    /// frames drawing the effects' targets (a placeholder in others); and
    /// what water's particles splat (`crate::water_particles`), the shares
    /// white water and foam cover and the waves' slopes, and those slopes'
    /// squares, with their coarser levels, and the sampler repeating them
    /// (placeholders where none run), which the particles' passes write
    /// outside it: 15 of a fragment stage's 16 sampled textures with water's
    /// first group's and water's (`CAMERA_WATER`, `WATER`).
    const WATER_BEHIND: Table = Table {
        label: "engine/water-behind-layout",
        serves: [SHADER @ 2],
        bindings: &[
            texture(22, "water_depth", Depth, D2, F),
            texture(23, "water_scene", Float, D2, F),
            texture(24, "water_particles", Float, D2, F),
            sampler(25, "water_particles_sampler", Filtering, F),
            texture(26, "water_particles_spread", Float, D2, F),
        ],
    };

    /// Group 2 of water's particles' passes
    /// (`meshes/water_particles.wgsl`), after an empty first group and the
    /// water's (`WATER`), through which they read the water as its shading
    /// does: their frame, their lattice, their pools twice over (the frame
    /// before's and this frame's), their counts, their splat's sums, and a
    /// level of each of the two textures water's shading reads (written) and
    /// the level finer than it (read): six of a compute stage's 8 storage
    /// buffers with the water's two, and two of its 4 storage textures.
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

    /// Water's simulation (`water_sim.wgsl`): the stage's frame, the
    /// ground and lakes' levels of the blocks the slots store, their cells'
    /// two states, their faces, the fluxes through them, the cells rivers
    /// bring water to and those just past the squares' edges as their water
    /// settled, the slots' tables of blocks, each cell's waves and seconds,
    /// the slots' squares, the sums settling squares keep (the live square's
    /// cells' looks in their place), and the solids
    /// standing in the live square's water: its eight storage buffers, the
    /// most a compute stage binds, and three uniforms.
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

    /// Group 2 of the live water's bed (`meshes/water_bed.wgsl`), after an
    /// empty first group and the terrain group (`TERRAIN`), through which
    /// it reads the ground's materials as their mesh band does: the blocks
    /// whose bed a frame raises and the meters across their cells, each
    /// one's slot, first corner and datum, the live water's ground it raises
    /// (the simulation's `sim_ground`) and the ground each slot's block was
    /// taken with, which it rises from: five of a compute stage's 8 storage
    /// buffers with the terrain group's two.
    const WATER_BED: Table = Table {
        label: "engine/water-bed-layout",
        serves: [SHADER @ 2],
        bindings: &[
            uniform_of(41, "bed_frame", size_of::<water::SimBed>(), C),
            read(42, "bed_taken", C),
            write(43, "bed_ground", C),
            read(44, "bed_kept", C),
        ],
    };

    /// Group 1 of ground cover's pipelines (`meshes/cover.wgsl`):
    /// the frame's blades and the wind and times they sway with.
    const COVER_DRAW: Table = Table {
        label: "engine/cover-layout",
        serves: [SHADER @ 1],
        bindings: &[
            read(33, "cover_blades", V),
            uniform_of(34, "cover_frame", size_of::<ground_cover::FrameUniform>(), V),
        ],
    };

    /// Placed objects' culling (`scatter_cull.wgsl`): the frame,
    /// every object's record and every draw group, each view's visible lists
    /// and the indirect draws they count in, the views, and the camera's
    /// impostor cards and their draw.
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

    /// Placed objects set on the terrain as drawn (
    /// `scatter_snap.wgsl`): the slots and terrains, every object's record and
    /// pose, each terrain's place in the directory, the directory and the
    /// resident heights; and every object's ground, whose color it writes
    /// from the resident control values, the terrain palette and the
    /// terrains' macro colors.
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

    /// Occupancy (`occupancy.wgsl`), one table for its selection and its
    /// draws, whose groups fill what each leaves alone: the map's window,
    /// cells redrawn and counts, the placed objects' records, poses and
    /// draw groups' covers; the footprints listed, which the selection
    /// writes with their draw's arguments and the draws read; and the finer
    /// map's coverage and heights, whose means the map takes where they
    /// overlap.
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

    /// Group 0 of depth passes: each view's matrix at a dynamic offset, with
    /// the wind and the eye placed objects sway by; and the routes moving
    /// casters draw through into this frame's moving tiles.
    const SHADOW_VIEW: Table = Table {
        label: "engine/shadow-view-layout",
        serves: [SHADOW_SHADER @ 0, TERRAIN_SHADOW_SHADER @ 0],
        bindings: &[
            dynamic_uniform(9, "view", 96, V),
            read_of(13, "routes", size_of::<RouteData>(), VF),
        ],
    };

    /// Group 1 of retained depth passes: slot poses, and slot colors and joint
    /// palettes for masked and skinned casters; and the regions of tiles
    /// this frame redrew, which composing reads into the atlas (a
    /// placeholder in passes drawing into them).
    const SHADOW_POSES: Table = Table {
        label: "engine/shadow-pose-layout",
        serves: [SHADOW_SHADER @ 1],
        bindings: &[
            read_of(0, "slot_poses", POSE, V),
            read_of(1, "slot_colors", COLOR, V),
            read_of(2, "slot_palette", POSE, V),
            texture(3, "shadow_regions", Depth, D2Array, F),
        ],
    };

    // Other passes.

    /// Light binning (`lights.wgsl`).
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

    /// The screen effects (`post.wgsl`), whose passes each read
    /// some of these. Textures the passes sample with the linear sampler
    /// filter; the others only load.
    const POST: Table = Table {
        label: "engine/post-layout",
        serves: [POST_SHADER @ 0],
        bindings: &[
            uniform(0, "post", F),
            texture(1, "depth_texture", Depth, D2, F),
            // Scene, glow, a level's source, filtered occlusion, bloom.
            texture(2, "scene", Float, D2, F),
            texture(3, "glow", Float, D2, F),
            texture(4, "source", Float, D2, F),
            texture(5, "occlusion", Float, D2, F),
            texture(6, "bloom", Float, D2, F),
            sampler(7, "linear", Filtering, F),
            texture(8, "reflections", Float, D2, F),
            // The haze's march and raw occlusion.
            texture(9, "marched", Unfilterable, D2, F),
            texture(12, "raw_occlusion", Unfilterable, D2, F),
            // A bloom level halved beside a level of the scene.
            texture(15, "bloom_source", Float, D2, F),
            // Sky occlusion's maps, their comparison sampler and where each
            // looks, which the occlusion pass reads.
            texture(16, "sky_maps", Depth, D2Array, F),
            sampler(17, "sky_sampler", Comparison, F),
            uniform_of(18, "sky", size_of::<SkyUniform>(), F),
            // The haze filtered over time: the history its filter
            // reprojects, or this frame's, which the composite lays over the
            // scene; the twelfth sampled texture.
            texture(19, "haze", Float, D2, F),
        ],
    };

    /// Group 1 of reflections (`reflections.wgsl`), after the main passes'
    /// first group as terrain binds it (`CAMERA_TERRAIN`): the screen's
    /// part, the effects' uniform, the depth buffer, the glow target, the
    /// linear sampler, the pyramid of nearest depths, the surface's packed
    /// words and the scene with its levels; and what the world trace walks,
    /// the terrain pass's heights with their peaks in their mips,
    /// directory, control values, palette, macro colors and sky maps, the
    /// trace's terrains and placements, the placed objects' records, poses
    /// and draw groups, and the placed objects' impostor atlas, every layer
    /// undecoded. With the first group, sixteen sampled textures and eight
    /// storage buffers: the atlas takes the texture the peaks took before
    /// they moved into the heights' mips.
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

    /// Group 1 of the haze's march (`haze.wgsl`).
    const HAZE: Table = Table {
        label: "engine/post-haze-layout",
        serves: [HAZE_SHADER @ 1],
        bindings: &[
            uniform_of(0, "post", size_of::<PostUniform>(), F),
            texture(1, "depth_texture", Depth, D2, F),
        ],
    };

    /// Temporal antialiasing's resolve (`temporal.wgsl`).
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
            // The nearest surface around each pixel, and what particles
            // leave visible.
            texture(9, "depth_texture", Depth, D2, F),
            texture(10, "revealed", Unfilterable, D2, F),
            // The haze filtered over time, which the canvas alone shows.
            texture(11, "haze", Float, D2, F),
        ],
    };

    /// The transparent composite (`transparent.wgsl`).
    const TRANSPARENT_COMPOSITE: Table = Table {
        label: "engine/transparent-composite-layout",
        serves: [TRANSPARENT_SHADER @ 0],
        bindings: &[
            texture(0, "accumulated", Unfilterable, D2, F),
            texture(1, "revealed", Unfilterable, D2, F),
        ],
    };

    /// Auto exposure (`exposure.wgsl`): its settings, the scene color and
    /// depth it meters, its histogram, its state and the scene's lighting.
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

    /// The debug views' pass (`debug_view.wgsl`): the view, and what drew
    /// each pixel with the fragments there and terrain's tiles and
    /// materials.
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

    /// Group 1 of particles (`particles.wgsl`).
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

    /// The environment maps' sky (`environment.wgsl`): the lighting, and
    /// what the sky's light along a view reads (`sky_light.wgsl`), at the
    /// camera group's numbers.
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

    /// Each environment level down from the level above it.
    const ENVIRONMENT_DOWN: Table = Table {
        label: "engine/environment-down-layout",
        serves: [ENVIRONMENT_SHADER @ 0],
        bindings: &[
            texture(20, "level_source", Float, Cube, F),
            sampler(22, "cube_sampler", Filtering, F),
        ],
    };

    /// The prefilter of every level.
    const ENVIRONMENT_PREFILTER: Table = Table {
        label: "engine/environment-prefilter-layout",
        serves: [ENVIRONMENT_SHADER @ 0],
        bindings: &[
            texture(21, "radiance", Float, Cube, F),
            sampler(22, "cube_sampler", Filtering, F),
        ],
    };

    /// Texture mipmaps (`mipmaps.wgsl`).
    const MIPMAPS: Table = Table {
        label: "engine/texture-mip-layout",
        serves: [MIPMAP_SHADER @ 0],
        bindings: &[
            texture(0, "source", Float, D2, F),
            sampler(1, "source_sampler", Filtering, F),
        ],
    };

    /// The depth pyramid's first level from the depth buffer (
    /// `depth_pyramid.wgsl`).
    const PYRAMID_FIRST: Table = Table {
        label: "engine/depth-pyramid-first-layout",
        serves: [DEPTH_PYRAMID_SHADER @ 0],
        bindings: &[
            texture(0, "source_depth", Depth, D2, C),
            bind(2, "target_level", Kind::Storage(R32Float), C),
            uniform_of(3, "extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };

    /// Each further level from the one above it.
    const PYRAMID_DOWN: Table = Table {
        label: "engine/depth-pyramid-down-layout",
        serves: [DEPTH_PYRAMID_SHADER @ 0],
        bindings: &[
            texture(1, "source_level", Unfilterable, D2, C),
            bind(2, "target_level", Kind::Storage(R32Float), C),
            uniform_of(3, "extent", PYRAMID_EXTENT_BYTES, C),
        ],
    };

    /// The visibility buffer's tile classification (
    /// `visibility.wgsl`).
    const VISIBILITY_CLASSIFY: Table = Table {
        label: "engine/visibility-classify-layout",
        serves: [VISIBILITY_SHADER @ 0],
        bindings: &[
            uniform(0, "classify", C),
            texture(1, "ids", Uint, D2, C),
            write(2, "counts", C),
            write(3, "tiles", C),
            write(4, "draw_args", C),
        ],
    };
}
