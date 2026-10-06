//! Bind group layouts as data (decision 049): every layout the renderer
//! builds is a table of the bindings its shaders read, each with its
//! number, WGSL name, kind and the stages reading it. Layouts and bind
//! groups are built from the tables (`crate::pipelines::layout` and
//! `group`), so a binding's number is written once in WGSL and once here,
//! groups bind resources by name, and a test reflects every shader to check
//! each table against the entry points of the pipelines using it.

use crate::gpu_data::{
    CameraUniform, ColorData, LightingUniform, MaterialUniform, PoseData, PostUniform,
    ResolveUniform, ShadowCullUniform, ShadowUniform,
};
use crate::impostors::{ImpostorBakeUniform, ImpostorRecord};

/// Shader stages, as bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stages(u8);

impl Stages {
    pub(crate) const VERTEX: Self = Self(1);
    pub(crate) const FRAGMENT: Self = Self(2);
    pub(crate) const COMPUTE: Self = Self(4);
    pub(crate) const VERTEX_FRAGMENT: Self = Self(3);

    pub(crate) fn vertex(self) -> bool {
        self.0 & 1 != 0
    }

    pub(crate) fn fragment(self) -> bool {
        self.0 & 2 != 0
    }

    pub(crate) fn compute(self) -> bool {
        self.0 & 4 != 0
    }
}

/// How shaders read a texture's texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sample {
    /// Floats, filtered by samplers.
    Float,
    /// Floats shaders only load.
    Unfilterable,
    Depth,
    Uint,
}

/// A texture view's dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum View {
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// How a sampler filters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sampling {
    Filtering,
    NonFiltering,
    Comparison,
}

/// What a binding holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A uniform buffer of at least `min_size` bytes (0 leaves it to
    /// draws), bound at a dynamic offset when `dynamic`.
    Uniform {
        min_size: u64,
        dynamic: bool,
    },
    /// A storage buffer shaders only read, of at least `min_size` bytes.
    Read {
        min_size: u64,
    },
    /// A storage buffer shaders write.
    Write,
    Texture {
        sample: Sample,
        view: View,
    },
    /// A 2D texture of this format shaders write.
    Storage(wgpu::TextureFormat),
    /// A 3D texture of this format shaders write.
    Storage3d(wgpu::TextureFormat),
    Sampler(Sampling),
}

/// One binding of a layout.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Binding {
    pub(crate) binding: u32,
    pub(crate) name: &'static str,
    pub(crate) kind: Kind,
    pub(crate) stages: Stages,
}

/// A bind group layout: its label and bindings.
#[derive(Debug)]
pub(crate) struct Table {
    pub(crate) label: &'static str,
    pub(crate) bindings: &'static [Binding],
}

impl Table {
    /// The binding named `name`; a name the table lacks is a programming
    /// error.
    pub(crate) fn binding(&self, name: &str) -> &Binding {
        self.bindings
            .iter()
            .find(|binding| binding.name == name)
            .unwrap_or_else(|| panic!("{} has no binding named {name}", self.label))
    }
}

const fn bind(binding: u32, name: &'static str, kind: Kind, stages: Stages) -> Binding {
    Binding {
        binding,
        name,
        kind,
        stages,
    }
}

const fn uniform(binding: u32, name: &'static str, stages: Stages) -> Binding {
    bind(
        binding,
        name,
        Kind::Uniform {
            min_size: 0,
            dynamic: false,
        },
        stages,
    )
}

const fn uniform_of(binding: u32, name: &'static str, size: usize, stages: Stages) -> Binding {
    bind(
        binding,
        name,
        Kind::Uniform {
            min_size: size as u64,
            dynamic: false,
        },
        stages,
    )
}

const fn dynamic_uniform(binding: u32, name: &'static str, size: u64, stages: Stages) -> Binding {
    bind(
        binding,
        name,
        Kind::Uniform {
            min_size: size,
            dynamic: true,
        },
        stages,
    )
}

const fn read(binding: u32, name: &'static str, stages: Stages) -> Binding {
    bind(binding, name, Kind::Read { min_size: 0 }, stages)
}

const fn read_of(binding: u32, name: &'static str, size: usize, stages: Stages) -> Binding {
    bind(
        binding,
        name,
        Kind::Read {
            min_size: size as u64,
        },
        stages,
    )
}

const fn write(binding: u32, name: &'static str, stages: Stages) -> Binding {
    bind(binding, name, Kind::Write, stages)
}

const fn texture(
    binding: u32,
    name: &'static str,
    sample: Sample,
    view: View,
    stages: Stages,
) -> Binding {
    bind(binding, name, Kind::Texture { sample, view }, stages)
}

const fn sampler(binding: u32, name: &'static str, sampling: Sampling, stages: Stages) -> Binding {
    bind(binding, name, Kind::Sampler(sampling), stages)
}

use Sample::{Depth, Float, Uint, Unfilterable};
use Sampling::{Comparison, Filtering, NonFiltering};
use View::{Cube, CubeArray, D2, D2Array, D3};
use wgpu::TextureFormat::{R32Float, Rgba8Snorm, Rgba8Unorm, Rgba16Float};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;
// Bytes of a pose, a slot color, and of virtual geometry's mesh, cluster
// and page records (decision 044).
const POSE: usize = size_of::<PoseData>();
const COLOR: usize = size_of::<ColorData>();
const VIRTUAL_MESH: usize = 80;
const VIRTUAL_CLUSTER: usize = 48;
const VIRTUAL_PAGE: usize = 16;

/// A group binding nothing, standing in for groups a pipeline skips.
pub(crate) const EMPTY: Table = Table {
    label: "engine/empty-layout",
    bindings: &[],
};

// The mesh shader (`gpu_data::SHADER`).

/// Group 0 of mesh, sky and effect pipelines: the camera, lighting and
/// shadow lookups (decisions 020 and 028), what glass refracts (decision
/// 038), the environment maps and reflection probes (decision 043).
pub(crate) const CAMERA: Table = Table {
    label: "engine/camera-layout",
    bindings: &[
        // Fragments read the eye, sky rays and the jitter.
        uniform_of(0, "camera", size_of::<CameraUniform>(), VF),
        uniform_of(1, "lighting", size_of::<LightingUniform>(), F),
        uniform_of(2, "shadow", size_of::<ShadowUniform>(), F),
        texture(3, "near_shadows", Depth, D2Array, F),
        texture(4, "kept_shadows", Depth, D2, F),
        texture(5, "moving_shadows", Depth, D2, F),
        sampler(6, "shadow_sampler", Comparison, F),
        // Receiver-plane lookups read four texels (decision 028).
        sampler(7, "shadow_texels", NonFiltering, F),
        texture(8, "refraction_source", Float, D2, F),
        sampler(9, "refraction_sampler", Filtering, F),
        texture(10, "environment_maps", Float, CubeArray, F),
        sampler(11, "environment_sampler", Filtering, F),
        read(12, "probe_words", F),
        // Terrain's long sun shadows (decision 052): each terrain's horizon
        // map, two texels of records per terrain, and the maps' sampler.
        texture(13, "terrain_horizons", Float, D2Array, F),
        texture(14, "terrain_horizon_records", Unfilterable, D2, F),
        sampler(15, "terrain_horizon_sampler", Filtering, F),
        // What placed objects cover around the camera (decision 059): its
        // density, filtered, and the covers' tops and bottoms.
        texture(16, "occupancy_coverage", Float, D2, F),
        texture(17, "occupancy_heights", Unfilterable, D2, F),
        // The kept shadow levels (decision 065), one layer each.
        texture(18, "kept_levels", Depth, D2Array, F),
        // Clouds' shadows (decision 084): the eighth storage buffer the
        // terrain's lit fragments read (a uniform array read at each
        // fragment's own index is many times slower).
        read_of(19, "cloud_shadows", crate::clouds::CLOUD_SHADOW_BYTES, F),
    ],
};

/// Group 1 of mesh pipelines: a material's maps and their samplers and its
/// map parameters (decision 031); shadow casters bind it as group 2.
pub(crate) const MATERIAL: Table = Table {
    label: "engine/material-layout",
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

/// Group 2 of retained mesh pipelines: slot poses, colors and the joint
/// palettes of skinned slots (decision 036).
pub(crate) const RETAINED_POSES: Table = Table {
    label: "engine/retained-pose-layout",
    bindings: &[
        read_of(0, "poses", POSE, V),
        read_of(1, "colors", COLOR, V),
        read_of(3, "palette", POSE, V),
    ],
};

/// Group 2 of temporal mesh pipelines (decision 017): the retained poses
/// and the previous frame's poses and palettes.
pub(crate) const TEMPORAL_POSES: Table = Table {
    label: "engine/temporal-pose-layout",
    bindings: &[
        read_of(0, "poses", POSE, V),
        read_of(1, "colors", COLOR, V),
        read_of(2, "previous_poses", POSE, V),
        read_of(3, "palette", POSE, V),
        read_of(4, "previous_palette", POSE, V),
    ],
};

/// Group 2 of virtual geometry's pipelines (decision 044): poses, colors,
/// previous poses and the arena's meshes, clusters and pages.
pub(crate) const VIRTUAL_POSES: Table = Table {
    label: "engine/virtual-pose-layout",
    bindings: &[
        read_of(0, "poses", POSE, V),
        read_of(1, "colors", COLOR, V),
        read_of(2, "previous_poses", POSE, V),
        read_of(5, "virtual_meshes", VIRTUAL_MESH, V),
        read_of(6, "virtual_clusters", VIRTUAL_CLUSTER, V),
        read_of(7, "virtual_pages", VIRTUAL_PAGE, V),
    ],
};

/// Group 2 of the visibility pass (decision 045): poses, the arena and each
/// draw's bin through a dynamic offset.
pub(crate) const VISIBILITY_PASS: Table = Table {
    label: "engine/visibility-layout",
    bindings: &[
        read_of(0, "poses", POSE, V),
        read_of(5, "virtual_meshes", VIRTUAL_MESH, V),
        read_of(6, "virtual_clusters", VIRTUAL_CLUSTER, V),
        read_of(7, "virtual_pages", VIRTUAL_PAGE, V),
        dynamic_uniform(8, "visibility_draw", 16, V),
    ],
};

/// Group 2 of the visibility buffer's resolves (decision 045): the poses,
/// arena and draw, the selected clusters, the ID target and the tile lists.
pub(crate) const VISIBILITY_RESOLVE: Table = Table {
    label: "engine/visibility-resolve-layout",
    bindings: &[
        read_of(0, "poses", POSE, F),
        read_of(1, "colors", COLOR, F),
        read_of(2, "previous_poses", POSE, F),
        read_of(5, "virtual_meshes", VIRTUAL_MESH, F),
        read_of(6, "virtual_clusters", VIRTUAL_CLUSTER, F),
        read_of(7, "virtual_pages", VIRTUAL_PAGE, F),
        dynamic_uniform(8, "visibility_draw", 16, VF),
        read_of(9, "visible_clusters", 8, F),
        texture(10, "visibility_ids", Uint, D2, VF),
        read_of(11, "resolve_tiles", 4, V),
    ],
};

/// Group 3 of lit pipelines: the clustered local lights (decision 023).
pub(crate) const LIGHT_SHADING: Table = Table {
    label: "engine/light-shading-layout",
    bindings: &[
        uniform(0, "clusters", F),
        read(1, "lights", F),
        read(2, "cluster_ranges", F),
        read(3, "cluster_lights", F),
    ],
};

/// Group 2 of decals (decision 039) beside the mesh poses' bindings: the
/// joint palette holds retained frames' decal poses.
pub(crate) const DECALS: Table = Table {
    label: "engine/decal-layout",
    bindings: &[
        read(3, "palette", V),
        read(8, "decal_records", VF),
        texture(9, "decal_depth", Depth, D2, F),
    ],
};

/// Group 1 of impostor cards (decision 048): the records and the four
/// arrays of baked frames.
pub(crate) const IMPOSTOR_DRAW: Table = Table {
    label: "engine/impostor-layout",
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
/// depths of its frames (decision 060).
pub(crate) const IMPOSTOR_BAKE: Table = Table {
    label: "engine/impostor-bake-layout",
    bindings: &[
        uniform_of(20, "impostor_bake", size_of::<ImpostorBakeUniform>(), VF),
        texture(21, "impostor_shade", Depth, D2, F),
    ],
};

// GPU presentation's families (decision 014; `presentation.wgsl`).

/// Parent composition, and skinned slots' palettes from the composed
/// parents.
pub(crate) const PRESENTATION_COMPOSE: Table = Table {
    label: "engine/presentation-compose-layout",
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
pub(crate) const PRESENTATION_LEVEL: Table = Table {
    label: "engine/presentation-level-layout",
    bindings: &[uniform(0, "level", C)],
};

pub(crate) const PRESENTATION_CULL: Table = Table {
    label: "engine/presentation-cull-layout",
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
pub(crate) const PRESENTATION_TESTS: Table = Table {
    label: "engine/presentation-test-layout",
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
pub(crate) const PRESENTATION_COMPACT: Table = Table {
    label: "engine/presentation-compact-layout",
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

/// Skinned slots' bounds (decision 036) from their palettes, culled.
pub(crate) const PRESENTATION_SKIN_BOUNDS: Table = Table {
    label: "engine/presentation-skin-bounds-layout",
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

// Virtual geometry's selection (decision 044; `virtual_geometry.wgsl`).
// Passes dispatching from the dispatch arguments (15) never bind them.

/// Seeding, and the dispatch arguments of each level, the work and the
/// retest.
pub(crate) const VIRTUAL_SEED: Table = Table {
    label: "engine/virtual-seed-layout",
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

pub(crate) const VIRTUAL_TRAVERSE: Table = Table {
    label: "engine/virtual-traverse-layout",
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
pub(crate) const VIRTUAL_SELECT: Table = Table {
    label: "engine/virtual-select-layout",
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

pub(crate) const VIRTUAL_BIN: Table = Table {
    label: "engine/virtual-bin-layout",
    bindings: &[
        uniform(0, "frame", C),
        write(12, "counters", C),
        read(14, "bins", C),
        write(16, "draw_args", C),
    ],
};

/// Depth-only drawing of shadow views' bins: the arena, the frame's poses
/// and the shadow maps' views at dynamic offsets.
pub(crate) const VIRTUAL_SHADOW: Table = Table {
    label: "engine/virtual-shadow-layout",
    bindings: &[
        read(2, "meshes", V),
        read(5, "clusters", V),
        read(6, "poses", V),
        read(17, "pages", V),
        dynamic_uniform(18, "shadow_view", 64, V),
    ],
};

// Shadows (decision 020; `shadows.wgsl`).

/// Caster selection and counting.
pub(crate) const SHADOW_SELECT: Table = Table {
    label: "engine/shadow-select-layout",
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
    ],
};

/// Skinned casters (decision 036) by their palette bounds: within eight
/// storage buffers.
pub(crate) const SHADOW_SKINNED: Table = Table {
    label: "engine/shadow-skinned-layout",
    bindings: &[
        uniform_of(0, "cull", size_of::<ShadowCullUniform>(), C),
        read(2, "batches", C),
        read(4, "flags", C),
        write(6, "ids", C),
        write(7, "args", C),
        write(8, "counters", C),
        read(10, "skinned", C),
        read(11, "skin_bounds", C),
    ],
};

/// Group 1 of terrain pipelines (decision 051; `meshes/terrain_vertex.wgsl`):
/// the height array, each view's eye-relative projection and each
/// terrain's anchor and material, the uniforms at dynamic offsets.
pub(crate) const TERRAIN: Table = Table {
    label: "engine/terrain-layout",
    bindings: &[
        texture(17, "terrain_heights", Unfilterable, D2Array, V),
        dynamic_uniform(18, "terrain_view", 80, V),
        dynamic_uniform(
            19,
            "terrain",
            crate::terrain::TERRAIN_UNIFORM_BYTES as u64,
            VF,
        ),
        read(20, "terrain_holes", F),
        texture(21, "terrain_normals", Float, D2Array, F),
        sampler(22, "terrain_normal_sampler", Filtering, F),
        read(23, "terrain_directory", F),
        texture(24, "terrain_control", Uint, D2Array, F),
        uniform_of(25, "terrain_palette", crate::terrain::PALETTE_BYTES, F),
        // Surface materials' maps (decision 054) and their sampler.
        texture(26, "terrain_albedo", Float, D2Array, F),
        texture(27, "terrain_material_normals", Float, D2Array, F),
        texture(28, "terrain_orm", Float, D2Array, F),
        sampler(29, "terrain_material_sampler", Filtering, F),
        // Each terrain's sky visibility map (decision 055), read through
        // the normals' sampler.
        texture(30, "terrain_sky", Float, D2Array, F),
        // Runtime deformation's map (decision 083).
        read(37, "terrain_deformation", F),
    ],
};

/// The clouds' noise volume (decision 084; `clouds.wgsl`), made once.
pub(crate) const CLOUD_NOISE: Table = Table {
    label: "engine/cloud-noise-layout",
    bindings: &[bind(4, "noise_out", Kind::Storage3d(Rgba8Unorm), C)],
};

/// The clouds' march into their dome (decision 084; `clouds.wgsl`): the
/// layer, sun and sky, the noise volume and its sampler, and the dome.
pub(crate) const CLOUD_MARCH: Table = Table {
    label: "engine/cloud-march-layout",
    bindings: &[
        uniform_of(0, "clouds", crate::clouds::CLOUD_UNIFORM_BYTES, C),
        texture(1, "noise_volume", Float, D3, C),
        sampler(2, "noise_sampler", Filtering, C),
        bind(3, "dome", Kind::Storage(Rgba16Float), C),
        write(5, "shadow_cells", C),
    ],
};

/// Group 1 of the sky's pipelines (decision 084; `meshes/sky.wgsl`): the
/// clouds' dome and its sampler.
pub(crate) const CLOUD_SKY: Table = Table {
    label: "engine/cloud-sky-layout",
    bindings: &[
        texture(38, "sky_clouds", Float, D2, F),
        sampler(39, "sky_cloud_sampler", Filtering, F),
    ],
};

/// Runtime deformation's map (decision 083; `deformation.wgsl`): each job's
/// uniform at a dynamic offset, the stamps it presses and the map.
pub(crate) const DEFORMATION: Table = Table {
    label: "engine/deformation-layout",
    bindings: &[
        dynamic_uniform(0, "job", 48, C),
        read(1, "stamps", C),
        write(2, "map", C),
    ],
};

/// A tile's normal map (decision 051; `terrain_normals.wgsl`): the height
/// array, the job's layer and spacing, and the layer of the normal array
/// written.
pub(crate) const TERRAIN_NORMALS: Table = Table {
    label: "engine/terrain-normals-layout",
    bindings: &[
        texture(0, "heights", Unfilterable, D2Array, C),
        uniform_of(1, "job", 16, C),
        bind(2, "normals", Kind::Storage(Rgba8Snorm), C),
    ],
};

/// A terrain horizon map's build (decision 052; `terrain_horizon.wgsl`):
/// the height array, which layer holds each source tile, the build's
/// uniform and the layer it writes.
pub(crate) const TERRAIN_HORIZON: Table = Table {
    label: "engine/terrain-horizon-layout",
    bindings: &[
        texture(0, "heights", Unfilterable, D2Array, C),
        read(1, "tile_layers", C),
        uniform_of(2, "build", 64, C),
        bind(3, "horizon", Kind::Storage(Rgba16Float), C),
    ],
};

/// A terrain sky visibility map's build (decision 055; `terrain_sky.wgsl`):
/// the height array, which layer holds each source tile, the build's
/// uniform and the layer it writes.
pub(crate) const TERRAIN_SKY: Table = Table {
    label: "engine/terrain-sky-layout",
    bindings: &[
        texture(0, "heights", Unfilterable, D2Array, C),
        read(1, "tile_layers", C),
        uniform_of(2, "build", 48, C),
        bind(3, "sky", Kind::Storage(Rgba8Unorm), C),
    ],
};

/// Ground cover's growth (decision 056; `cover_grow.wgsl`): terrain's
/// height and control arrays, directory, palette and covers, each
/// terrain's growth at a dynamic offset, the blades and their counts, and
/// the indirect draw's arguments.
pub(crate) const COVER_GROW: Table = Table {
    label: "engine/cover-grow-layout",
    bindings: &[
        texture(0, "heights", Unfilterable, D2Array, C),
        texture(1, "control", Uint, D2Array, C),
        read(2, "directory", C),
        uniform_of(3, "palette", crate::terrain::PALETTE_BYTES, C),
        read(4, "covers", C),
        dynamic_uniform(5, "grow", crate::ground_cover::GROW_BYTES, C),
        write(6, "blades", C),
        write(7, "counts", C),
        write(8, "args", C),
        // Runtime deformation's map (decision 083): blades lie down in
        // tracks.
        read(9, "deformation", C),
    ],
};

/// Group 1 of ground cover's pipelines (decision 056; `meshes/cover.wgsl`):
/// the frame's blades and the wind and times they sway with.
/// Group 1 of water's pipelines (decision 067; `meshes/water.wgsl`): one
/// terrain's water surface heights, its depth and flow, and its uniform;
/// the sea's (decision 076) has placeholder textures.
pub(crate) const WATER: Table = Table {
    label: "engine/water-layout",
    bindings: &[
        texture(33, "water_surface", Unfilterable, D2, V),
        texture(34, "water_info", Unfilterable, D2, V),
        uniform_of(35, "water", crate::water::WATER_UNIFORM_BYTES, VF),
    ],
};

/// Group 2 of water's pipelines (decision 074): a copy of the opaque
/// scene's depth, the ground behind the water.
pub(crate) const WATER_DEPTH: Table = Table {
    label: "engine/water-depth-layout",
    bindings: &[texture(36, "water_depth", Depth, D2, F)],
};

pub(crate) const COVER_DRAW: Table = Table {
    label: "engine/cover-layout",
    bindings: &[
        read(31, "cover_blades", V),
        uniform_of(32, "cover_frame", 16, V),
    ],
};

/// Placed objects' culling (decision 057; `scatter_cull.wgsl`): the frame,
/// every object's record and every draw group, each view's visible lists
/// and the indirect draws they count in, the views, and the camera's
/// impostor cards and their draw.
pub(crate) const SCATTER_CULL: Table = Table {
    label: "engine/scatter-cull-layout",
    bindings: &[
        uniform_of(0, "frame", 48, C),
        read(1, "records", C),
        read(2, "groups", C),
        write(3, "ids", C),
        write(4, "draws", C),
        read(5, "views", C),
        write(6, "cards", C),
        write(7, "card_args", C),
    ],
};

/// Placed objects set on the terrain as drawn (decision 057;
/// `scatter_snap.wgsl`): the slots and terrains, every object's record and
/// pose, each terrain's place in the directory, the directory and the
/// resident heights.
pub(crate) const SCATTER_SNAP: Table = Table {
    label: "engine/scatter-snap-layout",
    bindings: &[
        uniform_of(0, "snap", 16, C),
        write(1, "records", C),
        write(2, "poses", C),
        read(3, "terrains", C),
        read(4, "directory", C),
        texture(5, "heights", Unfilterable, D2Array, C),
    ],
};

/// Occupancy (decision 059; `occupancy.wgsl`): the map's area and counts,
/// and the placed objects' records, poses and draw groups' covers.
pub(crate) const OCCUPANCY: Table = Table {
    label: "engine/occupancy-layout",
    bindings: &[
        uniform_of(0, "splat", 32, V),
        read(1, "records", V),
        read(2, "poses", V),
        read(3, "covers", V),
    ],
};

/// Group 0 of depth passes: each map's view matrix at a dynamic offset.
pub(crate) const SHADOW_VIEW: Table = Table {
    label: "engine/shadow-view-layout",
    bindings: &[dynamic_uniform(9, "view", 80, V)],
};

/// Group 1 of retained depth passes: slot poses, and slot colors and joint
/// palettes for masked and skinned casters.
pub(crate) const SHADOW_POSES: Table = Table {
    label: "engine/shadow-pose-layout",
    bindings: &[
        read_of(0, "slot_poses", 48, V),
        read_of(1, "slot_colors", 16, V),
        read_of(2, "slot_palette", 48, V),
    ],
};

// Other passes.

/// Light binning (decision 023; `lights.wgsl`).
pub(crate) const LIGHT_BINNING: Table = Table {
    label: "engine/light-binning-layout",
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

/// The screen effects (decision 021; `post.wgsl`), whose passes each read
/// some of these. Textures the passes sample with the linear sampler
/// filter; the others only load.
pub(crate) const POST: Table = Table {
    label: "engine/post-layout",
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
        // Shafts, nearest depths, traced reflections, raw occlusion, surface.
        texture(9, "shafts", Unfilterable, D2, F),
        texture(10, "nearest_depth", Unfilterable, D2, F),
        texture(11, "traced", Unfilterable, D2, F),
        texture(12, "raw_occlusion", Unfilterable, D2, F),
        texture(13, "surface_texture", Unfilterable, D2, F),
        // Scene levels and a bloom level halved beside them.
        texture(14, "scene_levels", Float, D2, F),
        texture(15, "bloom_source", Float, D2, F),
    ],
};

/// Group 1 of sun shafts (decision 024; `shafts.wgsl`).
pub(crate) const SHAFTS: Table = Table {
    label: "engine/post-shafts-layout",
    bindings: &[
        uniform_of(0, "post", size_of::<PostUniform>(), F),
        texture(1, "depth_texture", Depth, D2, F),
    ],
};

/// Temporal antialiasing's resolve (decision 017; `temporal.wgsl`).
pub(crate) const TAA_RESOLVE: Table = Table {
    label: "engine/taa-resolve-layout",
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
    ],
};

/// The transparent composite (decision 026; `transparent.wgsl`).
pub(crate) const TRANSPARENT_COMPOSITE: Table = Table {
    label: "engine/transparent-composite-layout",
    bindings: &[
        texture(0, "accumulated", Unfilterable, D2, F),
        texture(1, "revealed", Unfilterable, D2, F),
    ],
};

/// Group 1 of particles (decision 026; `particles.wgsl`).
pub(crate) const PARTICLES: Table = Table {
    label: "engine/particle-layout",
    bindings: &[
        uniform(0, "particles", VF),
        read(1, "emitters", VF),
        read(2, "owners", VF),
        texture(3, "depth_texture", Depth, D2, F),
    ],
};

/// The environment maps' sky (decision 043; `environment.wgsl`).
pub(crate) const ENVIRONMENT_SKY: Table = Table {
    label: "engine/environment-sky-layout",
    bindings: &[
        uniform(1, "lighting", F),
        // The clouds' dome (decision 084).
        texture(38, "environment_clouds", Float, D2, F),
        sampler(39, "environment_cloud_sampler", Filtering, F),
    ],
};

/// Each environment level down from the level above it.
pub(crate) const ENVIRONMENT_DOWN: Table = Table {
    label: "engine/environment-down-layout",
    bindings: &[
        texture(20, "level_source", Float, Cube, F),
        sampler(22, "cube_sampler", Filtering, F),
    ],
};

/// The prefilter of every level.
pub(crate) const ENVIRONMENT_PREFILTER: Table = Table {
    label: "engine/environment-prefilter-layout",
    bindings: &[
        texture(21, "radiance", Float, Cube, F),
        sampler(22, "cube_sampler", Filtering, F),
    ],
};

/// Texture mipmaps (`mipmaps.wgsl`).
pub(crate) const MIPMAPS: Table = Table {
    label: "engine/texture-mip-layout",
    bindings: &[
        texture(0, "source", Float, D2, F),
        sampler(1, "source_sampler", Filtering, F),
    ],
};

/// The depth pyramid's first level from the depth buffer (decision 040;
/// `depth_pyramid.wgsl`).
pub(crate) const PYRAMID_FIRST: Table = Table {
    label: "engine/depth-pyramid-first-layout",
    bindings: &[
        texture(0, "source_depth", Depth, D2, C),
        bind(2, "target_level", Kind::Storage(R32Float), C),
        uniform_of(3, "extent", PYRAMID_EXTENT_BYTES, C),
    ],
};

/// A pyramid level's `Extent` of `depth_pyramid.wgsl`.
pub(crate) const PYRAMID_EXTENT_BYTES: usize = 16;

/// Each further level from the one above it.
pub(crate) const PYRAMID_DOWN: Table = Table {
    label: "engine/depth-pyramid-down-layout",
    bindings: &[
        texture(1, "source_level", Unfilterable, D2, C),
        bind(2, "target_level", Kind::Storage(R32Float), C),
        uniform_of(3, "extent", PYRAMID_EXTENT_BYTES, C),
    ],
};

/// The visibility buffer's tile classification (decision 045;
/// `visibility.wgsl`).
pub(crate) const VISIBILITY_CLASSIFY: Table = Table {
    label: "engine/visibility-classify-layout",
    bindings: &[
        uniform(0, "classify", C),
        texture(1, "ids", Uint, D2, C),
        write(2, "counts", C),
        write(3, "tiles", C),
        write(4, "draw_args", C),
    ],
};
