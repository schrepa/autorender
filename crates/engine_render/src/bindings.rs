//! Bind group layouts as data: every layout the renderer
//! builds is a table of the bindings its shaders read, each with its
//! number, WGSL name, kind and the stages reading it. Layouts and bind
//! groups are built from the tables (`crate::pipelines::layout` and
//! `group`), so a binding's number is written once in WGSL and once here,
//! and groups bind resources by name.

// Native builds read only the tables; Wasm builds build layouts from them.

pub(crate) use crate::gpu_layouts::constants::PYRAMID_EXTENT_BYTES;
use crate::gpu_layouts::impostors::{ImpostorBakeUniform, ImpostorRecord};
use crate::gpu_layouts::scene::{
    CameraUniform, ColorData, GroundData, LightingUniform, MaterialUniform, PoseData, PostUniform,
    ResolveUniform, ShadowCullUniform, ShadowUniform,
};
use crate::gpu_layouts::virtual_geometry::MeshRecord;
use crate::gpu_layouts::{
    clouds, deformation, ground_cover, occupancy, scatter, sky, terrain, terrain_horizons,
    terrain_sky, water,
};

/// Shader stages, as bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stages(u8);

impl Stages {
    pub const VERTEX: Self = Self(1);
    pub const FRAGMENT: Self = Self(2);
    pub const COMPUTE: Self = Self(4);
    pub const VERTEX_FRAGMENT: Self = Self(3);

    pub fn vertex(self) -> bool {
        self.0 & 1 != 0
    }

    pub fn fragment(self) -> bool {
        self.0 & 2 != 0
    }

    pub fn compute(self) -> bool {
        self.0 & 4 != 0
    }
}

/// How shaders read a texture's texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sample {
    /// Floats, filtered by samplers.
    Float,
    /// Floats shaders only load.
    Unfilterable,
    Depth,
    Uint,
}

/// A texture view's dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// How a sampler filters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Filtering,
    NonFiltering,
    Comparison,
}

/// A storage texture's texel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    R32Float,
    R32Uint,
    Rgba8Snorm,
    Rgba8Unorm,
    Rgba16Float,
}

/// What a binding holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
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
    Storage(Format),
    /// A 3D texture of this format shaders write.
    Storage3d(Format),
    Sampler(Sampling),
}

/// One binding of a layout.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub binding: u32,
    pub name: &'static str,
    pub kind: Kind,
    pub stages: Stages,
}

/// A bind group slot a table serves: group `group` of shader module
/// `module` (`crate::shaders`).
#[derive(Debug)]
pub struct Serves {
    pub module: &'static str,
    pub group: u32,
}

/// A bind group layout: its label, the shader groups it serves and its
/// bindings. Tables are declared with `tables!` alone, which registers
/// each in [`TABLES`] for the GPU check.
#[derive(Debug)]
pub struct Table {
    pub label: &'static str,
    pub serves: &'static [Serves],
    pub bindings: &'static [Binding],
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

const fn dynamic_uniform(binding: u32, name: &'static str, size: usize, stages: Stages) -> Binding {
    bind(
        binding,
        name,
        Kind::Uniform {
            min_size: size as u64,
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

use Format::{R32Float, R32Uint, Rgba8Snorm, Rgba8Unorm, Rgba16Float};
use Sample::{Depth, Float, Uint, Unfilterable};
use Sampling::{Comparison, Filtering, NonFiltering};
use View::{Cube, CubeArray, D2, D2Array, D3};

const C: Stages = Stages::COMPUTE;
const F: Stages = Stages::FRAGMENT;
const V: Stages = Stages::VERTEX;
const VF: Stages = Stages::VERTEX_FRAGMENT;
// Bytes of a pose, a slot color, a placed object's ground,
// and of virtual geometry's mesh, cluster and page pool word.
const POSE: usize = size_of::<PoseData>();
const COLOR: usize = size_of::<ColorData>();
const GROUND: usize = size_of::<GroundData>();
const VIRTUAL_MESH: usize = size_of::<MeshRecord>();
const VIRTUAL_CLUSTER: usize = engine_core::CLUSTER_WORDS * 4;
const VIRTUAL_PAGE: usize = 4;

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
    /// the environment maps and reflection probes.
    const CAMERA: Table = Table {
        label: "engine/camera-layout",
        serves: [SHADER @ 0, PARTICLE_SHADER @ 0, SHAFTS_SHADER @ 0],
        bindings: &[
            // Fragments read the eye, sky rays and the jitter.
            uniform_of(0, "camera", size_of::<CameraUniform>(), VF),
            uniform_of(1, "lighting", size_of::<LightingUniform>(), F),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), F),
            texture(3, "near_shadows", Depth, D2Array, F),
            texture(4, "kept_shadows", Depth, D2, F),
            texture(5, "moving_shadows", Depth, D2, F),
            sampler(6, "shadow_sampler", Comparison, F),
            // Receiver-plane lookups read four texels.
            sampler(7, "shadow_texels", NonFiltering, F),
            // The kept shadow levels, one layer each.
            texture(8, "kept_levels", Depth, D2Array, F),
            texture(9, "refraction_source", Float, D2, F),
            sampler(10, "refraction_sampler", Filtering, F),
            texture(11, "environment_maps", Float, CubeArray, F),
            sampler(12, "environment_sampler", Filtering, F),
            read(13, "probe_words", F),
            // Terrain's long sun shadows: each terrain's horizon
            // map, two texels of records per terrain, and the maps' sampler.
            texture(14, "terrain_horizons", Float, D2Array, F),
            texture(15, "terrain_horizon_records", Unfilterable, D2, F),
            sampler(16, "terrain_horizon_sampler", Filtering, F),
            // What placed objects cover around the camera: its
            // density, filtered, and the covers' tops and bottoms; the near map
            // in the first layer, the far one in the second.
            texture(17, "occupancy_coverage", Float, D2Array, F),
            texture(18, "occupancy_heights", Unfilterable, D2Array, F),
            // Clouds' shadows: the eighth storage buffer the
            // terrain's lit fragments read (a uniform array read at each
            // fragment's own index is many times slower).
            read_of(19, "cloud_shadows", clouds::CLOUD_SHADOW_BYTES, F),
        ],
    };

    /// Group 0 of terrain's resolves: the main passes' without
    /// what glass refracts, which keeps them within the 16 sampled textures a
    /// fragment stage may bind.
    const CAMERA_TERRAIN: Table = Table {
        label: "engine/camera-terrain-layout",
        serves: [SHADER @ 0],
        bindings: &[
            uniform_of(0, "camera", size_of::<CameraUniform>(), VF),
            uniform_of(1, "lighting", size_of::<LightingUniform>(), F),
            uniform_of(2, "shadow", size_of::<ShadowUniform>(), F),
            texture(3, "near_shadows", Depth, D2Array, F),
            texture(4, "kept_shadows", Depth, D2, F),
            texture(5, "moving_shadows", Depth, D2, F),
            sampler(6, "shadow_sampler", Comparison, F),
            sampler(7, "shadow_texels", NonFiltering, F),
            texture(8, "kept_levels", Depth, D2Array, F),
            texture(11, "environment_maps", Float, CubeArray, F),
            sampler(12, "environment_sampler", Filtering, F),
            read(13, "probe_words", F),
            texture(14, "terrain_horizons", Float, D2Array, F),
            texture(15, "terrain_horizon_records", Unfilterable, D2, F),
            sampler(16, "terrain_horizon_sampler", Filtering, F),
            texture(17, "occupancy_coverage", Float, D2Array, F),
            texture(18, "occupancy_heights", Unfilterable, D2Array, F),
            read_of(19, "cloud_shadows", clouds::CLOUD_SHADOW_BYTES, F),
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
    /// palettes of skinned slots and placed objects' grounds.
    const RETAINED_POSES: Table = Table {
        label: "engine/retained-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, V),
            read_of(1, "colors", COLOR, V),
            read_of(3, "palette", POSE, V),
            read_of(5, "grounds", GROUND, V),
        ],
    };

    /// Group 2 of temporal mesh pipelines: the retained poses
    /// and the previous frame's poses and palettes.
    const TEMPORAL_POSES: Table = Table {
        label: "engine/temporal-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, V),
            read_of(1, "colors", COLOR, V),
            read_of(2, "previous_poses", POSE, V),
            read_of(3, "palette", POSE, V),
            read_of(4, "previous_palette", POSE, V),
            read_of(5, "grounds", GROUND, V),
        ],
    };

    /// Group 2 of virtual geometry's pipelines: poses, colors,
    /// previous poses and the arena's meshes, clusters and pages.
    const VIRTUAL_POSES: Table = Table {
        label: "engine/virtual-pose-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read_of(0, "poses", POSE, V),
            read_of(1, "colors", COLOR, V),
            read_of(2, "previous_poses", POSE, V),
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

    /// Caster selection and counting.
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
        ],
    };

    /// Group 1 of terrain pipelines (`meshes/terrain_vertex.wgsl`):
    /// the height array, each view's eye-relative projection and each
    /// terrain's anchor and material, the uniforms at dynamic offsets.
    const TERRAIN: Table = Table {
        label: "engine/terrain-layout",
        serves: [SHADER @ 1, TERRAIN_SHADOW_SHADER @ 1],
        bindings: &[
            texture(17, "terrain_heights", Unfilterable, D2Array, V),
            dynamic_uniform(18, "terrain_view", size_of::<terrain::ViewUniform>(), VF),
            dynamic_uniform(
                19,
                "terrain",
                size_of::<terrain::TerrainUniform>(),
                VF,
            ),
            read(20, "terrain_holes", F),
            texture(21, "terrain_normals", Float, D2Array, F),
            sampler(22, "terrain_normal_sampler", Filtering, F),
            read(23, "terrain_directory", F),
            texture(24, "terrain_control", Uint, D2Array, F),
            uniform_of(25, "terrain_palette", terrain::PALETTE_BYTES, F),
            // Surface materials' maps and their sampler.
            texture(26, "terrain_albedo", Float, D2Array, F),
            texture(27, "terrain_material_normals", Float, D2Array, F),
            texture(28, "terrain_orm", Float, D2Array, F),
            sampler(29, "terrain_material_sampler", Filtering, F),
            // Each terrain's sky visibility map, read through
            // the normals' sampler.
            texture(30, "terrain_sky", Float, D2Array, F),
            // Runtime deformation's map.
            read(31, "terrain_deformation", F),
            // Gathers of the control values (nearest).
            sampler(32, "terrain_control_sampler", NonFiltering, F),
        ],
    };

    /// Group 2 of terrain's resolves: each pixel's terrain and
    /// depth, as its visibility pass wrote them.
    const TERRAIN_RESOLVE: Table = Table {
        label: "engine/terrain-resolve-layout",
        serves: [SHADER @ 2],
        bindings: &[texture(18, "terrain_ids_texture", Uint, D2, F)],
    };

    /// Group 2 of the runtime virtual texture's bakes: the
    /// pages to bake, and every resident tile's heights, which the terrain
    /// group gives vertex stages alone.
    const TERRAIN_VT_BAKE: Table = Table {
        label: "engine/terrain-vt-bake-layout",
        serves: [SHADER @ 2],
        bindings: &[
            read(20, "terrain_vt_bakes", VF),
            texture(21, "terrain_vt_heights", Unfilterable, D2Array, F),
        ],
    };

    /// The runtime virtual texture's feedback, a compute pass
    /// over terrain's visibility target: the camera (group 0); the view, the
    /// terrain and the directory with the texture's windows and tables (group
    /// 1, as the terrain group binds them); the visibility target and the
    /// pages wanted (group 2).
    const TERRAIN_VT_CAMERA: Table = Table {
        label: "engine/terrain-vt-camera-layout",
        serves: [SHADER @ 0],
        bindings: &[uniform_of(0, "camera", size_of::<CameraUniform>(), C)],
    };

    const TERRAIN_VT_TERRAIN: Table = Table {
        label: "engine/terrain-vt-terrain-layout",
        serves: [SHADER @ 1],
        bindings: &[
            dynamic_uniform(18, "terrain_view", size_of::<terrain::ViewUniform>(), C),
            dynamic_uniform(
                19,
                "terrain",
                size_of::<terrain::TerrainUniform>(),
                C,
            ),
            read(23, "terrain_directory", C),
        ],
    };

    const TERRAIN_VT_FEEDBACK: Table = Table {
        label: "engine/terrain-vt-feedback-layout",
        serves: [SHADER @ 2],
        bindings: &[
            texture(18, "terrain_ids_texture", Uint, D2, C),
            write(19, "terrain_vt_wanted", C),
        ],
    };

    /// The clouds' noise volume (`clouds.wgsl`), made once.
    const CLOUD_NOISE: Table = Table {
        label: "engine/cloud-noise-layout",
        serves: [CLOUDS_SHADER @ 0],
        bindings: &[bind(4, "noise_out", Kind::Storage3d(Rgba8Unorm), C)],
    };

    /// The clouds' march into their dome (`clouds.wgsl`): the
    /// layer, sun and sky, the noise volume and its sampler, and the dome.
    const CLOUD_MARCH: Table = Table {
        label: "engine/cloud-march-layout",
        serves: [CLOUDS_SHADER @ 0],
        bindings: &[
            uniform_of(0, "clouds", size_of::<clouds::CloudUniform>(), C),
            texture(1, "noise_volume", Float, D3, C),
            sampler(2, "noise_sampler", Filtering, C),
            bind(3, "dome", Kind::Storage(Rgba16Float), C),
            write(5, "shadow_cells", C),
        ],
    };

    /// The view's march of the clouds (`clouds.wgsl`'s
    /// `march_view`): the march from the camera, the noise volume and its
    /// sampler, the view, its depth buffer, and the marched blocks' clouds and
    /// their distances.
    const CLOUD_VIEW: Table = Table {
        label: "engine/cloud-view-layout",
        serves: [CLOUDS_SHADER @ 0],
        bindings: &[
            uniform_of(0, "clouds", size_of::<clouds::CloudUniform>(), C),
            texture(1, "noise_volume", Float, D3, C),
            sampler(2, "noise_sampler", Filtering, C),
            uniform_of(6, "cloud_view", size_of::<clouds::ViewUniform>(), C),
            texture(7, "view_depth", Depth, D2, C),
            bind(8, "view_out", Kind::Storage(Rgba16Float), C),
            bind(9, "front_out", Kind::Storage(R32Float), C),
            write(10, "composite_draw", C),
        ],
    };

    /// Group 1 of the view's clouds' composite (
    /// `meshes/sky.wgsl`): the marched blocks' clouds and distances, the depth
    /// buffer and the blocks' sampler.
    const CLOUD_COMPOSITE: Table = Table {
        label: "engine/cloud-composite-layout",
        serves: [SHADER @ 1],
        bindings: &[
            texture(46, "view_clouds", Float, D2, F),
            texture(47, "view_cloud_front", Unfilterable, D2, F),
            texture(48, "view_cloud_depth", Depth, D2, F),
            sampler(49, "view_cloud_sampler", Filtering, F),
        ],
    };

    /// Group 1 of the sky's pipelines (`meshes/sky.wgsl`): the
    /// clouds' dome and its sampler; the physically based sky's uniform, its
    /// sky-view and transmittance tables and their sampler.
    const CLOUD_SKY: Table = Table {
        label: "engine/cloud-sky-layout",
        serves: [SHADER @ 1],
        bindings: &[
            texture(38, "sky_clouds", Float, D2, F),
            sampler(39, "sky_cloud_sampler", Filtering, F),
            uniform_of(40, "sky_air", size_of::<sky::SkyUniform>(), F),
            texture(41, "sky_view_table", Float, D2, F),
            texture(42, "sky_transmittance_table", Float, D2, F),
            sampler(43, "sky_table_sampler", Filtering, F),
            // The older shown dome and the domes' uniform.
            texture(44, "sky_clouds_old", Float, D2, F),
            uniform_of(45, "sky_domes", size_of::<clouds::DomeUniform>(), F),
        ],
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
            bind(2, "normals", Kind::Storage(Rgba8Snorm), C),
        ],
    };

    /// A tile's control values from its terrain's surface (
    /// `terrain_control.wgsl`): the height array, the job's uniform, the
    /// surface's data and the control layer it writes.
    const TERRAIN_CONTROL: Table = Table {
        label: "engine/terrain-control-layout",
        serves: [TERRAIN_CONTROL_SHADER @ 0],
        bindings: &[
            texture(0, "heights", Unfilterable, D2Array, C),
            uniform_of(1, "job", 32, C),
            bind(2, "data", Kind::Read { min_size: 4 }, C),
            bind(3, "control", Kind::Storage(R32Uint), C),
        ],
    };

    /// A terrain horizon map's build (`terrain_horizon.wgsl`):
    /// the height array, which layer holds each source tile, the build's
    /// uniform and the layer it writes.
    const TERRAIN_HORIZON: Table = Table {
        label: "engine/terrain-horizon-layout",
        serves: [TERRAIN_HORIZON_SHADER @ 0],
        bindings: &[
            texture(0, "heights", Unfilterable, D2Array, C),
            read(1, "tile_layers", C),
            uniform_of(2, "build", size_of::<terrain_horizons::BuildUniform>(), C),
            bind(3, "horizon", Kind::Storage(Rgba16Float), C),
        ],
    };

    /// A terrain sky visibility map's build (`terrain_sky.wgsl`):
    /// the height array, which layer holds each source tile, the build's
    /// uniform and the layer it writes.
    const TERRAIN_SKY: Table = Table {
        label: "engine/terrain-sky-layout",
        serves: [TERRAIN_SKY_SHADER @ 0],
        bindings: &[
            texture(0, "heights", Unfilterable, D2Array, C),
            read(1, "tile_layers", C),
            uniform_of(2, "build", size_of::<terrain_sky::BuildUniform>(), C),
            bind(3, "sky", Kind::Storage(Rgba8Unorm), C),
        ],
    };

    /// Ground cover's growth (`cover_grow.wgsl`): terrain's
    /// height and control arrays, directory, palette and covers, each
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
            // The terrains' sky maps: their variety tints blades as it does the
            // ground.
            texture(10, "sky_maps", Float, D2Array, C),
        ],
    };

    /// Group 1 of ground cover's pipelines (`meshes/cover.wgsl`):
    /// the frame's blades and the wind and times they sway with.
    /// Group 1 of water's pipelines (`meshes/water.wgsl`): one
    /// terrain's water surface heights, its depth and flow, and its uniform;
    /// the sea's has placeholder textures.
    const WATER: Table = Table {
        label: "engine/water-layout",
        serves: [SHADER @ 1],
        bindings: &[
            texture(35, "water_surface", Unfilterable, D2, V),
            texture(36, "water_info", Unfilterable, D2, V),
            uniform_of(37, "water", size_of::<water::WaterUniform>(), VF),
        ],
    };

    /// Group 2 of water's pipelines: a copy of the opaque
    /// scene's depth, the ground behind the water.
    const WATER_DEPTH: Table = Table {
        label: "engine/water-depth-layout",
        serves: [SHADER @ 2],
        bindings: &[texture(17, "water_depth", Depth, D2, F)],
    };

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
            read(8, "thins", C),
        ],
    };

    /// Placed objects set on the terrain as drawn (
    /// `scatter_snap.wgsl`): the slots and terrains, every object's record and
    /// pose, each terrain's place in the directory, the directory and the
    /// resident heights.
    const SCATTER_SNAP: Table = Table {
        label: "engine/scatter-snap-layout",
        serves: [SCATTER_SNAP_SHADER @ 0],
        bindings: &[
            uniform_of(0, "snap", 16, C),
            write(1, "records", C),
            write(2, "poses", C),
            read(3, "terrains", C),
            read(4, "directory", C),
            texture(5, "heights", Unfilterable, D2Array, C),
        ],
    };

    /// Occupancy (`occupancy.wgsl`): the map's area and counts,
    /// and the placed objects' records, poses and draw groups' covers, and how
    /// their layers thin, which the far map weighs far objects
    /// by.
    const OCCUPANCY: Table = Table {
        label: "engine/occupancy-layout",
        serves: [OCCUPANCY_SHADER @ 0],
        bindings: &[
            uniform_of(0, "splat", size_of::<occupancy::SplatUniform>(), V),
            read(1, "records", V),
            read(2, "poses", V),
            read(3, "covers", V),
            read(4, "thins", V),
        ],
    };

    /// Group 0 of depth passes: each map's view matrix at a dynamic offset.
    const SHADOW_VIEW: Table = Table {
        label: "engine/shadow-view-layout",
        serves: [SHADOW_SHADER @ 0, TERRAIN_SHADOW_SHADER @ 0],
        bindings: &[dynamic_uniform(9, "view", 80, V)],
    };

    /// Group 1 of retained depth passes: slot poses, and slot colors and joint
    /// palettes for masked and skinned casters.
    const SHADOW_POSES: Table = Table {
        label: "engine/shadow-pose-layout",
        serves: [SHADOW_SHADER @ 1],
        bindings: &[
            read_of(0, "slot_poses", POSE, V),
            read_of(1, "slot_colors", COLOR, V),
            read_of(2, "slot_palette", POSE, V),
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

    /// Group 1 of sun shafts (`shafts.wgsl`).
    const SHAFTS: Table = Table {
        label: "engine/post-shafts-layout",
        serves: [SHAFTS_SHADER @ 1],
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
            texture(9, "coarse_history", Float, D2, F),
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

    /// The environment maps' sky (`environment.wgsl`).
    const ENVIRONMENT_SKY: Table = Table {
        label: "engine/environment-sky-layout",
        serves: [ENVIRONMENT_SHADER @ 0],
        bindings: &[
            uniform(1, "lighting", F),
            // The clouds' dome.
            texture(38, "environment_clouds", Float, D2, F),
            sampler(39, "environment_cloud_sampler", Filtering, F),
            // The physically based sky.
            uniform_of(40, "environment_air", size_of::<sky::SkyUniform>(), F),
            texture(41, "environment_view_table", Float, D2, F),
            texture(42, "environment_transmittance_table", Float, D2, F),
            sampler(43, "environment_table_sampler", Filtering, F),
            // The older shown dome and the domes' uniform.
            texture(44, "environment_clouds_old", Float, D2, F),
            uniform_of(
                45,
                "environment_domes",
                size_of::<clouds::DomeUniform>(),
                F,
            ),
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

/// The bindings one feature takes in a group features share: its numbers
/// (a range of them) and its bindings' names.
#[derive(Debug)]
pub struct FeatureRange {
    pub feature: &'static str,
    pub numbers: core::ops::Range<u32>,
    pub names: &'static [&'static str],
}

/// The groups several features share and each feature's range in them:
/// group `group` of module `module`, every table serving it holding each
/// binding within its feature's range.
#[derive(Debug)]
pub struct SharedGroup {
    pub module: &'static str,
    pub group: u32,
    pub ranges: &'static [FeatureRange],
}

/// Declares [`SHARED_GROUPS`]: `MODULE @ group { feature: start..end [names], .. }`.
macro_rules! shared_groups {
    ($($module:ident @ $group:literal {
        $($feature:ident: $start:literal..$end:literal [$($name:ident),* $(,)?]),* $(,)?
    })*) => {
        /// The mesh module's groups, which every surface pipeline shares,
        /// each feature in a range of its own: `cargo xtask lint` holds every
        /// table serving one to the ranges, so a feature adding a binding
        /// takes a number in its range, and features never collide.
        pub const SHARED_GROUPS: &[SharedGroup] = &[$(SharedGroup {
            module: stringify!($module),
            group: $group,
            ranges: &[$(FeatureRange {
                feature: stringify!($feature),
                numbers: $start..$end,
                names: &[$(stringify!($name)),*],
            }),*],
        }),*];
    };
}

shared_groups! {
    SHADER @ 0 {
        frame: 0..2 [camera, lighting],
        shadows: 2..9 [
            shadow, near_shadows, kept_shadows, moving_shadows, shadow_sampler, shadow_texels,
            kept_levels
        ],
        refraction: 9..11 [refraction_source, refraction_sampler],
        environment: 11..13 [environment_maps, environment_sampler],
        probes: 13..14 [probe_words],
        horizons: 14..17 [terrain_horizons, terrain_horizon_records, terrain_horizon_sampler],
        occupancy: 17..19 [occupancy_coverage, occupancy_heights],
        clouds: 19..20 [cloud_shadows],
    }
    SHADER @ 1 {
        material: 0..11 [
            base_color_texture, base_color_sampler, metallic_roughness_texture,
            metallic_roughness_sampler, normal_texture, normal_sampler, occlusion_texture,
            occlusion_sampler, emissive_texture, emissive_sampler, material_maps
        ],
        impostors: 11..17 [
            impostor_records, impostor_albedo, impostor_emissive, impostor_geometry,
            impostor_surface, impostor_sampler
        ],
        terrain: 17..33 [
            terrain_heights, terrain_view, terrain, terrain_holes, terrain_normals,
            terrain_normal_sampler, terrain_directory, terrain_control, terrain_palette,
            terrain_albedo, terrain_material_normals, terrain_orm, terrain_material_sampler,
            terrain_sky, terrain_deformation, terrain_control_sampler
        ],
        cover: 33..35 [cover_blades, cover_frame],
        water: 35..38 [water_surface, water_info, water],
        sky: 38..46 [
            sky_clouds, sky_cloud_sampler, sky_air, sky_view_table, sky_transmittance_table,
            sky_table_sampler, sky_clouds_old, sky_domes
        ],
        view_clouds: 46..50 [view_clouds, view_cloud_front, view_cloud_depth, view_cloud_sampler],
    }
    SHADER @ 2 {
        poses: 0..6 [poses, colors, previous_poses, palette, previous_palette, grounds],
        virtual_geometry: 6..9 [virtual_meshes, virtual_clusters, virtual_pages],
        visibility: 9..13 [visibility_draw, visible_clusters, visibility_ids, resolve_tiles],
        decals: 13..15 [decal_records, decal_depth],
        impostor_bakes: 15..17 [impostor_bake, impostor_shade],
        water: 17..18 [water_depth],
        terrain: 18..19 [terrain_ids_texture],
        virtual_texture: 19..22 [terrain_vt_wanted, terrain_vt_bakes, terrain_vt_heights],
    }
    SHADER @ 3 {
        lights: 0..4 [clusters, lights, cluster_ranges, cluster_lights],
    }
}
