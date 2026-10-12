use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

use engine_core::{MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, TextureHandle};
use wgpu::util::DeviceExt;

use crate::platform::{RenderError, RenderTarget, render_error};

use crate::RenderStats;
use crate::batches::{BatchPlan, DrawBatch};
use crate::cached::Cached;
use crate::decals::DecalPass;
use crate::depth_pyramid::{DepthPyramid, Reduction};
use crate::entries::{MeshPass, PassKind, Streams};
use crate::environment::{ENVIRONMENT_FORMAT, Environment};
use crate::gpu_data::{
    CameraUniform, ColorData, ExtraVertex, FAR_DEPTH, GroundData, LOD_LEVELS, LOD_STATE,
    LightingUniform, PoseData, SHADER, SkinVertex, Variant, Vertex, clamped_extent,
};
use crate::gpu_presentation::{BatchMeta, GpuPresentationPass};
use crate::gpu_uploads;
use crate::impostors::{
    AtlasShape, BakeSource, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas, ImpostorPipelines,
    ImpostorRecord,
};
use crate::lights::{LightFrame, LightPass};
use crate::materials::MaterialStore;
use crate::mesh_arena::{Bindings, MeshArena, MeshDraw, Placement};
use crate::particles::ParticlePass;
use crate::pipelines::PipelineCounts;
use crate::post::PostPass;
use crate::probes::{FACES, ProbePass, face_view_projection};
use crate::profiling::{GpuProfiler, Span};
use crate::retained_uploads::RetainedUploads;
use crate::shadows::{CasterDraw, ShadowPass};
use crate::staging::{self, StagingRing};
use crate::targets::Targets;
use crate::temporal::{TemporalFrame, TemporalPass};
use crate::transparent::{TransparentPass, accumulation_targets};
use crate::uploads::UploadCache;
use crate::virtual_geometry::{
    CameraTests, Cone, IdBuffers, MAX_VIEWS, SelectionFrame, VirtualBatch, VirtualGeometryPass,
    VirtualView,
};
use crate::visibility::{
    ID_FORMAT, RESOLVE_LIT_STORAGE_BUFFERS, RESOLVE_STORAGE_BUFFERS, VisibilityBuffer,
};

mod camera;
mod display;
mod frame;
mod frame_stats;
mod impostor_draws;
mod instances;
mod meshes;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod probe_captures;
mod queries;
mod scatter_draws;
mod scatter_pool;
mod startup;
mod terrain_draws;
mod virtual_draws;
mod visibility_draws;

use camera::CameraBindings;
use display::{DepthTarget, Display, DisplayFrame};
use impostor_draws::ImpostorDraws;
pub(crate) use instances::FramePoses;
use instances::Instances;
use meshes::{BatchBinds, DrawOrder, MeshPipelines, SkyTargets};
pub(crate) use meshes::{extra_layout, skin_layout, vertex_layout};
#[cfg(not(target_arch = "wasm32"))]
pub use native::GpuMark;
pub use startup::StartingRenderer;
use virtual_draws::{VIRTUAL_ATTRIBUTES, VirtualDraws, VirtualStages};
use visibility_draws::VisibilityDraws;

const INITIAL_CAPACITY: usize = 1_024;
// Reversed depth: nearer surfaces keep larger values, which
// floating-point depth resolves about evenly at every distance. The buffer
// clears to the far plane's 0 and depth tests keep the greater value.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct GpuMesh {
    // Where the mesh arena holds it, or the buffers of its own when the
    // arena had no room.
    placement: Result<Placement, OwnMesh>,
    // Whether it has vertex colors or second texture coordinates (or
    // joints), and joints and weights.
    extended: bool,
    skinned: bool,
    index_count: u32,
    byte_len: u64,
    // Coarser levels of detail, after the mesh's own indices
    // in its range: each one's first index there, index count and geometric
    // error in mesh units, finest first.
    lods: Vec<(u32, u32, f32)>,
}

/// A mesh's own buffers: positions, extras, joints and indices.
struct OwnMesh {
    vertices: wgpu::Buffer,
    extras: Option<wgpu::Buffer>,
    skin: Option<wgpu::Buffer>,
    indices: wgpu::Buffer,
}

impl GpuMesh {
    fn draw<'a>(&'a self, arena: &'a MeshArena) -> MeshDraw<'a> {
        match &self.placement {
            // The range holds the coarser levels after the mesh's indices.
            Ok(placement) => MeshDraw {
                index_count: self.index_count,
                ..arena.draw(*placement)
            },
            Err(own) => MeshDraw {
                vertices: &own.vertices,
                extras: own.extras.as_ref(),
                skin: own.skin.as_ref(),
                indices: &own.indices,
                first_index: 0,
                base_vertex: 0,
                index_count: self.index_count,
            },
        }
    }
}

/// The levels of detail `batch` draws at on GPU-presented frames:
/// its mesh's where it draws the mesh opaquely, otherwise only the
/// mesh itself. Batch records and draw loops both follow this.
fn lod_levels(batch: &DrawBatch, mesh: &GpuMesh, virtual_geometry: bool) -> usize {
    if batch.transparent || virtual_geometry || mesh.skinned {
        1
    } else {
        1 + mesh.lods.len().min(LOD_LEVELS - 1)
    }
}

/// `source`'s indices followed by `levels` levels of detail at most,
/// and each level's first index in them, index count and
/// error.
fn lod_chain(source: &engine_core::Mesh, levels: usize) -> (Vec<u32>, Vec<(u32, u32, f32)>) {
    let lods = engine_core::mesh_lods(source, levels);
    let levels = lods
        .levels
        .iter()
        .map(|level| (level.first_index, level.index_count, level.error))
        .collect();
    (lods.indices, levels)
}

/// Generations of the shadow maps, environment maps and reflection probe
/// grid the camera groups bind.
type CameraGroupsKey = (u64, u64, u64, u64, u64);

/// Occlusion culling's depth pyramid, kept while the `OcclusionCulling`
/// settings have it run.
struct Culling {
    pyramid: Option<DepthPyramid>,
}

/// The plan's draws, the mesh layout and the material groups: what a frame's
/// structure-only work reads.
type StructureKey = (u64, u64, u64);

/// What per-frame work reading only the frame's structure last ran for;
/// work whose key stands keeps its results.
#[derive(Default)]
struct StructureKeys {
    // The opaque draw order.
    order: Option<StructureKey>,
    // The material variants' pipelines of the opaque and transparent passes,
    // and of reflection probe faces.
    variants: Option<(StructureKey, PassKind, PassKind)>,
    probes: Option<(StructureKey, PassKind)>,
    // Impostor records, by the level of detail settings,
    // whether the frame was GPU-presented and the atlas generation.
    impostors: Option<(StructureKey, engine_core::LevelOfDetail, bool, u64)>,
    // Mesh residency: the plan's rebuild and the caster groups' generation
    // it was retired for, and those with the mesh layout it uploaded for.
    residency: Option<(u64, u64, u64)>,
    uploads: Option<(u64, u64, u64, u64)>,
}

/// Main-thread browser renderer. Never transfer it or its GPU objects to workers.
///
/// Visible slots are grouped by mesh and material into reusable draw ranges.
/// Immutable assets upload once per residency lifetime; unused asset resources
/// retire at each valid render boundary, even when suspended. Slot storage
/// grows geometrically up to what one storage binding holds.
/// Model transforms must be finite, nonsingular, and preserve triangle winding.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: crate::AdapterInfo,
    // A graphics debugger's capture runs.
    #[cfg(not(target_arch = "wasm32"))]
    debugger_capture: bool,
    // The canvas, its surface and the depth buffer at its size.
    display: Display,
    depth: DepthTarget,
    // Terrain's visibility target and its resolves' group.
    terrain_ids: Option<terrain_draws::TerrainIds>,
    // Resident meshes, in the shared arena's buffers (or their own when it
    // had no room), and a generation that advances whenever a mesh is
    // placed or retired.
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    mesh_arena: MeshArena,
    mesh_layout: u64,
    vertex_staging: Vec<Vertex>,
    // Material textures, samplers and bind groups.
    materials: MaterialStore,
    // The frame's draw batches, and the opaque ones' drawing order.
    plan: BatchPlan,
    draw_order: DrawOrder,
    structure_keys: StructureKeys,
    // Slots' poses, colors and palettes, and the camera bytes last written.
    instances: Instances,
    uploads: UploadCache,
    // The mesh shader, the main passes' first group, and the pipelines of
    // material variants, created with their first draw.
    mesh_shader: wgpu::ShaderModule,
    camera: CameraBindings,
    mesh_pipelines: MeshPipelines,
    pipeline_counts: PipelineCounts,
    // Blending, composition and culling for GPU-presented frames.
    gpu_presentation: GpuPresentationPass,
    culling: Culling,
    // Sun shadows.
    shadows: ShadowPass,
    // Local lights: binning and the lit mesh pipeline
    // layouts, created with the first frame that has lights. Lit
    // public-instance pipelines bind an empty group 2.
    lights: Option<LightPass>,
    // The environment maps surfaces reflect, redrawn when the
    // sky changes, and the reflection probes holding their cubes after the
    // sky's.
    environment: Environment,
    probes: ProbePass,
    // Clouds: their dome, which the sky and its environment
    // map show.
    clouds: crate::clouds::CloudPass,
    // The physically based sky.
    sky: crate::sky::SkyPass,
    // Virtual geometry and the visibility buffer resolving
    // its clusters.
    virtual_draws: VirtualDraws,
    visibility: VisibilityDraws,
    // Impostors.
    impostors: ImpostorDraws,
    // Terrain: heights of resident tiles, patches, pipelines.
    terrain: crate::terrain::TerrainPass,
    // Ground cover grown each frame on terrain.
    ground_cover: crate::ground_cover::GroundCoverPass,
    // Terrains' lakes and rivers.
    water: crate::water::WaterPass,
    // Objects placed on terrain: their pass, their meshes
    // and a counter advancing when that set changes, and their draw groups
    // and slots.
    scatter: crate::scatter::ScatterPass,
    scatter_meshes: (BTreeSet<MeshHandle>, u64),
    scatter_draws: scatter_draws::ScatterDraws,
    // How far switches between how objects show fade this frame.
    switch_fades: frame::SwitchFades,
    // What placed objects cover around the camera.
    occupancy: crate::occupancy::OccupancyPass,
    // The presentation time of the last frame written, which temporal
    // frames sway placed objects by for their motion.
    last_seconds: Option<f32>,
    // The screen effects, created with the first frame that has one.
    post: Option<PostPass>,
    // Temporal antialiasing of GPU-presented display frames.
    temporal: Option<TemporalPass>,
    // Transparent surfaces and particles: targets, the
    // composite and particle pipelines, created with the first frame that
    // draws them.
    transparent: Option<TransparentPass>,
    particles: Option<ParticlePass>,
    // Decals: records and pipelines, created with the first
    // frame that draws them.
    decals: Option<DecalPass>,
    // The frame's passes, declared anew each frame.
    graph: crate::render_graph::RenderGraph,
    // Whether the previous frame was GPU-presented; its poses were GPU-written.
    gpu_frame: bool,
    // The only callback-shared state is a rare error message, never ECS data.
    failure: Arc<Mutex<Option<String>>>,
    // Times the frame's passes during profiling captures (a stand-in that
    // records nothing in other builds).
    profiler: GpuProfiler,
}

impl Renderer {
    /// The frame's structure as structure-only work reads it.
    fn structure_key(&self) -> StructureKey {
        (
            self.plan.generation(),
            self.mesh_layout,
            self.materials.generation(),
        )
    }

    /// Resize in physical pixels, preserving aspect ratio within device limits.
    /// A zero dimension suspends rendering without configuring an invalid surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.display.resize(width, height);
    }
}
