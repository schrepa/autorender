use engine_core::collections::HashMap;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use engine_core::{
    MaterialHandle, MeshHandle, RecordKey, RenderFrame, RenderSnapshot, TextureHandle,
};
use wgpu::util::DeviceExt;

use crate::platform::{RenderError, RenderTarget, render_error};

use crate::RenderStats;
use crate::batches::{BatchPlan, DrawBatch};
use crate::cached::{Bound, Versioned};
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
    AtlasShape, BakeGpu, BakeSource, Contents, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas,
    ImpostorPipelines, ImpostorRecord, Supplied,
};
use crate::lights::{LightFrame, LightPass};
use crate::materials::MaterialStore;
use crate::mesh_arena::{Bindings, MeshArena, MeshDraw, Placement};
use crate::pipelines::PipelineCounts;
use crate::post::PostPass;
use crate::probes::{FACES, ProbePass, face_view_projection};
use crate::profiling::GpuProfiler;
use crate::render_features::{Eager, FeatureFrames, Lazy, Ready, RenderFeature, RenderFeatures};
use crate::retained_uploads::RetainedUploads;
use crate::scene_shading::SceneShading;
use crate::shadows::{CasterDraw, ShadowPass};
use crate::staging::{self, StagingRing};
use crate::targets::{FrameSize, ScreenTarget, Targets};
use crate::temporal::{TemporalFrame, TemporalPass};
use crate::transparent::accumulation_targets;
use crate::uploads::UploadCache;
use crate::virtual_geometry::{
    CameraTests, Cone, IdBuffers, MAX_VIEWS, SelectionFrame, VirtualBatch, VirtualGeometryPass,
    VirtualView,
};
use crate::visibility::{ID_FORMAT, VisibilityBuffer};

mod camera;
mod display;
mod frame;
mod frame_stats;
mod health;
mod impostor_draws;
mod instances;
mod meshes;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(feature = "dev")]
mod probe;
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
    // Its cooked levels of detail, after the mesh's own indices in its
    // range: each one's first index there, index count and geometric error
    // in mesh units, finest first; and those plan batches draw at, its
    // first simplification and its empty level ([`plan_levels`]).
    lods: Vec<(u32, u32, f32)>,
    plan: Vec<(u32, u32, f32)>,
    // The triangles shadow maps draw of each level, the mesh's first;
    // empty where they are the levels' own.
    casters: Vec<(u32, u32)>,
    // Whether it could have levels but came without: no pack or store held
    // the cook's.
    without_levels: bool,
}

/// What uploading new meshes keeps from one to the next.
#[derive(Default)]
struct MeshUploads {
    // The vertices of the mesh being uploaded, as the GPU reads them.
    staging: Vec<Vertex>,
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

    /// The first index (in the mesh's range) and index count of the
    /// triangles shadow maps draw of level `level` (0 the mesh itself),
    /// whose own are `own`.
    fn caster_range(&self, level: usize, own: (u32, u32)) -> (u32, u32) {
        self.casters.get(level).copied().unwrap_or(own)
    }
}

/// The levels of detail `batch` draws at on GPU-presented frames:
/// its mesh's plan levels where it draws the mesh opaquely, otherwise only
/// the mesh itself. Batch records and draw loops both follow this.
fn lod_levels(batch: &DrawBatch, mesh: &GpuMesh, virtual_geometry: bool) -> usize {
    if batch.transparent || virtual_geometry || mesh.skinned {
        1
    } else {
        1 + mesh.plan.len().min(LOD_LEVELS - 1)
    }
}

/// The levels plan batches draw a mesh at of its cooked `lods`: its first
/// simplification, and its empty level, past which it draws nothing; as
/// many as their draw records hold ([`LOD_LEVELS`] with the mesh).
fn plan_levels(lods: &[(u32, u32, f32)]) -> Vec<(u32, u32, f32)> {
    let first = lods.first().copied().filter(|&(_, count, _)| count > 0);
    let empty = lods.last().copied().filter(|&(_, count, _)| count == 0);
    first.into_iter().chain(empty).collect()
}

/// A mesh's indices followed by those its levels of detail add, its levels
/// (each one's first index there, index count and error) and the ranges
/// of the triangles shadow maps draw of each.
type WithLevels = (Vec<u32>, Vec<(u32, u32, f32)>, Vec<(u32, u32)>);

/// `mesh` with its cooked levels `lods`, as [`WithLevels`] lays them out.
fn with_levels(mesh: &engine_core::Mesh, lods: Option<&engine_core::MeshLods>) -> WithLevels {
    let mut indices =
        Vec::with_capacity(mesh.indices().len() + lods.map_or(0, |lods| lods.indices.len()));
    indices.extend_from_slice(mesh.indices());
    let Some(lods) = lods else {
        return (indices, Vec::new(), Vec::new());
    };
    indices.extend_from_slice(&lods.indices);
    let levels = lods
        .levels
        .iter()
        .map(|level| (level.first_index, level.index_count, level.error))
        .collect();
    (indices, levels, lods.casters.clone())
}

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
    queue: crate::gpu_calls::Queue,
    adapter: crate::AdapterInfo,
    // A graphics debugger's capture runs.
    #[cfg(not(target_arch = "wasm32"))]
    debugger_capture: bool,
    // The canvas, its surface and the depth buffer at its size.
    display: Display,
    depth: DepthTarget,
    // Terrain's visibility target and its resolves' group.
    terrain_ids: ScreenTarget<terrain_draws::TerrainIds>,
    // Resident meshes, in the shared arena's buffers (or their own when it
    // had no room), and a generation that advances whenever a mesh is
    // placed or retired.
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    mesh_arena: MeshArena,
    mesh_layout: u64,
    mesh_uploads: MeshUploads,
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
    // Virtual geometry and the visibility buffer resolving
    // its clusters.
    virtual_draws: VirtualDraws,
    visibility: VisibilityDraws,
    // Impostors.
    impostors: ImpostorDraws,
    // Terrain: heights of resident tiles, patches, pipelines.
    terrain: crate::terrain::TerrainPass,
    // Objects placed on terrain: their meshes and a counter advancing when
    // that set changes, and their draw groups and slots (the pass is a
    // feature's).
    scatter_meshes: (BTreeSet<MeshHandle>, u64),
    scatter_draws: scatter_draws::ScatterDraws,
    // How far switches between how objects show fade this frame.
    switch_fades: frame::SwitchFades,
    // The presentation time of the last frame written, which temporal
    // frames sway placed objects by for their motion.
    last_seconds: Option<f32>,
    // The features render_features! lists, each created with the first
    // frame needing it.
    features: RenderFeatures,
    // The frame's passes, declared anew each frame.
    graph: crate::render_graph::RenderGraph,
    // Whether the previous frame was GPU-presented; its poses were GPU-written.
    gpu_frame: bool,
    // What the device's callbacks report, which each frame reads first.
    health: Arc<Mutex<health::Reports>>,
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

    /// Whether the session cooks its scene's pipeline list: frames then
    /// create the pipelines of every pass kind their content may draw
    /// with, lit or not, beyond those they draw.
    pub fn set_cooking(&mut self, cooking: bool) {
        self.mesh_pipelines.cooking = cooking;
    }
}
