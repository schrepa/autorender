use engine_core::collections::HashMap;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use engine_core::{
    MaterialHandle, MeshHandle, RecordKey, RenderFrame, RenderSnapshot, TextureHandle,
};

use crate::platform::{RenderError, RenderTarget, render_error};

use crate::RenderStats;
use crate::batches::{BatchPlan, DrawBatch};
use crate::cached::{Bound, Versioned};
use crate::depth_pyramid::{DepthPyramid, Reduction};
use crate::entries::{MeshPass, PassKind, Streams};
use crate::environment::{ENVIRONMENT_FORMAT, Environment};
use crate::gpu_data::{
    CameraUniform, ColorData, ExtraVertex, FAR_DEPTH, GroundData, LOD_LEVELS, LOD_STATE,
    LightingUniform, PoseData, SHADER, Variant, Vertex, clamped_extent,
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
use crate::uploads::UploadCache;

mod attribution;
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
mod records;
mod scatter_draws;
mod scatter_pool;
mod startup;
mod terrain_draws;

pub use attribution::GpuPassTimes;
use camera::CameraBindings;
use display::{DepthTarget, Display, DisplayFrame};
use impostor_draws::ImpostorDraws;
pub(crate) use instances::FramePoses;
use instances::Instances;
use meshes::{BatchBinds, DrawOrder, MeshPipelines, SkyTargets};
pub(crate) use meshes::{extra_layout, vertex_layout};
#[cfg(not(target_arch = "wasm32"))]
pub use native::GpuMark;
pub use startup::StartingRenderer;

const INITIAL_CAPACITY: usize = 1_024;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct GpuMesh {
    placement: Result<Placement, OwnMesh>,
    extended: bool,
    index_count: u32,
    byte_len: u64,
    lods: Vec<(u32, u32, f32)>,
    plan: Vec<(u32, u32, f32)>,
    casters: Vec<(u32, u32)>,
    without_levels: bool,
}

#[derive(Default)]
struct MeshUploads {
    staging: Vec<Vertex>,
}

struct OwnMesh {
    vertices: wgpu::Buffer,
    extras: Option<wgpu::Buffer>,
    indices: wgpu::Buffer,
}

impl GpuMesh {
    fn draw<'a>(&'a self, arena: &'a MeshArena) -> MeshDraw<'a> {
        match &self.placement {
            Ok(placement) => MeshDraw {
                index_count: self.index_count,
                ..arena.draw(*placement)
            },
            Err(own) => MeshDraw {
                vertices: &own.vertices,
                extras: own.extras.as_ref(),
                indices: &own.indices,
                first_index: 0,
                base_vertex: 0,
                index_count: self.index_count,
            },
        }
    }

    fn caster_range(&self, level: usize, own: (u32, u32)) -> (u32, u32) {
        self.casters.get(level).copied().unwrap_or(own)
    }
}

fn lod_levels(mesh: &GpuMesh) -> usize {
    1 + mesh.plan.len().min(LOD_LEVELS - 1)
}

fn plan_levels(lods: &[(u32, u32, f32)]) -> Vec<(u32, u32, f32)> {
    let first = lods.first().copied().filter(|&(_, count, _)| count > 0);
    let empty = lods.last().copied().filter(|&(_, count, _)| count == 0);
    first.into_iter().chain(empty).collect()
}

type WithLevels = (Vec<u32>, Vec<(u32, u32, f32)>, Vec<(u32, u32)>);

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

struct Culling {
    pyramid: Option<DepthPyramid>,
}

type StructureKey = (u64, u64, u64);

#[derive(Default)]
struct StructureKeys {
    order: Option<StructureKey>,
    variants: Option<(StructureKey, PassKind)>,
    probes: Option<(StructureKey, PassKind)>,
    impostors: Option<(StructureKey, engine_core::LevelOfDetail, bool, u64)>,
    residency: Option<(u64, u64, u64)>,
    uploads: Option<(u64, u64, u64, u64)>,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: crate::gpu_calls::Queue,
    adapter: crate::AdapterInfo,
    #[cfg(not(target_arch = "wasm32"))]
    debugger_capture: bool,
    display: Display,
    depth: DepthTarget,
    terrain_ids: ScreenTarget<terrain_draws::TerrainIds>,
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    mesh_arena: MeshArena,
    mesh_layout: u64,
    mesh_uploads: MeshUploads,
    materials: MaterialStore,
    plan: BatchPlan,
    draw_order: DrawOrder,
    structure_keys: StructureKeys,
    instances: Instances,
    uploads: UploadCache,
    mesh_shader: wgpu::ShaderModule,
    camera: CameraBindings,
    mesh_pipelines: MeshPipelines,
    pipeline_counts: PipelineCounts,
    gpu_presentation: GpuPresentationPass,
    culling: Culling,
    shadows: ShadowPass,
    impostors: ImpostorDraws,
    water_settle: crate::water_settle::WaterSettle,
    terrain: crate::terrain::TerrainPass,
    scatter_meshes: (BTreeSet<MeshHandle>, u64),
    scatter_draws: scatter_draws::ScatterDraws,
    switch_fades: frame::SwitchFades,
    last_seconds: Option<f32>,
    features: RenderFeatures,
    graph: crate::render_graph::RenderGraph,
    gpu_frame: bool,
    health: Arc<Mutex<health::Reports>>,
    profiler: GpuProfiler,
}

impl Renderer {
    fn structure_key(&self) -> StructureKey {
        (
            self.plan.generation(),
            self.mesh_layout,
            self.materials.generation(),
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.display.resize(width, height);
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.display.ask_render_size(width, height);
    }

    pub fn set_cooking(&mut self, cooking: bool) {
        self.mesh_pipelines.cooking = cooking;
        self.terrain.cooking = cooking;
        self.impostors.atlas.set_cooking(cooking);
        self.impostors.placed.set_cooking(cooking);
    }
}
