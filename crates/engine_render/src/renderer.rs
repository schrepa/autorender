use engine_core::collections::HashMap;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use engine_core::{
    MaterialHandle, MeshHandle, RecordKey, RenderFrame, RenderSnapshot, TextureHandle,
};

use crate::platform::{RenderTarget, failed, failed_with};
use engine_core::{Error, Expected};

use crate::RenderStats;
use crate::batches::{BatchPlan, DrawBatch};
use crate::cached::{Bound, Versioned};
use crate::depth_pyramid::DepthPyramid;
use crate::entries::{MeshPass, PassKind, Streams};
use crate::gpu_data::{
    CameraUniform, ColorData, ExtraVertex, GroundData, LOD_LEVELS, LOD_STATE, PoseData, Variant,
    Vertex, clamped_extent,
};
use crate::gpu_presentation::GpuPresentationPass;
use crate::gpu_uploads;
use crate::graph::{FrameGraph, GraphTable};
use crate::impostors::{
    AtlasShape, BakeGpu, BakeSource, Contents, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas,
    ImpostorPipelines, ImpostorRecord, Supplied,
};
use crate::materials::MaterialStore;
use crate::mesh_arena::{Bindings, MeshArena, MeshDraw, Placement};
use crate::parts::{FrameInfo, FrameKind, GpuPart, PartEntry, Parts, Passes};
use crate::pipelines::PipelineCounts;
use crate::profiling::GpuProfiler;
use crate::retained_uploads::RetainedUploads;
use crate::scene_shading::SceneShading;
use crate::staging::{self, StagingRing};
use crate::targets::{FrameSize, Targets};
use crate::uploads::UploadCache;

mod attribution;
mod camera;
mod core_state;
mod display;
mod extra_views;
mod frame;
mod frame_stats;
mod health;
mod impostor_draws;
mod install;
mod instances;
mod meshes;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod opaque;
mod previous;
#[cfg(feature = "dev")]
mod probe;
pub(crate) mod probe_captures;
mod queries;
mod records;
pub(crate) mod scatter_draws;
pub(crate) mod scatter_pool;
mod startup;

pub use attribution::GpuPassTimes;
pub use camera::CameraBindings;
pub use core_state::Core;
pub(crate) use display::DisplayFrame;
pub use display::{DepthTarget, Display};
pub use impostor_draws::ImpostorDraws;
pub use extra_views::CasterDraw;
pub use instances::FramePoses;
pub use instances::Instances;
pub use meshes::{BatchBinds, DrawOrder, MeshPipelines, ShadingSet};
pub use meshes::{extra_layout, vertex_layout};
#[cfg(not(target_arch = "wasm32"))]
pub use native::GpuMark;
pub(crate) use opaque::{OPAQUE_PART, OpaqueReads, encode_phase};
pub(crate) use previous::{History, PreviousFrame};
pub use previous::{MOTION_FORMAT, Motion};
pub use startup::StartingRenderer;

const INITIAL_CAPACITY: usize = 1_024;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct GpuMesh {
    pub(crate) placement: Result<Placement, OwnMesh>,
    pub extended: bool,
    pub index_count: u32,
    pub byte_len: u64,
    pub lods: Vec<(u32, u32, f32)>,
    pub plan: Vec<(u32, u32, f32)>,
    pub casters: Vec<(u32, u32)>,
    pub without_levels: bool,
}

#[derive(Default)]
pub(crate) struct MeshUploads {
    staging: Vec<Vertex>,
}

pub(crate) struct OwnMesh {
    vertices: wgpu::Buffer,
    extras: Option<wgpu::Buffer>,
    indices: wgpu::Buffer,
}

impl GpuMesh {
    pub fn draw<'a>(&'a self, arena: &'a MeshArena) -> MeshDraw<'a> {
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

    pub fn caster_range(&self, level: usize, own: (u32, u32)) -> (u32, u32) {
        self.casters.get(level).copied().unwrap_or(own)
    }
}

pub fn lod_levels(mesh: &GpuMesh) -> usize {
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

pub(crate) struct Culling {
    pub(crate) pyramid: Option<DepthPyramid>,
}

pub(crate) type StructureKey = (u64, u64, u64);

#[derive(Default)]
pub(crate) struct StructureKeys {
    pub(crate) order: Option<StructureKey>,
    pub(crate) variants: Option<(StructureKey, PassKind)>,
    pub(crate) probes: Option<(StructureKey, PassKind)>,
    pub(crate) impostors: Option<(StructureKey, engine_core::LevelOfDetail, bool, u64)>,
    pub(crate) residency: Option<(u64, u64)>,
    pub(crate) uploads: Option<(u64, u64, u64)>,
}

type Validation = std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>;

pub struct Renderer {
    core: Core,
    parts: Parts,
    table: GraphTable,
    graph: FrameGraph,
    installing: Option<Validation>,
    ran: Vec<&'static [&'static str]>,
    #[cfg(feature = "profiling")]
    prelude_owners: Vec<(&'static str, &'static str)>,
}

impl Renderer {
    pub fn installed_passes(&self) -> &'static [&'static str] {
        self.table.names()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.core.display.resize(width, height);
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.core.display.ask_render_size(width, height);
    }

    pub fn set_cooking(&mut self, cooking: bool) {
        self.core.mesh_pipelines.cooking = cooking;
        self.parts.set_cooking(cooking);
        self.core.impostors.atlas.set_cooking(cooking);
    }

    fn view<'f>(&'f self, info: &'f FrameInfo<'f>) -> crate::parts::View<'f> {
        crate::parts::View {
            core: &self.core,
            parts: &self.parts,
            table: &self.table,
            frame: info,
        }
    }

    fn steps(
        &mut self,
        info: &FrameInfo<'_>,
        mut act: impl FnMut(&mut dyn GpuPart, &mut crate::parts::Step<'_>),
    ) {
        let (parts, core, table) = (&mut self.parts, &mut self.core, &self.table);
        for index in 0..parts.len() {
            let _owner = crate::memory::owned(parts.decl(index).brick);
            parts.with_part(index, |part, parts| {
                act(
                    part,
                    &mut crate::parts::Step {
                        core,
                        parts,
                        table,
                        frame: info,
                    },
                );
            });
        }
    }

}
