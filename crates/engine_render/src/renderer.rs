use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use engine_core::{Error, Expected, MeshHandle};
use glam::Mat4;

use crate::encode::{Assets, Run};
use crate::gpu_data::{ExtraVertex, Vertex};
use crate::graph::PlanCache;
use crate::materials::MaterialStore;
use crate::memory::buffer_init;
use crate::mesh_arena::{MeshArena, MeshDraw, Placement};
use crate::platform::{RenderTarget, failed, failed_with};
use crate::profiling::GpuProfiler;

mod attribution;
mod display;
mod frame;
mod health;
mod install;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod queries;
mod startup;

pub use attribution::GpuPassTimes;
pub(crate) use display::{Display, DisplayFrame};
#[cfg(not(target_arch = "wasm32"))]
pub use native::GpuMark;
pub use startup::StartingRenderer;

pub struct GpuMesh {
    pub(crate) placement: Result<Placement, OwnMesh>,
    pub extended: bool,
    pub index_count: u32,
    pub byte_len: u64,
    pub lods: Vec<(u32, u32, f32)>,
    pub casters: Vec<(u32, u32)>,
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

pub(crate) fn upload_meshes(
    device: &wgpu::Device,
    queue: &crate::gpu_calls::Queue,
    assets: &mut Assets,
    meshes: &[engine_core::RenderMesh],
) -> u64 {
    let mut uploaded = 0;
    let mut staging: Vec<Vertex> = Vec::new();
    for rendered in meshes {
        let std::collections::btree_map::Entry::Vacant(entry) = assets.meshes.entry(rendered.handle) else {
            continue;
        };
        let source = &rendered.mesh;
        staging.clear();
        staging.extend(source.vertices().iter().map(Vertex::from));
        let extras = ExtraVertex::stream(source);
        let mut streams: Vec<&[u8]> = vec![bytemuck::cast_slice(&staging)];
        if let Some(extras) = &extras {
            streams.push(bytemuck::cast_slice(extras));
        }
        let (indices, lods, casters) = with_levels(source, rendered.lods.as_deref());
        let index_count = source.indices().len() as u32;
        let placement = assets
            .arena
            .insert(device, queue, &streams, staging.len() as u32, &indices)
            .ok_or_else(|| {
                let buffer = |label, contents: &[u8], usage| {
                    buffer_init(
                        device,
                        &wgpu::util::BufferInitDescriptor {
                            label: Some(label),
                            contents,
                            usage,
                        },
                    )
                };
                let vertex = wgpu::BufferUsages::VERTEX;
                OwnMesh {
                    vertices: buffer("engine/mesh-vertices", streams[0], vertex),
                    extras: extras.as_ref().map(|stream| {
                        buffer("engine/mesh-extras", bytemuck::cast_slice(stream), vertex)
                    }),
                    indices: buffer(
                        "engine/mesh-indices",
                        bytemuck::cast_slice(&indices),
                        wgpu::BufferUsages::INDEX,
                    ),
                }
            });
        let byte_len = source.byte_len() as u64;
        entry.insert(GpuMesh {
            placement,
            extended: extras.is_some(),
            index_count,
            byte_len,
            lods,
            casters,
        });
        uploaded += byte_len;
    }
    uploaded
}

pub(crate) fn new_assets(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> Assets {
    Assets {
        meshes: BTreeMap::<MeshHandle, GpuMesh>::new(),
        arena: MeshArena::new(
            device,
            [size_of::<Vertex>(), size_of::<ExtraVertex>()].map(|bytes| bytes as u64),
        ),
        materials: MaterialStore::new(device, queue),
    }
}

pub fn inverse_or_identity(matrix: Mat4) -> Mat4 {
    let inverse = matrix.inverse();
    if inverse.is_finite() {
        inverse
    } else {
        Mat4::IDENTITY
    }
}

type Validation = std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>;

pub(crate) struct Core {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: crate::gpu_calls::Queue,
    pub(crate) adapter: crate::AdapterInfo,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) debugger_capture: bool,
    pub(crate) display: Display,
    pub(crate) health: Arc<Mutex<health::Reports>>,
    pub(crate) profiler: GpuProfiler,
}

pub struct Renderer {
    core: Core,
    run: Option<Run>,
    assets: Option<Assets>,
    plans: PlanCache,
    passes: Vec<&'static str>,
    installing: Option<Validation>,
    previous: Option<Mat4>,
    seconds: Option<f32>,
}

impl Renderer {
    pub fn installed_passes(&self) -> &[&'static str] {
        &self.passes
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.core.display.resize(width, height);
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.core.display.ask_render_size(width, height);
    }
}
