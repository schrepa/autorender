use std::sync::{Arc, Mutex};

use engine_core::{Error, Expected};
use glam::Mat4;

use crate::encode::{Assets, Run};
use crate::graph::PlanCache;
use crate::materials::MaterialStore;
use crate::mesh_arena::GpuMeshes;
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
pub(crate) use display::Display;
use display::DisplayFrame;
pub use install::{Installation, InstalledPart, InstalledPiece, install_noop};
#[cfg(not(target_arch = "wasm32"))]
pub use native::GpuMark;
pub use startup::StartingRenderer;

pub(crate) fn new_assets(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> Assets {
    Assets {
        meshes: GpuMeshes::new(device),
        materials: MaterialStore::new(device, queue),
        impostors: crate::impostors::Impostors::new(device),
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
    pub(crate) pipeline_counts: crate::pipelines::PipelineCounts,
}

pub struct Renderer {
    surface: Option<DisplayFrame>,
    core: Core,
    run: Option<Run>,
    assets: Option<Assets>,
    plans: PlanCache,
    passes: &'static [&'static str],
    spans: Vec<&'static str>,
    #[cfg(feature = "profiling")]
    prelude_owners: Vec<(&'static str, &'static str)>,
    installing: Option<Validation>,
    previous: Option<(Mat4, (u32, u32), u64)>,
    seconds: Option<f32>,
    part_names: &'static [&'static str],
    linked: engine_core::LinkedStats,
    resolver: Option<(&'static str, &'static str)>,
}

impl Renderer {
    pub fn installed_passes(&self) -> &[&'static str] {
        self.passes
    }

    pub fn pass_owner(&self, pass: &str) -> Option<&'static str> {
        if let Some((resolved, package)) = self.resolver
            && resolved == pass
        {
            return Some(package);
        }
        let machine = &self.run.as_ref()?.machine;
        let declared = machine
            .declared
            .passes
            .iter()
            .find(|known| known.group.unwrap_or(known.name) == pass || known.name == pass)?;
        machine
            .declared
            .parts
            .get(declared.part as usize)
            .map(|part| part.package)
    }

    pub fn readback_overflows(&self) -> u64 {
        self.run.as_ref().map_or(0, |run| run.overflows)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.core.display.resize(width, height);
    }
}
