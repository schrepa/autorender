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
#[cfg(not(target_arch = "wasm32"))]
mod cache;
mod display;
mod frame;
mod health;
mod install;
mod kind_passes;
#[cfg(not(target_arch = "wasm32"))]
mod native;
mod queries;
#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
mod reload;
mod startup;

pub use attribution::GpuPassTimes;
pub(crate) use display::Display;
use display::DisplayFrame;
pub use health::DeviceHealth;
pub use install::{Ahead, Installation, InstalledPart, device_limits, install_noop};
pub use startup::StartingRenderer;

pub(crate) fn new_assets(device: &wgpu::Device, queue: &crate::gpu_calls::Queue) -> Assets {
    Assets {
        meshes: GpuMeshes::new(device),
        materials: MaterialStore::new(device, queue),
    }
}

type Validation = std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>;

fn validate_also(installing: &mut Option<Validation>, popped: Validation) {
    *installing = Some(match installing.take() {
        None => popped,
        Some(earlier) => Box::pin(async move {
            let first = earlier.await;
            let second = popped.await;
            first.or(second)
        }),
    });
}

pub(crate) struct Core {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: crate::gpu_calls::Queue,
    pub(crate) adapter: crate::AdapterInfo,
    pub(crate) display: Display,
    pub(crate) health: Arc<Mutex<health::Reports>>,
    pub(crate) profiler: GpuProfiler,
    pub(crate) ledger: crate::pipelines::Ledger,
    pub(crate) limits: engine_core::DeviceLimits,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) pipeline_cache: Option<cache::KeptCache>,
}

pub struct Renderer {
    surface: Option<DisplayFrame>,
    hidden: frame::Hidden,
    core: Core,
    run: Option<Run>,
    assets: Option<Assets>,
    plans: PlanCache,
    passes: &'static [&'static str],
    spans: Vec<&'static str>,
    #[cfg(feature = "profiling")]
    prelude_owners: Vec<(&'static str, &'static str)>,
    installing: Option<Validation>,
    part_names: &'static [&'static str],
    recovering: Option<startup::Opening>,
    lost_bricks: Vec<engine_registry::BrickId>,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    watched: reload::Watched,
}

impl Renderer {
    pub fn pass_owner(&self, pass: &str) -> Option<&'static str> {
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

    pub fn device_limits(&self) -> engine_core::DeviceLimits {
        self.core.limits
    }

    pub fn readback_overflows(&self) -> u64 {
        self.run
            .as_ref()
            .map_or(0, |run| run.reading.list.overflows())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.core.display.resize(width, height);
    }
}
