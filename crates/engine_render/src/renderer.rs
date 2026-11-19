use std::sync::{Arc, Mutex};

use engine_core::{Error, Expected};
use glam::Mat4;

use crate::encode::{Assets, Run};
use crate::mesh_arena::GpuMeshes;
use crate::textures::TextureStore;
use crate::platform::{RenderTarget, failed, failed_with};
use crate::profiling::GpuProfiler;

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

pub(crate) use display::Display;
use display::DisplayFrame;
#[cfg(any(feature = "dev", feature = "profiling"))]
pub use frame::names::gpu_names;
pub use health::DeviceHealth;
pub use install::{Ahead, Installation, device_limits, install_noop};
pub use startup::StartingRenderer;

pub(crate) fn new_assets(device: &wgpu::Device) -> Assets {
    Assets {
        meshes: GpuMeshes::new(device),
        textures: TextureStore::default(),
        batches: crate::encode::HeldBatches::default(),
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
    pub(crate) pipeline_cache: Result<cache::KeptCache, &'static str>,
}

pub struct Renderer {
    surface: Option<DisplayFrame>,
    hidden: frame::Hidden,
    picture: frame::Picture,
    core: Core,
    run: Option<Run>,
    assets: Option<Assets>,
    passes: &'static [&'static str],
    spans: Vec<&'static str>,
    installing: Option<Validation>,
    recovering: Option<startup::Opening>,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    watched: reload::Watched,
}

impl Renderer {
    pub fn device_limits(&self) -> engine_core::DeviceLimits {
        self.core.limits
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.core.display.resize(width, height);
    }
}
