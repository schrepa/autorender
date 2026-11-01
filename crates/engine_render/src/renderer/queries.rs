use super::*;

impl Renderer {
    pub fn adapter(&self) -> &crate::AdapterInfo {
        &self.core.adapter
    }

    pub fn device_limits(&self) -> wgpu::Limits {
        self.core.device.limits()
    }

    pub fn device_features(&self) -> wgpu::Features {
        self.core.device.features()
    }

    pub fn texture_sampling(&self) -> engine_core::TextureSampling {
        use engine_core::{TextureFamily, TextureSampling};
        let features = self.core.device.features();
        if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            TextureSampling::Family(TextureFamily::Bc7)
        } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
            TextureSampling::Family(TextureFamily::Astc)
        } else {
            TextureSampling::Decompressed
        }
    }

    pub fn view_size(&self) -> Option<(u32, u32)> {
        (!self.core.display.suspended).then(|| self.core.display.size().pixels())
    }

    pub fn render_size(&self) -> Option<(u32, u32)> {
        (!self.core.display.suspended).then(|| self.core.display.region.render.pixels())
    }

    pub fn times_frames(&self) -> bool {
        self.core.display.times_frames()
    }

    pub fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.core.display.take_frame_time()
    }

    #[cfg(feature = "profiling")]
    pub fn profile_begin(&mut self, capture_id: u64) {
        self.core.profiler.begin(&self.core.device, capture_id);
    }

    #[cfg(feature = "profiling")]
    pub fn profile_poll(&mut self) {
        self.core.profiler.poll();
    }

    #[cfg(feature = "profiling")]
    pub fn profile_status(&self) -> serde_json::Value {
        self.core.profiler.status()
    }

    #[cfg(feature = "profiling")]
    pub fn profile_report(&mut self) -> serde_json::Value {
        self.core.profiler.report()
    }

    pub fn is_texture_resident(&self, handle: engine_core::TextureHandle) -> bool {
        let assets = self.run.as_ref().map(|run| &run.assets).or(self.assets.as_ref());
        assets.is_some_and(|assets| assets.materials.is_texture_resident(handle))
    }
}
