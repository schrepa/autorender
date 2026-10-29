use super::*;

impl Renderer {
    pub fn adapter(&self) -> &crate::AdapterInfo {
        &self.adapter
    }

    pub fn terrain_tile_capacity(&self) -> u32 {
        self.display.max_layers
    }

    pub fn texture_sampling(&self) -> engine_core::TextureSampling {
        use engine_core::{TextureFamily, TextureSampling};
        let features = self.device.features();
        if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            TextureSampling::Family(TextureFamily::Bc7)
        } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
            TextureSampling::Family(TextureFamily::Astc)
        } else {
            TextureSampling::Decompressed
        }
    }

    pub fn view_size(&self) -> Option<(u32, u32)> {
        (!self.display.suspended).then(|| self.display.size().pixels())
    }

    pub fn render_size(&self) -> Option<(u32, u32)> {
        (!self.display.suspended).then(|| self.display.region.render.pixels())
    }

    pub fn times_frames(&self) -> bool {
        self.display.times_frames()
    }

    pub fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.display.take_frame_time()
    }

    #[cfg(feature = "profiling")]
    pub fn profile_begin(&mut self, capture_id: u64) {
        self.profiler.begin(&self.device, capture_id);
    }

    #[cfg(feature = "profiling")]
    pub fn profile_poll(&mut self) {
        self.profiler.poll();
    }

    #[cfg(feature = "profiling")]
    pub fn profile_status(&self) -> serde_json::Value {
        self.profiler.status()
    }

    #[cfg(feature = "profiling")]
    pub fn profile_report(&mut self) -> serde_json::Value {
        self.profiler.report()
    }

    pub fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.materials.is_texture_resident(handle)
    }

    pub fn mesh_levels(&self, handle: MeshHandle) -> Option<Vec<(u32, f32)>> {
        let mesh = self.meshes.get(&handle)?;
        let levels = mesh
            .lods
            .iter()
            .map(|&(_, count, error)| (count, error))
            .collect();
        Some(levels)
    }

    pub fn drawn_level(&self, slot: u32) -> Option<u32> {
        self.gpu_presentation.watch(slot);
        match self.gpu_presentation.watched() {
            Some((watched, state)) if watched == slot && state & LOD_STATE != 0 => Some(state & 3),
            _ => None,
        }
    }

    pub fn slot_read_back(&self, slot: u32) -> Option<(Option<u32>, u64)> {
        self.gpu_presentation.watch(slot);
        let (watched, state) = self.gpu_presentation.watched()?;
        let age = self.gpu_presentation.read_back_age()?;
        (watched == slot).then(|| ((state & LOD_STATE != 0).then_some(state & 3), age))
    }

    pub fn impostor_resident(&self, mesh: MeshHandle, material: MaterialHandle) -> Option<bool> {
        let gpu = self.meshes.get(&mesh)?;
        if (gpu.index_count as usize) / 3 < IMPOSTOR_MIN_TRIANGLES {
            return None;
        }
        Some(self.impostors.atlas.contains((mesh, Some(material))))
    }

    pub fn pipelines_compiling(&self) -> bool {
        crate::platform::poll(&self.device);
        self.pipeline_counts.compiling()
    }

    pub fn pipeline_list(&self) -> Vec<u8> {
        self.pipeline_counts.list()
    }

    pub fn create_listed_pipelines(&mut self, record: &[u8]) -> Result<u32, RenderError> {
        self.pipeline_counts
            .create_listed(record)
            .map_err(|error| render_error("The pipeline list does not read", error))
    }

    pub fn pipelines_listed(&self) -> (u32, u32) {
        self.pipeline_counts.listed()
    }

    pub(super) fn residency(&self) -> Residency {
        let (textures, texture_bytes, _) = self.materials.texture_totals();
        Residency {
            meshes: self.meshes.len() as u32,
            mesh_bytes: self.meshes.values().map(|mesh| mesh.byte_len).sum(),
            textures,
            texture_bytes,
        }
    }

    pub(super) fn residency_stats(&self) -> RenderStats {
        let residency = self.residency();
        RenderStats {
            resident_meshes: residency.meshes,
            resident_mesh_bytes: residency.mesh_bytes,
            resident_textures: residency.textures,
            resident_texture_bytes: residency.texture_bytes,
            pipelines: self.pipeline_counts.created(),
            ..RenderStats::default()
        }
    }
}

pub(super) struct Residency {
    pub(super) meshes: u32,
    pub(super) mesh_bytes: u64,
    pub(super) textures: u32,
    pub(super) texture_bytes: u64,
}
