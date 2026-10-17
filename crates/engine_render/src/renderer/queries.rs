//! What hosts ask a renderer between frames: its GPU and capacities, what
//! is resident, how meshes draw (levels of detail, impostors), whether
//! pipelines still compile, and its profiling captures.

use super::*;

impl Renderer {
    /// The GPU the renderer draws on.
    pub fn adapter(&self) -> &crate::AdapterInfo {
        &self.adapter
    }

    /// Terrain tiles the device's texture array holds.
    pub fn terrain_tile_capacity(&self) -> u32 {
        self.display.max_layers
    }

    /// The size frames draw at, in physical pixels: the size asked for,
    /// fitted within the device's largest texture; `None` while a view
    /// without area suspends drawing.
    pub fn view_size(&self) -> Option<(u32, u32)> {
        (!self.display.suspended).then(|| self.display.size().pixels())
    }

    /// Start a bounded GPU capture. Allocate its reusable storage outside frame work.
    /// The caller supplies the matching CPU capture ID.
    #[cfg(feature = "profiling")]
    pub fn profile_begin(&mut self, capture_id: u64) {
        self.profiler.begin(&self.device, capture_id);
    }

    /// Consume completed asynchronous readbacks without waiting or submitting work.
    #[cfg(feature = "profiling")]
    pub fn profile_poll(&mut self) {
        self.profiler.poll();
    }

    /// Bounded metadata and completeness counters, without copying sample arrays.
    #[cfg(feature = "profiling")]
    pub fn profile_status(&self) -> serde_json::Value {
        self.profiler.status()
    }

    /// Export GPU durations for the retained capture. Late maps may settle after
    /// CPU capture stops; call again after yielding to the browser event loop.
    #[cfg(feature = "profiling")]
    pub fn profile_report(&mut self) -> serde_json::Value {
        self.profiler.report()
    }

    /// Whether this immutable texture currently has uploaded GPU storage.
    /// Skipped frames do not upload missing sources; unused handles retire even
    /// when suspended. Device loss requires replacing this renderer.
    pub fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.materials.is_texture_resident(handle)
    }

    /// The current material's coverage curve or legacy fallback reason.
    /// None until an eligible binding uploads. Equivalent material handles
    /// share these facts, just as they share their GPU binding.
    pub fn material_alpha_coverage(
        &self,
        material: &engine_core::Material,
    ) -> Option<engine_core::AlphaCoverageState> {
        self.materials.alpha_coverage(material)
    }

    /// Mesh `handle` as uploaded: whether with skin joints, and its coarser
    /// levels of detail, finest first, each one's index count and largest
    /// deviation in mesh units. None before it uploads.
    pub fn mesh_levels(&self, handle: MeshHandle) -> Option<(bool, Vec<(u32, f32)>)> {
        let mesh = self.meshes.get(&handle)?;
        let levels = mesh
            .lods
            .iter()
            .map(|&(_, count, error)| (count, error))
            .collect();
        Some((mesh.skinned, levels))
    }

    /// The level of detail the slot `slot` drew at, read back a few frames
    /// late: 0 the mesh, 1 and 2 its simplifications, 3 its impostor card.
    /// Asking watches the slot; None until a GPU-culled frame with levels of
    /// detail on draws it after that.
    pub fn drawn_level(&self, slot: u32) -> Option<u32> {
        self.gpu_presentation.watch(slot);
        match self.gpu_presentation.watched() {
            // States carry their level once set; zero is undrawn.
            Some((watched, state)) if watched == slot && state & LOD_STATE != 0 => Some(state & 3),
            _ => None,
        }
    }

    /// Whether an impostor card of mesh `mesh` drawn with `material` is
    /// baked. None where it never draws as one: before upload, and for
    /// skinned meshes and those of too few triangles.
    pub fn impostor_baked(&self, mesh: MeshHandle, material: MaterialHandle) -> Option<bool> {
        let gpu = self.meshes.get(&mesh)?;
        if gpu.skinned || (gpu.index_count as usize) / 3 < IMPOSTOR_MIN_TRIANGLES {
            return None;
        }
        Some(self.impostors.atlas.contains((mesh, Some(material))))
    }

    /// Why the latest failed virtual geometry page read did, if any failed
    /// (frame statistics' `virtual_geometry.failed_pages` counts them).
    pub fn latest_page_failure(&self) -> Option<&str> {
        self.virtual_draws.pass.page_failure()
    }

    /// Why simplification refused the latest mesh whose levels of detail it
    /// left out, if it refused any (frame statistics'
    /// `level_of_detail.failed_meshes` counts them).
    pub fn latest_lod_failure(&self) -> Option<&str> {
        self.mesh_uploads.lod_failures.1.as_deref()
    }

    /// Whether pipelines created by earlier frames may still be compiling:
    /// the GPU has not yet completed the last frame that created some.
    pub fn pipelines_compiling(&self) -> bool {
        // Native interfaces report finished work, compiles among it, when
        // polled.
        crate::platform::poll(&self.device);
        self.pipeline_counts.compiling()
    }

    /// The meshes and textures resident on the GPU.
    pub(super) fn residency(&self) -> Residency {
        let (textures, texture_bytes, _) = self.materials.texture_totals();
        Residency {
            meshes: self.meshes.len() as u32,
            mesh_bytes: self.meshes.values().map(|mesh| mesh.byte_len).sum(),
            textures,
            texture_bytes,
        }
    }

    /// A skipped frame's stats: no work, and the current residency.
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

/// Meshes and textures resident on the GPU, and their bytes.
pub(super) struct Residency {
    pub(super) meshes: u32,
    pub(super) mesh_bytes: u64,
    pub(super) textures: u32,
    pub(super) texture_bytes: u64,
}
