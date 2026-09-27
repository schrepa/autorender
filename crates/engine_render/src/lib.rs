//! WebGPU rendering of owned snapshots, without access to the gameplay world.
//!
//! The browser renderer owns one canvas and all its GPU resources. CPU upload
//! records and WGSL live together; ordinary gameplay never handles GPU objects.

/// Work submitted by one frame, plus GPU asset residency after retirement.
/// Skipped frames return zero work counters and current residency counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// Indexed draw commands of the camera pass; `shadows` counts the
    /// shadow maps' separately.
    pub draw_calls: u32,
    /// Actual bytes uploaded or initialized: changed camera/instances and new assets.
    /// An unchanged rendered frame can submit draws with zero upload bytes.
    /// Excludes renderer startup resources such as the four-byte white texture.
    pub upload_bytes: u64,
    /// Number of mesh instances submitted (before GPU clipping). GPU-culled
    /// frames (decision 014) report the read-back visible count limited by the
    /// instance budget, or the budgeted candidate count before any read-back.
    pub instances: u32,
    /// GPU-culled frames only: the pre-budget visible count read back from the
    /// latest completed frame, a few frames old; `None` before the first one.
    pub gpu_visible: Option<u32>,
    /// Number of distinct meshes retained on the GPU.
    pub resident_meshes: u32,
    /// Vertex/index bytes initialized for meshes first seen this frame.
    pub mesh_upload_bytes: u64,
    /// Vertex/index bytes retained on the GPU after this frame.
    pub resident_mesh_bytes: u64,
    /// Number of asset textures retained on the GPU, excluding renderer white.
    pub resident_textures: u32,
    /// RGBA8 texel bytes uploaded for textures first seen this frame.
    pub texture_upload_bytes: u64,
    /// RGBA8 asset texel bytes retained on the GPU after this frame.
    pub resident_texture_bytes: u64,
    /// Whether the frame was submitted. A suspended, occluded or recovering
    /// surface skips drawing and uploads, which later frames then catch up.
    pub submitted: bool,
    /// This frame's sun shadow work (decision 020); `None` without shadows.
    pub shadows: Option<ShadowStats>,
    /// Whether this frame applied ambient occlusion and bloom (decision 021).
    pub ambient_occlusion: bool,
    pub bloom: bool,
    /// This frame's local lights (decision 023); `None` when it shaded none.
    pub lights: Option<LightStats>,
    /// Whether this frame applied sun shafts (decision 024) and screen-space
    /// reflections (decision 025).
    pub sun_shafts: bool,
    pub reflections: bool,
    /// Transparent draws (decision 026), also counted in `draw_calls`, and
    /// the particles this frame drew; dropped particles are those a budget
    /// or the GPU's bindings left out.
    pub transparent_draws: u32,
    pub particles: u32,
    pub dropped_particles: u32,
}

/// Local lights of one frame (decision 023).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LightStats {
    /// Lights the frame shaded, within the budget.
    pub lights: u32,
    /// Usable lights beyond the budget, or without a valid pose.
    pub dropped: u32,
    /// Lights that reached a view cluster, and cluster entries lost to full
    /// cluster lists, read back a few frames late; `None` before the first
    /// read-back.
    pub visible: Option<u32>,
    pub overflow: Option<u32>,
}

/// Sun shadow work of one frame (decision 020).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowStats {
    /// Indexed draws into the shadow maps, in addition to `draw_calls`.
    pub draws: u32,
    /// Kept-map redraw: 0 none, 1 one strip, 2 the whole map (every still
    /// caster).
    pub kept_redraw: u8,
    /// Casters the near, moving and kept maps drew, read back a few frames
    /// late; `None` before the first read-back. In a settled scene the moving
    /// count is the moving renderables; the kept count covers redraws only.
    pub casters: Option<[u32; 3]>,
}

#[cfg(any(target_arch = "wasm32", test))]
mod gpu_data;

#[cfg(any(target_arch = "wasm32", test))]
mod batches;
#[cfg(any(target_arch = "wasm32", test))]
mod gpu_uploads;
#[cfg(any(target_arch = "wasm32", test))]
mod lights;
#[cfg(any(target_arch = "wasm32", test))]
mod particles;
#[cfg(any(target_arch = "wasm32", test))]
mod retained_uploads;
#[cfg(any(target_arch = "wasm32", test))]
mod shadow_fit;
#[cfg(any(target_arch = "wasm32", test))]
mod uploads;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod gpu_presentation;
#[cfg(target_arch = "wasm32")]
mod mipmaps;
#[cfg(target_arch = "wasm32")]
mod post;
#[cfg(all(target_arch = "wasm32", feature = "profiling"))]
mod profiling;
#[cfg(target_arch = "wasm32")]
mod shadows;
#[cfg(target_arch = "wasm32")]
mod staging;
#[cfg(target_arch = "wasm32")]
mod temporal;
#[cfg(target_arch = "wasm32")]
mod transparent;
#[cfg(target_arch = "wasm32")]
pub use browser::Renderer;
