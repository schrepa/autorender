//! WebGPU rendering of owned snapshots, without access to the gameplay world.
//!
//! The browser renderer owns one canvas and all its GPU resources. CPU upload
//! records and WGSL live together; ordinary gameplay never handles GPU objects.

/// Maximum instances accepted in a snapshot; bounds CPU staging and GPU memory.
pub const MAX_INSTANCES: usize = 262_144;

/// Maximum distinct meshes referenced by a snapshot and retained on the GPU.
pub const MAX_RESIDENT_MESHES: usize = 128;

/// Maximum source vertex/index bytes resident on the GPU (32 MiB).
pub const MAX_RESIDENT_MESH_BYTES: usize = 32 * 1024 * 1024;

/// Maximum distinct asset textures referenced by a snapshot and retained on the GPU.
pub const MAX_RESIDENT_TEXTURES: usize = 64;

/// Maximum RGBA8 asset texels resident on the GPU (32 MiB), excluding renderer white.
pub const MAX_RESIDENT_TEXTURE_BYTES: usize = 32 * 1024 * 1024;

/// Work submitted by one frame, plus GPU asset residency after retirement.
/// Skipped frames return zero work counters and current residency counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// Number of indexed draw commands submitted.
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
}

#[cfg(any(target_arch = "wasm32", test))]
mod gpu_data;

#[cfg(any(target_arch = "wasm32", test))]
mod batches;
#[cfg(any(target_arch = "wasm32", test))]
mod gpu_uploads;
#[cfg(any(target_arch = "wasm32", test))]
mod retained_uploads;
#[cfg(any(target_arch = "wasm32", test))]
mod uploads;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod gpu_presentation;
#[cfg(target_arch = "wasm32")]
mod mipmaps;
#[cfg(all(target_arch = "wasm32", feature = "profiling"))]
mod profiling;
#[cfg(target_arch = "wasm32")]
mod staging;
#[cfg(target_arch = "wasm32")]
mod temporal;
#[cfg(target_arch = "wasm32")]
pub use browser::Renderer;
