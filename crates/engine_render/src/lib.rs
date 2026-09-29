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
    /// GPU-culled frames with occlusion culling (decision 040): visible
    /// instances the occlusion test left undrawn, read back with
    /// `gpu_visible`; `None` otherwise.
    pub occluded_instances: Option<u32>,
    /// GPU-culled frames with sub-pixel culling (decision 041): visible
    /// instances whose bounds cover no pixel center, left undrawn, read back
    /// with `gpu_visible`; `None` otherwise.
    pub subpixel_instances: Option<u32>,
    /// GPU-culled frames with levels of detail (decision 047): instances
    /// drawn as their mesh, its two simplifications and impostors, read
    /// back with `gpu_visible`; `None` otherwise.
    pub lod_instances: Option<[u32; 4]>,
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
    /// Whether this frame's glass refracted the opaque scene (decision 038).
    pub refraction: bool,
    /// Whether the visibility buffer resolved the camera's virtual geometry
    /// (decision 045) rather than drawing it forward.
    pub visibility_buffer: bool,
    /// Transparent draws (decision 026), also counted in `draw_calls`, and
    /// the particles this frame drew; dropped particles are those a budget
    /// or the GPU's bindings left out.
    pub transparent_draws: u32,
    pub particles: u32,
    pub dropped_particles: u32,
    /// Decals this frame drew (decision 039); dropped decals are those a
    /// budget or the GPU's binding left out.
    pub decals: u32,
    pub dropped_decals: u32,
    /// Impostors baked and resident (decision 048), and opaque draw groups
    /// that qualified but draw their meshes because a budget or the GPU's
    /// texture array layers left them none.
    pub impostors: u32,
    pub dropped_impostors: u32,
    /// Reflection probes (decision 043) surfaces could reflect this frame,
    /// the cube faces it captured, and probes a budget or the GPU's texture
    /// array layers left out.
    pub reflection_probes: u32,
    pub reflection_probe_faces: u32,
    pub dropped_reflection_probes: u32,
    /// Clusters and triangles virtual geometry (decision 044) drew for the
    /// camera and into shadow maps, the camera's clusters occlusion culling
    /// left undrawn, and selections dropped while their storage grew, read
    /// back a few frames late.
    pub virtual_clusters: u32,
    pub virtual_triangles: u32,
    pub virtual_shadow_clusters: u32,
    pub virtual_shadow_triangles: u32,
    pub virtual_occluded_clusters: u32,
    pub dropped_virtual: u32,
    /// Streamed pages (decision 044): resident, still loading, wanted and
    /// lacking in the last read-back frame (0 once its view had all the
    /// detail it asked for), and those of them (with arrivals since) that
    /// found no room under the budget or the device's limits.
    pub virtual_resident_pages: u32,
    pub virtual_loading_pages: u32,
    pub virtual_missing_pages: u32,
    pub dropped_virtual_pages: u32,
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
    /// Kept-map redraw: 0 none, 1 a rectangle of dirty tiles, 2 the whole
    /// map (every still caster).
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
mod decals;
#[cfg(target_arch = "wasm32")]
mod depth_pyramid;
#[cfg(any(target_arch = "wasm32", test))]
mod gpu_uploads;
#[cfg(any(target_arch = "wasm32", test))]
mod impostors;
#[cfg(any(target_arch = "wasm32", test))]
mod lights;
#[cfg(any(target_arch = "wasm32", test))]
mod particles;
#[cfg(any(target_arch = "wasm32", test))]
mod probes;
#[cfg(any(target_arch = "wasm32", test))]
mod retained_uploads;
#[cfg(any(target_arch = "wasm32", test))]
mod shadow_fit;
#[cfg(any(target_arch = "wasm32", test))]
mod uploads;
#[cfg(any(target_arch = "wasm32", test))]
mod virtual_streaming;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod environment;
#[cfg(target_arch = "wasm32")]
mod gpu_presentation;
#[cfg(target_arch = "wasm32")]
mod materials;
#[cfg(any(target_arch = "wasm32", test))]
mod mesh_arena;
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
mod virtual_geometry;
#[cfg(target_arch = "wasm32")]
mod visibility;
#[cfg(target_arch = "wasm32")]
pub use browser::Renderer;
