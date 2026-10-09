//! WebGPU rendering of engine frames, without access to the gameplay world.
//!
//! The browser renderer owns one canvas and all its GPU resources. CPU upload
//! records and WGSL live together; ordinary gameplay never handles GPU objects.

use serde::Serialize;

/// Work submitted by one frame, plus GPU asset residency after retirement.
/// Skipped frames return zero work counters and current residency counters.
///
/// This is the one definition of the renderer's statistics: hosts and agents
/// read them as the JSON object it serializes to, under these names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RenderStats {
    /// Draw commands of the camera's mesh passes, each counted once however
    /// many occlusion phases issue it: each opaque mesh
    /// group's, one per level of detail on GPU-presented frames,
    /// the impostors', virtual geometry's forward bins
    /// and the transparent groups'. `shadow_draws` counts the
    /// shadow maps'.
    pub draw_calls: u32,
    /// Actual bytes uploaded or initialized: changed camera/instances and new assets.
    /// An unchanged rendered frame can submit draws with zero upload bytes.
    /// Excludes renderer startup resources such as the four-byte white texture.
    pub upload_bytes: u64,
    /// Number of mesh instances submitted (before GPU clipping). GPU-culled
    /// frames report the read-back visible count limited by the
    /// instance budget, or the budgeted candidate count before any read-back.
    pub instances: u32,
    /// GPU-culled frames only: the pre-budget visible count read back from the
    /// latest completed frame, a few frames old; `None` before the first one.
    #[serde(skip)]
    pub gpu_visible: Option<u32>,
    /// GPU-culled frames with occlusion culling: visible
    /// instances the occlusion test left undrawn, read back with
    /// `gpu_visible`; `None` otherwise.
    pub occluded_instances: Option<u32>,
    /// GPU-culled frames with sub-pixel culling: visible
    /// instances whose bounds cover no pixel center, left undrawn, read back
    /// with `gpu_visible`; `None` otherwise.
    pub subpixel_instances: Option<u32>,
    /// GPU-culled frames with levels of detail: instances
    /// drawn as their mesh, its two simplifications and impostors, read
    /// back with `gpu_visible`; `None` otherwise.
    pub lod0_instances: Option<u32>,
    pub lod1_instances: Option<u32>,
    pub lod2_instances: Option<u32>,
    pub impostor_instances: Option<u32>,
    /// Triangles the camera's mesh draws submitted, at their levels of
    /// detail and impostors' two each: exact on CPU-prepared frames, read
    /// back with `gpu_visible` on GPU-culled ones. Virtual geometry counts
    /// its own in `virtual_triangles`.
    pub triangles: Option<u32>,
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
    #[serde(skip)]
    pub submitted: bool,
    /// Whether the sun cast shadows, the indexed draws into
    /// the shadow maps (beside the camera pass's `draw_calls`), and the kept
    /// map's redraw: 0 none, 1 a rectangle of dirty tiles, 2 the whole map
    /// (every still caster).
    pub shadows_enabled: bool,
    pub shadow_draws: u32,
    pub shadow_kept_redraw: u32,
    /// Casters the near, moving and kept maps drew, read back a few frames
    /// late; `None` before the first read-back. In a settled scene the moving
    /// count is the moving renderables; the kept count covers redraws only.
    pub shadow_near_casters: Option<u32>,
    pub shadow_moving_casters: Option<u32>,
    pub shadow_kept_casters: Option<u32>,
    /// Kept levels: tiles this frame redrew, tiles no level
    /// holds yet (lookups read coarser maps there), and casters the levels
    /// drew, read back like the others.
    pub shadow_level_tiles: u32,
    pub shadow_levels_pending: u32,
    pub shadow_level_casters: Option<u32>,
    /// Whether this frame applied ambient occlusion and bloom.
    pub ambient_occlusion: bool,
    /// Whether the frame traced contact shadows.
    pub contact_shadows: bool,
    pub bloom: bool,
    /// Local lights the frame shaded within the budget, and
    /// usable lights beyond it or without a valid pose.
    pub local_lights: u32,
    pub dropped_lights: u32,
    /// Lights that reached a view cluster, and cluster entries lost to full
    /// cluster lists, read back a few frames late; `None` before the first
    /// read-back or without lights.
    pub lights_visible: Option<u32>,
    pub light_cluster_overflow: Option<u32>,
    /// Whether this frame applied sun shafts and screen-space
    /// reflections.
    pub sun_shafts: bool,
    pub reflections: bool,
    /// Whether this frame's glass refracted the opaque scene.
    pub refraction: bool,
    /// Whether the visibility buffer resolved the camera's virtual geometry
    /// rather than drawing it forward.
    pub visibility_buffer: bool,
    /// Transparent draws, also counted in `draw_calls`, and
    /// the particles this frame drew; dropped particles are those a budget
    /// or the GPU's bindings left out.
    pub transparent_draws: u32,
    pub particles: u32,
    pub dropped_particles: u32,
    /// Decals this frame drew; dropped decals are those a
    /// budget or the GPU's binding left out.
    pub decals: u32,
    pub dropped_decals: u32,
    /// Impostors baked and resident, and opaque draw groups
    /// that qualified but draw their meshes because a budget or the GPU's
    /// texture array layers left them none.
    pub impostors: u32,
    pub dropped_impostors: u32,
    /// Reflection probes surfaces could reflect this frame,
    /// the cube faces it captured, and probes a budget or the GPU's texture
    /// array layers left out.
    pub reflection_probes: u32,
    pub reflection_probe_faces: u32,
    pub dropped_reflection_probes: u32,
    /// Pipelines the renderer has created, and those this frame created:
    /// 0 once everything the scene draws is compiled.
    pub pipelines: u32,
    pub pipelines_created: u32,
    /// Clusters and triangles virtual geometry drew for the
    /// camera and into shadow maps, the camera's clusters occlusion culling
    /// left undrawn, and selections dropped while their storage grew, read
    /// back a few frames late.
    pub virtual_clusters: u32,
    pub virtual_triangles: u32,
    pub virtual_shadow_clusters: u32,
    pub virtual_shadow_triangles: u32,
    pub virtual_occluded_clusters: u32,
    pub dropped_virtual: u32,
    /// Streamed pages: resident, still loading, wanted and
    /// lacking in the last read-back frame (0 once its view had all the
    /// detail it asked for), and those of them (with arrivals since) that
    /// found no room under the budget or the device's limits.
    pub virtual_resident_pages: u32,
    pub virtual_loading_pages: u32,
    pub virtual_missing_pages: u32,
    pub dropped_virtual_pages: u32,
    /// Terrain: patches and triangles the camera drew, and
    /// patches drawn into shadow maps; tiles resident on the GPU, tiles
    /// still generating or loading, and tiles dropped by the tile budget or
    /// the device's texture array layers; patches standing in for finer
    /// levels not resident yet.
    pub terrain_patches: u32,
    pub terrain_triangles: u32,
    pub terrain_shadow_patches: u32,
    pub terrain_tiles: u32,
    pub terrain_loading_tiles: u32,
    /// Tiles the GPU gave control values this frame.
    pub terrain_gpu_controls: u32,
    pub dropped_terrain_tiles: u32,
    pub terrain_fallback_patches: u32,
    /// Terrains casting long sun shadows through horizon maps,
    /// and the maps this frame rebuilt.
    pub terrain_horizons: u32,
    pub terrain_horizon_builds: u32,
    /// Terrains whose sky visibility maps are built, and
    /// the rows of them this frame built.
    pub terrain_sky_maps: u32,
    pub terrain_sky_rows: u32,
    /// The runtime virtual texture: the pages its sheets
    /// hold, pages resident, baked this frame (absent or stale), asked for
    /// but absent (in the last feedback read back), asked for with no slot
    /// left under its budget or the device's texture array layers, and
    /// evicted for others this frame.
    pub terrain_vt_slots: u32,
    pub terrain_vt_pages: u32,
    pub terrain_vt_bakes: u32,
    pub terrain_vt_missing: u32,
    pub dropped_terrain_vt_pages: u32,
    pub terrain_vt_evictions: u32,
    /// Ground cover: blades the last read-back frame drew,
    /// and those past the capacity or budget, which dropped.
    pub ground_cover_blades: u32,
    pub dropped_ground_cover: u32,
    /// Meters from the eye ground cover reached this frame (rounded), and
    /// the most tiers it grew in: the radius's and doublings
    /// past it.
    pub ground_cover_reach: u32,
    pub ground_cover_tiers: u32,
    /// Objects terrains' placement layers stand on resident level-0 tiles,
    /// each culled on the GPU every frame.
    pub placed_objects: u32,
    /// Of those, objects standing far away, which hide where
    /// the tiles holding them instead are resident.
    pub placed_far_objects: u32,
    /// Chunks of terrains' water drawn: lakes' and rivers'
    /// ribbons.
    pub water_chunks: u32,
    /// The sea's sectors drawn, and their triangles.
    pub ocean_sectors: u32,
    pub ocean_triangles: u32,
    /// Runtime deformation: stamps this frame pressed into
    /// the map around the camera, and its texels along a side (0 while
    /// off; fewer than the settings' where the device's buffers cannot
    /// hold them).
    pub deformation_stamps: u32,
    pub deformation_texels: u32,
    /// Rows of the clouds' dome this frame marched: all of
    /// them on exact frames, a slice on ordinary ones, none without clouds.
    pub cloud_rows: u32,
    /// Blocks of the view whose clouds this frame marched:
    /// none without clouds or with their march in the view off.
    pub cloud_view_blocks: u32,
    /// The physically based sky's tables this frame built:
    /// two when its air changed, one more when the sun, the camera's height
    /// or the air changed; none without the sky.
    pub sky_tables: u32,
    /// World streaming: cells holding streamed items, those
    /// loaded, and those within range still waiting; entities streamed
    /// items hold spawned; and the entities this frame spawned and
    /// despawned.
    pub cells: u32,
    pub cells_loaded: u32,
    pub cells_pending: u32,
    pub streamed_entities: u32,
    pub cell_spawns: u32,
    pub cell_despawns: u32,
    /// Placed objects' impostors baked, which the camera
    /// draws far away and shadow maps cast as cards; counted in
    /// `impostors` too.
    pub placed_impostors: u32,
    /// Times the occupancy map around the camera was drawn:
    /// it follows the camera in steps and redraws when objects change.
    pub occupancy_redraws: u32,
    /// Terrain surface materials past the palette's
    /// capacity, or whose maps the shared arrays cannot hold (block
    /// compressed, or sized unlike the first material's), which show a
    /// plain color instead.
    pub dropped_terrain_materials: u32,
    /// The passes this frame ran, in order, named in the
    /// JSON.
    #[serde(
        rename = "render_passes",
        serialize_with = "render_graph::serialize_passes"
    )]
    pub passes: u32,
}

#[cfg(target_arch = "wasm32")]
mod bindings;
#[cfg(target_arch = "wasm32")]
mod cached;
#[cfg(target_arch = "wasm32")]
mod gpu_data;

#[cfg(target_arch = "wasm32")]
mod batches;
#[cfg(target_arch = "wasm32")]
mod clouds;
#[cfg(target_arch = "wasm32")]
mod decals;
#[cfg(target_arch = "wasm32")]
mod deformation;
#[cfg(target_arch = "wasm32")]
mod depth_pyramid;
#[cfg(target_arch = "wasm32")]
mod entries;
#[cfg(target_arch = "wasm32")]
mod gpu_uploads;
#[cfg(target_arch = "wasm32")]
mod impostors;
#[cfg(target_arch = "wasm32")]
mod lights;
#[cfg(target_arch = "wasm32")]
mod particles;
#[cfg(target_arch = "wasm32")]
mod pipelines;
#[cfg(target_arch = "wasm32")]
mod probes;
mod render_graph;
#[cfg(target_arch = "wasm32")]
mod retained_uploads;
#[cfg(target_arch = "wasm32")]
mod shadow_fit;
#[cfg(target_arch = "wasm32")]
mod sky;
#[cfg(target_arch = "wasm32")]
mod uploads;
#[cfg(target_arch = "wasm32")]
mod virtual_streaming;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(target_arch = "wasm32")]
mod buffers;
#[cfg(target_arch = "wasm32")]
mod environment;
#[cfg(target_arch = "wasm32")]
mod gpu_presentation;
#[cfg(target_arch = "wasm32")]
mod ground_cover;
#[cfg(target_arch = "wasm32")]
mod materials;
#[cfg(target_arch = "wasm32")]
mod mesh_arena;
#[cfg(target_arch = "wasm32")]
mod mipmaps;
#[cfg(target_arch = "wasm32")]
mod occupancy;
#[cfg(target_arch = "wasm32")]
mod post;
#[cfg(target_arch = "wasm32")]
mod profiling;
#[cfg(target_arch = "wasm32")]
mod scatter;
#[cfg(target_arch = "wasm32")]
mod shadows;
#[cfg(target_arch = "wasm32")]
mod staging;
#[cfg(target_arch = "wasm32")]
mod targets;
#[cfg(target_arch = "wasm32")]
mod temporal;
#[cfg(target_arch = "wasm32")]
mod terrain;
#[cfg(target_arch = "wasm32")]
mod terrain_horizons;
#[cfg(target_arch = "wasm32")]
mod terrain_maps;
#[cfg(target_arch = "wasm32")]
mod terrain_sky;
#[cfg(target_arch = "wasm32")]
mod terrain_vt;
#[cfg(target_arch = "wasm32")]
mod transparent;
#[cfg(target_arch = "wasm32")]
mod virtual_geometry;
#[cfg(target_arch = "wasm32")]
mod visibility;
#[cfg(target_arch = "wasm32")]
mod water;
#[cfg(target_arch = "wasm32")]
pub use browser::{Renderer, StartingRenderer};
