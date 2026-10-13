//! The one definition of a frame's observations: hosts and agents read them
//! as the JSON object [`FrameStats`] serializes to, under these names; the
//! development protocol describes them from this file's documentation, and
//! profiles count every number in them.
//!
//! A feature's numbers live under its environment section's key (the build
//! checks it), and what a budget or the device's limits left out is named
//! `dropped` or `dropped_*`.
#![deny(missing_docs)]

use serde::{Serialize, Serializer};

/// CPU observations from the last submitted frame and the renderer's report
/// of it. Render time is not GPU time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FrameStats {
    /// The frame these stats describe, as `FrameTime::frame` counts it.
    pub frame: u64,
    /// The last fixed tick that frame completed, as `FixedTime::tick`
    /// counts it.
    pub tick: u64,
    /// Milliseconds the frame spent in fixed and variable simulation.
    pub simulation_ms: f64,
    /// Milliseconds the frame spent extracting the render snapshot.
    pub extraction_ms: f64,
    /// Milliseconds the frame spent preparing and submitting GPU work.
    pub render_ms: f64,
    /// Renderable entities skipped because their derived transform is
    /// invalid.
    pub invalid_transforms: u32,
    /// Every entity carrying a mesh, before transform and asset checks.
    pub render_candidates: u32,
    /// Valid instances surviving culling, before the instance budget;
    /// GPU-culled frames report the latest read-back count.
    pub visible_instances: u32,
    /// Valid instances culling proved outside the view.
    pub culled_instances: u32,
    /// Visible instances the instance budget left undrawn.
    pub dropped_instances: u32,
    /// Renderable entities skipped because a mesh, material, or texture is
    /// absent.
    pub missing_assets: u32,
    /// Whether frustum culling is on.
    pub culling_enabled: bool,
    /// Fixed simulation ticks this frame ran.
    pub fixed_steps: u32,
    /// Total simulation time discarded by the fixed-step budget in this session.
    pub dropped_seconds: f64,
    /// Whether the GPU may still be compiling pipelines earlier frames
    /// created, read when the stats are.
    pub pipelines_compiling: bool,
    /// Whether terrains have yet to load the camera's view whole, read
    /// when the stats are: hosts keep preparing meanwhile.
    pub loading: bool,
    /// The view's width in physical pixels, as the frame drew it: the
    /// host's, unless the device's largest texture holds less, which the
    /// view then fits within at its aspect.
    pub view_width: u32,
    /// The view's height in physical pixels, as the frame drew it.
    pub view_height: u32,
    /// Pixels of the view the host asked for that the device's largest
    /// texture left out.
    pub dropped_view_pixels: u64,
    /// The host's workers and their jobs, read when the stats are.
    pub jobs: crate::JobStats,
    /// Navigation agents at the last fixed tick, read when the stats are.
    pub navigation: crate::NavStats,
    /// The renderer's features it disabled for good, by name: one whose
    /// creation failed validation, or that an uncaptured GPU error named
    /// by its objects' labels. Frames render without them.
    pub disabled_features: Names,
    /// The renderer's report of the frame.
    #[serde(flatten)]
    pub render: RenderStats,
}

/// Work submitted by one frame, plus GPU asset residency after retirement.
/// Skipped frames return zero work counters and current residency counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RenderStats {
    /// Draw commands of the camera's mesh passes, each counted once however
    /// many occlusion phases issue it: each opaque mesh group's, one per
    /// level of detail on GPU-presented frames, the impostors', virtual
    /// geometry's forward bins and the transparent groups'. `shadows.draws`
    /// counts the shadow maps'.
    pub draw_calls: u32,
    /// Actual bytes uploaded or initialized: changed camera/instances and new
    /// assets. An unchanged rendered frame can submit draws with zero upload
    /// bytes. Excludes renderer startup resources such as the four-byte white
    /// texture.
    pub upload_bytes: u64,
    /// Number of mesh instances submitted (before GPU clipping). GPU-culled
    /// frames report the read-back visible count limited by the instance
    /// budget, or the budgeted candidate count before any read-back.
    pub instances: u32,
    /// GPU-culled frames only: the pre-budget visible count read back from the
    /// latest completed frame, a few frames old; `None` before the first one.
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub gpu_visible: Option<u32>,
    /// Triangles the camera's mesh draws submitted, at their levels of
    /// detail and impostors' two each: exact on CPU-prepared frames, read
    /// back with the visible count on GPU-culled ones. Virtual geometry
    /// counts its own in `virtual_geometry.triangles`.
    pub triangles: Option<u32>,
    /// GPU-culled frames only: how many GPU-presented frames late the counts
    /// read back with the visible count (`instances`, `triangles`,
    /// `occlusion_culling`, `subpixel_culling` and `level_of_detail`'s
    /// instances) are, 1 when they arrived as soon as they could, more while
    /// the GPU falls behind; `None` before the first read-back. A check that
    /// changed what they count presents until this is below the frames it
    /// presented since.
    pub read_back_age: Option<u32>,
    /// Number of distinct meshes retained on the GPU.
    pub resident_meshes: u32,
    /// Vertex/index bytes initialized for meshes first seen this frame.
    pub mesh_upload_bytes: u64,
    /// Vertex/index bytes retained on the GPU after this frame.
    pub resident_mesh_bytes: u64,
    /// Number of asset textures retained on the GPU, excluding renderer white.
    pub resident_textures: u32,
    /// Texel bytes uploaded for textures first seen this frame.
    pub texture_upload_bytes: u64,
    /// Asset texel bytes retained on the GPU after this frame.
    pub resident_texture_bytes: u64,
    /// Whether the frame was submitted. A suspended, occluded or recovering
    /// surface skips drawing and uploads, which later frames then catch up.
    #[serde(skip)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub submitted: bool,
    /// Transparent draws, also counted in `draw_calls`.
    pub transparent_draws: u32,
    /// Pipelines the renderer has created.
    pub pipelines: u32,
    /// Pipelines this frame created: 0 once everything the scene draws is
    /// compiled.
    pub pipelines_created: u32,
    /// The passes this frame ran, in order.
    pub render_passes: Names,
    /// Sun shadows.
    pub shadows: ShadowStats,
    /// Ambient occlusion.
    pub ambient_occlusion: Applied,
    /// Contact shadows.
    pub contact_shadows: Applied,
    /// Bloom.
    pub bloom: Applied,
    /// Local lights.
    pub local_lights: LightStats,
    /// Sun shafts.
    pub sun_shafts: Applied,
    /// Screen-space reflections.
    pub reflections: Applied,
    /// Particles.
    pub particles: Drawn,
    /// Glass refraction.
    pub refraction: Applied,
    /// Decals.
    pub decals: Drawn,
    /// Reflection probes.
    pub reflection_probes: ProbeStats,
    /// Virtual geometry.
    pub virtual_geometry: VirtualGeometryStats,
    /// Levels of detail and impostors.
    pub level_of_detail: DetailStats,
    /// Terrains.
    pub terrain: TerrainRenderStats,
    /// Ground cover.
    pub ground_cover: GroundCoverStats,
    /// The occupancy map.
    pub occupancy: OccupancyStats,
    /// Lakes and rivers.
    pub water: WaterStats,
    /// The sea.
    pub ocean: OceanStats,
    /// Runtime deformation.
    pub deformation: DeformationStats,
    /// Clouds.
    pub clouds: CloudStats,
    /// World streaming.
    pub world_streaming: crate::CellStats,
    /// The physically based sky.
    pub sky: SkyStats,
    /// Occlusion culling.
    pub occlusion_culling: CullingStats,
    /// Sub-pixel culling.
    pub subpixel_culling: CullingStats,
    /// Temporal antialiasing.
    pub temporal_antialiasing: Applied,
}

/// A set of names, such as the render passes a frame ran, serialized as
/// the names of its bits, in order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Names {
    /// One bit per name, by its index in `names`.
    pub bits: u64,
    /// Every name the set may hold, in order.
    pub names: &'static [&'static str],
}

impl Serialize for Names {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.names
                .iter()
                .enumerate()
                .filter(|(index, _)| self.bits & (1 << index) != 0)
                .map(|(_, name)| name),
        )
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for Names {
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Names".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        <Vec<String>>::json_schema(generator)
    }
}

/// Whether the frame applied an effect its settings ask for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Applied {
    /// Whether this frame applied it.
    pub active: bool,
}

/// What a feature drew, and what a budget or the GPU's bindings left out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Drawn {
    /// Items this frame drew.
    pub drawn: u32,
    /// Items a budget or the GPU's bindings left out.
    pub dropped: u32,
    /// Items breaking their rules, which no frame draws.
    pub invalid: u32,
}

/// What GPU culling ran and left undrawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CullingStats {
    /// Whether this GPU-presented display frame ran it, without an instance
    /// budget (occlusion culling draws in two phases around its test).
    pub active: bool,
    /// Visible instances it left undrawn (occluded, or covering no pixel
    /// center), read back with the visible count; `None` otherwise.
    pub instances: Option<u32>,
}

/// The sun's shadow maps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ShadowStats {
    /// Whether the sun cast shadows this frame.
    pub active: bool,
    /// Indexed draws into the shadow maps, beside the camera pass's
    /// `draw_calls`.
    pub draws: u32,
    /// The kept map's redraw: 0 none, 1 a rectangle of dirty tiles, 2 the
    /// whole map (every still caster).
    pub kept_redraw: u32,
    /// Casters the near map drew, read back a few frames late; `None` before
    /// the first read-back.
    pub near_casters: Option<u32>,
    /// Casters the moving map drew, read back like the near map's: in a
    /// settled scene, the moving renderables.
    pub moving_casters: Option<u32>,
    /// Casters the kept map drew, read back like the near map's: redraws
    /// only.
    pub kept_casters: Option<u32>,
    /// Kept-level tiles this frame redrew.
    pub level_tiles: u32,
    /// Kept-level tiles no level holds yet: lookups read coarser maps there.
    pub levels_pending: u32,
    /// Casters the kept levels drew, read back like the near map's.
    pub level_casters: Option<u32>,
}

/// Local lights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LightStats {
    /// Local lights the frame shaded within the budget.
    pub shaded: u32,
    /// Usable lights beyond the budget or without a valid pose.
    pub dropped: u32,
    /// Lights breaking their rules, which no frame draws.
    pub invalid: u32,
    /// Lights that reached a view cluster, read back a few frames late;
    /// `None` before the first read-back or without lights.
    pub visible: Option<u32>,
    /// Cluster entries beyond the clusters' budget or the storage still
    /// growing, read back like `visible`.
    pub cluster_overflow: Option<u32>,
}

/// Reflection probes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ProbeStats {
    /// Probes surfaces could reflect this frame.
    pub ready: u32,
    /// Cube faces this frame captured.
    pub faces: u32,
    /// Probes a budget or the GPU's texture array layers left out.
    pub dropped: u32,
}

/// Virtual geometry, read back a few frames late.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct VirtualGeometryStats {
    /// Clusters drawn for the camera.
    pub clusters: u32,
    /// Triangles drawn for the camera.
    pub triangles: u32,
    /// Clusters drawn into shadow maps.
    pub shadow_clusters: u32,
    /// Triangles drawn into shadow maps.
    pub shadow_triangles: u32,
    /// The camera's clusters occlusion culling left undrawn.
    pub occluded_clusters: u32,
    /// Cluster selections dropped while their storage grew.
    pub dropped: u32,
    /// Streamed pages resident.
    pub resident_pages: u32,
    /// Streamed pages still loading.
    pub loading_pages: u32,
    /// Pages the last read-back frame wanted and lacked: 0 once its view had
    /// all the detail it asked for.
    pub missing_pages: u32,
    /// Wanted pages (with arrivals since) that found no room under the
    /// budget or the device's limits.
    pub dropped_pages: u32,
    /// Page reads that failed or held no valid page since streaming
    /// started, each read again later; the renderer's
    /// `latest_page_failure` says why the latest did.
    pub failed_pages: u32,
    /// Whether the visibility buffer resolved the camera's virtual geometry
    /// rather than drawing it forward.
    pub visibility_buffer: bool,
}

/// Levels of detail and impostors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DetailStats {
    /// GPU-culled frames with levels of detail: instances drawn as their
    /// mesh, read back with the visible count; `None` otherwise.
    pub lod0_instances: Option<u32>,
    /// Instances drawn as their mesh's first simplification, read back like
    /// `lod0_instances`.
    pub lod1_instances: Option<u32>,
    /// Instances drawn as their mesh's second simplification, read back like
    /// `lod0_instances`.
    pub lod2_instances: Option<u32>,
    /// Instances drawn as impostors, read back like `lod0_instances`.
    pub impostor_instances: Option<u32>,
    /// Impostors baked and resident.
    pub impostors: u32,
    /// Opaque draw groups that qualified for impostors but draw their meshes
    /// because a budget or the GPU's texture array layers left them none.
    pub dropped_impostors: u32,
    /// Meshes whose levels of detail simplification refused since the
    /// renderer started, each drawn with the levels made before (none when
    /// it refused the first): only meshes beyond what 32-bit indices
    /// address; the renderer's `latest_lod_failure` says why the latest was.
    pub failed_meshes: u32,
}

/// Terrains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainRenderStats {
    /// Patches the camera drew.
    pub patches: u32,
    /// Triangles the camera drew.
    pub triangles: u32,
    /// Patches drawn into shadow maps.
    pub shadow_patches: u32,
    /// Tiles resident on the GPU.
    pub tiles: u32,
    /// Tiles still generating or loading.
    pub loading_tiles: u32,
    /// Resident tiles lacking objects their levels place with nothing
    /// under way to place them: none unless streaming lost track of them.
    pub unplaced_tiles: u32,
    /// Tiles the GPU gave control values this frame.
    pub gpu_controls: u32,
    /// Tiles the tile budget or the device's texture array layers dropped.
    pub dropped_tiles: u32,
    /// Patches standing in for finer levels not resident yet.
    pub fallback_patches: u32,
    /// Terrains casting long sun shadows through horizon maps.
    pub horizons: u32,
    /// Horizon maps this frame rebuilt.
    pub horizon_builds: u32,
    /// Terrains whose sky visibility maps are built.
    pub sky_maps: u32,
    /// Rows of sky visibility maps this frame built.
    pub sky_rows: u32,
    /// Surface materials past the palette's capacity, or whose maps the
    /// shared arrays cannot hold (block compressed, or sized unlike the first
    /// material's), which show a plain color instead.
    pub dropped_materials: u32,
    /// Pages the runtime virtual texture's sheets hold.
    pub vt_slots: u32,
    /// Virtual texture pages resident.
    pub vt_pages: u32,
    /// Virtual texture pages baked this frame (absent or stale).
    pub vt_bakes: u32,
    /// Virtual texture pages asked for but absent, in the last feedback read
    /// back.
    pub vt_missing: u32,
    /// Virtual texture pages asked for with no slot left under its budget or
    /// the device's texture array layers.
    pub dropped_vt_pages: u32,
    /// Virtual texture pages evicted for others this frame.
    pub vt_evictions: u32,
    /// Objects placement layers stand on resident level-0 tiles, each culled
    /// on the GPU every frame.
    pub placed_objects: u32,
    /// Of those, objects standing far away, which hide where the tiles
    /// holding them instead are resident.
    pub placed_far_objects: u32,
    /// Placed objects' impostors baked, which the camera draws far away and
    /// shadow maps cast as cards; counted in `level_of_detail.impostors` too.
    pub placed_impostors: u32,
    /// Placed objects the device's storage bindings left out: those past
    /// what the objects' buffers hold, and entries past what the views'
    /// lists hold, once each view and level leaving them out, read back a
    /// few frames late.
    pub dropped_placed_objects: u32,
}

/// Ground cover.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GroundCoverStats {
    /// Blades the last read-back frame drew.
    pub blades: u32,
    /// Blades past the capacity or budget, which dropped.
    pub dropped: u32,
    /// Meters from the eye ground cover reached this frame, rounded.
    pub reach: u32,
    /// The most tiers it grew in: the radius's and doublings past it.
    pub tiers: u32,
}

/// The occupancy map around the camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OccupancyStats {
    /// Times the map was drawn: it follows the camera in steps and redraws
    /// when objects change.
    pub redraws: u32,
}

/// Terrains' lakes and rivers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WaterStats {
    /// Chunks of lakes' and rivers' ribbons drawn.
    pub chunks: u32,
}

/// The sea.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OceanStats {
    /// Sectors drawn.
    pub sectors: u32,
    /// Triangles of the sectors drawn.
    pub triangles: u32,
}

/// Runtime deformation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DeformationStats {
    /// Stamps this frame pressed into the map around the camera.
    pub stamps: u32,
    /// The map's texels along a side: 0 while off, fewer than the settings'
    /// where the device's buffers cannot hold them.
    pub texels: u32,
}

/// Clouds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudStats {
    /// Rows of the clouds' dome this frame marched: all of them on exact
    /// frames, a slice on ordinary ones, none without clouds.
    pub rows: u32,
    /// Blocks of the view whose clouds this frame marched: none without
    /// clouds or with their march in the view off.
    pub view_blocks: u32,
}

/// The physically based sky.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SkyStats {
    /// Tables this frame built: two when its air changed, one more when the
    /// sun, the camera's height or the air changed; none without the sky.
    pub tables: u32,
}
