use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

use engine_core::{MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, TextureHandle};
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

use crate::RenderStats;
use crate::batches::{BatchPlan, DrawBatch};
use crate::cached::Cached;
use crate::decals::DecalPass;
use crate::depth_pyramid::{DepthPyramid, Reduction};
use crate::entries::{MeshPass, PassKind, Streams};
use crate::environment::{ENVIRONMENT_FORMAT, Environment};
use crate::gpu_data::{
    CameraUniform, ColorData, ExtraVertex, FAR_DEPTH, GroundData, LOD_LEVELS, LOD_STATE,
    LightingUniform, PoseData, SHADER, SkinVertex, Variant, Vertex, clamped_extent,
};
use crate::gpu_presentation::{BatchMeta, GpuPresentationPass};
use crate::gpu_uploads;
use crate::impostors::{
    AtlasShape, BakeSource, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas, ImpostorPipelines,
    ImpostorRecord,
};
use crate::lights::{LightFrame, LightPass};
use crate::materials::MaterialStore;
use crate::mesh_arena::{Bindings, MeshArena, MeshDraw, Placement};
use crate::particles::ParticlePass;
use crate::pipelines::PipelineCounts;
use crate::post::PostPass;
use crate::probes::{FACES, ProbePass, face_view_projection};
use crate::profiling::{GpuProfiler, Span};
use crate::retained_uploads::RetainedUploads;
use crate::shadows::{CasterDraw, ShadowPass};
use crate::staging::{self, StagingRing};
use crate::targets::Targets;
use crate::temporal::{TemporalFrame, TemporalPass};
use crate::transparent::{TransparentPass, accumulation_targets};
use crate::uploads::UploadCache;
use crate::virtual_geometry::{
    CameraTests, Cone, IdBuffers, MAX_VIEWS, SelectionFrame, VirtualBatch, VirtualGeometryPass,
    VirtualView,
};
use crate::visibility::{
    ID_FORMAT, RESOLVE_LIT_STORAGE_BUFFERS, RESOLVE_STORAGE_BUFFERS, VisibilityBuffer,
};

mod camera;
mod display;
mod frame;
mod frame_stats;
mod impostor_draws;
mod instances;
mod meshes;
mod probe_captures;
mod scatter_draws;
mod scatter_pool;
mod terrain_draws;
mod virtual_draws;
mod visibility_draws;

use camera::CameraBindings;
use display::{DepthTarget, Display};
use impostor_draws::ImpostorDraws;
pub(crate) use instances::FramePoses;
use instances::Instances;
use meshes::{BatchBinds, DrawOrder, MeshPipelines, SkyTargets};
pub(crate) use meshes::{extra_layout, skin_layout, vertex_layout};
use virtual_draws::{VIRTUAL_ATTRIBUTES, VirtualDraws, VirtualStages};
use visibility_draws::VisibilityDraws;

const INITIAL_CAPACITY: usize = 1_024;
// Reversed depth: nearer surfaces keep larger values, which
// floating-point depth resolves about evenly at every distance. The buffer
// clears to the far plane's 0 and depth tests keep the greater value.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct GpuMesh {
    // Where the mesh arena holds it, or the buffers of its own when the
    // arena had no room.
    placement: Result<Placement, OwnMesh>,
    // Whether it has vertex colors or second texture coordinates (or
    // joints), and joints and weights.
    extended: bool,
    skinned: bool,
    index_count: u32,
    byte_len: u64,
    // Coarser levels of detail, after the mesh's own indices
    // in its range: each one's first index there, index count and geometric
    // error in mesh units, finest first.
    lods: Vec<(u32, u32, f32)>,
}

/// A mesh's own buffers: positions, extras, joints and indices.
struct OwnMesh {
    vertices: wgpu::Buffer,
    extras: Option<wgpu::Buffer>,
    skin: Option<wgpu::Buffer>,
    indices: wgpu::Buffer,
}

impl GpuMesh {
    fn draw<'a>(&'a self, arena: &'a MeshArena) -> MeshDraw<'a> {
        match &self.placement {
            // The range holds the coarser levels after the mesh's indices.
            Ok(placement) => MeshDraw {
                index_count: self.index_count,
                ..arena.draw(*placement)
            },
            Err(own) => MeshDraw {
                vertices: &own.vertices,
                extras: own.extras.as_ref(),
                skin: own.skin.as_ref(),
                indices: &own.indices,
                first_index: 0,
                base_vertex: 0,
                index_count: self.index_count,
            },
        }
    }
}

/// The levels of detail `batch` draws at on GPU-presented frames:
/// its mesh's where it draws the mesh opaquely, otherwise only the
/// mesh itself. Batch records and draw loops both follow this.
fn lod_levels(batch: &DrawBatch, mesh: &GpuMesh, virtual_geometry: bool) -> usize {
    if batch.transparent || virtual_geometry || mesh.skinned {
        1
    } else {
        1 + mesh.lods.len().min(LOD_LEVELS - 1)
    }
}

/// `source`'s indices followed by `levels` levels of detail at most,
/// and each level's first index in them, index count and
/// error.
fn lod_chain(source: &engine_core::Mesh, levels: usize) -> (Vec<u32>, Vec<(u32, u32, f32)>) {
    let lods = engine_core::mesh_lods(source, levels);
    let levels = lods
        .levels
        .iter()
        .map(|level| (level.first_index, level.index_count, level.error))
        .collect();
    (lods.indices, levels)
}

/// Generations of the shadow maps, environment maps and reflection probe
/// grid the camera groups bind.
type CameraGroupsKey = (u64, u64, u64, u64, u64);

/// Occlusion culling's depth pyramid, kept while the `OcclusionCulling`
/// settings have it run.
struct Culling {
    pyramid: Option<DepthPyramid>,
}

/// The plan's draws, the mesh layout and the material groups: what a frame's
/// structure-only work reads.
type StructureKey = (u64, u64, u64);

/// What per-frame work reading only the frame's structure last ran for;
/// work whose key stands keeps its results.
#[derive(Default)]
struct StructureKeys {
    // The opaque draw order.
    order: Option<StructureKey>,
    // The material variants' pipelines of the opaque and transparent passes,
    // and of reflection probe faces.
    variants: Option<(StructureKey, PassKind, PassKind)>,
    probes: Option<(StructureKey, PassKind)>,
    // Impostor records, by the level of detail settings,
    // whether the frame was GPU-presented and the atlas generation.
    impostors: Option<(StructureKey, engine_core::LevelOfDetail, bool, u64)>,
    // Mesh residency: the plan's rebuild and the caster groups' generation
    // it was retired for, and those with the mesh layout it uploaded for.
    residency: Option<(u64, u64, u64)>,
    uploads: Option<(u64, u64, u64, u64)>,
}

/// Main-thread browser renderer. Never transfer it or its GPU objects to workers.
///
/// Visible slots are grouped by mesh and material into reusable draw ranges.
/// Immutable assets upload once per residency lifetime; unused asset resources
/// retire at each valid render boundary, even when suspended. Slot storage
/// grows geometrically up to what one storage binding holds.
/// Model transforms must be finite, nonsingular, and preserve triangle winding.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // The canvas, its surface and the depth buffer at its size.
    display: Display,
    depth: DepthTarget,
    // Terrain's visibility target and its resolves' group.
    terrain_ids: Option<terrain_draws::TerrainIds>,
    // Resident meshes, in the shared arena's buffers (or their own when it
    // had no room), and a generation that advances whenever a mesh is
    // placed or retired.
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    mesh_arena: MeshArena,
    mesh_layout: u64,
    vertex_staging: Vec<Vertex>,
    // Material textures, samplers and bind groups.
    materials: MaterialStore,
    // The frame's draw batches, and the opaque ones' drawing order.
    plan: BatchPlan,
    draw_order: DrawOrder,
    structure_keys: StructureKeys,
    // Slots' poses, colors and palettes, and the camera bytes last written.
    instances: Instances,
    uploads: UploadCache,
    // The mesh shader, the main passes' first group, and the pipelines of
    // material variants, created with their first draw.
    mesh_shader: wgpu::ShaderModule,
    camera: CameraBindings,
    mesh_pipelines: MeshPipelines,
    pipeline_counts: PipelineCounts,
    // Blending, composition and culling for GPU-presented frames.
    gpu_presentation: GpuPresentationPass,
    culling: Culling,
    // Sun shadows.
    shadows: ShadowPass,
    // Local lights: binning and the lit mesh pipeline
    // layouts, created with the first frame that has lights. Lit
    // public-instance pipelines bind an empty group 2.
    lights: Option<LightPass>,
    // The environment maps surfaces reflect, redrawn when the
    // sky changes, and the reflection probes holding their cubes after the
    // sky's.
    environment: Environment,
    probes: ProbePass,
    // Clouds: their dome, which the sky and its environment
    // map show.
    clouds: crate::clouds::CloudPass,
    // The physically based sky.
    sky: crate::sky::SkyPass,
    // Virtual geometry and the visibility buffer resolving
    // its clusters.
    virtual_draws: VirtualDraws,
    visibility: VisibilityDraws,
    // Impostors.
    impostors: ImpostorDraws,
    // Terrain: heights of resident tiles, patches, pipelines.
    terrain: crate::terrain::TerrainPass,
    // Ground cover grown each frame on terrain.
    ground_cover: crate::ground_cover::GroundCoverPass,
    // Terrains' lakes and rivers.
    water: crate::water::WaterPass,
    // Objects placed on terrain: their pass, their meshes
    // and a counter advancing when that set changes, and their draw groups
    // and slots.
    scatter: crate::scatter::ScatterPass,
    scatter_meshes: (BTreeSet<MeshHandle>, u64),
    scatter_draws: scatter_draws::ScatterDraws,
    // How far switches between how objects show fade this frame.
    switch_fades: frame::SwitchFades,
    // What placed objects cover around the camera.
    occupancy: crate::occupancy::OccupancyPass,
    // The presentation time of the last frame written, which temporal
    // frames sway placed objects by for their motion.
    last_seconds: Option<f32>,
    // The screen effects, created with the first frame that has one.
    post: Option<PostPass>,
    // Temporal antialiasing of GPU-presented display frames.
    temporal: Option<TemporalPass>,
    // Transparent surfaces and particles: targets, the
    // composite and particle pipelines, created with the first frame that
    // draws them.
    transparent: Option<TransparentPass>,
    particles: Option<ParticlePass>,
    // Decals: records and pipelines, created with the first
    // frame that draws them.
    decals: Option<DecalPass>,
    // The frame's passes, declared anew each frame.
    graph: crate::render_graph::RenderGraph,
    // Whether the previous frame was GPU-presented; its poses were GPU-written.
    gpu_frame: bool,
    // The only callback-shared state is a rare error message, never ECS data.
    failure: Arc<Mutex<Option<String>>>,
    // Times the frame's passes during profiling captures (a stand-in that
    // records nothing in other builds).
    profiler: GpuProfiler,
}

/// A renderer whose startup pipelines the browser may still be compiling
/// and validating: build the world meanwhile, then [`Self::finish`].
pub struct StartingRenderer {
    renderer: Renderer,
    validation: std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>,
}

impl StartingRenderer {
    /// The renderer, once its startup pipelines passed validation.
    pub async fn finish(self) -> Result<Renderer, JsValue> {
        if let Some(error) = self.validation.await {
            return Err(js_error("WebGPU pipeline initialization failed", error));
        }
        Ok(self.renderer)
    }
}

/// Lets the browser run one task: commands issued before it reach the GPU
/// process before the caller's next synchronous work.
async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let scheduled = web_sys::window()
            .is_some_and(|window| window.set_timeout_with_callback(&resolve).is_ok());
        if !scheduled {
            let _ = resolve.call0(&JsValue::NULL);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

impl Renderer {
    /// Create an engine-owned WebGPU context on a canvas in a secure context.
    ///
    /// Returns an actionable error when no compatible WebGPU adapter exists,
    /// device creation fails, or pipeline validation fails. Ordinary builds require
    /// no optional GPU features. Profiling builds request timestamps when supported;
    /// there is deliberately no WebGL fallback in this milestone.
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        Self::start(canvas).await?.finish().await
    }

    /// [`Self::new`] without waiting for its startup pipelines: the browser
    /// compiles and validates them while the caller builds its world, the
    /// usual startup cost, until [`StartingRenderer::finish`].
    pub async fn start(canvas: HtmlCanvasElement) -> Result<StartingRenderer, JsValue> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|error| js_error("Could not create the WebGPU canvas", error))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| js_error("No WebGPU adapter is available. Use a WebGPU-capable browser on HTTPS or localhost", error))?;
        let adapter_limits = adapter.limits();
        // Buffers as large as the adapter allows: content, not constants,
        // sizes them.
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d.min(8_192),
            // GPU presentation's cull pass binds eight, the WebGPU default;
            // the visibility buffer's resolve up to eleven
            // where the adapter allows them.
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .max(8),
            // Terrain tiles, impostors and reflection probes
            // take a layer each.
            max_texture_array_layers: adapter_limits.max_texture_array_layers,
            ..wgpu::Limits::downlevel_defaults()
        };
        // Block-compressed textures in every format the
        // adapter samples; hosts transcode to one of them.
        let compression = adapter.features()
            & (wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC);
        // Indirect draws name their visible-ID regions as first instances
        // where the adapter allows it, so passes bind each ID buffer once.
        // Bloom keeps its levels in 32 bits where the device
        // renders them.
        let features = compression
            | (adapter.features()
                & (wgpu::Features::INDIRECT_FIRST_INSTANCE
                    | wgpu::Features::RG11B10UFLOAT_RENDERABLE));
        #[cfg(not(feature = "profiling"))]
        let required_features = features;
        #[cfg(feature = "profiling")]
        let required_features = features | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("engine/device"),
                required_features,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|error| js_error("Could not initialize the WebGPU device", error))?;

        let failure = Arc::new(Mutex::new(None));
        let lost_failure = Arc::clone(&failure);
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut slot) = lost_failure.lock() {
                *slot = Some(format!(
                    "GPU device lost ({reason:?}): {message}. Reload to recreate the engine."
                ));
            }
        });
        let uncaptured_failure = Arc::clone(&failure);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            if let Ok(mut slot) = uncaptured_failure.lock() {
                *slot = Some(format!(
                    "WebGPU rendering failed: {error}. Reload to recreate the engine."
                ));
            }
        }));

        let device_limits = device.limits();
        let display = Display::new(
            instance,
            surface,
            canvas,
            &adapter,
            (
                device_limits.max_texture_dimension_2d,
                device_limits.max_texture_array_layers,
            ),
        )?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        display.configure(&device);
        let depth = DepthTarget::new(&device, display.size());
        let lighting_buffer = crate::buffers::uniform(
            &device,
            "engine/lighting-uniform",
            size_of::<LightingUniform>() as u64,
        );
        let materials = MaterialStore::new(&device, &queue);
        let shadows = ShadowPass::new(&device, materials.layout());
        let clouds = crate::clouds::CloudPass::new(&device, &queue);
        let sky = crate::sky::SkyPass::new(&device);
        let environment = Environment::new(
            &device,
            &lighting_buffer,
            engine_core::ReflectionProbeSettings::default().resolution,
            clouds.dome(),
            sky.bindings(),
        );
        let probes = ProbePass::new(&device, size_of::<CameraUniform>() as u64);
        let terrain = crate::terrain::TerrainPass::new(&device, &queue);
        let ground_cover = crate::ground_cover::GroundCoverPass::new(&device);
        let water = crate::water::WaterPass::new(&device);
        let scatter = crate::scatter::ScatterPass::new(&device);
        let occupancy = crate::occupancy::OccupancyPass::new(&device);
        let camera = CameraBindings::new(
            &device,
            &queue,
            lighting_buffer,
            &camera::GroupSources {
                shadows: &shadows,
                environment: &environment,
                probes: &probes,
                horizons: &terrain.horizons,
                occupancy: &occupancy,
            },
        );
        let virtual_draws = VirtualDraws::new(&device);
        let visibility = VisibilityDraws::new(&device);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/meshes"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let instances = Instances::new(&device);
        let mesh_pipelines = MeshPipelines::new(&device);
        let impostor_pipelines =
            ImpostorPipelines::new(&device, materials.layout(), &mesh_pipelines.empty_layout);
        let impostor_atlases = (
            ImpostorAtlas::new(&device, AtlasShape::SPHERE),
            ImpostorAtlas::new(&device, AtlasShape::PLACED),
        );
        let gpu_presentation = GpuPresentationPass::new(&device);
        // Popping takes effect now; validation resolves once they compiled.
        let validation = Box::pin(scope.pop());

        let profiler = GpuProfiler::new(
            required_features.contains(wgpu::Features::TIMESTAMP_QUERY),
            f64::from(queue.get_timestamp_period()),
        );
        let mesh_arena = MeshArena::new(
            &device,
            [
                size_of::<Vertex>(),
                size_of::<ExtraVertex>(),
                size_of::<SkinVertex>(),
            ]
            .map(|bytes| bytes as u64),
        );
        let renderer = Self {
            device,
            queue,
            display,
            depth,
            terrain_ids: None,
            meshes: BTreeMap::new(),
            mesh_arena,
            mesh_layout: 0,
            vertex_staging: Vec::new(),
            materials,
            plan: BatchPlan::with_instance_limit(instances.limit),
            draw_order: DrawOrder::default(),
            structure_keys: StructureKeys::default(),
            instances,
            uploads: UploadCache::default(),
            mesh_shader: shader,
            camera,
            mesh_pipelines,
            pipeline_counts: PipelineCounts::default(),
            gpu_presentation,
            culling: Culling { pyramid: None },
            shadows,
            lights: None,
            environment,
            probes,
            clouds,
            sky,
            virtual_draws,
            visibility,
            impostors: ImpostorDraws {
                atlas: impostor_atlases.0,
                placed: impostor_atlases.1,
                pipelines: impostor_pipelines,
                records: Vec::new(),
                dropped: 0,
            },
            terrain,
            ground_cover,
            water,
            scatter,
            scatter_meshes: (BTreeSet::new(), 0),
            scatter_draws: scatter_draws::ScatterDraws::default(),
            switch_fades: frame::SwitchFades::default(),
            occupancy,
            last_seconds: None,
            post: None,
            temporal: None,
            transparent: None,
            particles: None,
            decals: None,
            graph: Default::default(),
            gpu_frame: false,
            failure,
            profiler,
        };
        yield_to_browser().await;
        Ok(StartingRenderer {
            renderer,
            validation,
        })
    }

    /// The frame's structure as structure-only work reads it.
    fn structure_key(&self) -> StructureKey {
        (
            self.plan.generation(),
            self.mesh_layout,
            self.materials.generation(),
        )
    }

    /// Terrain tiles the device's texture array holds.
    pub fn terrain_tile_capacity(&self) -> u32 {
        self.display.max_layers
    }

    /// Resize in physical pixels, preserving aspect ratio within device limits.
    /// A zero dimension suspends rendering without configuring an invalid surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.display.resize(width, height);
    }

    /// Renders a borrowed core frame using its retained structural and pose
    /// revisions. Lost or outdated surfaces recover on a later frame; device
    /// loss and validation errors permanently fail this renderer, and frame
    /// source errors return before uploads or cache changes. The frame is
    /// exact: never jittered or blended with earlier frames.
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, JsValue> {
        self.render_inner(frame, false)
    }

    /// [`Self::render_frame`] for ordinary display frames. With temporal
    /// antialiasing enabled, GPU-presented frames rasterize with a sub-pixel
    /// jitter and blend with the reprojected output of the previous display
    /// frame. Any other frame ends that history.
    pub fn render_display_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, JsValue> {
        self.render_inner(frame, true)
    }

    /// Follows the `TemporalAntialiasing` settings: enabling creates the
    /// pipelines of temporally antialiased display frames, disabling releases
    /// their targets and pose buffer. Frame targets follow the surface size,
    /// 16 bytes per pixel.
    fn apply_temporal_antialiasing(&mut self, enabled: bool) {
        if enabled == self.temporal.is_some() {
            return;
        }
        if enabled {
            self.temporal = Some(TemporalPass::new(
                &self.device,
                &self.camera.layout,
                self.materials.layout(),
                self.display.format(),
            ));
        } else {
            self.temporal = None;
            // Pipelines over the temporal pass's layouts go with it.
            let current =
                |&(kind, _): &(PassKind, Variant)| kind.targets() != Some(Targets::Temporal);
            self.mesh_pipelines.variants.retain(|key, _| current(key));
            self.virtual_draws.pipelines.retain(|key, _| current(key));
            self.visibility
                .resolve_pipelines
                .retain(|key, _| current(key));
            self.impostors.pipelines.forget_pipelines(Targets::Temporal);
            self.mesh_pipelines
                .forget_sky(SkyTargets::Frame(Targets::Temporal));
            self.instances.release_alternate();
        }
    }

    /// Copies GPU presentation's current poses into a staging buffer between
    /// frames, e.g. right after a fixed tick completes ahead of its due frame.
    /// The next GPU frame copies them on the GPU instead of staging them again
    /// when its gather still has the same version and uploads every current
    /// pose. Poses whose gather stamped fewer than three quarters of them
    /// upload record by record instead, so they are not staged here.
    pub fn prestage_gpu_current(&mut self, current: engine_core::GpuCurrent<'_>) {
        let bytes: &[u8] = bytemuck::cast_slice(current.poses);
        if self.display.suspended
            || !self.gpu_frame
            || bytes.len() < staging::MIN_STAGED_BYTES
            || !gpu_uploads::whole_stream(current.stamped, current.poses.len())
        {
            return;
        }
        let _profile = engine_profile::span("render", "prestage_current");
        if let Some(prestaged) = self.instances.pose_uploads.prestage(&self.device, bytes) {
            self.gpu_presentation.prestage(prestaged, current.version);
        }
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

    /// Whether pipelines created by earlier frames may still be compiling:
    /// the GPU has not yet completed the last frame that created some.
    pub fn pipelines_compiling(&self) -> bool {
        self.pipeline_counts.compiling()
    }

    /// The meshes and textures resident on the GPU.
    fn residency(&self) -> Residency {
        let (textures, texture_bytes, _) = self.materials.texture_totals();
        Residency {
            meshes: self.meshes.len() as u32,
            mesh_bytes: self.meshes.values().map(|mesh| mesh.byte_len).sum(),
            textures,
            texture_bytes,
        }
    }

    /// A skipped frame's stats: no work, and the current residency.
    fn residency_stats(&self) -> RenderStats {
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
struct Residency {
    meshes: u32,
    mesh_bytes: u64,
    textures: u32,
    texture_bytes: u64,
}

/// The canvas's clear color: the scene's in linear light, encoded for the
/// UNORM canvas.
fn canvas_clear() -> [f64; 4] {
    let encode = |linear: f64| 1.055 * linear.powf(1.0 / 2.4) - 0.055;
    let [r, g, b, a] = crate::render_graph::SCENE_CLEAR;
    [encode(r), encode(g), encode(b), a]
}

fn js_error(context: &str, error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{context}: {error}"))
}
