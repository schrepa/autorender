use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

use engine_core::{MeshHandle, RenderFrame, RenderSnapshot, TextureHandle};
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

use crate::batches::BatchPlan;
use crate::gpu_data::{
    CameraUniform, ColorData, ExtraVertex, InstanceData, LightingUniform, PoseData, SHADER,
    ShadowUniform, SkinVertex, Variant, Vertex, clamped_extent,
};
use crate::gpu_presentation::GpuPresentationPass;
use crate::gpu_uploads;
use crate::lights::{LightPass, LightPoses};
use crate::materials::MaterialStore;
use crate::particles::ParticlePass;
use crate::post::{GLOW_FORMAT, PostPass, SCENE_FORMAT};
use crate::retained_uploads::RetainedUploads;
use crate::shadows::{Poses, ShadowPass};
use crate::staging::{self, StagingRing};
use crate::temporal::{TemporalFrame, TemporalPass};
use crate::transparent::{TransparentPass, accumulation_targets};
use crate::uploads::UploadCache;
use crate::{LightStats, RenderStats, ShadowStats};

const INITIAL_CAPACITY: usize = 1_024;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
// Position, octahedral normal and texture coordinates.
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Snorm16x2, 7 => Float32x2];
// Columns, packed color and packed surface of public instance records.
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Uint32x2, 8 => Uint32];
const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];
// Vertex color and second texture coordinates (decision 034).
const EXTRA_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![9 => Unorm16x4, 10 => Float32x2];

// Joints and weights of meshes with joints (decision 036).
const SKIN_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![11 => Uint16x4, 12 => Unorm16x4];

/// The skin stream of skinned draws, after the second stream.
pub(crate) fn skin_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<SkinVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &SKIN_ATTRIBUTES,
    })
}

/// The second vertex stream of extended meshes.
pub(crate) fn extra_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<ExtraVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &EXTRA_ATTRIBUTES,
    })
}

fn vertex_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
    Some(wgpu::VertexBufferLayout {
        array_stride: size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &VERTEX_ATTRIBUTES,
    })
}

/// Mesh vertices and the slot ID stream of retained frames.
fn affine_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        vertex_layout(),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &SLOT_ATTRIBUTES,
        }),
    ]
}

struct DepthTarget {
    // Keep texture ownership explicit alongside its reusable view.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    // Vertex colors and second texture coordinates, when the mesh has them
    // or joints.
    extras: Option<wgpu::Buffer>,
    // Joints and weights, when the mesh has them (decision 036).
    skin: Option<wgpu::Buffer>,
    indices: wgpu::Buffer,
    index_count: u32,
    byte_len: u64,
}

/// The pipelines a material variant (decision 031) rebuilds: opaque meshes
/// into the canvas or the screen effects' targets, temporal frames' opaque
/// meshes, and transparent meshes, lit or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PassKind {
    Opaque { post: bool, lit: bool, affine: bool },
    Temporal { lit: bool },
    Transparent { lit: bool, affine: bool },
}

/// Main-thread browser renderer. Never transfer it or its GPU objects to workers.
///
/// Opaque instances are sorted by mesh/material into reusable CPU upload storage.
/// Immutable assets upload once per residency lifetime; unused asset resources
/// retire at each valid render boundary, even when suspended.
/// Instance buffers grow geometrically, up to 100,000.
/// Model transforms must be finite, nonsingular, and preserve triangle winding.
pub struct Renderer {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    canvas: HtmlCanvasElement,
    device: wgpu::Device,
    queue: wgpu::Queue,
    configuration: wgpu::SurfaceConfiguration,
    max_dimension: u32,
    suspended: bool,
    reconfigure_next_frame: bool,
    depth: DepthTarget,
    // Advances when the depth buffer is replaced (ambient occlusion reads it).
    depth_generation: u64,
    pipeline: wgpu::RenderPipeline,
    affine_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    affine_pipeline_layout: wgpu::PipelineLayout,
    // The same draws into the screen effects' scene and glow targets, and the
    // effects themselves, created with the first frame that has an effect.
    post_pipelines: Option<PostPipelines>,
    post: Option<PostPass>,
    mesh_shader: wgpu::ShaderModule,
    camera_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
    // Camera, lighting and shadow lookups, for the shadow maps of
    // `shadow_generation`.
    camera_bind_group: wgpu::BindGroup,
    shadow_generation: u64,
    // The scene's atmosphere (decision 018), written when it changes.
    lighting_buffer: wgpu::Buffer,
    lighting: Option<LightingUniform>,
    // The sky behind all geometry, drawn last in the opaque pass.
    sky_pipeline: wgpu::RenderPipeline,
    // Sun shadows (decision 020).
    shadows: ShadowPass,
    // Local lights (decision 023): binning and the lit mesh pipelines,
    // created with the first frame that has lights. Lit public-instance
    // pipelines bind an empty group 2.
    lights: Option<LightPass>,
    lit: Option<LitPipelines>,
    empty_layout: wgpu::BindGroupLayout,
    empty_group: wgpu::BindGroup,
    // Transparent surfaces and particles (decision 026): targets, the
    // composite and particle pipelines, created with the first frame that
    // draws them; transparent mesh pipelines `[unlit, lit][full, affine]`.
    transparent: Option<TransparentPass>,
    transparent_pipelines: [[Option<wgpu::RenderPipeline>; 2]; 2],
    particles: Option<ParticlePass>,
    // Poses of odd temporal frames with the colors, for transparent draws
    // of pose storage generation `.0`.
    alternate_pose_group: Option<(u64, wgpu::BindGroup)>,
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    // Material textures, samplers and bind groups (decision 031), and the
    // pipelines of material variants, created with their first draw.
    materials: MaterialStore,
    variant_pipelines: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    plan: BatchPlan,
    // Most instances this GPU's instance, pose and shadow buffers hold.
    instance_limit: usize,
    vertex_staging: Vec<Vertex>,
    instance_buffer: wgpu::Buffer,
    color_buffer: Option<wgpu::Buffer>,
    pose_layout: wgpu::BindGroupLayout,
    pose_bind_group: Option<wgpu::BindGroup>,
    slot_buffer: Option<wgpu::Buffer>,
    slot_capacity: usize,
    slot_staging: Vec<u32>,
    slot_revision: Option<(u64, u64)>,
    affine_layout: bool,
    instance_capacity: usize,
    staging: Vec<InstanceData>,
    // Retained poses upload straight from `RenderFrame::pose_rows`.
    color_staging: Vec<ColorData>,
    uploads: UploadCache,
    retained_uploads: RetainedUploads,
    pose_uploads: StagingRing,
    // Blending, composition and culling for GPU-presented frames (decision 014).
    gpu_presentation: GpuPresentationPass,
    // Temporal antialiasing of GPU-presented display frames (decision 017).
    temporal: Option<TemporalPass>,
    // Temporal frames cull into this buffer and `instance_buffer` in turn.
    alternate_poses: Option<wgpu::Buffer>,
    // Joint palettes of skinned slots (decision 036), one per pose buffer,
    // with room for `palette_capacity` joints.
    palettes: [wgpu::Buffer; 2],
    palette_capacity: usize,
    palette_staging: Vec<engine_core::PoseRows>,
    // Advances when retained pose storage is replaced (compute bindings follow).
    pose_generation: u64,
    // Whether the previous frame was GPU-presented; its poses were GPU-written.
    gpu_frame: bool,
    // The only callback-shared state is a rare error message, never ECS data.
    failure: Arc<Mutex<Option<String>>>,
    #[cfg(feature = "profiling")]
    profiler: crate::profiling::GpuProfiler,
}

impl Renderer {
    /// Create an engine-owned WebGPU context on a canvas in a secure context.
    ///
    /// Returns an actionable error when no compatible WebGPU adapter exists,
    /// device creation fails, or pipeline validation fails. Ordinary builds require
    /// no optional GPU features. Profiling builds request timestamps when supported;
    /// there is deliberately no WebGL fallback in this milestone.
    pub async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
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
            // GPU presentation's cull pass binds eight; the WebGPU default.
            max_storage_buffers_per_shader_stage: 8,
            ..wgpu::Limits::downlevel_defaults()
        };
        #[cfg(not(feature = "profiling"))]
        let required_features = wgpu::Features::empty();
        #[cfg(feature = "profiling")]
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
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

        let max_dimension = device.limits().max_texture_dimension_2d;
        // Instances: what the largest per-instance storage (poses, colors,
        // six shadow ID lists) holds within one binding, below the moving
        // flag of visible IDs.
        let binding = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        let record = size_of::<InstanceData>()
            .max(size_of::<PoseData>())
            .max(6 * size_of::<u32>()) as u64;
        let instance_limit =
            (binding / record).min(u64::from(crate::gpu_data::MOVED_SLOT)) as usize;
        let requested_width = canvas.width();
        let requested_height = canvas.height();
        let extent = clamped_extent(requested_width, requested_height, max_dimension);
        let (width, height) = extent.unwrap_or((1, 1));
        let mut configuration = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| {
                JsValue::from_str("This adapter cannot present to the supplied canvas.")
            })?;
        configuration.present_mode = wgpu::PresentMode::Fifo;
        configuration.desired_maximum_frame_latency = 2;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        if extent.is_some() {
            canvas.set_width(width);
            canvas.set_height(height);
            surface.configure(&device, &configuration);
        }
        let depth = create_depth(&device, width, height);
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/camera-uniform"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lighting_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/lighting-uniform"),
            size: size_of::<LightingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_entry = |binding, visibility, size: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size as u64),
            },
            count: None,
        };
        let shadow_map_entry = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/camera-layout"),
            entries: &[
                // Fragments read the eye, sky rays and the jitter.
                uniform_entry(
                    0,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    size_of::<CameraUniform>(),
                ),
                uniform_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    size_of::<LightingUniform>(),
                ),
                // Shadow lookups (decisions 020 and 028): the near cascades,
                // then the kept and moving maps.
                uniform_entry(2, wgpu::ShaderStages::FRAGMENT, size_of::<ShadowUniform>()),
                shadow_map_entry(3, wgpu::TextureViewDimension::D2Array),
                shadow_map_entry(4, wgpu::TextureViewDimension::D2),
                shadow_map_entry(5, wgpu::TextureViewDimension::D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // Receiver-plane lookups read four texels (decision 028).
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let materials = MaterialStore::new(&device, &queue);
        let shadows = ShadowPass::new(
            &device,
            (
                size_of::<Vertex>() as u64,
                size_of::<InstanceData>() as u64,
                std::mem::offset_of!(Vertex, uv) as u64,
            ),
            materials.layout(),
        );
        let camera_bind_group = camera_bind_group(
            &device,
            &camera_layout,
            &camera_buffer,
            &lighting_buffer,
            &shadows,
        );
        let texture_layout = materials.layout();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/mesh-pipeline-layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(texture_layout)],
            immediate_size: 0,
        });
        let storage_entry = |binding, stride| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(stride),
            },
            count: None,
        };
        let pose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/retained-pose-layout"),
            entries: &[
                storage_entry(0, size_of::<PoseData>() as u64),
                storage_entry(1, size_of::<ColorData>() as u64),
                // Joint palettes of skinned slots (decision 036).
                storage_entry(3, size_of::<PoseData>() as u64),
            ],
        });
        let affine_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/retained-mesh-pipeline-layout"),
                bind_group_layouts: &[
                    Some(&camera_layout),
                    Some(texture_layout),
                    Some(&pose_layout),
                ],
                immediate_size: 0,
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/meshes.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let canvas_target = [Some(wgpu::ColorTargetState {
            format: configuration.format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let canvas_constants = [(
            "SURFACE_IS_SRGB",
            if configuration.format.is_srgb() {
                1.0
            } else {
                0.0
            },
        )];
        let direct = MeshTargets {
            fragment: "fs_main",
            targets: &canvas_target,
            constants: &canvas_constants,
            depth_write: true,
        };
        let pipeline = mesh_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            false,
            &direct,
            Variant::BASIC,
        );
        let affine_pipeline = mesh_pipeline(
            &device,
            &shader,
            &affine_pipeline_layout,
            true,
            &direct,
            Variant::BASIC,
        );
        let sky_pipeline = sky_pipeline(
            &device,
            &shader,
            &camera_layout,
            "fs_sky",
            &[Some(configuration.format.into())],
            Some(configuration.format.is_srgb()),
        );
        let instance_buffer =
            create_instance_buffer(&device, INITIAL_CAPACITY, size_of::<InstanceData>());
        let palettes = [0, 1].map(|_| create_storage_buffer(&device, 1, size_of::<PoseData>()));
        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/empty-layout"),
            entries: &[],
        });
        let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/empty-group"),
            layout: &empty_layout,
            entries: &[],
        });
        if let Some(error) = scope.pop().await {
            return Err(js_error("WebGPU pipeline initialization failed", error));
        }

        let gpu_presentation = GpuPresentationPass::new(&device);
        #[cfg(feature = "profiling")]
        let profiler = crate::profiling::GpuProfiler::new(
            required_features.contains(wgpu::Features::TIMESTAMP_QUERY),
            f64::from(queue.get_timestamp_period()),
        );
        Ok(Self {
            instance,
            surface,
            canvas,
            device,
            queue,
            configuration,
            max_dimension,
            suspended: extent.is_none(),
            reconfigure_next_frame: false,
            depth,
            depth_generation: 0,
            pipeline,
            affine_pipeline,
            pipeline_layout,
            affine_pipeline_layout,
            post_pipelines: None,
            post: None,
            mesh_shader: shader,
            camera_layout,
            camera_buffer,
            shadow_generation: shadows.generation(),
            camera_bind_group,
            lighting_buffer,
            lighting: None,
            sky_pipeline,
            shadows,
            lights: None,
            lit: None,
            empty_layout,
            empty_group,
            transparent: None,
            transparent_pipelines: Default::default(),
            particles: None,
            alternate_pose_group: None,
            palettes,
            palette_capacity: 1,
            palette_staging: Vec::new(),
            meshes: BTreeMap::new(),
            materials,
            variant_pipelines: HashMap::new(),
            plan: BatchPlan::with_instance_limit(instance_limit),
            instance_limit,
            vertex_staging: Vec::new(),
            instance_buffer,
            color_buffer: None,
            pose_layout,
            pose_bind_group: None,
            slot_buffer: None,
            slot_capacity: 0,
            slot_staging: Vec::new(),
            slot_revision: None,
            affine_layout: false,
            instance_capacity: INITIAL_CAPACITY,
            staging: Vec::with_capacity(INITIAL_CAPACITY),
            color_staging: Vec::new(),
            uploads: UploadCache::default(),
            retained_uploads: RetainedUploads::default(),
            pose_uploads: StagingRing::default(),
            gpu_presentation,
            temporal: None,
            alternate_poses: None,
            pose_generation: 0,
            gpu_frame: false,
            failure,
            #[cfg(feature = "profiling")]
            profiler,
        })
    }

    /// Resize in physical pixels, preserving aspect ratio within device limits.
    /// A zero dimension suspends rendering without configuring an invalid surface.
    pub fn resize(&mut self, width: u32, height: u32) {
        let Some((width, height)) = clamped_extent(width, height, self.max_dimension) else {
            self.suspended = true;
            return;
        };
        if !self.suspended
            && self.configuration.width == width
            && self.configuration.height == height
        {
            return;
        }
        self.suspended = false;
        self.configuration.width = width;
        self.configuration.height = height;
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.reconfigure_next_frame = true;
        // Reconfiguration happens at the next render boundary after checking
        // device loss, so resize never touches a known-lost GPU device.
    }

    /// Submit one opaque instanced draw per mesh/texture pair. Lost/outdated surfaces recover on a later
    /// frame; device loss and validation errors permanently fail this renderer.
    /// Snapshot budget/source errors return before uploads or cache changes.
    pub fn render(&mut self, snapshot: &RenderSnapshot) -> Result<RenderStats, JsValue> {
        self.retained_uploads.invalidate();
        self.render_inner(snapshot, None, false)
    }

    /// Render a borrowed core-owned frame using retained structural/pose revisions.
    /// Untrusted public snapshots remain on the full-matrix validated path above.
    /// The frame is exact: never jittered or blended with earlier frames.
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, JsValue> {
        self.render_inner(frame.snapshot(), Some(&frame), false)
    }

    /// [`Self::render_frame`] for ordinary display frames. With temporal
    /// antialiasing enabled, GPU-presented frames rasterize with a sub-pixel
    /// jitter and blend with the reprojected output of the previous display
    /// frame (decision 017). Any other frame ends that history.
    pub fn render_display_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, JsValue> {
        self.render_inner(frame.snapshot(), Some(&frame), true)
    }

    /// Enables temporal antialiasing of GPU-presented display frames, creating
    /// its pipelines now; disabling releases its targets and pose buffer.
    /// Frame targets follow the surface size, 16 bytes per pixel.
    pub fn set_temporal_antialiasing(&mut self, enabled: bool) {
        if enabled == self.temporal.is_some() {
            return;
        }
        if enabled {
            self.temporal = Some(TemporalPass::new(
                &self.device,
                &self.mesh_shader,
                &self.camera_layout,
                self.materials.layout(),
                &affine_layouts(),
                DEPTH_FORMAT,
                self.configuration.format,
            ));
        } else {
            self.temporal = None;
            self.variant_pipelines
                .retain(|(kind, _), _| !matches!(kind, PassKind::Temporal { .. }));
            if self.alternate_poses.take().is_some() {
                // Culling must not keep writing the released buffer.
                self.pose_generation += 1;
            }
        }
    }

    pub fn temporal_antialiasing(&self) -> bool {
        self.temporal.is_some()
    }

    /// Copies GPU presentation's current poses into a staging buffer between
    /// frames, e.g. right after a fixed tick completes ahead of its due frame.
    /// The next GPU frame copies them on the GPU instead of staging them again
    /// when its gather still has the same version and uploads every current
    /// pose. Poses whose gather stamped fewer than three quarters of them
    /// upload record by record instead, so they are not staged here.
    pub fn prestage_gpu_current(&mut self, current: engine_core::GpuCurrent<'_>) {
        let bytes: &[u8] = bytemuck::cast_slice(current.poses);
        if self.suspended
            || !self.gpu_frame
            || bytes.len() < staging::MIN_STAGED_BYTES
            || !gpu_uploads::whole_stream(current.stamped, current.poses.len())
        {
            return;
        }
        let _profile = engine_profile::span("render", "prestage_current");
        if let Some(prestaged) = self.pose_uploads.prestage(&self.device, bytes) {
            self.gpu_presentation.prestage(prestaged, current.version);
        }
    }

    fn render_inner(
        &mut self,
        snapshot: &RenderSnapshot,
        retained: Option<&RenderFrame<'_>>,
        display: bool,
    ) -> Result<RenderStats, JsValue> {
        let _profile = engine_profile::span("render", "render_snapshot");
        if let Ok(failure) = self.failure.lock()
            && let Some(message) = failure.as_ref()
        {
            return Err(JsValue::from_str(message));
        }
        {
            let _profile = engine_profile::span("render", "batch_plan");
            let planned = if let Some(frame) = retained {
                self.plan.prepare_frame(frame)
            } else {
                self.plan.prepare(snapshot)
            };
            planned.map_err(|error| js_error("Invalid render snapshot", error))?;
        }
        let gpu = retained.and_then(|frame| frame.gpu());
        let gpu_mode = gpu.is_some();
        if self.gpu_frame && !gpu_mode {
            // GPU frames wrote poses and skipped the slot-ID stream; CPU frames
            // must not trust either upload cache.
            self.retained_uploads.invalidate();
            self.slot_revision = None;
        }
        self.gpu_frame = gpu_mode;
        let temporal = display && gpu_mode && self.temporal.is_some();
        if !temporal && let Some(pass) = &mut self.temporal {
            pass.reset();
        }
        let profile_residency = engine_profile::span("render", "validate_and_retire");
        for source in &snapshot.textures {
            if source.texture.width().max(source.texture.height()) > self.max_dimension {
                return Err(JsValue::from_str(&format!(
                    "Texture {:?} exceeds the device's {}-pixel dimension limit.",
                    source.handle, self.max_dimension,
                )));
            }
        }
        // Shadow draw groups (decision 020): CPU-prepared frames also draw
        // culled casters, whose meshes stay resident and upload with the
        // batches' meshes.
        self.shadows
            .plan_groups(snapshot, &self.plan, retained, gpu_mode);
        let shadow_meshes: BTreeSet<MeshHandle> = self
            .shadows
            .groups()
            .iter()
            .map(|group| group.mesh)
            .collect();
        // Supplied sources keep residency across culling; uploads below follow
        // only emitted draw batches, shadow casters and referenced textures.
        self.meshes
            .retain(|handle, _| self.plan.contains_mesh(*handle) || shadow_meshes.contains(handle));
        self.materials.retain(&self.plan);
        drop(profile_residency);
        if self.suspended {
            return Ok(self.residency_stats());
        }
        let instance_count = self.plan.ordered_instances.len();
        let profile_surface = engine_profile::span("render", "surface_acquisition");
        if self.reconfigure_next_frame {
            self.surface.configure(&self.device, &self.configuration);
            self.depth = create_depth(
                &self.device,
                self.configuration.width,
                self.configuration.height,
            );
            self.depth_generation += 1;
            self.reconfigure_next_frame = false;
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.reconfigure_next_frame = true;
                frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(self.residency_stats());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.reconfigure_next_frame = true;
                return Ok(self.residency_stats());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                    .map_err(|error| js_error("Could not recover the WebGPU surface", error))?;
                self.reconfigure_next_frame = true;
                return Ok(self.residency_stats());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(JsValue::from_str(
                    "WebGPU surface validation failed. Reload to recreate the engine.",
                ));
            }
        };
        drop(profile_surface);

        let profile_upload = engine_profile::span("render", "asset_uploads");
        let mut mesh_upload_bytes = 0_u64;
        let sources = self
            .plan
            .batches
            .iter()
            .map(|batch| (batch.mesh, batch.source_index))
            .chain(
                self.shadows
                    .groups()
                    .iter()
                    .map(|group| (group.mesh, group.source_index)),
            );
        for (handle, source_index) in sources {
            if let std::collections::btree_map::Entry::Vacant(entry) = self.meshes.entry(handle) {
                let source = &snapshot.meshes[source_index].mesh;
                self.vertex_staging.clear();
                self.vertex_staging.reserve_exact(source.vertices().len());
                self.vertex_staging
                    .extend(source.vertices().iter().map(Vertex::from));
                let vertices = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("engine/mesh-vertices"),
                        contents: bytemuck::cast_slice(&self.vertex_staging),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                let indices = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("engine/mesh-indices"),
                        contents: bytemuck::cast_slice(source.indices()),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                let extras = ExtraVertex::stream(source).map(|stream| {
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("engine/mesh-extras"),
                            contents: bytemuck::cast_slice(&stream),
                            usage: wgpu::BufferUsages::VERTEX,
                        })
                });
                let skin = SkinVertex::stream(source).map(|stream| {
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("engine/mesh-skin"),
                            contents: bytemuck::cast_slice(&stream),
                            usage: wgpu::BufferUsages::VERTEX,
                        })
                });
                let byte_len = source.byte_len() as u64;
                entry.insert(GpuMesh {
                    vertices,
                    extras,
                    skin,
                    indices,
                    index_count: source.indices().len() as u32,
                    byte_len,
                });
                mesh_upload_bytes += byte_len;
            }
        }
        // New textures' mip levels render from their uploaded base level in
        // a command buffer submitted ahead of this frame's.
        let mut mip_encoder = None;
        // Masked shadow casters bind their materials even when culled.
        let shadow_materials = self
            .shadows
            .groups()
            .iter()
            .filter_map(|group| group.material);
        let (texture_upload_bytes, mip_levels) = self.materials.upload(
            &self.device,
            &self.queue,
            snapshot,
            &self.plan,
            shadow_materials,
            &mut mip_encoder,
        );
        drop(profile_upload);
        let profile_packing = engine_profile::span("render", "instance_buffer_and_packing");
        let affine = retained.is_some();
        let payload_count = if affine {
            snapshot.instances.len()
        } else {
            instance_count
        };
        let layout_changed = self.affine_layout != affine;
        let grew = payload_count > self.instance_capacity;
        if layout_changed {
            self.affine_layout = affine;
            // A caller switching contracts replaces the old staging layout;
            // do not retain a second full population of instance payloads.
            self.staging = Vec::new();
            self.color_staging = Vec::new();
            self.retained_uploads = RetainedUploads::default();
            self.pose_uploads.clear();
            self.slot_buffer = None;
            self.slot_capacity = 0;
            self.slot_staging = Vec::new();
            self.slot_revision = None;
        }
        if grew {
            self.instance_capacity = payload_count.next_power_of_two().min(self.instance_limit);
            engine_profile::counter("render.instance_buffer_growth", 1.0);
        }
        // Joint palettes (decision 036) grow with the frame's joints.
        let joints = if affine { snapshot.joints.len() } else { 0 };
        let palettes_grew = joints > self.palette_capacity;
        if palettes_grew {
            self.palette_capacity = joints.next_power_of_two();
            self.palettes = [0, 1].map(|_| {
                create_storage_buffer(&self.device, self.palette_capacity, size_of::<PoseData>())
            });
        }
        if grew || layout_changed {
            let stride = if affine {
                size_of::<PoseData>()
            } else {
                size_of::<InstanceData>()
            };
            self.instance_buffer = if affine {
                create_storage_buffer(&self.device, self.instance_capacity, stride)
            } else {
                create_instance_buffer(&self.device, self.instance_capacity, stride)
            };
            self.color_buffer = affine.then(|| {
                create_storage_buffer(&self.device, self.instance_capacity, size_of::<ColorData>())
            });
            self.uploads.invalidate_instances();
            self.retained_uploads.invalidate();
            self.alternate_poses = None;
        }
        if grew || layout_changed || palettes_grew {
            self.pose_bind_group = self.color_buffer.as_ref().map(|color| {
                pose_group(
                    &self.device,
                    &self.pose_layout,
                    &self.instance_buffer,
                    color,
                    &self.palettes[0],
                )
            });
            self.pose_generation += 1;
        }
        if temporal && self.alternate_poses.is_none() {
            self.alternate_poses = Some(create_storage_buffer(
                &self.device,
                self.instance_capacity,
                size_of::<PoseData>(),
            ));
            self.pose_generation += 1;
        }
        let temporal_frame: Option<TemporalFrame> = match (self.temporal.as_mut(), retained) {
            (Some(pass), Some(frame)) if temporal => Some(pass.begin(
                &self.device,
                &self.queue,
                frame,
                (self.configuration.width, self.configuration.height),
                self.pose_generation,
            )),
            _ => None,
        };
        // Pose/color buffers retain core slot positions. Only compact IDs follow
        // changing visibility and mesh/texture grouping, so order changes never
        // shift payload records or force otherwise unchanged colors to upload.
        let slot_key = retained.map(|frame| (frame.identity(), frame.structure_revision()));
        let mut write_slots = false;
        if affine && !gpu_mode {
            if self.slot_buffer.is_none() || instance_count > self.slot_capacity {
                self.slot_capacity = instance_count
                    .max(INITIAL_CAPACITY)
                    .next_power_of_two()
                    .min(self.instance_limit);
                self.slot_buffer = Some(create_instance_buffer(
                    &self.device,
                    self.slot_capacity,
                    size_of::<u32>(),
                ));
                self.slot_revision = None;
            }
            if self.slot_revision != slot_key {
                self.slot_staging.clear();
                self.slot_staging.reserve_exact(instance_count);
                self.slot_staging
                    .extend(self.plan.ordered_instances.iter().map(|&slot| slot as u32));
                write_slots = !self.slot_staging.is_empty();
            }
        }
        let (repacked, poses_repacked, colors_repacked, full_upload, fallback) =
            if let Some(frame) = retained {
                if self.color_staging.capacity() < self.instance_capacity {
                    self.color_staging
                        .reserve_exact(self.instance_capacity - self.color_staging.len());
                }
                let prepared = self
                    .retained_uploads
                    .prepare(frame, &mut self.color_staging);
                (
                    prepared.repacked,
                    prepared.poses_repacked,
                    prepared.colors_repacked,
                    prepared.full_upload,
                    prepared.fallback,
                )
            } else {
                if self.staging.capacity() < self.instance_capacity {
                    self.staging
                        .reserve_exact(self.instance_capacity - self.staging.len());
                }
                let prepared = self.uploads.prepare_instances(
                    &snapshot.instances,
                    &self.plan.ordered_instances,
                    &mut self.staging,
                );
                debug_assert_eq!(prepared.write, !self.uploads.instance_ranges().is_empty());
                (
                    prepared.repacked,
                    prepared.repacked,
                    prepared.repacked,
                    prepared.full_upload,
                    prepared.fallback,
                )
            };
        drop(profile_packing);
        let profile_writes = engine_profile::span("render", "queue_writes");
        let camera = temporal_frame.as_ref().map_or_else(
            || {
                CameraUniform::exact(
                    snapshot.view_projection,
                    snapshot.camera_position,
                    snapshot.elapsed_seconds,
                )
            },
            |frame| frame.camera,
        );
        let mut camera_upload_bytes = 0_u64;
        if self.uploads.camera_needs_write(&camera) {
            self.queue
                .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera));
            self.uploads.commit_camera(camera);
            camera_upload_bytes = size_of::<CameraUniform>() as u64;
        }
        let lighting =
            LightingUniform::new(&snapshot.atmosphere, &snapshot.bloom, &snapshot.reflections);
        if self.lighting != Some(lighting) {
            self.queue
                .write_buffer(&self.lighting_buffer, 0, bytemuck::bytes_of(&lighting));
            self.lighting = Some(lighting);
            camera_upload_bytes += size_of::<LightingUniform>() as u64;
        }
        let mut instance_upload_bytes = 0_u64;
        let mut pose_upload_bytes = 0_u64;
        let mut color_upload_bytes = 0_u64;
        let mut slot_upload_bytes = 0_u64;
        let instance_write_ranges;
        // Large pose uploads copy into a persistently mapped staging buffer
        // and a GPU copy submitted before this frame's draws.
        let mut staged_poses = None;
        if let Some(frame) = retained {
            // GPU-presented frames write poses on the GPU.
            let pose_ranges = if gpu_mode {
                &[][..]
            } else {
                self.retained_uploads.pose_ranges()
            };
            instance_write_ranges = pose_ranges.len()
                + self.retained_uploads.color_ranges().len()
                + usize::from(write_slots);
            // Core keeps each slot's pose in this upload layout.
            let pose_rows: &[u8] = bytemuck::cast_slice(frame.pose_rows());
            let stride = size_of::<PoseData>();
            pose_upload_bytes = pose_ranges
                .iter()
                .map(|range| ((range.end - range.start) * stride) as u64)
                .sum();
            let mut staged = false;
            if pose_upload_bytes as usize >= staging::MIN_STAGED_BYTES {
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("engine/pose-staging-copies"),
                        });
                let outcome = self.pose_uploads.record(
                    &self.device,
                    &mut encoder,
                    &self.instance_buffer,
                    pose_rows,
                    stride,
                    pose_ranges,
                );
                staged = outcome == staging::Staged::Recorded;
                engine_profile::counter("render.pose_staged", f64::from(u8::from(staged)));
                engine_profile::counter(
                    "render.pose_staging_unavailable",
                    f64::from(u8::from(!staged)),
                );
                staged_poses = staged.then(|| encoder.finish());
            }
            if !staged {
                for range in pose_ranges {
                    self.queue.write_buffer(
                        &self.instance_buffer,
                        (range.start * stride) as u64,
                        &pose_rows[range.start * stride..range.end * stride],
                    );
                }
            }
            for range in self.retained_uploads.color_ranges() {
                let records = &self.color_staging[range.start..range.end];
                self.queue.write_buffer(
                    self.color_buffer.as_ref().expect("affine color buffer"),
                    (range.start * size_of::<ColorData>()) as u64,
                    bytemuck::cast_slice(records),
                );
                color_upload_bytes += std::mem::size_of_val(records) as u64;
            }
            // CPU-prepared frames upload skinned slots' palettes (decision
            // 036); GPU-presented ones compute them.
            if !gpu_mode && !snapshot.joints.is_empty() {
                self.palette_staging.clear();
                self.palette_staging
                    .extend(snapshot.joints.iter().map(|joint| joint.matrix));
                self.queue.write_buffer(
                    &self.palettes[0],
                    0,
                    bytemuck::cast_slice(&self.palette_staging),
                );
                pose_upload_bytes += std::mem::size_of_val(&self.palette_staging[..]) as u64;
            }
            if write_slots {
                self.queue.write_buffer(
                    self.slot_buffer.as_ref().expect("retained slot buffer"),
                    0,
                    bytemuck::cast_slice(&self.slot_staging),
                );
                slot_upload_bytes = (self.slot_staging.len() * size_of::<u32>()) as u64;
            }
            instance_upload_bytes = pose_upload_bytes + color_upload_bytes + slot_upload_bytes;
            self.retained_uploads.commit();
            self.slot_revision = slot_key;
        } else {
            instance_write_ranges = self.uploads.instance_ranges().len();
            for range in self.uploads.instance_ranges() {
                let records = &self.staging[range.start..range.end];
                self.queue.write_buffer(
                    &self.instance_buffer,
                    (range.start * size_of::<InstanceData>()) as u64,
                    bytemuck::cast_slice(records),
                );
                instance_upload_bytes += std::mem::size_of_val(records) as u64;
            }
            self.uploads.commit_instances();
        }
        let mut input_copies = None;
        #[cfg(feature = "profiling")]
        let query_slot = self.profiler.reserve(&self.device);
        let gpu_prepared = if let (Some(frame), Some(gpu)) = (retained, gpu.as_ref()) {
            let meshes = &self.meshes;
            let shadows = &self.shadows;
            // Temporal frames alternate pose buffers; the other keeps the
            // previous frame's poses.
            let (poses, poses_index) = match &temporal_frame {
                Some(temporal) if temporal.parity == 1 => {
                    (self.alternate_poses.as_ref().expect("created above"), 1)
                }
                _ => (&self.instance_buffer, 0),
            };
            #[cfg(feature = "profiling")]
            let stamps = self.profiler.stage_stamps(query_slot);
            #[cfg(not(feature = "profiling"))]
            let stamps = None;
            Some(
                self.gpu_presentation.prepare(
                    &self.device,
                    &self.queue,
                    &mut self.pose_uploads,
                    &mut input_copies,
                    frame,
                    gpu,
                    &self.plan,
                    |index, batch| {
                        let mesh = &snapshot.meshes[batch.source_index].mesh;
                        (
                            meshes[&batch.mesh].index_count,
                            mesh.bounds(),
                            shadows.batch_group(index),
                            mesh.joint_count() != 0,
                        )
                    },
                    !shadows.groups().is_empty(),
                    poses,
                    &self.palettes[poses_index],
                    poses_index,
                    self.pose_generation,
                    temporal_frame
                        .as_ref()
                        .map_or(snapshot.view_projection, |temporal| temporal.rasterized),
                    temporal_frame
                        .as_ref()
                        .is_some_and(|temporal| temporal.motion_valid),
                    stamps,
                ),
            )
        } else {
            None
        };
        #[cfg(feature = "profiling")]
        if let Some(prepared) = &gpu_prepared {
            self.profiler.timed_stages(query_slot, prepared.timed);
        }
        // Shadow maps (decision 020) select from this frame's poses.
        let shadow_frame = {
            let poses = affine.then(|| Poses {
                buffers: [Some(&self.instance_buffer), self.alternate_poses.as_ref()],
                colors: self.color_buffer.as_ref().expect("affine color buffer"),
                palettes: [&self.palettes[0], &self.palettes[1]],
                index: temporal_frame.as_ref().map_or(0, |frame| frame.parity),
                generation: self.pose_generation,
            });
            let meshes = &self.meshes;
            self.shadows.prepare(
                &self.device,
                &self.queue,
                snapshot,
                gpu_mode.then(|| self.gpu_presentation.shadow_inputs()),
                poses,
                self.instance_capacity,
                |mesh| meshes.get(&mesh).map_or(0, |mesh| mesh.index_count),
            )
        };
        if self.shadows.generation() != self.shadow_generation {
            self.shadow_generation = self.shadows.generation();
            self.camera_bind_group = camera_bind_group(
                &self.device,
                &self.camera_layout,
                &self.camera_buffer,
                &self.lighting_buffer,
                &self.shadows,
            );
        }
        // Local lights (decision 023): binned on the GPU from this frame's
        // poses, so slot lights follow GPU-presented meshes.
        let light_frame = if snapshot.local_lights.active() && !snapshot.lights.is_empty() {
            let device = &self.device;
            let lights = self.lights.get_or_insert_with(|| LightPass::new(device));
            if self.lit.is_none() {
                self.lit = Some(LitPipelines::new(
                    device,
                    &self.mesh_shader,
                    [
                        &self.camera_layout,
                        self.materials.layout(),
                        &self.pose_layout,
                        &self.empty_layout,
                        lights.fragment_layout(),
                    ],
                    self.configuration.format,
                ));
            }
            if temporal && let Some(pass) = self.temporal.as_mut() {
                pass.ensure_lit(
                    device,
                    &self.mesh_shader,
                    [
                        &self.camera_layout,
                        self.materials.layout(),
                        lights.fragment_layout(),
                    ],
                    &affine_layouts(),
                    DEPTH_FORMAT,
                );
            }
            let (buffer, index) = match &temporal_frame {
                Some(frame) if frame.parity == 1 => {
                    (self.alternate_poses.as_ref().expect("created above"), 1)
                }
                _ => (&self.instance_buffer, 0),
            };
            let poses = affine.then_some(LightPoses {
                buffer,
                generation: self.pose_generation,
                index,
            });
            lights.prepare(
                device,
                &self.queue,
                snapshot,
                (self.configuration.width, self.configuration.height),
                poses,
            )
        } else {
            None
        };
        // Screen effects (decision 021); temporal frames apply them in their
        // resolve, whose history averages the occlusion noise.
        let effects = snapshot.ambient_occlusion.sanitized().enabled()
            || snapshot.bloom.sanitized().enabled()
            || snapshot.reflections.sanitized().enabled()
            || (snapshot.sun_shafts.sanitized().enabled() && snapshot.shadows.enabled);
        let post_frame = if temporal_frame.is_some() || effects {
            if self.post_pipelines.is_none() {
                self.post_pipelines = Some(PostPipelines::new(
                    &self.device,
                    &self.mesh_shader,
                    &self.camera_layout,
                    [&self.pipeline_layout, &self.affine_pipeline_layout],
                ));
            }
            let format = self.configuration.format;
            let (device, camera_layout) = (&self.device, &self.camera_layout);
            let post = self
                .post
                .get_or_insert_with(|| PostPass::new(device, format, camera_layout));
            Some(post.begin(
                &self.device,
                &self.queue,
                snapshot,
                (self.configuration.width, self.configuration.height),
                &self.depth.view,
                self.depth_generation,
            ))
        } else {
            // Frames without effects release their targets.
            if let Some(post) = self.post.as_mut() {
                post.release_targets();
            }
            None
        };
        if let (Some(_), Some(pass), Some(post)) =
            (&temporal_frame, self.temporal.as_mut(), self.post.as_ref())
        {
            pass.prepare_resolve(&self.device, post);
        }
        // Transparent surfaces and particles (decision 026). Particles
        // sample the depth buffer for their soft edges.
        let transparent_draws = self.plan.batches.iter().any(|batch| batch.transparent);
        let particle_frame = if snapshot.particles.enabled && !snapshot.emitters.is_empty() {
            let device = &self.device;
            let (camera_layout, format) = (&self.camera_layout, self.configuration.format);
            let particles = self.particles.get_or_insert_with(|| {
                ParticlePass::new(
                    device,
                    camera_layout,
                    format,
                    DEPTH_FORMAT,
                    &accumulation_targets(),
                )
            });
            let prepared = particles.prepare(
                device,
                &self.queue,
                snapshot,
                (self.configuration.width, self.configuration.height),
                &self.depth.view,
                self.depth_generation,
            );
            if prepared.is_some() {
                particles.ensure_additive(device, post_frame.is_some());
            }
            prepared
        } else {
            None
        };
        let transparency = transparent_draws || particle_frame.is_some();
        if transparency {
            let (device, format) = (&self.device, self.configuration.format);
            let pass = self
                .transparent
                .get_or_insert_with(|| TransparentPass::new(device, format, DEPTH_FORMAT));
            pass.begin(
                device,
                (self.configuration.width, self.configuration.height),
            );
            pass.ensure_composite(device, post_frame.is_some());
            if transparent_draws {
                let lit = light_frame.is_some();
                let pipeline =
                    &mut self.transparent_pipelines[usize::from(lit)][usize::from(affine)];
                if pipeline.is_none() {
                    let layout = match (lit, affine) {
                        (false, false) => &self.pipeline_layout,
                        (false, true) => &self.affine_pipeline_layout,
                        (true, _) => {
                            &self.lit.as_ref().expect("created with lights").layouts
                                [usize::from(affine)]
                        }
                    };
                    let targets = accumulation_targets();
                    *pipeline = Some(mesh_pipeline(
                        device,
                        &self.mesh_shader,
                        layout,
                        affine,
                        &MeshTargets {
                            fragment: if lit {
                                "fs_transparent_lit"
                            } else {
                                "fs_transparent"
                            },
                            targets: &targets,
                            constants: &[],
                            depth_write: false,
                        },
                        Variant::BASIC,
                    ));
                }
                if temporal_frame
                    .as_ref()
                    .is_some_and(|frame| frame.parity == 1)
                    && self
                        .alternate_pose_group
                        .as_ref()
                        .is_none_or(|(built, _)| *built != self.pose_generation)
                {
                    let group = pose_group(
                        device,
                        &self.pose_layout,
                        self.alternate_poses.as_ref().expect("created above"),
                        self.color_buffer.as_ref().expect("affine color buffer"),
                        &self.palettes[1],
                    );
                    self.alternate_pose_group = Some((self.pose_generation, group));
                }
            }
        } else if let Some(pass) = self.transparent.as_mut() {
            pass.release_targets();
        }
        drop(profile_writes);
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            });
        // Pipelines of the material variants (decision 031) this frame draws.
        let lit_frame = light_frame.is_some() && self.lit.is_some() && self.lights.is_some();
        let opaque_kind = if temporal_frame.is_some() && self.temporal.is_some() {
            PassKind::Temporal { lit: lit_frame }
        } else {
            PassKind::Opaque {
                post: post_frame.is_some() && self.post_pipelines.is_some(),
                lit: lit_frame,
                affine,
            }
        };
        let transparent_kind = PassKind::Transparent {
            lit: lit_frame,
            affine,
        };
        self.ensure_variants(opaque_kind, transparent_kind);
        if gpu_mode {
            #[cfg(feature = "profiling")]
            let stamps = self.profiler.stage_stamps(query_slot);
            #[cfg(not(feature = "profiling"))]
            let stamps = None;
            let _timed = self
                .gpu_presentation
                .encode(&self.device, &mut encoder, stamps);
            #[cfg(feature = "profiling")]
            self.profiler.timed_stages(query_slot, _timed);
        }
        #[cfg(feature = "profiling")]
        let effect_queries = self.profiler.effect_queries(query_slot);
        #[cfg(not(feature = "profiling"))]
        let effect_queries: Option<(&wgpu::QuerySet, u32)> = None;
        let mut _effects_timed = 0_u8;
        let mut shadow_draws = 0;
        if let Some(shadow_frame) = &shadow_frame {
            let meshes = &self.meshes;
            let encoded = self.shadows.encode(
                &self.device,
                &mut encoder,
                shadow_frame,
                &self.plan,
                &self.instance_buffer,
                |mesh| {
                    meshes.get(&mesh).map(|mesh| {
                        (
                            &mesh.vertices,
                            &mesh.indices,
                            mesh.index_count,
                            mesh.skin.as_ref(),
                        )
                    })
                },
                |material| self.materials.group(material),
                effect_queries,
            );
            _effects_timed |= encoded.timed;
            shadow_draws = encoded.draws;
        }
        if let (Some(frame), Some(lights)) = (&light_frame, self.lights.as_mut()) {
            lights.encode(&self.device, &mut encoder, frame);
        }
        if let (Some(_), Some(pass)) = (&temporal_frame, self.temporal.as_mut()) {
            pass.prepare_pose_groups(
                &self.device,
                self.pose_generation,
                [
                    &self.instance_buffer,
                    self.alternate_poses.as_ref().expect("created above"),
                ],
                self.color_buffer.as_ref().expect("affine color buffer"),
                [&self.palettes[0], &self.palettes[1]],
            );
        }
        let temporal_pass = temporal_frame.as_ref().zip(self.temporal.as_ref());
        // Lit frames shade with the lit pipelines and the lights' group.
        let lit = light_frame
            .as_ref()
            .and(self.lit.as_ref())
            .zip(self.lights.as_ref());
        let post = self.post.as_ref().filter(|_| post_frame.is_some());
        let post_pipelines = self
            .post_pipelines
            .as_ref()
            .filter(|_| post_frame.is_some());
        {
            let attachment = |view, clear| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            // Frames with screen effects draw into their sRGB scene and glow
            // targets, temporal frames into their motion target as well.
            let mut targets = [None, None, None];
            let count = match (temporal_pass, post) {
                (Some((_, pass)), Some(post)) => {
                    let (scene, glow) = post.targets();
                    targets = [
                        attachment(scene, clear_color(true)),
                        attachment(pass.motion(), wgpu::Color::TRANSPARENT),
                        attachment(glow, wgpu::Color::TRANSPARENT),
                    ];
                    3
                }
                (_, Some(post)) => {
                    let (scene, glow) = post.targets();
                    targets[0] = attachment(scene, clear_color(true));
                    targets[1] = attachment(glow, wgpu::Color::TRANSPARENT);
                    2
                }
                (_, None) => {
                    targets[0] =
                        attachment(&view, clear_color(self.configuration.format.is_srgb()));
                    1
                }
            };
            // Ambient occlusion and the transparent passes read the depth
            // buffer afterwards.
            let keep_depth = post_frame
                .is_some_and(|frame| frame.occlusion || frame.reflections || frame.shafts)
                || transparency;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/opaque-meshes"),
                color_attachments: &targets[..count],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: if keep_depth {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: None,
                }),
                #[cfg(not(feature = "profiling"))]
                timestamp_writes: None,
                #[cfg(feature = "profiling")]
                timestamp_writes: self.profiler.writes(query_slot),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if instance_count > 0 {
                let opaque_base = match (temporal_pass, lit) {
                    (Some((_, temporal)), Some(_)) => temporal.lit_mesh_pipeline(),
                    (Some((_, temporal)), None) => temporal.mesh_pipeline(),
                    (None, Some((pipelines, _))) => {
                        &pipelines.variants[usize::from(post_pipelines.is_some())]
                            [usize::from(affine)]
                    }
                    (None, None) => match post_pipelines {
                        Some(pipelines) if affine => &pipelines.affine,
                        Some(pipelines) => &pipelines.full,
                        None if affine => &self.affine_pipeline,
                        None => &self.pipeline,
                    },
                };
                pass.set_pipeline(opaque_base);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                if let Some((_, lights)) = lit {
                    pass.set_bind_group(3, lights.fragment_group(), &[]);
                    if !affine {
                        pass.set_bind_group(2, &self.empty_group, &[]);
                    }
                }
                if affine {
                    pass.set_bind_group(
                        2,
                        temporal_pass.map_or_else(
                            || {
                                self.pose_bind_group
                                    .as_ref()
                                    .expect("retained pose bindings")
                            },
                            |(frame, temporal)| temporal.pose_group(frame),
                        ),
                        &[],
                    );
                    if !gpu_mode {
                        pass.set_vertex_buffer(
                            1,
                            self.slot_buffer
                                .as_ref()
                                .expect("retained slot buffer")
                                .slice(..),
                        );
                    }
                } else {
                    pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                }
                let mut bound = Variant::BASIC;
                for (index, batch) in self.plan.batches.iter().enumerate() {
                    // Transparent batches draw after the opaque pass.
                    if batch.transparent {
                        continue;
                    }
                    // Every planned source was validated and uploaded above.
                    let mesh = &self.meshes[&batch.mesh];
                    let (group, variant) = self.materials.binding(batch.material);
                    let skin = mesh.skin.as_ref().filter(|_| affine);
                    let variant = variant
                        .with_extended(mesh.extras.is_some())
                        .with_skinned(skin.is_some());
                    if variant != bound {
                        pass.set_pipeline(if variant == Variant::BASIC {
                            opaque_base
                        } else {
                            &self.variant_pipelines[&(opaque_kind, variant)]
                        });
                        bound = variant;
                    }
                    pass.set_bind_group(1, group, &[]);
                    pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    if let Some(extras) = &mesh.extras {
                        pass.set_vertex_buffer(2, extras.slice(..));
                    }
                    if let Some(skin) = skin {
                        pass.set_vertex_buffer(3, skin.slice(..));
                    }
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    if gpu_mode {
                        // Visible slots of this group, compacted and counted on the GPU.
                        let (ids, draws, offset) = self.gpu_presentation.draw_source(batch, index);
                        pass.set_vertex_buffer(1, ids);
                        pass.draw_indexed_indirect(draws, offset);
                    } else {
                        pass.draw_indexed(0..mesh.index_count, 0, batch.instances.clone());
                    }
                }
            }
            // Last, so depth testing leaves the sky only uncovered pixels.
            pass.set_pipeline(match temporal_pass {
                Some((_, temporal)) => temporal.sky_pipeline(),
                None => post_pipelines.map_or(&self.sky_pipeline, |pipelines| &pipelines.sky),
            });
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(transparent) = self.transparent.as_ref().filter(|_| transparency) {
            // Read-only depth: surfaces behind the opaque ones stay hidden,
            // and particles sample it.
            let depth = Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth.view,
                depth_ops: None,
                stencil_ops: None,
            });
            {
                let attachments = transparent.attachments();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/transparent-accumulation"),
                    color_attachments: &attachments,
                    depth_stencil_attachment: depth.clone(),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if transparent_draws && instance_count > 0 {
                    let transparent_base = self.transparent_pipelines[usize::from(lit.is_some())]
                        [usize::from(affine)]
                    .as_ref()
                    .expect("created with the transparent frame");
                    pass.set_pipeline(transparent_base);
                    pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    if let Some((_, lights)) = lit {
                        pass.set_bind_group(3, lights.fragment_group(), &[]);
                        if !affine {
                            pass.set_bind_group(2, &self.empty_group, &[]);
                        }
                    }
                    if affine {
                        let odd = temporal_frame
                            .as_ref()
                            .is_some_and(|frame| frame.parity == 1);
                        let group = if odd {
                            &self.alternate_pose_group.as_ref().expect("built above").1
                        } else {
                            self.pose_bind_group
                                .as_ref()
                                .expect("retained pose bindings")
                        };
                        pass.set_bind_group(2, group, &[]);
                        if !gpu_mode {
                            pass.set_vertex_buffer(
                                1,
                                self.slot_buffer
                                    .as_ref()
                                    .expect("retained slot buffer")
                                    .slice(..),
                            );
                        }
                    } else {
                        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
                    }
                    let mut bound = Variant::BASIC;
                    for (index, batch) in self.plan.batches.iter().enumerate() {
                        if !batch.transparent {
                            continue;
                        }
                        let mesh = &self.meshes[&batch.mesh];
                        let (group, variant) = self.materials.binding(batch.material);
                        let skin = mesh.skin.as_ref().filter(|_| affine);
                        let variant = variant
                            .with_extended(mesh.extras.is_some())
                            .with_skinned(skin.is_some());
                        if variant != bound {
                            pass.set_pipeline(if variant == Variant::BASIC {
                                transparent_base
                            } else {
                                &self.variant_pipelines[&(transparent_kind, variant)]
                            });
                            bound = variant;
                        }
                        pass.set_bind_group(1, group, &[]);
                        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                        if let Some(extras) = &mesh.extras {
                            pass.set_vertex_buffer(2, extras.slice(..));
                        }
                        if let Some(skin) = skin {
                            pass.set_vertex_buffer(3, skin.slice(..));
                        }
                        pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                        if gpu_mode {
                            let (ids, draws, offset) =
                                self.gpu_presentation.draw_source(batch, index);
                            pass.set_vertex_buffer(1, ids);
                            pass.draw_indexed_indirect(draws, offset);
                        } else {
                            pass.draw_indexed(0..mesh.index_count, 0, batch.instances.clone());
                        }
                    }
                }
                if let (Some(frame), Some(particles)) = (&particle_frame, self.particles.as_ref()) {
                    particles.draw_alpha(&mut pass, &self.camera_bind_group, frame);
                }
            }
            // The composite over the frame's color, then additive particles.
            let load = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let attachments = match post {
                Some(post) => {
                    let (scene, glow) = post.targets();
                    [load(scene), load(glow)]
                }
                None => [load(&view), None],
            };
            let count = if post.is_some() { 2 } else { 1 };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/transparent-composite"),
                color_attachments: &attachments[..count],
                depth_stencil_attachment: depth,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            transparent.composite(&mut pass, post.is_some());
            if let (Some(frame), Some(particles)) = (&particle_frame, self.particles.as_ref()) {
                particles.draw_additive(&mut pass, &self.camera_bind_group, frame, post.is_some());
            }
        }
        if let (Some(post_frame), Some(post)) = (post_frame, post) {
            let queries = effect_queries.map(|(set, first)| (set, first + 4));
            post.encode_effects(&mut encoder, post_frame, &self.camera_bind_group, queries);
            _effects_timed |= u8::from(post_frame.occlusion) << 2 | u8::from(post_frame.bloom) << 3;
            if temporal_pass.is_none() {
                let queries = effect_queries.map(|(set, first)| (set, first + 8));
                post.encode_composite(&mut encoder, &view, queries);
                _effects_timed |= 1 << 4;
            }
        }
        #[cfg(feature = "profiling")]
        if effect_queries.is_some() {
            self.profiler.timed_effects(query_slot, _effects_timed);
        }
        if let Some((frame, pass)) = temporal_pass {
            #[cfg(not(feature = "profiling"))]
            let writes = None;
            #[cfg(feature = "profiling")]
            let writes = self.profiler.resolve_writes(query_slot);
            pass.encode_resolve(&mut encoder, &view, frame, writes);
            #[cfg(feature = "profiling")]
            self.profiler.timed_resolve(query_slot);
        }
        #[cfg(feature = "profiling")]
        self.profiler.resolve(query_slot, &mut encoder);
        drop(profile_encoding);
        let profile_submit = engine_profile::span("render", "submit_and_present");
        // Staged pose/input copies precede the passes that read them.
        self.queue.submit(
            mip_encoder
                .map(wgpu::CommandEncoder::finish)
                .into_iter()
                .chain(staged_poses)
                .chain(input_copies.map(wgpu::CommandEncoder::finish))
                .chain([encoder.finish()]),
        );
        self.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
        }
        self.shadows.after_submit();
        if let Some(lights) = self.lights.as_mut() {
            lights.after_submit();
        }
        if let (Some(frame), Some(pass)) = (&temporal_frame, &mut self.temporal) {
            pass.submitted(frame);
        }
        #[cfg(feature = "profiling")]
        self.profiler.map_after_submit(query_slot);
        self.queue.present(frame);
        drop(profile_submit);
        let gpu_visible = if gpu_mode {
            self.gpu_presentation.visible()
        } else {
            None
        };
        let stats = RenderStats {
            draw_calls: self.plan.batches.len() as u32,
            upload_bytes: camera_upload_bytes
                + instance_upload_bytes
                + mesh_upload_bytes
                + texture_upload_bytes,
            // GPU-culled frames report the latest read-back visible count
            // within the budget the GPU applied.
            instances: match (retained, gpu_visible) {
                (Some(frame), Some(visible)) if gpu_mode => {
                    visible.min(frame.instance_budget().min(u32::MAX as usize) as u32)
                }
                (Some(frame), None) if gpu_mode => {
                    instance_count.min(frame.instance_budget()) as u32
                }
                _ => instance_count as u32,
            },
            gpu_visible,
            mesh_upload_bytes,
            texture_upload_bytes,
            submitted: true,
            shadows: shadow_frame.as_ref().map(|frame| ShadowStats {
                draws: shadow_draws,
                kept_redraw: frame.refresh_kind(),
                casters: self.shadows.caster_counts(),
            }),
            ambient_occlusion: post_frame.is_some_and(|frame| frame.occlusion),
            bloom: post_frame.is_some_and(|frame| frame.bloom),
            lights: light_frame.as_ref().map(|_| {
                let totals = self.lights.as_ref().and_then(LightPass::totals);
                LightStats {
                    lights: snapshot.lights.len() as u32,
                    dropped: snapshot.dropped_lights.min(u32::MAX as usize) as u32,
                    visible: totals.map(|totals| totals[0]),
                    overflow: totals.map(|totals| totals[1]),
                }
            }),
            transparent_draws: if instance_count > 0 {
                self.plan
                    .batches
                    .iter()
                    .filter(|batch| batch.transparent)
                    .count() as u32
            } else {
                0
            },
            particles: particle_frame.as_ref().map_or(0, |frame| frame.total),
            dropped_particles: (snapshot.dropped_particles as u64
                + particle_frame.as_ref().map_or(0, |frame| frame.dropped))
            .min(u64::from(u32::MAX)) as u32,
            sun_shafts: post_frame.is_some_and(|frame| frame.shafts),
            reflections: post_frame.is_some_and(|frame| frame.reflections),
            ..self.residency_stats()
        };
        if engine_profile::active() {
            engine_profile::counter("render.instance_capacity", self.instance_capacity as f64);
            engine_profile::counter("render.instances", f64::from(stats.instances));
            engine_profile::counter("render.draw_calls", self.plan.batches.len() as f64);
            engine_profile::counter("render.mesh_upload_bytes", mesh_upload_bytes as f64);
            engine_profile::counter("render.texture_upload_bytes", texture_upload_bytes as f64);
            engine_profile::counter("render.texture_mip_levels", f64::from(mip_levels));
            engine_profile::counter("render.camera_write_bytes", camera_upload_bytes as f64);
            engine_profile::counter("render.instance_write_bytes", instance_upload_bytes as f64);
            engine_profile::counter("render.pose_write_bytes", pose_upload_bytes as f64);
            engine_profile::counter("render.color_write_bytes", color_upload_bytes as f64);
            engine_profile::counter("render.instance_id_write_bytes", slot_upload_bytes as f64);
            engine_profile::counter("render.retained_frame", f64::from(u8::from(affine)));
            engine_profile::counter("render.gpu_presentation", f64::from(u8::from(gpu_mode)));
            engine_profile::counter("render.taa", f64::from(u8::from(temporal_frame.is_some())));
            engine_profile::counter(
                "render.shadow_refresh",
                shadow_frame
                    .as_ref()
                    .map_or(0.0, |frame| f64::from(frame.refresh_kind())),
            );
            engine_profile::counter("render.shadow_groups", self.shadows.groups().len() as f64);
            engine_profile::counter("render.shadow_draws", f64::from(shadow_draws));
            if let Some([near, moving, kept]) = self.shadows.caster_counts() {
                engine_profile::counter("render.shadow_near_casters", f64::from(near));
                engine_profile::counter("render.shadow_moving_casters", f64::from(moving));
                engine_profile::counter("render.shadow_kept_casters", f64::from(kept));
            }
            engine_profile::counter(
                "render.ambient_occlusion",
                f64::from(u8::from(post_frame.is_some_and(|frame| frame.occlusion))),
            );
            engine_profile::counter(
                "render.bloom",
                f64::from(u8::from(post_frame.is_some_and(|frame| frame.bloom))),
            );
            engine_profile::counter(
                "memory.gpu_shadow_bytes",
                self.shadows.capacity_bytes() as f64,
            );
            engine_profile::counter(
                "memory.gpu_post_bytes",
                self.post.as_ref().map_or(0, PostPass::capacity_bytes) as f64,
            );
            if let Some(lights) = &stats.lights {
                engine_profile::counter("render.local_lights", f64::from(lights.lights));
                if let Some([visible, overflow]) = self.lights.as_ref().and_then(LightPass::totals)
                {
                    engine_profile::counter("render.lights_visible", f64::from(visible));
                    engine_profile::counter("render.light_cluster_overflow", f64::from(overflow));
                }
            }
            engine_profile::counter(
                "memory.gpu_light_bytes",
                self.lights.as_ref().map_or(0, LightPass::capacity_bytes) as f64,
            );
            engine_profile::counter(
                "render.transparent_draws",
                f64::from(stats.transparent_draws),
            );
            engine_profile::counter("render.particles", f64::from(stats.particles));
            engine_profile::counter(
                "memory.gpu_transparent_bytes",
                (self
                    .transparent
                    .as_ref()
                    .map_or(0, TransparentPass::capacity_bytes)
                    + self
                        .particles
                        .as_ref()
                        .map_or(0, ParticlePass::capacity_bytes)) as f64,
            );
            if let Some(frame) = &temporal_frame {
                engine_profile::counter(
                    "render.taa_history",
                    f64::from(u8::from(frame.history_valid)),
                );
                engine_profile::counter(
                    "render.taa_object_motion",
                    f64::from(u8::from(frame.motion_valid)),
                );
            }
            engine_profile::counter(
                "memory.gpu_temporal_bytes",
                (self
                    .temporal
                    .as_ref()
                    .map_or(0, TemporalPass::capacity_bytes)
                    + self.alternate_poses.as_ref().map_or(0, wgpu::Buffer::size))
                    as f64,
            );
            if let Some(prepared) = &gpu_prepared {
                engine_profile::counter("render.gpu_candidates", f64::from(prepared.candidates));
                engine_profile::counter(
                    "render.gpu_node_upload_bytes",
                    prepared.node_upload_bytes as f64,
                );
                engine_profile::counter(
                    "render.gpu_group_upload_bytes",
                    prepared.group_upload_bytes as f64,
                );
                engine_profile::counter(
                    "render.gpu_full_node_upload",
                    f64::from(u8::from(prepared.full_node_upload)),
                );
                engine_profile::counter(
                    "render.gpu_node_upload_staged",
                    f64::from(u8::from(prepared.staged)),
                );
                engine_profile::counter("render.gpu_parent_levels", prepared.parent_levels as f64);
                engine_profile::counter("render.gpu_shift", f64::from(u8::from(prepared.shifted)));
                engine_profile::counter("render.gpu_blend", f64::from(u8::from(prepared.blend)));
                engine_profile::counter(
                    "render.gpu_current_prestaged",
                    f64::from(u8::from(prepared.prestaged)),
                );
                engine_profile::counter("render.gpu_held", f64::from(u8::from(prepared.held)));
                if let Some(visible) = gpu_visible {
                    engine_profile::counter("render.gpu_visible_readback", f64::from(visible));
                }
            }
            engine_profile::counter(
                "memory.gpu_presentation_bytes",
                self.gpu_presentation.capacity_bytes() as f64,
            );
            engine_profile::counter("render.instance_write_ranges", instance_write_ranges as f64);
            engine_profile::counter(
                "render.instance_full_upload",
                f64::from(u8::from(full_upload)),
            );
            engine_profile::counter(
                "render.instance_upload_fallback",
                f64::from(u8::from(fallback)),
            );
            engine_profile::counter(
                "render.camera_upload_reused",
                f64::from(u8::from(camera_upload_bytes == 0)),
            );
            engine_profile::counter(
                "render.instance_upload_reused",
                f64::from(u8::from(instance_count != 0 && instance_upload_bytes == 0)),
            );
            engine_profile::counter("render.instances_repacked", repacked as f64);
            engine_profile::counter("render.poses_repacked", poses_repacked as f64);
            engine_profile::counter("render.colors_repacked", colors_repacked as f64);
            engine_profile::counter(
                "memory.render_upload_cache_bytes",
                (size_of::<UploadCache>() + self.retained_uploads.capacity_bytes()) as f64,
            );
            engine_profile::counter(
                "memory.render_upload_range_bytes",
                (UploadCache::range_scratch_bytes() + RetainedUploads::range_scratch_bytes())
                    as f64,
            );
            engine_profile::counter(
                "memory.render_staging_capacity_bytes",
                (self.staging.capacity() * size_of::<InstanceData>()
                    + self.color_staging.capacity() * size_of::<ColorData>()
                    + self.slot_staging.capacity() * size_of::<u32>()
                    + self.vertex_staging.capacity() * size_of::<Vertex>()) as f64,
            );
            engine_profile::counter(
                "memory.gpu_pose_staging_bytes",
                self.pose_uploads.capacity_bytes() as f64,
            );
            engine_profile::counter(
                "memory.gpu_instance_buffer_bytes",
                (self.instance_capacity
                    * if affine {
                        size_of::<PoseData>() + size_of::<ColorData>()
                    } else {
                        size_of::<InstanceData>()
                    }
                    + self.slot_capacity * size_of::<u32>()) as f64,
            );
            engine_profile::counter("render.upload_bytes", stats.upload_bytes as f64);
            engine_profile::counter(
                "memory.gpu_mesh_payload_bytes",
                stats.resident_mesh_bytes as f64,
            );
            engine_profile::counter(
                "memory.gpu_texture_payload_bytes",
                stats.resident_texture_bytes as f64,
            );
            engine_profile::counter(
                "memory.gpu_texture_mip_bytes",
                self.materials.texture_totals().2 as f64,
            );
        }
        Ok(stats)
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

    /// Creates the pipelines of the plan's material variants for this frame's
    /// opaque and transparent passes.
    fn ensure_variants(&mut self, opaque: PassKind, transparent: PassKind) {
        let mut missing = Vec::new();
        for batch in &self.plan.batches {
            let (_, variant) = self.materials.binding(batch.material);
            let kind = if batch.transparent {
                transparent
            } else {
                opaque
            };
            // Only retained slots skin (decision 036).
            let affine = match kind {
                PassKind::Opaque { affine, .. } | PassKind::Transparent { affine, .. } => affine,
                PassKind::Temporal { .. } => true,
            };
            let mesh = self.meshes.get(&batch.mesh);
            let variant = variant
                .with_extended(mesh.is_some_and(|mesh| mesh.extras.is_some()))
                .with_skinned(affine && mesh.is_some_and(|mesh| mesh.skin.is_some()));
            if variant != Variant::BASIC
                && !self.variant_pipelines.contains_key(&(kind, variant))
                && !missing.contains(&(kind, variant))
            {
                missing.push((kind, variant));
            }
        }
        for (kind, variant) in missing {
            let pipeline = self.create_variant(kind, variant);
            self.variant_pipelines.insert((kind, variant), pipeline);
        }
    }

    /// The pipeline of `kind` shading materials of `variant`.
    fn create_variant(&self, kind: PassKind, variant: Variant) -> wgpu::RenderPipeline {
        let (device, shader) = (&self.device, &self.mesh_shader);
        let layout = |lit: bool, affine: bool| {
            if lit {
                &self.lit.as_ref().expect("lit frames create it").layouts[usize::from(affine)]
            } else if affine {
                &self.affine_pipeline_layout
            } else {
                &self.pipeline_layout
            }
        };
        match kind {
            PassKind::Opaque { post, lit, affine } => {
                let canvas_target = [Some(wgpu::ColorTargetState {
                    format: self.configuration.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })];
                let canvas_constants = [(
                    "SURFACE_IS_SRGB",
                    if self.configuration.format.is_srgb() {
                        1.0
                    } else {
                        0.0
                    },
                )];
                let post_targets = [Some(SCENE_FORMAT.into()), Some(GLOW_FORMAT.into())];
                let targets = MeshTargets {
                    fragment: match (post, lit) {
                        (false, false) => "fs_main",
                        (false, true) => "fs_main_lit",
                        (true, false) => "fs_post",
                        (true, true) => "fs_post_lit",
                    },
                    targets: if post { &post_targets } else { &canvas_target },
                    constants: if post { &[] } else { &canvas_constants },
                    depth_write: true,
                };
                mesh_pipeline(
                    device,
                    shader,
                    layout(lit, affine),
                    affine,
                    &targets,
                    variant,
                )
            }
            PassKind::Transparent { lit, affine } => {
                let targets = accumulation_targets();
                let targets = MeshTargets {
                    fragment: if lit {
                        "fs_transparent_lit"
                    } else {
                        "fs_transparent"
                    },
                    targets: &targets,
                    constants: &[],
                    depth_write: false,
                };
                mesh_pipeline(
                    device,
                    shader,
                    layout(lit, affine),
                    affine,
                    &targets,
                    variant,
                )
            }
            PassKind::Temporal { lit } => self
                .temporal
                .as_ref()
                .expect("temporal frames create it")
                .variant(
                    device,
                    shader,
                    lit,
                    variant,
                    &affine_layouts(),
                    DEPTH_FORMAT,
                ),
        }
    }

    /// Whether this immutable handle currently has GPU buffers. This observes
    /// residency, not completion of submitted GPU work, and never waits/readbacks.
    pub fn is_mesh_resident(&self, handle: MeshHandle) -> bool {
        self.meshes.contains_key(&handle)
    }

    /// Whether this immutable texture currently has uploaded GPU storage.
    /// Skipped frames do not upload missing sources; unused handles retire even
    /// when suspended. Device loss requires replacing this renderer.
    pub fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.materials.is_texture_resident(handle)
    }

    fn residency_stats(&self) -> RenderStats {
        let (resident_textures, resident_texture_bytes, _) = self.materials.texture_totals();
        RenderStats {
            resident_meshes: self.meshes.len() as u32,
            resident_mesh_bytes: self.meshes.values().map(|mesh| mesh.byte_len).sum(),
            resident_textures,
            resident_texture_bytes,
            ..RenderStats::default()
        }
    }
}

fn create_instance_buffer(device: &wgpu::Device, capacity: usize, stride: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/instance-buffer"),
        size: (capacity * stride) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Group 2 of retained mesh pipelines: slot poses, colors and the joint
/// palettes of skinned slots (decision 036).
fn pose_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    poses: &wgpu::Buffer,
    colors: &wgpu::Buffer,
    palette: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine/retained-pose-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: poses.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: colors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: palette.as_entire_binding(),
            },
        ],
    })
}

fn create_storage_buffer(device: &wgpu::Device, capacity: usize, stride: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/retained-instance-storage"),
        size: (capacity * stride) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The sky pass of `fragment` into `targets`: one triangle at the far
/// plane, tested but not written against depth, over the camera bindings.
/// `surface_srgb` tells a pass into the canvas whether it is sRGB.
pub(crate) fn sky_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    camera_layout: &wgpu::BindGroupLayout,
    fragment: &str,
    targets: &[Option<wgpu::ColorTargetState>],
    surface_srgb: Option<bool>,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/sky-pipeline-layout"),
        bind_group_layouts: &[Some(camera_layout)],
        immediate_size: 0,
    });
    let constants = surface_srgb.map(|srgb| [("SURFACE_IS_SRGB", if srgb { 1.0 } else { 0.0 })]);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("engine/sky"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_sky"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: constants.as_ref().map_or(&[], |constants| &constants[..]),
                ..Default::default()
            },
            targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Fragment stage of a mesh pipeline: entry point, color targets and
/// pipeline constants, and whether it writes depth (transparent surfaces do
/// not).
struct MeshTargets<'a> {
    fragment: &'a str,
    targets: &'a [Option<wgpu::ColorTargetState>],
    constants: &'a [(&'a str, f64)],
    depth_write: bool,
}

/// A mesh pipeline over `layout`: affine slot-stream inputs or public
/// instance records, shading materials of `variant` (decision 031).
fn mesh_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    affine: bool,
    fragment: &MeshTargets<'_>,
    variant: Variant,
) -> wgpu::RenderPipeline {
    let constants: Vec<(&str, f64)> = fragment
        .constants
        .iter()
        .copied()
        .chain(variant.constants())
        .collect();
    // Extended meshes (decision 034) read a third vertex buffer through
    // their own entry points; skinned slots' meshes (decision 036) a fourth.
    let extended = variant.extended();
    let skinned = affine && variant.skinned();
    let suffix = if extended { "_extended" } else { "" };
    let vertex_entry = if skinned {
        "vs_affine_skinned".to_owned()
    } else {
        format!("{}{suffix}", if affine { "vs_affine" } else { "vs_main" })
    };
    let fragment_entry = format!("{}{suffix}", fragment.fragment);
    let full_layouts = [
        vertex_layout(),
        Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }),
        extra_layout(),
    ];
    let [mesh, slots] = affine_layouts();
    let affine_layouts = [mesh, slots, extra_layout(), skin_layout()];
    let streams = if skinned {
        4
    } else if extended {
        3
    } else {
        2
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("engine/instanced-meshes"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(&vertex_entry),
            compilation_options: Default::default(),
            buffers: if affine {
                &affine_layouts[..streams]
            } else {
                &full_layouts[..streams]
            },
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(fragment.depth_write),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(&fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            targets: fragment.targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Mesh pipelines shading local lights (decision 023): canvas and scene
/// target variants, each over public instances and retained slots.
struct LitPipelines {
    /// `[direct, post][full, affine]`.
    variants: [[wgpu::RenderPipeline; 2]; 2],
    /// Full and affine pipeline layouts with the lights' group.
    layouts: [wgpu::PipelineLayout; 2],
}

impl LitPipelines {
    /// `layouts`: camera, texture, pose, empty and lights layouts.
    fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layouts: [&wgpu::BindGroupLayout; 5],
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let [camera, texture, pose, empty, lights] = layouts;
        let layout = |group_two: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/lit-mesh-pipeline-layout"),
                bind_group_layouts: &[Some(camera), Some(texture), Some(group_two), Some(lights)],
                immediate_size: 0,
            })
        };
        let full_layout = layout(empty);
        let affine_layout = layout(pose);
        let canvas_target = [Some(wgpu::ColorTargetState {
            format: surface_format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let canvas_constants = [(
            "SURFACE_IS_SRGB",
            if surface_format.is_srgb() { 1.0 } else { 0.0 },
        )];
        let direct = MeshTargets {
            fragment: "fs_main_lit",
            targets: &canvas_target,
            constants: &canvas_constants,
            depth_write: true,
        };
        let post_targets = [Some(SCENE_FORMAT.into()), Some(GLOW_FORMAT.into())];
        let post = MeshTargets {
            fragment: "fs_post_lit",
            targets: &post_targets,
            constants: &[],
            depth_write: true,
        };
        let basic = Variant::BASIC;
        let variants = [
            [
                mesh_pipeline(device, shader, &full_layout, false, &direct, basic),
                mesh_pipeline(device, shader, &affine_layout, true, &direct, basic),
            ],
            [
                mesh_pipeline(device, shader, &full_layout, false, &post, basic),
                mesh_pipeline(device, shader, &affine_layout, true, &post, basic),
            ],
        ];
        Self {
            variants,
            layouts: [full_layout, affine_layout],
        }
    }
}

/// Mesh and sky pipelines drawing into the screen effects' scene and glow
/// targets (decision 021).
struct PostPipelines {
    full: wgpu::RenderPipeline,
    affine: wgpu::RenderPipeline,
    sky: wgpu::RenderPipeline,
}

impl PostPipelines {
    /// `layouts` are the public-instance and affine mesh pipeline layouts.
    fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera_layout: &wgpu::BindGroupLayout,
        layouts: [&wgpu::PipelineLayout; 2],
    ) -> Self {
        let targets = [Some(SCENE_FORMAT.into()), Some(GLOW_FORMAT.into())];
        let post = MeshTargets {
            fragment: "fs_post",
            targets: &targets,
            constants: &[],
            depth_write: true,
        };
        Self {
            full: mesh_pipeline(device, shader, layouts[0], false, &post, Variant::BASIC),
            affine: mesh_pipeline(device, shader, layouts[1], true, &post, Variant::BASIC),
            sky: sky_pipeline(device, shader, camera_layout, "fs_sky_post", &targets, None),
        }
    }
}

/// The main passes' first group: camera, lighting and shadow lookups.
fn camera_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &wgpu::Buffer,
    lighting: &wgpu::Buffer,
    shadows: &ShadowPass,
) -> wgpu::BindGroup {
    let [shadow, near, kept, moving, sampler, texel_sampler] = shadows.entries();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine/camera-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lighting.as_entire_binding(),
            },
            shadow,
            near,
            kept,
            moving,
            sampler,
            texel_sampler,
        ],
    })
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> DepthTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine/depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // Ambient occlusion (decision 021) samples it after the opaque pass.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    DepthTarget {
        _texture: texture,
        view,
    }
}

fn clear_color(srgb_target: bool) -> wgpu::Color {
    let encode = |linear: f64| {
        if srgb_target {
            linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        }
    };
    wgpu::Color {
        r: encode(0.009),
        g: encode(0.015),
        b: encode(0.025),
        a: 1.0,
    }
}

fn js_error(context: &str, error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{context}: {error}"))
}
