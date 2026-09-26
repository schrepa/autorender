use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use engine_core::{MeshHandle, RenderFrame, RenderSnapshot, Texture, TextureHandle};
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

use crate::batches::BatchPlan;
use crate::gpu_data::{
    CameraUniform, ColorData, InstanceData, PoseData, SHADER, Vertex, clamped_extent,
};
use crate::gpu_presentation::GpuPresentationPass;
use crate::retained_uploads::RetainedUploads;
use crate::staging::{self, StagingRing};
use crate::uploads::UploadCache;
use crate::{MAX_INSTANCES, RenderStats};

const INITIAL_CAPACITY: usize = 1_024;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 7 => Float32x2];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4];
const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];

struct DepthTarget {
    // Keep texture ownership explicit alongside its reusable view.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    byte_len: u64,
}

struct GpuTexture {
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    byte_len: u64,
}

/// Main-thread browser renderer. Never transfer it or its GPU objects to workers.
///
/// Opaque instances are sorted by mesh/texture into reusable CPU upload storage.
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
    pipeline: wgpu::RenderPipeline,
    affine_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    textures: BTreeMap<TextureHandle, GpuTexture>,
    texture_layout: wgpu::BindGroupLayout,
    texture_sampler: wgpu::Sampler,
    white_texture: GpuTexture,
    plan: BatchPlan,
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
        let limits = wgpu::Limits {
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
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/camera-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size_of::<CameraUniform>() as u64),
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/camera-bind-group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/base-color-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/base-color-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let white_texture = create_texture(
            &device,
            &queue,
            &texture_layout,
            &texture_sampler,
            &Texture::new(1, 1, vec![255; 4]).expect("opaque white texel"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/mesh-pipeline-layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&texture_layout)],
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
            ],
        });
        let affine_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/retained-mesh-pipeline-layout"),
                bind_group_layouts: &[
                    Some(&camera_layout),
                    Some(&texture_layout),
                    Some(&pose_layout),
                ],
                immediate_size: 0,
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/meshes.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let vertex_layout = || {
            Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &VERTEX_ATTRIBUTES,
            })
        };
        let full_layouts = [
            vertex_layout(),
            Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<InstanceData>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &INSTANCE_ATTRIBUTES,
            }),
        ];
        let affine_layouts = [
            vertex_layout(),
            Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<u32>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &SLOT_ATTRIBUTES,
            }),
        ];
        let make_pipeline = |affine| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("engine/instanced-meshes"),
                layout: Some(if affine {
                    &affine_pipeline_layout
                } else {
                    &pipeline_layout
                }),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(if affine { "vs_affine" } else { "vs_main" }),
                    compilation_options: Default::default(),
                    buffers: if affine {
                        &affine_layouts
                    } else {
                        &full_layouts
                    },
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[(
                            "SURFACE_IS_SRGB",
                            if configuration.format.is_srgb() {
                                1.0
                            } else {
                                0.0
                            },
                        )],
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format: configuration.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make_pipeline(false);
        let affine_pipeline = make_pipeline(true);
        let instance_buffer =
            create_instance_buffer(&device, INITIAL_CAPACITY, size_of::<InstanceData>());
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
            pipeline,
            affine_pipeline,
            camera_buffer,
            camera_bind_group,
            meshes: BTreeMap::new(),
            textures: BTreeMap::new(),
            texture_layout,
            texture_sampler,
            white_texture,
            plan: BatchPlan::default(),
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
        self.render_inner(snapshot, None)
    }

    /// Render a borrowed core-owned frame using retained structural/pose revisions.
    /// Untrusted public snapshots remain on the full-matrix validated path above.
    pub fn render_frame(&mut self, frame: RenderFrame<'_>) -> Result<RenderStats, JsValue> {
        self.render_inner(frame.snapshot(), Some(&frame))
    }

    /// Copies GPU presentation's current poses into a staging buffer between
    /// frames, e.g. right after a fixed tick completes ahead of its due frame.
    /// The next GPU frame copies them on the GPU instead of staging them again
    /// when its gather still has `version` and uploads every current pose.
    pub fn prestage_gpu_current(&mut self, poses: &[engine_core::NodePose], version: (u64, u64)) {
        let bytes: &[u8] = bytemuck::cast_slice(poses);
        if self.suspended || !self.gpu_frame || bytes.len() < staging::MIN_STAGED_BYTES {
            return;
        }
        let _profile = engine_profile::span("render", "prestage_current");
        if let Some(prestaged) = self.pose_uploads.prestage(&self.device, bytes) {
            self.gpu_presentation.prestage(prestaged, version);
        }
    }

    fn render_inner(
        &mut self,
        snapshot: &RenderSnapshot,
        retained: Option<&RenderFrame<'_>>,
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
        let profile_residency = engine_profile::span("render", "validate_and_retire");
        for source in &snapshot.textures {
            if source.texture.width().max(source.texture.height()) > self.max_dimension {
                return Err(JsValue::from_str(&format!(
                    "Texture {:?} exceeds the device's {}-pixel dimension limit.",
                    source.handle, self.max_dimension,
                )));
            }
        }
        // Supplied sources keep residency across culling; uploads below follow
        // only emitted draw batches and their referenced textures.
        self.meshes
            .retain(|handle, _| self.plan.contains_mesh(*handle));
        self.textures
            .retain(|handle, _| self.plan.contains_texture(*handle));
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
        for batch in &self.plan.batches {
            if let std::collections::btree_map::Entry::Vacant(entry) = self.meshes.entry(batch.mesh)
            {
                let source = &snapshot.meshes[batch.source_index].mesh;
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
                let byte_len = source.byte_len() as u64;
                entry.insert(GpuMesh {
                    vertices,
                    indices,
                    index_count: source.indices().len() as u32,
                    byte_len,
                });
                mesh_upload_bytes += byte_len;
            }
        }
        let mut texture_upload_bytes = 0_u64;
        for &index in &self.plan.texture_sources {
            let source = &snapshot.textures[index];
            if let std::collections::btree_map::Entry::Vacant(entry) =
                self.textures.entry(source.handle)
            {
                let texture = create_texture(
                    &self.device,
                    &self.queue,
                    &self.texture_layout,
                    &self.texture_sampler,
                    &source.texture,
                );
                texture_upload_bytes += texture.byte_len;
                entry.insert(texture);
            }
        }
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
            self.instance_capacity = payload_count.next_power_of_two().min(MAX_INSTANCES);
            engine_profile::counter("render.instance_buffer_growth", 1.0);
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
            self.pose_bind_group = self.color_buffer.as_ref().map(|color| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/retained-pose-bind-group"),
                    layout: &self.pose_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.instance_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: color.as_entire_binding(),
                        },
                    ],
                })
            });
            self.uploads.invalidate_instances();
            self.retained_uploads.invalidate();
            self.pose_generation += 1;
        }
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
                    .min(MAX_INSTANCES);
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
        let camera = CameraUniform {
            view_projection: snapshot.view_projection.to_cols_array_2d(),
        };
        let mut camera_upload_bytes = 0_u64;
        if self.uploads.camera_needs_write(&camera) {
            self.queue
                .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera));
            self.uploads.commit_camera(camera);
            camera_upload_bytes = size_of::<CameraUniform>() as u64;
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
        let gpu_prepared = if let (Some(frame), Some(gpu)) = (retained, gpu.as_ref()) {
            let meshes = &self.meshes;
            Some(self.gpu_presentation.prepare(
                &self.device,
                &self.queue,
                &mut self.pose_uploads,
                &mut input_copies,
                frame,
                gpu,
                &self.plan,
                |batch| {
                    (
                        meshes[&batch.mesh].index_count,
                        snapshot.meshes[batch.source_index].mesh.bounds(),
                    )
                },
                &self.instance_buffer,
                self.pose_generation,
            ))
        } else {
            None
        };
        drop(profile_writes);
        #[cfg(feature = "profiling")]
        let query_slot = self.profiler.reserve(&self.device);
        let profile_encoding = engine_profile::span("render", "command_encoding");
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/frame-encoder"),
            });
        if gpu_mode {
            #[cfg(feature = "profiling")]
            let timestamps = self.profiler.compute_writes(query_slot);
            #[cfg(not(feature = "profiling"))]
            let timestamps = None;
            self.gpu_presentation
                .encode(&self.device, &mut encoder, timestamps);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/opaque-meshes"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color(self.configuration.format.is_srgb())),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
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
                pass.set_pipeline(if affine {
                    &self.affine_pipeline
                } else {
                    &self.pipeline
                });
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                if affine {
                    pass.set_bind_group(
                        2,
                        self.pose_bind_group
                            .as_ref()
                            .expect("retained pose bindings"),
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
                for (index, batch) in self.plan.batches.iter().enumerate() {
                    // Every planned source was validated and uploaded above.
                    let mesh = &self.meshes[&batch.mesh];
                    let texture = batch
                        .texture
                        .map_or(&self.white_texture, |handle| &self.textures[&handle]);
                    pass.set_bind_group(1, &texture.bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh.vertices.slice(..));
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
        }
        #[cfg(feature = "profiling")]
        self.profiler.resolve(query_slot, &mut encoder);
        drop(profile_encoding);
        let profile_submit = engine_profile::span("render", "submit_and_present");
        // Staged pose/input copies precede the passes that read them.
        self.queue.submit(
            staged_poses
                .into_iter()
                .chain(input_copies.map(wgpu::CommandEncoder::finish))
                .chain([encoder.finish()]),
        );
        self.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
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
            ..self.residency_stats()
        };
        if engine_profile::active() {
            engine_profile::counter("render.instance_capacity", self.instance_capacity as f64);
            engine_profile::counter("render.instances", f64::from(stats.instances));
            engine_profile::counter("render.draw_calls", self.plan.batches.len() as f64);
            engine_profile::counter("render.mesh_upload_bytes", mesh_upload_bytes as f64);
            engine_profile::counter("render.texture_upload_bytes", texture_upload_bytes as f64);
            engine_profile::counter("render.camera_write_bytes", camera_upload_bytes as f64);
            engine_profile::counter("render.instance_write_bytes", instance_upload_bytes as f64);
            engine_profile::counter("render.pose_write_bytes", pose_upload_bytes as f64);
            engine_profile::counter("render.color_write_bytes", color_upload_bytes as f64);
            engine_profile::counter("render.instance_id_write_bytes", slot_upload_bytes as f64);
            engine_profile::counter("render.retained_frame", f64::from(u8::from(affine)));
            engine_profile::counter("render.gpu_presentation", f64::from(u8::from(gpu_mode)));
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

    /// Whether this immutable handle currently has GPU buffers. This observes
    /// residency, not completion of submitted GPU work, and never waits/readbacks.
    pub fn is_mesh_resident(&self, handle: MeshHandle) -> bool {
        self.meshes.contains_key(&handle)
    }

    /// Whether this immutable texture currently has uploaded GPU storage.
    /// Skipped frames do not upload missing sources; unused handles retire even
    /// when suspended. Device loss requires replacing this renderer.
    pub fn is_texture_resident(&self, handle: TextureHandle) -> bool {
        self.textures.contains_key(&handle)
    }

    fn residency_stats(&self) -> RenderStats {
        RenderStats {
            resident_meshes: self.meshes.len() as u32,
            resident_mesh_bytes: self.meshes.values().map(|mesh| mesh.byte_len).sum(),
            resident_textures: self.textures.len() as u32,
            resident_texture_bytes: self.textures.values().map(|texture| texture.byte_len).sum(),
            ..RenderStats::default()
        }
    }
}

fn create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    source: &Texture,
) -> GpuTexture {
    let extent = wgpu::Extent3d {
        width: source.width(),
        height: source.height(),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine/base-color-texture"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Queue writes accept tightly packed rows; no 256-byte buffer-copy padding.
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        source.pixels(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(source.width() * 4),
            rows_per_image: Some(source.height()),
        },
        extent,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("engine/base-color-bind-group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    GpuTexture {
        _texture: texture,
        bind_group,
        byte_len: source.byte_len() as u64,
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

fn create_storage_buffer(device: &wgpu::Device, capacity: usize, stride: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/retained-instance-storage"),
        size: (capacity * stride) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
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
