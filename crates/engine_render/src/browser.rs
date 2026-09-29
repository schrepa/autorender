use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::{Arc, Mutex},
};

use engine_core::{MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, TextureHandle};
use wasm_bindgen::JsValue;
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

use crate::batches::{BatchPlan, DrawBatch};
use crate::decals::DecalPass;
use crate::depth_pyramid::{DepthPyramid, Reduction};
use crate::environment::{ENVIRONMENT_FORMAT, Environment};
use crate::gpu_data::{
    CameraUniform, ColorData, ExtraVertex, FAR_DEPTH, InstanceData, LOD_LEVELS, LightingUniform,
    PoseData, SHADER, ShadowUniform, SkinVertex, Variant, Vertex, clamped_extent,
};
use crate::gpu_presentation::{BatchMeta, GpuPresentationPass};
use crate::gpu_uploads;
use crate::impostors::{
    BakeSource, IMPOSTOR_MIN_TRIANGLES, ImpostorAtlas, ImpostorRecord, ImpostorTargets,
};
use crate::lights::{LightFrame, LightPass, LightPoses};
use crate::materials::MaterialStore;
use crate::mesh_arena::{Bindings, MeshArena, MeshDraw, Placement};
use crate::particles::ParticlePass;
use crate::post::{GLOW_FORMAT, PostPass, SCENE_FORMAT, SURFACE_FORMAT};
use crate::probes::{FACES, ProbePass, face_view_projection};
use crate::retained_uploads::RetainedUploads;
use crate::shadows::{CasterDraw, Poses, ShadowPass};
use crate::staging::{self, StagingRing};
use crate::temporal::{TemporalFrame, TemporalPass};
use crate::transparent::{TransparentPass, accumulation_targets};
use crate::uploads::UploadCache;
use crate::virtual_geometry::{
    CameraTests, Cone, IdBuffers, MAX_VIEWS, VirtualBatch, VirtualGeometryPass, VirtualView,
};
use crate::visibility::{
    ID_FORMAT, RESOLVE_LIT_STORAGE_BUFFERS, RESOLVE_STORAGE_BUFFERS, VisibilityBuffer,
};
use crate::{LightStats, RenderStats, ShadowStats};

const INITIAL_CAPACITY: usize = 1_024;
// Reversed depth (decision 042): nearer surfaces keep larger values, which
// floating-point depth resolves about evenly at every distance. The buffer
// clears to the far plane's 0 and depth tests keep the greater value.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
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

pub(crate) fn vertex_layout() -> Option<wgpu::VertexBufferLayout<'static>> {
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
    // Where the mesh arena holds it, or the buffers of its own when the
    // arena had no room.
    placement: Result<Placement, OwnMesh>,
    // Whether it has vertex colors or second texture coordinates (or
    // joints: decision 034), and joints and weights (decision 036).
    extended: bool,
    skinned: bool,
    index_count: u32,
    byte_len: u64,
    // Coarser levels of detail (decision 047), after the mesh's own indices
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

/// The levels of detail `batch` draws at on GPU-presented frames (decision
/// 047): its mesh's where it draws the mesh opaquely, otherwise only the
/// mesh itself. Batch records and draw loops both follow this.
fn lod_levels(batch: &DrawBatch, mesh: &GpuMesh, virtual_geometry: bool) -> usize {
    if batch.transparent || virtual_geometry || mesh.skinned {
        1
    } else {
        1 + mesh.lods.len().min(LOD_LEVELS - 1)
    }
}

/// `source`'s indices followed by its levels of detail (decision 047), and
/// each level's first index in them, index count and error.
fn lod_chain(source: &engine_core::Mesh) -> (Vec<u32>, Vec<(u32, u32, f32)>) {
    let lods = engine_core::mesh_lods(source, LOD_LEVELS - 1);
    let levels = lods
        .levels
        .iter()
        .map(|level| (level.first_index, level.index_count, level.error))
        .collect();
    (lods.indices, levels)
}

/// The pipelines a material variant (decision 031) rebuilds: opaque meshes
/// into the canvas or the screen effects' targets, temporal frames' opaque
/// meshes, and transparent meshes, lit or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PassKind {
    Opaque { post: bool, lit: bool, affine: bool },
    Temporal { lit: bool },
    Transparent { lit: bool, affine: bool },
    // Reflection probe captures (decision 043): linear light into a cube
    // face, drawn mirrored.
    Probe { lit: bool },
}

/// Generations of the shadow maps, environment maps and reflection probe
/// grid (decision 043) the camera groups bind.
type CameraGroupsKey = (u64, u64, u64);

/// What the visibility buffer's groups (decision 045) bind: the pose
/// buffers' generation and whether alternate poses exist, and the
/// generations of the virtual arena, the selected clusters and the
/// visibility buffer's target and buffers.
type VisibilityGroupsKey = (u64, bool, u64, u64, u64);

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
    // Camera, lighting, shadow lookups, environment maps and probe grid, for
    // the shadow maps, maps and grid of these generations.
    camera_bind_group: wgpu::BindGroup,
    camera_groups_key: CameraGroupsKey,
    // The scene's atmosphere (decision 018), written when it changes.
    lighting_buffer: wgpu::Buffer,
    lighting: Option<LightingUniform>,
    // The environment maps surfaces reflect (decision 043), redrawn when the
    // sky changes, and the reflection probes holding their cubes after the
    // sky's: each face's camera group (built with the main group's key) and
    // the captures' sky pipeline.
    environment: Environment,
    probes: ProbePass,
    probe_camera_groups: Vec<wgpu::BindGroup>,
    probe_sky_pipeline: Option<wgpu::RenderPipeline>,
    // Virtual geometry (decision 044): the arena and cluster selection;
    // pipelines per pass kind and material variant over their layouts
    // (unlit, lit) and pose groups (poses, colors, previous poses and the
    // arena); this frame's camera bins (material and variant), the plan
    // batches whose clusters draw instead of their fallback meshes, each
    // shadow map's bin and view offset, and the first of the camera bins'
    // phase-two twins; the depth pyramid generation and projection of the
    // last frame that drew two phases.
    virtual_geometry: VirtualGeometryPass,
    virtual_pipelines: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    virtual_pose_layout: wgpu::BindGroupLayout,
    virtual_layouts: [Option<wgpu::PipelineLayout>; 2],
    virtual_pose_groups: Option<((u64, u64, bool), [wgpu::BindGroup; 4])>,
    virtual_bins: Vec<(Option<MaterialHandle>, Variant)>,
    virtual_drawn: Vec<bool>,
    // Plan batches in opaque drawing order and its sort keys.
    draw_order: Vec<u32>,
    draw_keys: Vec<(Variant, Option<MaterialHandle>, u8, u32)>,
    virtual_shadow_bins: Vec<(usize, usize, u32)>,
    virtual_phase_two: Option<usize>,
    virtual_previous: Option<(u64, engine_core::glam::Mat4)>,
    // The visibility buffer (decision 045), created with the first frame
    // resolving through it: its passes' group layouts, pipelines (the
    // visibility pass's single- and double-sided; resolves per pass kind and
    // material variant over their layouts, unlit and lit) and groups (the
    // visibility pass's per current pose buffer, the resolves' per current
    // and previous one); and which of this frame's camera bins it resolves.
    visibility: Option<VisibilityBuffer>,
    visibility_layout: wgpu::BindGroupLayout,
    resolve_layout: wgpu::BindGroupLayout,
    visibility_pipelines: Option<[wgpu::RenderPipeline; 2]>,
    resolve_layouts: [Option<wgpu::PipelineLayout>; 2],
    resolve_pipelines: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    visibility_groups: Option<(
        VisibilityGroupsKey,
        [wgpu::BindGroup; 2],
        [wgpu::BindGroup; 4],
    )>,
    visibility_bins: Vec<bool>,
    // The sky behind all geometry, drawn last in the opaque pass.
    sky_pipeline: wgpu::RenderPipeline,
    // Refraction through glass (decision 038): the white texel group 0 binds
    // where no opaque color is, its sampler, and group 0 of transparent
    // passes over the opaque color of post target generation `.0` and
    // shadow generation `.1`.
    refraction_fallback: wgpu::TextureView,
    refraction_sampler: wgpu::Sampler,
    refraction_group: Option<((u64, CameraGroupsKey), wgpu::BindGroup)>,
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
    // Decals (decision 039): records and pipelines, created with the first
    // frame that draws them.
    decals: Option<DecalPass>,
    // Impostors (decision 048): the baked layers and their draw pipelines,
    // this frame's record per plan batch (zeroed without an impostor), and
    // the draw groups a budget or the GPU left without one.
    impostors: ImpostorAtlas,
    impostor_records: Vec<ImpostorRecord>,
    dropped_impostors: u32,
    // Occlusion culling of GPU-presented frames (decision 040), on unless
    // turned off, and its depth pyramid.
    occlusion_culling: bool,
    depth_pyramid: Option<DepthPyramid>,
    // Sub-pixel culling of GPU-presented frames (decision 041), on unless
    // turned off.
    subpixel_culling: bool,
    // Poses of odd temporal frames with the colors, for transparent draws
    // of pose storage generation `.0`.
    alternate_pose_group: Option<(u64, wgpu::BindGroup)>,
    meshes: BTreeMap<MeshHandle, GpuMesh>,
    // Shared vertex and index buffers of resident meshes, and a generation
    // that advances whenever a mesh is placed or retired.
    mesh_arena: MeshArena,
    mesh_layout: u64,
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
            // GPU presentation's cull pass binds eight, the WebGPU default;
            // the visibility buffer's resolve (decision 045) up to eleven
            // where the adapter allows them.
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .max(8),
            ..wgpu::Limits::downlevel_defaults()
        };
        // Block-compressed textures (decision 037) in every format the
        // adapter samples; hosts transcode to one of them.
        let compression = adapter.features()
            & (wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC);
        // Indirect draws name their visible-ID regions as first instances
        // where the adapter allows it, so passes bind each ID buffer once.
        let features = compression | (adapter.features() & wgpu::Features::INDIRECT_FIRST_INSTANCE);
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
                // What glass refracts (decision 038) and its sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The environment maps surfaces reflect (decision 043) and
                // their sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The grid of reflection probes each cell's surfaces read.
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let refraction_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/refraction-fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            refraction_texture.as_image_copy(),
            &[255; 4],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let refraction_fallback =
            refraction_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let refraction_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/refraction-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
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
        let environment = Environment::new(
            &device,
            &lighting_buffer,
            engine_core::ReflectionProbes::default().resolution,
        );
        let probes = ProbePass::new(&device, size_of::<CameraUniform>() as u64);
        let virtual_geometry = VirtualGeometryPass::new(&device);
        let camera_bind_group = camera_bind_group(
            &device,
            &camera_layout,
            &camera_buffer,
            &lighting_buffer,
            &shadows,
            (&refraction_fallback, &refraction_sampler),
            (environment.binding(), probes.words().0),
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
        // Virtual geometry (decision 044): poses, colors, previous poses and
        // the arena's meshes, clusters and pages.
        let virtual_pose_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/virtual-pose-layout"),
                entries: &[
                    storage_entry(0, size_of::<PoseData>() as u64),
                    storage_entry(1, size_of::<ColorData>() as u64),
                    storage_entry(2, size_of::<PoseData>() as u64),
                    storage_entry(5, 80),
                    storage_entry(6, 48),
                    storage_entry(7, 16),
                ],
            });
        // The visibility buffer (decision 045): its pass poses clusters from
        // the arena, and its resolves read them, the selected clusters and
        // the target in fragments, the tile lists in vertices, and each
        // draw's bin through a dynamic offset.
        let draw_entry = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        };
        let visibility_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/visibility-layout"),
            entries: &[
                storage_entry(0, size_of::<PoseData>() as u64),
                storage_entry(5, 80),
                storage_entry(6, 48),
                storage_entry(7, 16),
                draw_entry(wgpu::ShaderStages::VERTEX),
            ],
        });
        let fragment_storage = |binding, stride| wgpu::BindGroupLayoutEntry {
            visibility: wgpu::ShaderStages::FRAGMENT,
            ..storage_entry(binding, stride)
        };
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/visibility-resolve-layout"),
            entries: &[
                fragment_storage(0, size_of::<PoseData>() as u64),
                fragment_storage(1, size_of::<ColorData>() as u64),
                fragment_storage(2, size_of::<PoseData>() as u64),
                fragment_storage(5, 80),
                fragment_storage(6, 48),
                fragment_storage(7, 16),
                draw_entry(wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                fragment_storage(9, 8),
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(11, 4),
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
            mirrored: false,
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
        let impostors = ImpostorAtlas::new(&device, materials.layout(), &empty_layout);
        if let Some(error) = scope.pop().await {
            return Err(js_error("WebGPU pipeline initialization failed", error));
        }

        let gpu_presentation = GpuPresentationPass::new(&device);
        #[cfg(feature = "profiling")]
        let profiler = crate::profiling::GpuProfiler::new(
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
            camera_groups_key: (shadows.generation(), 0, 0),
            camera_bind_group,
            lighting_buffer,
            lighting: None,
            environment,
            probes,
            probe_camera_groups: Vec::new(),
            probe_sky_pipeline: None,
            virtual_geometry,
            virtual_pipelines: HashMap::new(),
            virtual_pose_layout,
            virtual_layouts: [None, None],
            virtual_pose_groups: None,
            virtual_bins: Vec::new(),
            virtual_drawn: Vec::new(),
            draw_order: Vec::new(),
            draw_keys: Vec::new(),
            virtual_shadow_bins: Vec::new(),
            virtual_phase_two: None,
            virtual_previous: None,
            visibility: None,
            visibility_layout,
            resolve_layout,
            visibility_pipelines: None,
            resolve_layouts: [None, None],
            resolve_pipelines: HashMap::new(),
            visibility_groups: None,
            visibility_bins: Vec::new(),
            sky_pipeline,
            refraction_fallback,
            refraction_sampler,
            refraction_group: None,
            shadows,
            lights: None,
            lit: None,
            empty_layout,
            empty_group,
            transparent: None,
            transparent_pipelines: Default::default(),
            particles: None,
            decals: None,
            impostors,
            impostor_records: Vec::new(),
            dropped_impostors: 0,
            occlusion_culling: true,
            subpixel_culling: true,
            depth_pyramid: None,
            alternate_pose_group: None,
            palettes,
            palette_capacity: 1,
            palette_staging: Vec::new(),
            meshes: BTreeMap::new(),
            mesh_arena,
            mesh_layout: 0,
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

    /// Enables occlusion culling of GPU-presented frames (decision 040), on
    /// by default: frames that cull without an instance budget draw what the
    /// last frame found visible, test every other visible candidate against
    /// a depth pyramid of those depths, then draw what it does not hide.
    /// Disabling releases the pyramid.
    pub fn set_occlusion_culling(&mut self, enabled: bool) {
        self.occlusion_culling = enabled;
        if !enabled {
            self.depth_pyramid = None;
        }
    }

    pub fn occlusion_culling(&self) -> bool {
        self.occlusion_culling
    }

    /// Enables sub-pixel culling of GPU-presented frames (decision 041), on
    /// by default: frames that cull without an instance budget leave
    /// visible instances whose projected bounds cover no pixel center
    /// undrawn. They could produce no fragments, so output is unchanged.
    pub fn set_subpixel_culling(&mut self, enabled: bool) {
        self.subpixel_culling = enabled;
    }

    pub fn subpixel_culling(&self) -> bool {
        self.subpixel_culling
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
        for source in snapshot.textures.iter().chain(&snapshot.decal_textures) {
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
        let (plan, arena) = (&self.plan, &mut self.mesh_arena);
        let resident = self.meshes.len();
        self.meshes.retain(|handle, mesh| {
            let keep = plan.contains_mesh(*handle) || shadow_meshes.contains(handle);
            if !keep && let Ok(placement) = mesh.placement {
                arena.remove(placement);
            }
            keep
        });
        if self.meshes.len() != resident {
            self.mesh_layout += 1;
        }
        let meshes = &self.meshes;
        self.virtual_geometry
            .retain(|handle| meshes.contains_key(&handle));
        self.materials.retain(&self.plan, snapshot);
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
                // The arena holds the mesh in its streams' class; the device's
                // limits alone leave a mesh buffers of its own.
                let extras = ExtraVertex::stream(source);
                let skin = SkinVertex::stream(source);
                let mut streams: Vec<&[u8]> = vec![bytemuck::cast_slice(&self.vertex_staging)];
                if let Some(extras) = &extras {
                    streams.push(bytemuck::cast_slice(extras));
                }
                if let Some(skin) = &skin {
                    streams.push(bytemuck::cast_slice(skin));
                }
                // Levels of detail (decision 047) follow the mesh's own
                // indices in its range.
                let (indices, lods) = lod_chain(source);
                let placement = self
                    .mesh_arena
                    .insert(
                        &self.device,
                        &self.queue,
                        &streams,
                        self.vertex_staging.len() as u32,
                        &indices,
                    )
                    .ok_or_else(|| {
                        let buffer = |label, contents: &[u8], usage| {
                            self.device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some(label),
                                    contents,
                                    usage,
                                })
                        };
                        let vertex = wgpu::BufferUsages::VERTEX;
                        OwnMesh {
                            vertices: buffer("engine/mesh-vertices", streams[0], vertex),
                            extras: extras.as_ref().map(|stream| {
                                buffer("engine/mesh-extras", bytemuck::cast_slice(stream), vertex)
                            }),
                            skin: skin.as_ref().map(|stream| {
                                buffer("engine/mesh-skin", bytemuck::cast_slice(stream), vertex)
                            }),
                            indices: buffer(
                                "engine/mesh-indices",
                                bytemuck::cast_slice(&indices),
                                wgpu::BufferUsages::INDEX,
                            ),
                        }
                    });
                self.mesh_layout += 1;
                let byte_len = source.byte_len() as u64;
                entry.insert(GpuMesh {
                    placement,
                    extended: extras.is_some(),
                    skinned: skin.is_some(),
                    index_count: source.indices().len() as u32,
                    byte_len,
                    lods,
                });
                mesh_upload_bytes += byte_len;
                if let Some(geometry) = source.virtual_geometry() {
                    self.virtual_geometry.insert(handle, geometry);
                }
            }
        }
        self.virtual_geometry.upload(
            &self.device,
            &self.queue,
            snapshot.virtual_geometry.max_resident_pages,
        );
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
        // Impostors (decision 048) bake after their materials' levels.
        self.prepare_impostors(snapshot, gpu_mode, &mut mip_encoder);
        drop(profile_upload);
        self.order_draws();
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
                // Virtual geometry selection (decision 044) reads the slots
                // of CPU-prepared frames.
                self.slot_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/retained-slots"),
                    size: (self.slot_capacity * size_of::<u32>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
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
        let lighting = LightingUniform::new(
            &snapshot.atmosphere,
            &snapshot.bloom,
            &snapshot.reflections,
            &snapshot.refraction,
        );
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
        // Occlusion culling (decision 040): the depth pyramid of this frame's
        // depth buffer, sized before presentation plans two phases.
        let occlusion = if gpu_mode && self.occlusion_culling {
            let device = &self.device;
            let pyramid = self
                .depth_pyramid
                .get_or_insert_with(|| DepthPyramid::new(device, Reduction::Farthest));
            let (width, height) = (self.configuration.width, self.configuration.height);
            let levels = pyramid.prepare(
                device,
                &self.depth.view,
                self.depth_generation,
                (width, height),
            );
            Some((width, height, levels))
        } else {
            None
        };
        // Batches drawing virtual geometry (decision 044) instead of their
        // fallback meshes, by the arena and whether it draws.
        let virtual_enabled = affine && snapshot.virtual_geometry.enabled;
        let arena = self
            .virtual_geometry
            .arena()
            .map_or(0, |(_, generation)| generation);
        let virtual_key = arena * 2 + u64::from(virtual_enabled);
        let gpu_prepared = if let (Some(frame), Some(gpu)) = (retained, gpu.as_ref()) {
            let meshes = &self.meshes;
            let arena = &self.mesh_arena;
            let shadows = &self.shadows;
            let virtual_meshes = &self.virtual_geometry;
            let impostor_records = &self.impostor_records;
            // Batch records carry impostor errors: rebuilt with the
            // impostors, and when turning them on or off.
            let impostor_key = if snapshot.level_of_detail.impostors {
                self.impostors.generation() + 1
            } else {
                0
            };
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
                        let gpu_mesh = &meshes[&batch.mesh];
                        let draw = gpu_mesh.draw(arena);
                        let virtual_geometry = virtual_enabled
                            && !batch.transparent
                            && virtual_meshes.mesh_index(batch.mesh).is_some();
                        let levels = lod_levels(batch, gpu_mesh, virtual_geometry);
                        BatchMeta {
                            index_count: draw.index_count,
                            first_index: draw.first_index,
                            base_vertex: draw.base_vertex,
                            bounds: mesh.bounds(),
                            group: shadows.batch_group(index),
                            joints: mesh.joint_count(),
                            virtual_geometry,
                            lods: gpu_mesh.lods[..levels - 1]
                                .iter()
                                .map(|&(offset, count, error)| {
                                    (draw.first_index + offset, count, error)
                                })
                                .collect(),
                            impostor: impostor_records
                                .get(index)
                                .map_or(0.0, ImpostorRecord::error),
                        }
                    },
                    !shadows.groups().is_empty(),
                    poses,
                    &self.palettes[poses_index],
                    self.color_buffer.as_ref().expect("affine color buffer"),
                    poses_index,
                    self.pose_generation,
                    temporal_frame
                        .as_ref()
                        .map_or(snapshot.view_projection, |temporal| temporal.rasterized),
                    snapshot.camera_position,
                    temporal_frame
                        .as_ref()
                        .is_some_and(|temporal| temporal.motion_valid),
                    occlusion,
                    self.subpixel_culling,
                    if snapshot.level_of_detail.enabled {
                        snapshot.level_of_detail.pixel_error
                    } else {
                        0.0
                    },
                    snapshot.level_of_detail.impostors,
                    // Reflection probes (decision 043) capture still slots,
                    // counted while captures wait.
                    (snapshot.reflection_probe_settings.enabled
                        && !snapshot.reflection_probes.is_empty())
                    .then(|| self.probes.slots.waiting()),
                    virtual_key ^ (self.mesh_layout << 32),
                    impostor_key,
                    (self.configuration.width, self.configuration.height),
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
            let arena = &self.mesh_arena;
            self.shadows.prepare(
                &self.device,
                &self.queue,
                snapshot,
                gpu_mode.then(|| self.gpu_presentation.shadow_inputs()),
                poses,
                self.instance_capacity,
                self.mesh_layout,
                |mesh| {
                    meshes.get(&mesh).map_or(
                        CasterDraw {
                            index_count: 0,
                            first_index: 0,
                            base_vertex: 0,
                            lods: Vec::new(),
                        },
                        |mesh| {
                            let draw = mesh.draw(arena);
                            // Levels of detail (decision 047) of meshes
                            // without joints, which move their vertices.
                            let lods = if mesh.skinned || !snapshot.level_of_detail.enabled {
                                Vec::new()
                            } else {
                                mesh.lods
                                    .iter()
                                    .map(|&(offset, count, error)| {
                                        (draw.first_index + offset, count, error)
                                    })
                                    .collect()
                            };
                            CasterDraw {
                                index_count: draw.index_count,
                                first_index: draw.first_index,
                                base_vertex: draw.base_vertex,
                                lods,
                            }
                        },
                    )
                },
            )
        };
        // Environment maps (decision 043): as many cubes as the reflection
        // probes kept need (the nearest first, as many as the GPU's texture
        // arrays hold), the sky's redrawn when it changes, then the probes'
        // cubes assigned and their grid uploaded.
        let probe_settings = snapshot.reflection_probe_settings;
        let probe_list: &[engine_core::RenderReflectionProbe] = if probe_settings.enabled {
            &snapshot.reflection_probes
        } else {
            &[]
        };
        let most_cubes = (self.device.limits().max_texture_array_layers / FACES as u32).max(1);
        let wanted_cubes = 1 + (probe_list.len() as u32).min(most_cubes - 1);
        let cubes = if wanted_cubes == 1 {
            1
        } else if wanted_cubes > self.environment.cubes() {
            wanted_cubes.next_power_of_two().min(most_cubes)
        } else {
            self.environment.cubes()
        };
        let mut environment_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/environment-encoder"),
                });
        let lost = self.environment.ensure(
            &self.device,
            &mut environment_encoder,
            probe_settings.resolution,
            cubes,
        );
        let sky_changed = self
            .environment
            .encode_sky(&mut environment_encoder, &lighting);
        self.probes
            .slots
            .plan(probe_list, (cubes - 1) as usize, lost, sky_changed);
        self.probes.upload(&self.device, &self.queue);
        let dropped_probes = snapshot.dropped_reflection_probes as u64
            + (probe_list.len() as u64).saturating_sub(u64::from(most_cubes - 1));
        let camera_key = (
            self.shadows.generation(),
            self.environment.generation(),
            self.probes.words().1,
        );
        if camera_key != self.camera_groups_key {
            self.camera_groups_key = camera_key;
            self.camera_bind_group = camera_bind_group(
                &self.device,
                &self.camera_layout,
                &self.camera_buffer,
                &self.lighting_buffer,
                &self.shadows,
                (&self.refraction_fallback, &self.refraction_sampler),
                (self.environment.binding(), self.probes.words().0),
            );
            self.probe_camera_groups.clear();
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
                retained.map(|frame| (frame.identity(), frame.lights_revision())),
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
        // Glass refracts the opaque color of the scene target (decision
        // 038): frames drawing it render into the target.
        let refraction = snapshot.refraction.enabled()
            && self.plan.batches.iter().any(|batch| {
                batch.transparent && self.materials.binding(batch.material).1.transmission()
            });
        let post_frame = if temporal_frame.is_some() || effects || refraction {
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
                temporal_frame.is_some(),
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
                            mirrored: false,
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
        // Decals (decision 039) cover the opaque surfaces the depth buffer
        // holds; retained frames read their displayed poses from the joint
        // palette.
        let decal_lights = light_frame
            .as_ref()
            .and(self.lights.as_ref())
            .map(LightPass::fragment_layout);
        let decal_frame = if snapshot.decal_settings.enabled && !snapshot.decals.is_empty() {
            let (device, format) = (&self.device, self.configuration.format);
            let decals = self
                .decals
                .get_or_insert_with(|| DecalPass::new(device, format));
            let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
            let prepared = decals.prepare(
                device,
                &self.queue,
                snapshot,
                (
                    &self.palettes[parity],
                    affine.then_some((self.pose_generation, parity)),
                ),
                &self.depth.view,
                self.depth_generation,
            );
            if prepared.is_some() {
                decals.ensure_pipelines(
                    device,
                    &self.mesh_shader,
                    [&self.camera_layout, self.materials.layout()],
                    decal_lights,
                    post_frame.is_some(),
                );
            }
            prepared
        } else {
            None
        };
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
        // Impostors (decision 048): this frame's records, group and pipeline.
        let impostor_pass = if gpu_mode
            && self
                .impostor_records
                .iter()
                .any(|record| record.sphere[3] > 0.0)
        {
            self.prepare_impostor_draws(opaque_kind)
        } else {
            None
        };
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
        // Virtual geometry (decision 044): the clusters the visible
        // instances of virtual meshes draw this frame.
        let odd = temporal_frame
            .as_ref()
            .is_some_and(|frame| frame.parity == 1);
        let rasterized = temporal_frame
            .as_ref()
            .map_or(snapshot.view_projection, |temporal| temporal.rasterized);
        let virtual_frame = affine
            && self.prepare_virtual(
                snapshot,
                gpu_mode,
                odd,
                (rasterized, occlusion.map(|(_, _, levels)| levels)),
            );
        if !affine {
            self.virtual_drawn.clear();
        }
        if virtual_frame {
            self.ensure_virtual_pipelines(opaque_kind);
            self.ensure_virtual_pose_groups();
        }
        // The visibility buffer (decision 045) resolves the camera's bins it
        // can shade; the others draw forward.
        let resolving =
            virtual_frame && instance_count > 0 && self.prepare_visibility(snapshot, opaque_kind);
        if resolving {
            self.ensure_visibility_pipelines(opaque_kind);
            self.ensure_visibility_groups();
        }
        #[cfg(feature = "profiling")]
        let effect_queries = self.profiler.effect_queries(query_slot);
        #[cfg(not(feature = "profiling"))]
        let effect_queries: Option<(&wgpu::QuerySet, u32)> = None;
        let mut _effects_timed = 0_u32;
        // Shadow casters first: virtual geometry selects the clusters of the
        // virtual ones (decision 044) with the camera's, then the maps draw.
        if let Some(shadow_frame) = &shadow_frame {
            _effects_timed |= u32::from(self.shadows.encode_selection(
                &self.device,
                &mut encoder,
                shadow_frame,
                effect_queries,
            ));
        }
        if virtual_frame {
            timestamp_pass(&mut encoder, effect_queries, VIRTUAL_SELECTION, false);
            self.virtual_geometry.encode(&self.device, &mut encoder);
            timestamp_pass(&mut encoder, effect_queries, VIRTUAL_SELECTION, true);
            _effects_timed |= u32::from(effect_queries.is_some()) << VIRTUAL_SELECTION;
        }
        // Timestamp writes of effect `index` of the profiler's `EFFECTS`.
        let effect_writes = |index: u32| {
            effect_queries.map(|(query_set, first)| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(first + 2 * index),
                end_of_pass_write_index: Some(first + 2 * index + 1),
            })
        };
        let mut shadow_draws = 0;
        if let Some(shadow_frame) = &shadow_frame {
            let meshes = &self.meshes;
            let arena = &self.mesh_arena;
            let (virtual_geometry, shadow_bins) =
                (&self.virtual_geometry, &self.virtual_shadow_bins);
            let encoded = self.shadows.encode_maps(
                &mut encoder,
                shadow_frame,
                &self.plan,
                &self.instance_buffer,
                |mesh| {
                    meshes
                        .get(&mesh)
                        .map(|mesh| (mesh.draw(arena), mesh.skinned))
                },
                |material| self.materials.group(material),
                |pass, map| {
                    shadow_bins
                        .iter()
                        .find(|(bin_map, _, _)| *bin_map == map)
                        .map_or(0, |&(_, bin, offset)| {
                            u32::from(
                                virtual_frame && virtual_geometry.draw_shadow(pass, bin, offset),
                            )
                        })
                },
                effect_queries,
            );
            _effects_timed |= u32::from(encoded.timed);
            shadow_draws = encoded.draws;
        }
        if let (Some(frame), Some(lights)) = (&light_frame, self.lights.as_mut()) {
            lights.encode(
                &self.device,
                &mut encoder,
                frame,
                effect_queries.map(|(set, first)| (set, first + 14)),
            );
            _effects_timed |= u32::from(effect_queries.is_some()) << 7;
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
        if refraction && let Some(post) = self.post.as_ref() {
            let key = (post.generation(), self.camera_groups_key);
            if self
                .refraction_group
                .as_ref()
                .is_none_or(|(built, _)| *built != key)
            {
                let (scene, ..) = post.targets();
                let group = camera_bind_group(
                    &self.device,
                    &self.camera_layout,
                    &self.camera_buffer,
                    &self.lighting_buffer,
                    &self.shadows,
                    (scene, &self.refraction_sampler),
                    (self.environment.binding(), self.probes.words().0),
                );
                self.refraction_group = Some((key, group));
            }
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
        // Occlusion culling (decision 040) draws in two phases: what the last
        // test left visible, then, after the depth pyramid of those depths and
        // the test, what the test newly finds visible.
        let phases = if gpu_mode && self.gpu_presentation.occlusion() {
            2
        } else {
            1
        };
        for phase in 0..phases {
            if phase == 1
                && let Some((pyramid, generation)) =
                    self.depth_pyramid.as_ref().and_then(DepthPyramid::view)
            {
                let odd = temporal_frame
                    .as_ref()
                    .is_some_and(|frame| frame.parity == 1);
                let poses = if odd {
                    self.alternate_poses.as_ref().expect("created above")
                } else {
                    &self.instance_buffer
                };
                let timed = |index: u32, begin: bool| {
                    effect_queries.map(|(set, first)| (set, first + 2 * index + u32::from(!begin)))
                };
                self.depth_pyramid
                    .as_ref()
                    .expect("viewed above")
                    .encode(&mut encoder, timed(14, true));
                // The camera's clusters the last frame's pyramid hid
                // (decision 044) are retested against this frame's within
                // the same span.
                self.gpu_presentation.encode_occlusion(
                    &self.device,
                    &mut encoder,
                    (pyramid, generation),
                    poses,
                    timed(14, false).filter(|_| !virtual_frame),
                );
                if virtual_frame {
                    self.virtual_geometry
                        .encode_retest(&self.device, &mut encoder);
                    timestamp_pass(&mut encoder, effect_queries, 14, true);
                }
                _effects_timed |= u32::from(effect_queries.is_some()) << 14;
            }
            // The first of this phase's camera bins (decision 044), and
            // whether the visibility buffer resolves them (decision 045).
            let virtual_first = match phase {
                0 => Some(0),
                _ => self.virtual_phase_two,
            }
            .filter(|_| virtual_frame);
            let resolve_phase = resolving && virtual_first.is_some();
            let load = |clear| {
                if phase == 0 {
                    wgpu::LoadOp::Clear(clear)
                } else {
                    wgpu::LoadOp::Load
                }
            };
            let attachment = |view, clear| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            // Frames with screen effects draw into their sRGB scene, glow and
            // surface targets, temporal frames into their motion target as
            // well.
            let mut targets = [None, None, None, None];
            let count = match (temporal_pass, post) {
                (Some((_, pass)), Some(post)) => {
                    let (scene, glow, surface) = post.targets();
                    targets = [
                        attachment(scene, clear_color(true)),
                        attachment(pass.motion(), wgpu::Color::TRANSPARENT),
                        attachment(glow, wgpu::Color::TRANSPARENT),
                        attachment(surface, wgpu::Color::TRANSPARENT),
                    ];
                    4
                }
                (_, Some(post)) => {
                    let (scene, glow, surface) = post.targets();
                    targets[0] = attachment(scene, clear_color(true));
                    targets[1] = attachment(glow, wgpu::Color::TRANSPARENT);
                    targets[2] = attachment(surface, wgpu::Color::TRANSPARENT);
                    3
                }
                (_, None) => {
                    targets[0] =
                        attachment(&view, clear_color(self.configuration.format.is_srgb()));
                    1
                }
            };
            // Ambient occlusion, decals, the transparent passes and a second
            // phase read the depth buffer afterwards.
            let keep_depth = post_frame
                .is_some_and(|frame| frame.occlusion || frame.reflections || frame.shafts)
                || transparency
                || decal_frame.is_some()
                || phase + 1 < phases
                || resolve_phase;
            // Profiled frames time both phases as one opaque span.
            #[cfg(feature = "profiling")]
            let timestamp_writes = self
                .profiler
                .writes(query_slot)
                .map(|mut writes| {
                    if phase != 0 {
                        writes.beginning_of_pass_write_index = None;
                    }
                    if phase + 1 != phases || resolve_phase {
                        writes.end_of_pass_write_index = None;
                    }
                    writes
                })
                .filter(timestamps_written);
            #[cfg(not(feature = "profiling"))]
            let timestamp_writes = None;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/opaque-meshes"),
                color_attachments: &targets[..count],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: if phase == 0 {
                            wgpu::LoadOp::Clear(FAR_DEPTH)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: if keep_depth {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
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
                let mut bindings = Bindings::default();
                for &index in &self.draw_order {
                    let index = index as usize;
                    let batch = &self.plan.batches[index];
                    // Batches drawing virtual geometry (decision 044) draw its
                    // clusters below.
                    if self.virtual_drawn.get(index) == Some(&true) {
                        continue;
                    }
                    // Every planned source was validated and uploaded above.
                    let mesh = &self.meshes[&batch.mesh];
                    let (group, variant) = self.materials.binding(batch.material);
                    let skinned = mesh.skinned && affine;
                    let variant = variant.with_extended(mesh.extended).with_skinned(skinned);
                    if variant != bound {
                        pass.set_pipeline(if variant == Variant::BASIC {
                            opaque_base
                        } else {
                            &self.variant_pipelines[&(opaque_kind, variant)]
                        });
                        bound = variant;
                    }
                    bindings.group(&mut pass, 1, group);
                    let draw = mesh.draw(&self.mesh_arena);
                    bindings.vertex(&mut pass, 0, draw.vertices);
                    if let Some(extras) = draw.extras.filter(|_| mesh.extended) {
                        bindings.vertex(&mut pass, 2, extras);
                    }
                    if let Some(skin) = draw.skin.filter(|_| skinned) {
                        bindings.vertex(&mut pass, 3, skin);
                    }
                    bindings.index(&mut pass, draw.indices);
                    if gpu_mode {
                        // Visible slots of this group, compacted and counted
                        // on the GPU, each level of detail (decision 047)
                        // with its own.
                        let virtual_geometry = self.virtual_drawn.get(index) == Some(&true);
                        for level in 0..lod_levels(batch, mesh, virtual_geometry) {
                            let (ids, draws, offset) = self
                                .gpu_presentation
                                .draw_source(batch, index, phase, level);
                            bindings.ids(&mut pass, 1, ids);
                            pass.draw_indexed_indirect(draws, offset);
                        }
                    } else {
                        pass.draw_indexed(
                            draw.index_range(),
                            draw.base_vertex,
                            batch.instances.clone(),
                        );
                    }
                }
                // Impostors (decision 048): every draw group's cards in one
                // draw, over the pose and light groups the meshes bound.
                if gpu_mode
                    && let Some((targets, lit)) = impostor_pass
                    && let (Some(pipeline), Some(group)) = (
                        self.impostors.pipeline(targets, lit),
                        self.impostors.group(),
                    )
                {
                    let (ids, args, offset) = self.gpu_presentation.impostor_source(phase);
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(1, group, &[]);
                    pass.set_vertex_buffer(0, ids.slice(..));
                    pass.draw_indirect(args, offset);
                }
                // Virtual geometry (decision 044): each bin's selected
                // clusters, in phase one, then each bin's twin of clusters
                // the retest found visible in phase two; bins the visibility
                // buffer resolves draw below.
                if let Some(first_bin) = virtual_first {
                    let groups = &self.virtual_pose_groups.as_ref().expect("built above").1;
                    let group = match temporal_pass {
                        Some((frame, _)) => {
                            let previous = if frame.motion_valid {
                                1 - frame.parity
                            } else {
                                frame.parity
                            };
                            &groups[2 * frame.parity + previous]
                        }
                        None => &groups[0],
                    };
                    pass.set_bind_group(2, group, &[]);
                    for (bin, (material, variant)) in self.virtual_bins.iter().enumerate() {
                        if self.visibility_bins.get(bin) == Some(&true) {
                            continue;
                        }
                        pass.set_pipeline(&self.virtual_pipelines[&(opaque_kind, *variant)]);
                        pass.set_bind_group(1, self.materials.binding(*material).0, &[]);
                        let (stream, args, offset) =
                            self.virtual_geometry.bin_source(first_bin + bin);
                        pass.set_vertex_buffer(0, stream);
                        pass.draw_indirect(args, offset);
                    }
                }
            }
            // Last, so depth testing leaves the sky only uncovered pixels.
            let sky = match temporal_pass {
                Some((_, temporal)) => temporal.sky_pipeline(),
                None => post_pipelines.map_or(&self.sky_pipeline, |pipelines| &pipelines.sky),
            };
            if phase + 1 == phases && !resolve_phase {
                pass.set_pipeline(sky);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            drop(pass);
            // The visibility buffer (decision 045): this phase's resolved
            // bins write which triangle covers each pixel, compute lists each
            // bin's tiles, and each bin's resolve shades its pixels; the sky
            // follows the last phase's.
            let Some(first_bin) = virtual_first.filter(|_| resolve_phase) else {
                continue;
            };
            let visibility = self.visibility.as_ref().expect("prepared above");
            let pipelines = self.visibility_pipelines.as_ref().expect("created above");
            let (_, pass_groups, resolve_groups) =
                self.visibility_groups.as_ref().expect("built above");
            let bins = self.virtual_bins.len();
            // Pose buffers as the virtual pose groups pick them.
            let (current, previous) = match temporal_pass {
                Some((frame, _)) if frame.motion_valid => (frame.parity, 1 - frame.parity),
                Some((frame, _)) => (frame.parity, frame.parity),
                None => (0, 0),
            };
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/visibility"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: visibility.ids_view(),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth.view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, &self.empty_group, &[]);
                for (bin, (_, variant)) in self.virtual_bins.iter().enumerate() {
                    if !self.visibility_bins[bin] {
                        continue;
                    }
                    pass.set_pipeline(&pipelines[usize::from(variant.double_sided())]);
                    pass.set_bind_group(
                        2,
                        &pass_groups[current],
                        &[VisibilityBuffer::draw_offset(phase, bins, bin)],
                    );
                    let (stream, args, offset) = self.virtual_geometry.bin_source(first_bin + bin);
                    pass.set_vertex_buffer(0, stream);
                    pass.draw_indirect(args, offset);
                }
            }
            visibility.encode_classify(&mut encoder, bins);
            let resolve_targets: Vec<_> = targets[..count]
                .iter()
                .map(|target| {
                    target
                        .as_ref()
                        .map(|target| wgpu::RenderPassColorAttachment {
                            view: target.view,
                            depth_slice: target.depth_slice,
                            resolve_target: target.resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })
                })
                .collect();
            #[cfg(feature = "profiling")]
            let timestamp_writes = self
                .profiler
                .writes(query_slot)
                .map(|mut writes| {
                    writes.beginning_of_pass_write_index = None;
                    if phase + 1 != phases {
                        writes.end_of_pass_write_index = None;
                    }
                    writes
                })
                .filter(timestamps_written);
            #[cfg(not(feature = "profiling"))]
            let timestamp_writes = None;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/visibility-resolve"),
                color_attachments: &resolve_targets,
                // Read-only: the resolves test no depth and the sky tests it.
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            if let Some((_, lights)) = lit {
                pass.set_bind_group(3, lights.fragment_group(), &[]);
            }
            let (_, _, resolve_args) = visibility.buffers();
            for (bin, (material, variant)) in self.virtual_bins.iter().enumerate() {
                if !self.visibility_bins[bin] {
                    continue;
                }
                pass.set_pipeline(&self.resolve_pipelines[&(opaque_kind, *variant)]);
                pass.set_bind_group(1, self.materials.binding(*material).0, &[]);
                pass.set_bind_group(
                    2,
                    &resolve_groups[2 * current + previous],
                    &[VisibilityBuffer::draw_offset(phase, bins, bin)],
                );
                pass.draw_indirect(resolve_args, bin as u64 * 16);
            }
            if phase + 1 == phases {
                pass.set_pipeline(sky);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        if let (Some(_), Some(decals)) = (&decal_frame, self.decals.as_ref()) {
            let target = post.map_or(&view, |post| post.targets().0);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/decals"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: effect_writes(8),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            _effects_timed |= u32::from(effect_queries.is_some()) << 8;
            decals.draw(
                &mut pass,
                &self.camera_bind_group,
                &self.materials,
                light_frame
                    .as_ref()
                    .and(self.lights.as_ref())
                    .map(LightPass::fragment_group),
                post.is_some(),
            );
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
                    timestamp_writes: effect_writes(9),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if transparent_draws && instance_count > 0 {
                    let transparent_base = self.transparent_pipelines[usize::from(lit.is_some())]
                        [usize::from(affine)]
                    .as_ref()
                    .expect("created with the transparent frame");
                    pass.set_pipeline(transparent_base);
                    // Glass reads the opaque color the pass draws over.
                    let camera_group = match &self.refraction_group {
                        Some((_, group)) if refraction && post_frame.is_some() => group,
                        _ => &self.camera_bind_group,
                    };
                    pass.set_bind_group(0, camera_group, &[]);
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
                    let mut bindings = Bindings::default();
                    for (index, batch) in self.plan.batches.iter().enumerate() {
                        if !batch.transparent {
                            continue;
                        }
                        let mesh = &self.meshes[&batch.mesh];
                        let (group, variant) = self.materials.binding(batch.material);
                        let skinned = mesh.skinned && affine;
                        let variant = variant.with_extended(mesh.extended).with_skinned(skinned);
                        if variant != bound {
                            pass.set_pipeline(if variant == Variant::BASIC {
                                transparent_base
                            } else {
                                &self.variant_pipelines[&(transparent_kind, variant)]
                            });
                            bound = variant;
                        }
                        bindings.group(&mut pass, 1, group);
                        let draw = mesh.draw(&self.mesh_arena);
                        bindings.vertex(&mut pass, 0, draw.vertices);
                        if let Some(extras) = draw.extras.filter(|_| mesh.extended) {
                            bindings.vertex(&mut pass, 2, extras);
                        }
                        if let Some(skin) = draw.skin.filter(|_| skinned) {
                            bindings.vertex(&mut pass, 3, skin);
                        }
                        bindings.index(&mut pass, draw.indices);
                        if gpu_mode {
                            let (ids, draws, offset) =
                                self.gpu_presentation.draw_source(batch, index, 0, 0);
                            bindings.ids(&mut pass, 1, ids);
                            pass.draw_indexed_indirect(draws, offset);
                        } else {
                            pass.draw_indexed(
                                draw.index_range(),
                                draw.base_vertex,
                                batch.instances.clone(),
                            );
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
                    let (scene, glow, _) = post.targets();
                    [load(scene), load(glow)]
                }
                None => [load(&view), None],
            };
            let count = if post.is_some() { 2 } else { 1 };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/transparent-composite"),
                color_attachments: &attachments[..count],
                depth_stencil_attachment: depth,
                timestamp_writes: effect_writes(10),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            transparent.composite(&mut pass, post.is_some());
            _effects_timed |= u32::from(effect_queries.is_some()) << 9
                | u32::from(effect_queries.is_some()) << 10;
            if let (Some(frame), Some(particles)) = (&particle_frame, self.particles.as_ref()) {
                particles.draw_additive(&mut pass, &self.camera_bind_group, frame, post.is_some());
            }
        }
        if let (Some(post_frame), Some(post)) = (post_frame, post) {
            let queries = effect_queries.map(|(set, first)| (set, first + 4));
            post.encode_effects(&mut encoder, post_frame, &self.camera_bind_group, queries);
            _effects_timed |= u32::from(post_frame.occlusion) << 2
                | u32::from(post_frame.bloom) << 3
                | u32::from(post_frame.reflections) << 5
                | u32::from(post_frame.shafts) << 6;
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
        // Reflection probe captures (decision 043), after the frame's own
        // drawing, from this frame's poses, once culling has counted still
        // slots long enough.
        let probe_faces = if gpu_mode && affine && !probe_list.is_empty() {
            self.probes.slots.tick();
            if self.gpu_presentation.stillness_settled() {
                let odd = temporal_frame
                    .as_ref()
                    .is_some_and(|frame| frame.parity == 1);
                #[cfg(feature = "profiling")]
                let effect_queries = self.profiler.effect_queries(query_slot);
                timestamp_pass(&mut encoder, effect_queries, PROBE_CAPTURES, false);
                let faces = self.encode_probe_captures(
                    &mut encoder,
                    probe_list,
                    probe_settings.faces_per_frame,
                    odd,
                    snapshot.elapsed_seconds,
                    light_frame
                        .as_ref()
                        .map(|frame| (frame, &snapshot.local_lights)),
                );
                #[cfg(feature = "profiling")]
                {
                    let queries = self.profiler.effect_queries(query_slot);
                    timestamp_pass(&mut encoder, queries, PROBE_CAPTURES, true);
                    if faces != 0 && queries.is_some() {
                        self.profiler.timed_effects(query_slot, 1 << PROBE_CAPTURES);
                    }
                }
                faces
            } else {
                0
            }
        } else {
            0
        };
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
                .chain([environment_encoder.finish(), encoder.finish()]),
        );
        self.pose_uploads.after_submit();
        if gpu_mode {
            self.gpu_presentation.after_submit();
        }
        self.shadows.after_submit();
        if let Some(lights) = self.lights.as_mut() {
            lights.after_submit();
        }
        if virtual_frame {
            self.virtual_geometry.after_submit();
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
        // Read back a few frames late (decision 044).
        let virtual_stats = virtual_frame
            .then(|| self.virtual_geometry.stats())
            .flatten();
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
            occluded_instances: if gpu_mode && self.occlusion_culling {
                self.gpu_presentation.occluded()
            } else {
                None
            },
            subpixel_instances: if gpu_mode && self.subpixel_culling {
                self.gpu_presentation.subpixel()
            } else {
                None
            },
            lod_instances: if gpu_mode && snapshot.level_of_detail.enabled {
                self.gpu_presentation.lod_instances()
            } else {
                None
            },
            triangles: if gpu_mode {
                self.gpu_presentation.triangles()
            } else {
                // CPU-prepared frames draw every planned instance as its
                // mesh, but virtual geometry as its clusters.
                let triangles: u64 = self
                    .plan
                    .batches
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| self.virtual_drawn.get(*index) != Some(&true))
                    .map(|(_, batch)| {
                        u64::from(batch.instances.end - batch.instances.start)
                            * u64::from(
                                self.meshes
                                    .get(&batch.mesh)
                                    .map_or(0, |mesh| mesh.index_count)
                                    / 3,
                            )
                    })
                    .sum();
                Some(triangles.min(u64::from(u32::MAX)) as u32)
            },
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
            refraction: refraction && post_frame.is_some(),
            visibility_buffer: resolving,
            decals: decal_frame.as_ref().map_or(0, |frame| frame.drawn),
            dropped_decals: (snapshot.dropped_decals as u64
                + decal_frame
                    .as_ref()
                    .map_or(0, |frame| u64::from(frame.dropped)))
            .min(u64::from(u32::MAX)) as u32,
            impostors: self.impostors.live(),
            dropped_impostors: self.dropped_impostors,
            reflection_probes: self.probes.slots.ready(),
            reflection_probe_faces: probe_faces,
            dropped_reflection_probes: dropped_probes.min(u64::from(u32::MAX)) as u32,
            virtual_clusters: virtual_stats.map_or(0, |stats| stats.clusters),
            virtual_triangles: virtual_stats.map_or(0, |stats| stats.triangles),
            virtual_shadow_clusters: virtual_stats.map_or(0, |stats| stats.shadow_clusters),
            virtual_shadow_triangles: virtual_stats.map_or(0, |stats| stats.shadow_triangles),
            virtual_occluded_clusters: virtual_stats.map_or(0, |stats| stats.occluded),
            virtual_resident_pages: virtual_stats.map_or(0, |stats| stats.resident_pages),
            virtual_loading_pages: virtual_stats.map_or(0, |stats| stats.loading_pages),
            virtual_missing_pages: virtual_stats.map_or(0, |stats| stats.missing_pages),
            dropped_virtual_pages: virtual_stats.map_or(0, |stats| stats.dropped_pages),
            dropped_virtual: virtual_stats.map_or(0, |stats| stats.dropped),
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
            engine_profile::counter(
                "memory.gpu_environment_bytes",
                (self.environment.capacity_bytes() + self.probes.capacity_bytes()) as f64,
            );
            engine_profile::counter(
                "memory.gpu_virtual_bytes",
                (self.virtual_geometry.capacity_bytes()
                    + self
                        .visibility
                        .as_ref()
                        .map_or(0, VisibilityBuffer::capacity_bytes)) as f64,
            );
            if let Some(virtual_stats) = virtual_stats {
                engine_profile::counter(
                    "render.virtual_clusters",
                    f64::from(virtual_stats.clusters),
                );
                engine_profile::counter(
                    "render.virtual_triangles",
                    f64::from(virtual_stats.triangles),
                );
                engine_profile::counter(
                    "render.virtual_shadow_triangles",
                    f64::from(virtual_stats.shadow_triangles),
                );
                engine_profile::counter(
                    "render.virtual_occluded_clusters",
                    f64::from(virtual_stats.occluded),
                );
                engine_profile::counter(
                    "render.virtual_resident_pages",
                    f64::from(virtual_stats.resident_pages),
                );
                engine_profile::counter(
                    "render.virtual_loading_pages",
                    f64::from(virtual_stats.loading_pages),
                );
                engine_profile::counter(
                    "render.virtual_missing_pages",
                    f64::from(virtual_stats.missing_pages),
                );
                engine_profile::counter(
                    "render.dropped_virtual_pages",
                    f64::from(virtual_stats.dropped_pages),
                );
                engine_profile::counter("render.dropped_virtual", f64::from(virtual_stats.dropped));
            }
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
            engine_profile::counter("render.decals", f64::from(stats.decals));
            engine_profile::counter("render.impostors", f64::from(stats.impostors));
            engine_profile::counter(
                "render.dropped_impostors",
                f64::from(stats.dropped_impostors),
            );
            engine_profile::counter(
                "memory.gpu_impostor_bytes",
                self.impostors.capacity_bytes() as f64,
            );
            if let Some(occluded) = stats.occluded_instances {
                engine_profile::counter("render.occluded_instances", f64::from(occluded));
            }
            if let Some(subpixel) = stats.subpixel_instances {
                engine_profile::counter("render.subpixel_instances", f64::from(subpixel));
            }
            if let Some(triangles) = stats.triangles {
                engine_profile::counter("render.triangles", f64::from(triangles));
            }
            if let Some(levels) = stats.lod_instances {
                for (name, count) in [
                    "render.lod0_instances",
                    "render.lod1_instances",
                    "render.lod2_instances",
                    "render.impostor_instances",
                ]
                .into_iter()
                .zip(levels)
                {
                    engine_profile::counter(name, f64::from(count));
                }
            }
            engine_profile::counter(
                "memory.gpu_transparent_bytes",
                (self
                    .transparent
                    .as_ref()
                    .map_or(0, TransparentPass::capacity_bytes)
                    + self
                        .particles
                        .as_ref()
                        .map_or(0, ParticlePass::capacity_bytes)
                    + self.decals.as_ref().map_or(0, DecalPass::capacity_bytes))
                    as f64,
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
                (self.gpu_presentation.capacity_bytes()
                    + self
                        .depth_pyramid
                        .as_ref()
                        .map_or(0, DepthPyramid::capacity_bytes)) as f64,
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
                "memory.gpu_mesh_arena_bytes",
                self.mesh_arena.capacity_bytes() as f64,
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
    /// Impostors (decision 048) of this frame: drops those of retired or
    /// replaced meshes and materials, bakes the missing ones of opaque draw
    /// groups whose meshes have enough triangles into `encoder` (after the
    /// materials' levels) on GPU-presented frames with them, and fills the
    /// record of every plan batch, zeroed without one.
    fn prepare_impostors(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        encoder: &mut Option<wgpu::CommandEncoder>,
    ) {
        self.impostor_records.clear();
        self.dropped_impostors = 0;
        let (meshes, materials, plan) = (&self.meshes, &self.materials, &self.plan);
        self.impostors.retain(
            |(mesh, material)| {
                meshes.contains_key(mesh)
                    && material.is_none_or(|handle| materials.group(handle).is_some())
            },
            |handle| plan.material(snapshot, handle).copied(),
        );
        let settings = snapshot.level_of_detail;
        if !(gpu_mode && settings.enabled && settings.impostors) {
            return;
        }
        self.impostors.enforce_budget(settings.max_impostors);
        let device = &self.device;
        for batch in &self.plan.batches {
            let mesh = &self.meshes[&batch.mesh];
            let source = &snapshot.meshes[batch.source_index].mesh;
            let (group, variant) = self.materials.binding(batch.material);
            // Virtual geometry (decision 044) bakes its fallback cut: its
            // camera clusters select only instances not drawn as impostors.
            let eligible = !batch.transparent
                && !mesh.skinned
                && !variant.transmission()
                && source.indices().len() / 3 >= IMPOSTOR_MIN_TRIANGLES;
            let mut record = ImpostorRecord::default();
            if eligible {
                let material = batch
                    .material
                    .and_then(|handle| self.plan.material(snapshot, handle))
                    .copied();
                let key = (batch.mesh, batch.material);
                if let Some(baked) = self.impostors.record(key, material.as_ref()) {
                    record = baked;
                } else {
                    let encoder = encoder.get_or_insert_with(|| {
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("engine/texture-mips"),
                        })
                    });
                    let bounds = source.bounds();
                    let source = BakeSource {
                        draw: mesh.draw(&self.mesh_arena),
                        extended: mesh.extended,
                        group,
                        variant,
                        bounds: (bounds.min(), bounds.max()),
                        material,
                    };
                    if self.impostors.bake(
                        device,
                        encoder,
                        self.materials.mips(),
                        &self.mesh_shader,
                        key,
                        &source,
                        settings.max_impostors,
                    ) {
                        record = self
                            .impostors
                            .record(key, material.as_ref())
                            .unwrap_or_default();
                    } else {
                        self.dropped_impostors += 1;
                    }
                }
            }
            self.impostor_records.push(record);
        }
    }

    /// Uploads this frame's impostor records (decision 048) and creates the
    /// pipeline drawing them in the opaque pass of `kind`; returns its key,
    /// `None` when impostors cannot draw.
    fn prepare_impostor_draws(&mut self, kind: PassKind) -> Option<(ImpostorTargets, bool)> {
        let (targets, lit) = match kind {
            PassKind::Opaque {
                post,
                lit,
                affine: true,
            } => (
                if post {
                    ImpostorTargets::Post
                } else {
                    ImpostorTargets::Canvas
                },
                lit,
            ),
            PassKind::Temporal { lit } => (ImpostorTargets::Temporal, lit),
            _ => return None,
        };
        if !self
            .impostors
            .prepare(&self.device, &self.queue, &self.impostor_records)
        {
            return None;
        }
        if self.impostors.pipeline(targets, lit).is_none() {
            let pose_layout = if targets == ImpostorTargets::Temporal {
                self.temporal.as_ref()?.pose_layout()
            } else {
                &self.pose_layout
            };
            let mut layouts = vec![
                Some(&self.camera_layout),
                Some(self.impostors.draw_layout()),
                Some(pose_layout),
            ];
            if lit {
                layouts.push(Some(self.lights.as_ref()?.fragment_layout()));
            }
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("engine/impostor-pipeline-layout"),
                    bind_group_layouts: &layouts,
                    immediate_size: 0,
                });
            let pipeline = crate::impostors::draw_pipeline(
                &self.device,
                &self.mesh_shader,
                &layout,
                targets,
                lit,
                self.configuration.format,
                DEPTH_FORMAT,
            );
            self.impostors.insert_pipeline(targets, lit, pipeline);
        }
        Some((targets, lit))
    }

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
                PassKind::Temporal { .. } | PassKind::Probe { .. } => true,
            };
            let mesh = self.meshes.get(&batch.mesh);
            let variant = variant
                .with_extended(mesh.is_some_and(|mesh| mesh.extended))
                .with_skinned(affine && mesh.is_some_and(|mesh| mesh.skinned));
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

    /// Records up to `faces` reflection probe cube faces (decision 043), the
    /// nearest probes waiting first and each face at most once a frame (each
    /// has one camera): this frame's still opaque instances (drawn from the
    /// alternate pose buffer when `odd`) and the sky's haze, in linear light,
    /// then a probe's blurred cube once its last face is drawn. Returns the
    /// faces drawn.
    fn encode_probe_captures(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        probes: &[engine_core::RenderReflectionProbe],
        faces: u32,
        odd: bool,
        seconds: f32,
        lights: Option<(&LightFrame, &engine_core::LocalLights)>,
    ) -> u32 {
        let lights = lights.filter(|_| self.lit.is_some() && self.lights.is_some());
        let kind = PassKind::Probe {
            lit: lights.is_some(),
        };
        self.ensure_probe_pipelines(kind);
        if self.probe_camera_groups.is_empty() {
            self.probe_camera_groups = (0..FACES)
                .map(|face| {
                    camera_bind_group(
                        &self.device,
                        &self.camera_layout,
                        self.probes.camera(face),
                        &self.lighting_buffer,
                        &self.shadows,
                        (&self.refraction_fallback, &self.refraction_sampler),
                        (self.environment.binding(), self.probes.words().0),
                    )
                })
                .collect();
        }
        if odd
            && self
                .alternate_pose_group
                .as_ref()
                .is_none_or(|(built, _)| *built != self.pose_generation)
        {
            let group = pose_group(
                &self.device,
                &self.pose_layout,
                self.alternate_poses
                    .as_ref()
                    .expect("temporal frames create it"),
                self.color_buffer.as_ref().expect("affine color buffer"),
                &self.palettes[1],
            );
            self.alternate_pose_group = Some((self.pose_generation, group));
        }
        let mut used = [false; FACES];
        let mut drawn = 0;
        while drawn < faces {
            let Some((cube, face, probe)) = self.probes.slots.next_face(probes) else {
                break;
            };
            if used[face] {
                break;
            }
            used[face] = true;
            let view_projection = face_view_projection(probe.center, face);
            let camera = CameraUniform::probe(view_projection, probe.center, seconds);
            self.queue
                .write_buffer(self.probes.camera(face), 0, bytemuck::bytes_of(&camera));
            let poses = if odd {
                self.alternate_poses
                    .as_ref()
                    .expect("temporal frames create it")
            } else {
                &self.instance_buffer
            };
            if !self.gpu_presentation.encode_probe_face(
                &self.device,
                &self.queue,
                encoder,
                face,
                self.environment.size(),
                view_projection,
                poses,
            ) {
                break;
            }
            // Still lamps light the capture: lights binned for this face.
            if let (Some((frame, settings)), Some(pass)) = (lights, self.lights.as_mut()) {
                pass.encode_probe_face(
                    &self.device,
                    &self.queue,
                    encoder,
                    face,
                    view_projection,
                    self.environment.size(),
                    settings,
                    frame,
                    LightPoses {
                        buffer: poses,
                        generation: self.pose_generation,
                        index: usize::from(odd),
                    },
                    self.gpu_presentation.still_frames(),
                );
            }
            let depth = self
                .probes
                .depth(&self.device, self.environment.size(), DEPTH_FORMAT)
                .clone();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("engine/probe-face"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: self.environment.radiance_face(face),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(FAR_DEPTH),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_bind_group(0, &self.probe_camera_groups[face], &[]);
                let poses_group = if odd {
                    &self.alternate_pose_group.as_ref().expect("built above").1
                } else {
                    self.pose_bind_group
                        .as_ref()
                        .expect("retained pose bindings")
                };
                pass.set_bind_group(2, poses_group, &[]);
                if lights.is_some() {
                    let lights = self.lights.as_ref().expect("checked above");
                    pass.set_bind_group(3, lights.probe_fragment_group(face), &[]);
                }
                let mut bound = None;
                let mut bindings = Bindings::default();
                for &index in &self.draw_order {
                    let index = index as usize;
                    let batch = &self.plan.batches[index];
                    // Only still opaque meshes without joints are captured.
                    let mesh = &self.meshes[&batch.mesh];
                    if mesh.skinned {
                        continue;
                    }
                    let (group, variant) = self.materials.binding(batch.material);
                    let variant = variant.with_extended(mesh.extended);
                    if bound != Some(variant) {
                        pass.set_pipeline(&self.variant_pipelines[&(kind, variant)]);
                        bound = Some(variant);
                    }
                    bindings.group(&mut pass, 1, group);
                    let draw = mesh.draw(&self.mesh_arena);
                    bindings.vertex(&mut pass, 0, draw.vertices);
                    if let Some(extras) = draw.extras.filter(|_| mesh.extended) {
                        bindings.vertex(&mut pass, 2, extras);
                    }
                    bindings.index(&mut pass, draw.indices);
                    let (ids, draws, offset) =
                        self.gpu_presentation.probe_draw_source(batch, index);
                    bindings.ids(&mut pass, 1, ids);
                    pass.draw_indexed_indirect(draws, offset);
                }
                pass.set_pipeline(self.probe_sky_pipeline.as_ref().expect("created above"));
                pass.draw(0..3, 0..1);
            }
            drawn += 1;
            if self.probes.slots.face_drawn() {
                self.environment.finish(encoder, cube);
            }
        }
        drawn
    }

    /// Orders the plan's opaque batches for drawing: by pipeline variant,
    /// then material and vertex buffers. Meshes share the arena's buffers,
    /// so passes then switch pipelines and bind groups once per run rather
    /// than per batch.
    fn order_draws(&mut self) {
        let (plan, meshes, materials) = (&self.plan, &self.meshes, &self.materials);
        self.draw_keys.clear();
        self.draw_keys.extend(
            plan.batches
                .iter()
                .enumerate()
                .filter_map(|(index, batch)| {
                    if batch.transparent {
                        return None;
                    }
                    let mesh = &meshes[&batch.mesh];
                    let (_, variant) = materials.binding(batch.material);
                    let variant = variant
                        .with_extended(mesh.extended)
                        .with_skinned(mesh.skinned);
                    let buffers = match &mesh.placement {
                        Ok(placement) => placement.streams as u8,
                        Err(_) => u8::MAX,
                    };
                    Some((variant, batch.material, buffers, index as u32))
                }),
        );
        self.draw_keys.sort_unstable();
        self.draw_order.clear();
        self.draw_order
            .extend(self.draw_keys.iter().map(|&(.., index)| index));
    }

    /// Plans this frame's virtual geometry (decision 044): the camera's
    /// batches of virtual meshes and their bins, each shadow map's virtual
    /// casters and bin, the views, the camera's occlusion and sub-pixel
    /// tests (with `rasterized`, the projection the frame rasterizes, and
    /// the depth pyramid's levels when occlusion culling runs), and the
    /// bindings selection reads (this frame's poses, GPU-compacted visible
    /// IDs or CPU-prepared slots, the shadow maps' casters and the depth
    /// pyramid). Returns whether any batch draws clusters.
    fn prepare_virtual(
        &mut self,
        snapshot: &RenderSnapshot,
        gpu_mode: bool,
        odd: bool,
        (rasterized, pyramid_levels): (engine_core::glam::Mat4, Option<u32>),
    ) -> bool {
        self.virtual_bins.clear();
        self.virtual_drawn.clear();
        self.virtual_drawn.resize(self.plan.batches.len(), false);
        self.virtual_shadow_bins.clear();
        self.virtual_phase_two = None;
        // Frames drawing two phases (decision 040) retest the camera's
        // clusters the last such frame's pyramid hid, if it is this one's.
        let two_phases = gpu_mode && self.gpu_presentation.occlusion();
        let pyramid = self
            .depth_pyramid
            .as_ref()
            .and_then(DepthPyramid::view)
            .filter(|_| two_phases);
        let previous = self
            .virtual_previous
            .take()
            .and_then(|(generation, projection)| {
                pyramid
                    .is_some_and(|(_, current)| current == generation)
                    .then_some(projection)
            });
        self.virtual_previous = pyramid.map(|(_, generation)| (generation, rasterized));
        if !snapshot.virtual_geometry.enabled {
            return false;
        }
        let mut batches = Vec::new();
        let mut max_instances = 0;
        for (index, batch) in self.plan.batches.iter().enumerate() {
            if batch.transparent {
                continue;
            }
            let Some(mesh) = self.virtual_geometry.mesh_index(batch.mesh) else {
                continue;
            };
            let (_, variant) = self.materials.binding(batch.material);
            let bin = match self
                .virtual_bins
                .iter()
                .position(|(material, _)| *material == batch.material)
            {
                Some(bin) => bin,
                None => {
                    self.virtual_bins.push((batch.material, variant));
                    self.virtual_bins.len() - 1
                }
            };
            let count = batch.instances.end - batch.instances.start;
            max_instances = max_instances.max(count);
            batches.push(VirtualBatch {
                start: batch.instances.start,
                count: (!gpu_mode).then_some(count),
                draw: index as u32,
                mesh,
                bin: bin as u32,
                double_sided: variant.double_sided(),
                view: 0,
            });
            self.virtual_drawn[index] = true;
        }
        if batches.is_empty() {
            return false;
        }
        let view_projection = snapshot.view_projection;
        let mut camera_planes = [[0.0, 0.0, 0.0, 1.0]; crate::shadows::VIRTUAL_SHADOW_PLANES];
        camera_planes[..6].copy_from_slice(&crate::virtual_geometry::planes(view_projection));
        let camera = VirtualView {
            planes: camera_planes,
            eye: snapshot.camera_position,
            min_distance: VIRTUAL_NEAREST,
            // The projection's y scale: its second row's length.
            projection: view_projection.row(1).truncate().length() * 0.5,
            threshold: snapshot.virtual_geometry.pixel_error
                / self.configuration.height.max(1) as f32,
            unseen_error: 0.0,
            cone: Cone::Eye,
        };
        // Shadow maps drawn every frame select their virtual casters'
        // clusters as the camera sees them (decision 044), from where their
        // receivers start, and where the camera sees nothing within as many
        // of the map's texels as pixels, into a bin each. Closed meshes cast
        // the clusters facing the sun, which hide the rest; others cast both
        // faces.
        let mut views = vec![camera];
        let mut shadow_batches = Vec::new();
        let mut shadow_instances = 0;
        let (_, _, id_capacity, _) = self.shadows.caster_buffers();
        let shadow_groups = self.shadows.groups();
        let group_count = shadow_groups.len() as u32;
        for shadow_view in self.shadows.virtual_views() {
            if views.len() == MAX_VIEWS {
                break;
            }
            let bin = self.virtual_bins.len() + self.virtual_shadow_bins.len();
            let view = views.len() as u32;
            let mut any = false;
            for (index, group) in shadow_groups.iter().enumerate() {
                if !group.virtual_geometry {
                    continue;
                }
                let Some(mesh) = self.virtual_geometry.mesh_index(group.mesh) else {
                    continue;
                };
                shadow_batches.push(VirtualBatch {
                    start: shadow_view.map as u32 * id_capacity + group.start,
                    count: None,
                    draw: shadow_view.map as u32 * group_count + index as u32,
                    mesh,
                    bin: bin as u32,
                    double_sided: !self.virtual_geometry.closed(mesh),
                    view,
                });
                shadow_instances = shadow_instances.max(group.capacity);
                any = true;
            }
            if any {
                self.virtual_shadow_bins
                    .push((shadow_view.map, bin, shadow_view.view_offset));
                views.push(VirtualView {
                    planes: shadow_view.planes,
                    min_distance: shadow_view.min_distance.unwrap_or(VIRTUAL_NEAREST),
                    unseen_error: snapshot.virtual_geometry.pixel_error * shadow_view.texel,
                    cone: Cone::Along(shadow_view.direction),
                    ..camera
                });
            }
        }
        // Phase two's twins of the camera's bins follow the shadow maps'.
        let mut bins = self.virtual_bins.len() + self.virtual_shadow_bins.len();
        if pyramid.is_some() {
            self.virtual_phase_two = Some(bins);
            bins += self.virtual_bins.len();
        }
        let tests = CameraTests {
            view_projection: rasterized,
            viewport: (
                self.configuration.width,
                self.configuration.height,
                pyramid_levels.unwrap_or(0),
            ),
            previous,
            phase_two: self.virtual_phase_two.map(|bin| bin as u32),
            subpixel: gpu_mode && self.gpu_presentation.culls_subpixel(),
        };
        self.virtual_geometry.prepare(
            &self.device,
            &self.queue,
            [&batches, &shadow_batches],
            bins,
            &views,
            &tests,
            snapshot.virtual_geometry.max_resident_pages,
        );
        let poses = if odd {
            self.alternate_poses
                .as_ref()
                .expect("temporal frames create it")
        } else {
            &self.instance_buffer
        };
        let camera_ids = if gpu_mode {
            let (ids, draws, generation) = self.gpu_presentation.visible_sources();
            IdBuffers {
                ids,
                draws,
                key: generation * 2,
            }
        } else {
            let slots = self.slot_buffer.as_ref().expect("retained slot buffer");
            IdBuffers {
                ids: slots,
                draws: slots,
                key: self.slot_capacity as u64 * 2 + 1,
            }
        };
        let (shadow_ids, shadow_args, _, shadow_key) = self.shadows.caster_buffers();
        self.virtual_geometry.bind(
            &self.device,
            (poses, self.pose_generation * 2 + u64::from(odd)),
            camera_ids,
            (!shadow_batches.is_empty()).then_some(IdBuffers {
                ids: shadow_ids,
                draws: shadow_args,
                key: shadow_key,
            }),
            [max_instances, shadow_instances],
            self.shadows.view_buffer(),
            pyramid,
        );
        true
    }

    /// Pose groups of virtual geometry pipelines (decision 044): each pair
    /// of this frame's and the previous frame's pose buffers, with the
    /// colors and the arena.
    fn ensure_virtual_pose_groups(&mut self) {
        let Some(([meshes, clusters, pages], arena)) = self.virtual_geometry.arena() else {
            return;
        };
        let key = (self.pose_generation, arena, self.alternate_poses.is_some());
        if self
            .virtual_pose_groups
            .as_ref()
            .is_some_and(|(built, _)| *built == key)
        {
            return;
        }
        let buffers = [
            &self.instance_buffer,
            self.alternate_poses
                .as_ref()
                .unwrap_or(&self.instance_buffer),
        ];
        let colors = self.color_buffer.as_ref().expect("affine color buffer");
        let group = |current: usize, previous: usize| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/virtual-pose-group"),
                layout: &self.virtual_pose_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers[current].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: colors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers[previous].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: meshes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: clusters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: pages.as_entire_binding(),
                    },
                ],
            })
        };
        let groups = [group(0, 0), group(0, 1), group(1, 0), group(1, 1)];
        self.virtual_pose_groups = Some((key, groups));
    }

    /// The virtual geometry pipelines (decision 044) of `kind` for this
    /// frame's bins' material variants.
    fn ensure_virtual_pipelines(&mut self, kind: PassKind) {
        let lit = matches!(
            kind,
            PassKind::Opaque { lit: true, .. } | PassKind::Temporal { lit: true }
        );
        if self.virtual_layouts[usize::from(lit)].is_none() {
            let lights = self.lights.as_ref().filter(|_| lit);
            let mut layouts = vec![
                Some(&self.camera_layout),
                Some(self.materials.layout()),
                Some(&self.virtual_pose_layout),
            ];
            if let Some(lights) = lights {
                layouts.push(Some(lights.fragment_layout()));
            }
            self.virtual_layouts[usize::from(lit)] = Some(self.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("engine/virtual-pipeline-layout"),
                    bind_group_layouts: &layouts,
                    immediate_size: 0,
                },
            ));
        }
        let missing: Vec<Variant> = self
            .virtual_bins
            .iter()
            .map(|(_, variant)| *variant)
            .filter(|variant| !self.virtual_pipelines.contains_key(&(kind, *variant)))
            .collect();
        for variant in missing {
            let pipeline = self.create_virtual(kind, variant, lit);
            self.virtual_pipelines.insert((kind, variant), pipeline);
        }
    }

    /// A virtual geometry pipeline (decision 044): `vs_virtual` (or its
    /// temporal twin) with `kind`'s fragment stage and targets.
    fn create_virtual(&self, kind: PassKind, variant: Variant, lit: bool) -> wgpu::RenderPipeline {
        let layout = self.virtual_layouts[usize::from(lit)]
            .as_ref()
            .expect("created above");
        let VirtualStages {
            vertex,
            fragment,
            targets,
            constants,
        } = self.virtual_stages(kind, variant, false);
        let targets = &targets[..];
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("engine/virtual-meshes"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &self.mesh_shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: crate::virtual_geometry::VISIBLE_BYTES,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &VIRTUAL_ATTRIBUTES,
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.mesh_shader,
                    entry_point: Some(fragment),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// The stages of `kind`'s virtual geometry pipelines for material
    /// `variant`: forward (decision 044) or resolving the visibility buffer
    /// (decision 045), with the targets and constants of `kind`.
    fn virtual_stages(&self, kind: PassKind, variant: Variant, resolve: bool) -> VirtualStages {
        let canvas = vec![Some(wgpu::ColorTargetState {
            format: self.configuration.format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let canvas_constants = vec![(
            "SURFACE_IS_SRGB",
            if self.configuration.format.is_srgb() {
                1.0
            } else {
                0.0
            },
        )];
        let post = vec![
            Some(SCENE_FORMAT.into()),
            Some(GLOW_FORMAT.into()),
            Some(SURFACE_FORMAT.into()),
        ];
        let temporal = vec![
            Some(SCENE_FORMAT.into()),
            Some(crate::temporal::MOTION_FORMAT.into()),
            Some(GLOW_FORMAT.into()),
            Some(SURFACE_FORMAT.into()),
        ];
        let (vertex, fragment, targets, constants) = match kind {
            PassKind::Opaque {
                post: false, lit, ..
            } => (
                "vs_virtual",
                match (lit, resolve) {
                    (false, false) => "fs_main",
                    (true, false) => "fs_main_lit",
                    (false, true) => "fs_resolve",
                    (true, true) => "fs_resolve_lit",
                },
                canvas,
                canvas_constants,
            ),
            PassKind::Opaque {
                post: true, lit, ..
            } => (
                "vs_virtual",
                match (lit, resolve) {
                    (false, false) => "fs_post",
                    (true, false) => "fs_post_lit",
                    (false, true) => "fs_resolve_post",
                    (true, true) => "fs_resolve_post_lit",
                },
                post,
                Vec::new(),
            ),
            PassKind::Temporal { lit } => (
                "vs_virtual_temporal",
                match (lit, resolve) {
                    (false, false) => "fs_temporal",
                    (true, false) => "fs_temporal_lit",
                    (false, true) => "fs_resolve_temporal",
                    (true, true) => "fs_resolve_temporal_lit",
                },
                temporal,
                Vec::new(),
            ),
            PassKind::Transparent { .. } | PassKind::Probe { .. } => {
                unreachable!("virtual geometry draws in opaque passes")
            }
        };
        VirtualStages {
            vertex: if resolve { "vs_resolve" } else { vertex },
            fragment,
            targets,
            constants: constants.into_iter().chain(variant.constants()).collect(),
        }
    }

    /// Readies the visibility buffer (decision 045) for this frame's camera
    /// bins and marks those it resolves: every bin but those of alpha-masked
    /// materials, whose coverage its pass does not test. False when it is
    /// off, resolves no bin, or the device binds too few storage buffers or
    /// too small ones for `kind`'s resolves; the bins then draw forward.
    fn prepare_visibility(&mut self, snapshot: &RenderSnapshot, kind: PassKind) -> bool {
        self.visibility_bins.clear();
        let lit = matches!(
            kind,
            PassKind::Opaque { lit: true, .. } | PassKind::Temporal { lit: true }
        );
        let needed = if lit {
            RESOLVE_LIT_STORAGE_BUFFERS
        } else {
            RESOLVE_STORAGE_BUFFERS
        };
        if !snapshot.virtual_geometry.visibility_buffer
            || self.device.limits().max_storage_buffers_per_shader_stage < needed
            || self
                .virtual_bins
                .iter()
                .all(|(_, variant)| variant.alpha_mask())
        {
            return false;
        }
        let bins = self.virtual_bins.len();
        let bases = [Some(0), self.virtual_phase_two].map(|first| {
            (0..bins)
                .map(|bin| first.map_or(0, |first| self.virtual_geometry.bin_base(first + bin)))
                .collect::<Vec<u32>>()
        });
        let size = (self.configuration.width, self.configuration.height);
        let visibility = self
            .visibility
            .get_or_insert_with(|| VisibilityBuffer::new(&self.device));
        if !visibility.prepare(&self.device, &self.queue, size, [&bases[0], &bases[1]]) {
            return false;
        }
        self.visibility_bins.extend(
            self.virtual_bins
                .iter()
                .map(|(_, variant)| !variant.alpha_mask()),
        );
        true
    }

    /// The visibility pass's pipelines and `kind`'s resolves (decision 045)
    /// for this frame's resolved bins' material variants.
    fn ensure_visibility_pipelines(&mut self, kind: PassKind) {
        if self.visibility_pipelines.is_none() {
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("engine/visibility-pipeline-layout"),
                    bind_group_layouts: &[
                        Some(&self.camera_layout),
                        Some(&self.empty_layout),
                        Some(&self.visibility_layout),
                    ],
                    immediate_size: 0,
                });
            let pipeline = |double_sided: bool| {
                self.device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("engine/visibility"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &self.mesh_shader,
                            entry_point: Some("vs_visibility"),
                            compilation_options: Default::default(),
                            buffers: &[Some(wgpu::VertexBufferLayout {
                                array_stride: crate::virtual_geometry::VISIBLE_BYTES,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &VIRTUAL_ATTRIBUTES,
                            })],
                        },
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
                            ..Default::default()
                        },
                        depth_stencil: Some(wgpu::DepthStencilState {
                            format: DEPTH_FORMAT,
                            depth_write_enabled: Some(true),
                            depth_compare: Some(wgpu::CompareFunction::Greater),
                            stencil: Default::default(),
                            bias: Default::default(),
                        }),
                        multisample: wgpu::MultisampleState::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &self.mesh_shader,
                            entry_point: Some("fs_visibility"),
                            compilation_options: Default::default(),
                            targets: &[Some(ID_FORMAT.into())],
                        }),
                        multiview_mask: None,
                        cache: None,
                    })
            };
            self.visibility_pipelines = Some([pipeline(false), pipeline(true)]);
        }
        let lit = matches!(
            kind,
            PassKind::Opaque { lit: true, .. } | PassKind::Temporal { lit: true }
        );
        if self.resolve_layouts[usize::from(lit)].is_none() {
            let lights = self.lights.as_ref().filter(|_| lit);
            let mut layouts = vec![
                Some(&self.camera_layout),
                Some(self.materials.layout()),
                Some(&self.resolve_layout),
            ];
            if let Some(lights) = lights {
                layouts.push(Some(lights.fragment_layout()));
            }
            self.resolve_layouts[usize::from(lit)] = Some(self.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: Some("engine/visibility-resolve-pipeline-layout"),
                    bind_group_layouts: &layouts,
                    immediate_size: 0,
                },
            ));
        }
        let missing: Vec<Variant> = self
            .virtual_bins
            .iter()
            .zip(&self.visibility_bins)
            .filter(|&(_, &resolved)| resolved)
            .map(|((_, variant), _)| *variant)
            .filter(|variant| !self.resolve_pipelines.contains_key(&(kind, *variant)))
            .collect();
        for variant in missing {
            let pipeline = self.create_resolve(kind, variant, lit);
            self.resolve_pipelines.insert((kind, variant), pipeline);
        }
    }

    /// A resolve of the visibility buffer (decision 045): `vs_resolve`'s
    /// tiles with `kind`'s resolving fragment stage and targets, testing no
    /// depth.
    fn create_resolve(&self, kind: PassKind, variant: Variant, lit: bool) -> wgpu::RenderPipeline {
        let layout = self.resolve_layouts[usize::from(lit)]
            .as_ref()
            .expect("created above");
        let VirtualStages {
            vertex,
            fragment,
            targets,
            constants,
        } = self.virtual_stages(kind, variant, true);
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("engine/visibility-resolve"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &self.mesh_shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.mesh_shader,
                    entry_point: Some(fragment),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// The visibility buffer's groups (decision 045): its pass's per current
    /// pose buffer, and the resolves' per current and previous one, as
    /// `virtual_pose_groups`.
    fn ensure_visibility_groups(&mut self) {
        let (Some(([meshes, clusters, pages], arena)), Some(visibility)) =
            (self.virtual_geometry.arena(), self.visibility.as_ref())
        else {
            return;
        };
        let (visible, selection) = self.virtual_geometry.visible();
        let key = (
            self.pose_generation,
            self.alternate_poses.is_some(),
            arena,
            selection,
            visibility.generation(),
        );
        if self
            .visibility_groups
            .as_ref()
            .is_some_and(|(built, ..)| *built == key)
        {
            return;
        }
        let buffers = [
            &self.instance_buffer,
            self.alternate_poses
                .as_ref()
                .unwrap_or(&self.instance_buffer),
        ];
        let colors = self.color_buffer.as_ref().expect("affine color buffer");
        let (draws, tiles, _) = visibility.buffers();
        let draw = wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: draws,
                offset: 0,
                size: wgpu::BufferSize::new(16),
            }),
        };
        let arena = [meshes, clusters, pages];
        let pass_group = |current: usize| {
            let [poses, meshes, clusters, pages] = arena_entries(buffers[current], arena);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/visibility-group"),
                layout: &self.visibility_layout,
                entries: &[poses, meshes, clusters, pages, draw.clone()],
            })
        };
        let resolve_group = |current: usize, previous: usize| {
            let [poses, meshes, clusters, pages] = arena_entries(buffers[current], arena);
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/visibility-resolve-group"),
                layout: &self.resolve_layout,
                entries: &[
                    poses,
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: colors.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers[previous].as_entire_binding(),
                    },
                    meshes,
                    clusters,
                    pages,
                    draw.clone(),
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: visible.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(visibility.ids_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: tiles.as_entire_binding(),
                    },
                ],
            })
        };
        let pass_groups = [pass_group(0), pass_group(1)];
        let resolve_groups = [
            resolve_group(0, 0),
            resolve_group(0, 1),
            resolve_group(1, 0),
            resolve_group(1, 1),
        ];
        self.visibility_groups = Some((key, pass_groups, resolve_groups));
    }

    /// The reflection probe captures' pipelines (decision 043): the sky's,
    /// and `kind`'s for the variants of this frame's still opaque meshes.
    fn ensure_probe_pipelines(&mut self, kind: PassKind) {
        if self.probe_sky_pipeline.is_none() {
            self.probe_sky_pipeline = Some(sky_pipeline(
                &self.device,
                &self.mesh_shader,
                &self.camera_layout,
                "fs_probe_sky",
                &[Some(ENVIRONMENT_FORMAT.into())],
                None,
            ));
        }
        let mut missing = Vec::new();
        for batch in &self.plan.batches {
            let Some(mesh) = self.meshes.get(&batch.mesh) else {
                continue;
            };
            if batch.transparent || mesh.skinned {
                continue;
            }
            let (_, variant) = self.materials.binding(batch.material);
            let variant = variant.with_extended(mesh.extended);
            if !self.variant_pipelines.contains_key(&(kind, variant)) && !missing.contains(&variant)
            {
                missing.push(variant);
            }
        }
        for variant in missing {
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
                let post_targets = [
                    Some(SCENE_FORMAT.into()),
                    Some(GLOW_FORMAT.into()),
                    Some(SURFACE_FORMAT.into()),
                ];
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
                    mirrored: false,
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
                    mirrored: false,
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
            PassKind::Probe { lit } => {
                let targets = [Some(ENVIRONMENT_FORMAT.into())];
                let targets = MeshTargets {
                    fragment: if lit { "fs_probe_lit" } else { "fs_probe" },
                    targets: &targets,
                    constants: &[],
                    depth_write: true,
                    mirrored: true,
                };
                mesh_pipeline(device, shader, layout(lit, true), true, &targets, variant)
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

/// A virtual geometry pipeline's entry points, color targets and constants
/// (decisions 044 and 045).
/// Whether a pass's timestamp writes write any query: passes of a span
/// between its first and last write none, and WebGPU refuses writes naming
/// neither.
#[cfg(feature = "profiling")]
fn timestamps_written(writes: &wgpu::RenderPassTimestampWrites<'_>) -> bool {
    writes.beginning_of_pass_write_index.is_some() || writes.end_of_pass_write_index.is_some()
}

struct VirtualStages {
    vertex: &'static str,
    fragment: &'static str,
    targets: Vec<Option<wgpu::ColorTargetState>>,
    constants: Vec<(&'static str, f64)>,
}

/// The smallest distance virtual geometry's errors project from (decision
/// 044): a guard for the eye's own position.
const VIRTUAL_NEAREST: f32 = 0.05;

/// Selected clusters of virtual geometry (decision 044): visible ID and
/// cluster, one entry per instance.
const VIRTUAL_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![2 => Uint32, 13 => Uint32];

/// Group entries of slot poses and the virtual arena's meshes, clusters and
/// pages (decision 044), at the bindings `meshes.wgsl` declares.
fn arena_entries<'a>(
    poses: &'a wgpu::Buffer,
    [meshes, clusters, pages]: [&'a wgpu::Buffer; 3],
) -> [wgpu::BindGroupEntry<'a>; 4] {
    [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: poses.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: meshes.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 6,
            resource: clusters.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: pages.as_entire_binding(),
        },
    ]
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

/// Reflection probe captures' entry in the profiler's effects (decision 043).
const PROBE_CAPTURES: u32 = 15;
/// Virtual geometry's cluster selection entry in the profiler's effects
/// (decision 044).
const VIRTUAL_SELECTION: u32 = 16;

/// Writes query `first + 2 * effect` (or the next one at its `end`) of
/// `queries`, the profiler's effect queries, from an empty pass: effects of
/// many passes are timed from one before them to one after.
fn timestamp_pass(
    encoder: &mut wgpu::CommandEncoder,
    queries: Option<(&wgpu::QuerySet, u32)>,
    effect: u32,
    end: bool,
) {
    if let Some((query_set, first)) = queries {
        let query = first + 2 * effect;
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/effect-timestamp"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: (!end).then_some(query),
                end_of_pass_write_index: end.then_some(query + 1),
            }),
        }));
    }
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
        // On the far plane: it shows where nothing was drawn.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
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
    // Drawn through a mirroring view (reflection probe faces): triangles
    // wind the other way.
    mirrored: bool,
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
            front_face: if fragment.mirrored {
                wgpu::FrontFace::Cw
            } else {
                wgpu::FrontFace::Ccw
            },
            cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(fragment.depth_write),
            depth_compare: Some(wgpu::CompareFunction::Greater),
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
            mirrored: false,
        };
        let post_targets = [
            Some(SCENE_FORMAT.into()),
            Some(GLOW_FORMAT.into()),
            Some(SURFACE_FORMAT.into()),
        ];
        let post = MeshTargets {
            fragment: "fs_post_lit",
            targets: &post_targets,
            constants: &[],
            depth_write: true,
            mirrored: false,
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

/// Mesh and sky pipelines drawing into the screen effects' scene, glow and
/// surface targets (decisions 021 and 025).
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
        let targets = [
            Some(SCENE_FORMAT.into()),
            Some(GLOW_FORMAT.into()),
            Some(SURFACE_FORMAT.into()),
        ];
        let post = MeshTargets {
            fragment: "fs_post",
            targets: &targets,
            constants: &[],
            depth_write: true,
            mirrored: false,
        };
        Self {
            full: mesh_pipeline(device, shader, layouts[0], false, &post, Variant::BASIC),
            affine: mesh_pipeline(device, shader, layouts[1], true, &post, Variant::BASIC),
            sky: sky_pipeline(device, shader, camera_layout, "fs_sky_post", &targets, None),
        }
    }
}

/// The main passes' first group: camera, lighting, shadow lookups, what
/// glass refracts (decision 038) with its sampler, and what surfaces reflect
/// (decision 043): the environment maps with their sampler and the
/// reflection probes' grid.
fn camera_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &wgpu::Buffer,
    lighting: &wgpu::Buffer,
    shadows: &ShadowPass,
    (refraction, refraction_sampler): (&wgpu::TextureView, &wgpu::Sampler),
    ((environment, environment_sampler), probe_words): (
        (&wgpu::TextureView, &wgpu::Sampler),
        &wgpu::Buffer,
    ),
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
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(refraction),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::Sampler(refraction_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(environment),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::Sampler(environment_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: probe_words.as_entire_binding(),
            },
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
