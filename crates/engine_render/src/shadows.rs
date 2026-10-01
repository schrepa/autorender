//! Sun shadows (decision 020): depth maps seen from the atmosphere's sun.
//!
//! Near cascades, layers of one texture, cover the first stretch of the view,
//! each reaching a third as far as the next (decision 028), and are redrawn
//! every frame. Two far maps share one projection over the shadow region: the kept
//! map holds casters whose pose stayed the same for a few frames and is
//! redrawn only where that changes, a rectangle of its 16x16 tiles per
//! frame, or whole after its projection or the sun moved; the moving map
//! holds every other caster and is redrawn every frame. One compute pass
//! selects each map's casters from every candidate, visible or culled, into
//! indirect draws per mesh; on GPU-presented frames it also keeps each
//! slot's stillness and marks dirty tiles, read back a few frames later. CPU-prepared frames draw
//! every caster into the moving map and leave the kept map unread; public
//! snapshots draw their instances without selection. Casters with virtual
//! geometry (decision 044) are selected into the maps drawn every frame,
//! whose clusters virtual geometry selects and draws after its own
//! selection: [`ShadowPass::encode_selection`], then
//! [`ShadowPass::encode_maps`].

use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use engine_core::{
    AlphaMode, MaterialHandle, MeshHandle, RenderFrame, RenderSnapshot, glam::Vec3, math,
};

use crate::batches::BatchPlan;
use crate::gpu_data::{
    BatchData, CandidateData, DRAW_ARGS_BYTES, MAX_CASCADES, SHADOW_SHADER, ShadowCullUniform,
    ShadowUniform,
};
use crate::mesh_arena::{Bindings, Ids, MeshDraw};
use crate::shadow_fit::{
    DirtyTiles, LightBasis, MapRegion, TILE_WORDS, TILES, TileRegion, caster_planes,
    frustum_footprint, view_sphere,
};

pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
// Near cascades are maps 0 to MAX_CASCADES - 1; the far maps follow.
const MOVING_MAP: usize = MAX_CASCADES;
const KEPT_MAP: usize = MAX_CASCADES + 1;
const MAPS: usize = MAX_CASCADES + 2;
// Uniform offset alignment between view matrices: each near cascade's, then
// the far maps'.
const VIEW_STRIDE: u64 = 256;
const FAR_VIEW: u64 = MAX_CASCADES as u64 * VIEW_STRIDE;
const GROUP: u32 = 256;
const SKINNED_GROUP: u32 = 64;
const READBACKS: usize = 3;
// Mode bits of the selection pass.
const STILLNESS: u32 = 1;
const REFRESH: u32 = 2;
// Lookups move this many texels along the surface normal. Receiver planes
// (decision 028) keep tilted surfaces off their own depths; the offset only
// covers where a lookup's texels meet another surface.
const NORMAL_OFFSET_TEXELS: f32 = 0.5;
// Receiver planes steeper than this, in texels of depth per texel (86
// degrees from the sun), are clamped.
const SLOPE_LIMIT_TEXELS: f32 = 16.0;
// Shadows fade over the last part of their distance.
const FADE_START: f32 = 0.85;
// Texels of margin around the far region.
const MARGIN_TEXELS: f32 = 2.0;
// Bindings need at least one element; exceeds every record stride.
const MIN_STORAGE_BYTES: u64 = 256;

/// One mesh drawn into shadow maps, with its casters' ID region per map.
#[derive(Clone, Copy)]
pub(crate) struct ShadowGroup {
    pub mesh: MeshHandle,
    /// Index of its source in the snapshot's meshes.
    pub source_index: usize,
    /// The alpha-masked material (decision 031) whose base color alpha
    /// these casters test; `None` casts the whole mesh.
    pub material: Option<MaterialHandle>,
    /// Whether virtual geometry (decision 044) draws its casters' clusters
    /// instead of the mesh.
    pub virtual_geometry: bool,
    /// Its ID region's start and size in each map.
    pub start: u32,
    pub capacity: u32,
}

/// Planes of a virtual shadow view: its selection region's five, and the
/// view frustum slice's that the sun's sweep never crosses.
pub(crate) const VIRTUAL_SHADOW_PLANES: usize = 12;

/// A shadow map drawn this frame whose virtual casters' clusters virtual
/// geometry selects (decision 044): its planes around where casters can
/// darken visible receivers (unused ones hold everything), the camera
/// distance its receivers start at (`None` at the camera), its texel's
/// world size, the direction it looks along (away from the sun) and the
/// offset of its view matrix.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualShadowView {
    pub map: usize,
    pub planes: [[f32; 4]; VIRTUAL_SHADOW_PLANES],
    pub min_distance: Option<f32>,
    pub texel: f32,
    pub direction: Vec3,
    pub view_offset: u32,
}

/// Where this frame's casters come from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// GPU presentation's candidates, batches and flags.
    Presented,
    /// Every caster of a CPU-prepared retained frame.
    Listed,
    /// The instances of a public snapshot, drawn from the instance buffer.
    Snapshot,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Refresh {
    Whole,
    Tiles(TileRegion),
}

/// Inputs of GPU-presented frames' selection, from GPU presentation.
pub(crate) struct Presented<'a> {
    pub candidates: &'a wgpu::Buffer,
    pub batches: &'a wgpu::Buffer,
    pub flags: &'a wgpu::Buffer,
    pub count: u32,
    /// Skinned slots (decision 036) and their world bounds from this
    /// frame's palettes.
    pub skinned: &'a wgpu::Buffer,
    pub skin_bounds: &'a wgpu::Buffer,
    pub skinned_count: u32,
    /// Advances when any of these buffers is replaced.
    pub generation: u64,
}

/// Pose storage of retained frames: the buffer pair of `generation` and the
/// one this frame's poses are in.
pub(crate) struct Poses<'a> {
    pub buffers: [Option<&'a wgpu::Buffer>; 2],
    /// Slot colors, whose alpha masked casters test and whose flags locate
    /// skinned slots' palettes (decision 036).
    pub colors: &'a wgpu::Buffer,
    /// Joint palettes, one per pose buffer.
    pub palettes: [&'a wgpu::Buffer; 2],
    pub index: usize,
    pub generation: u64,
}

/// How a caster group's mesh draws (decision 047): its index count, first
/// index and base vertex, and its coarser levels of detail (first index,
/// index count and geometric error in mesh units, finest first).
pub(crate) struct CasterDraw {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub lods: Vec<(u32, u32, f32)>,
}

impl CasterDraw {
    /// The coarsest level whose error is at most `error` world units at unit
    /// scale; 0 for the mesh itself.
    fn level_within(&self, error: f32) -> u8 {
        self.lods
            .iter()
            .take_while(|&&(_, _, lod_error)| lod_error <= error)
            .count() as u8
    }

    /// Index count, first index and base vertex of level `level`.
    fn range(&self, level: u8) -> (u32, u32, i32) {
        match level.checked_sub(1) {
            Some(lod) => {
                let (first_index, index_count, _) = self.lods[lod as usize];
                (index_count, first_index, self.base_vertex)
            }
            None => (self.index_count, self.first_index, self.base_vertex),
        }
    }
}

/// A retained caster group as depth passes draw it: its pipeline choice
/// (masked, skinned), mesh, joints stream, material and group index.
struct Caster<'m> {
    choice: usize,
    draw: MeshDraw<'m>,
    skin: Option<&'m wgpu::Buffer>,
    material: Option<&'m wgpu::BindGroup>,
    index: usize,
}

/// What [`ShadowPass::encode`] recorded.
#[derive(Clone, Copy, Default)]
pub(crate) struct ShadowEncoded {
    /// Passes timed (bit 0 selection, bit 1 depth maps).
    pub timed: u16,
    /// Indexed draws, one per mesh group and map drawn.
    pub draws: u32,
}

/// One frame's shadow work, from [`ShadowPass::prepare`].
pub(crate) struct ShadowFrame {
    source: Source,
    candidates: u32,
    // Skinned slots selected by their palette bounds (GPU-presented frames).
    skinned: u32,
    // Near cascades drawn and looked up.
    cascades: u32,
    refresh: Option<Refresh>,
    pose_index: usize,
    read_dirty: bool,
}

impl ShadowFrame {
    /// Kept-map redraw of this frame, for profiling counters: 0 none, 1
    /// tiles, 2 the whole map.
    pub(crate) fn refresh_kind(&self) -> u8 {
        match self.refresh {
            None => 0,
            Some(Refresh::Tiles(_)) => 1,
            Some(Refresh::Whole) => 2,
        }
    }
}

struct Maps {
    resolution: u32,
    cascades: u32,
    // The near cascades as layers of one texture, and the moving and kept
    // far maps.
    textures: [wgpu::Texture; 3],
    // The near cascades as one array (lookups) and each layer (drawing).
    near_array: wgpu::TextureView,
    near_layers: Vec<wgpu::TextureView>,
    moving: wgpu::TextureView,
    kept: wgpu::TextureView,
}

impl Drop for Maps {
    fn drop(&mut self) {
        // Browsers otherwise keep the memory until garbage collection;
        // submitted work that still reads them completes first.
        for texture in &self.textures {
            texture.destroy();
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadbackState {
    Idle,
    Recorded,
    Pending,
}

struct DirtyReadback {
    buffer: wgpu::Buffer,
    state: Rc<Cell<ReadbackState>>,
}

// Dirty tiles, then the three caster totals and a pad word, read back
// together.
const READBACK_BYTES: u64 = (TILE_WORDS as u64 + 4) * 4;

#[derive(Clone, Copy, PartialEq, Eq)]
struct SelectKey {
    source: Source,
    buffers: u64,
    presented: u64,
    poses: u64,
    pose_index: usize,
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(MIN_STORAGE_BYTES).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Group keys: a mesh, and the masked material its casters test.
type GroupIndices = HashMap<(MeshHandle, Option<MaterialHandle>), u32>;

/// Adds `count` casters of `mesh` (masked by `material`) to their group,
/// creating it at the end.
fn add_to_group(
    groups: &mut Vec<ShadowGroup>,
    counts: &mut Vec<u32>,
    indices: &mut GroupIndices,
    (mesh, material): (MeshHandle, Option<MaterialHandle>),
    source_index: usize,
    count: u32,
) -> u32 {
    let index = match indices.get(&(mesh, material)) {
        Some(&index) => index,
        None => {
            groups.push(ShadowGroup {
                mesh,
                source_index,
                material,
                virtual_geometry: false,
                start: 0,
                capacity: 0,
            });
            counts.push(0);
            let index = groups.len() as u32 - 1;
            indices.insert((mesh, material), index);
            index
        }
    };
    counts[index as usize] += count;
    index
}

/// Shadow pipelines and their fixed bindings, created with the first frame
/// that has shadows: games without them never compile them.
struct ShadowPipelines {
    select: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    select_layout: wgpu::BindGroupLayout,
    // Skinned casters of GPU-presented frames (decision 036).
    select_skinned: wgpu::ComputePipeline,
    skinned_layout: wgpu::BindGroupLayout,
    affine: wgpu::RenderPipeline,
    full: wgpu::RenderPipeline,
    // Alpha-masked casters (decision 031), with a material group.
    affine_masked: wgpu::RenderPipeline,
    full_masked: wgpu::RenderPipeline,
    // Retained casters of meshes with joints (decision 036), plain and
    // masked, which read a skin stream and the joint palettes.
    affine_skinned: wgpu::RenderPipeline,
    affine_masked_skinned: wgpu::RenderPipeline,
    clear: wgpu::RenderPipeline,
    view_buffer: wgpu::Buffer,
    view_group: wgpu::BindGroup,
    pose_layout: wgpu::BindGroupLayout,
    cull: wgpu::Buffer,
}

impl ShadowPipelines {
    /// `vertex_stride` and `instance_stride` are the renderer's mesh vertex
    /// and public instance record sizes.
    /// `layout`: the vertex and instance strides and the texture coordinate
    /// offset in a vertex.
    fn new(
        device: &wgpu::Device,
        (vertex_stride, instance_stride, uv_offset): (u64, u64, u64),
        material_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/shadows.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });
        // Selection and counting share one explicit layout, and so one group.
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let cull_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size_of::<ShadowCullUniform>() as u64),
            },
            count: None,
        };
        let select_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/shadow-select-layout"),
            entries: &[
                cull_entry,
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/shadow-select-pipeline-layout"),
            bind_group_layouts: &[Some(&select_layout)],
            immediate_size: 0,
        });
        let compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&compute_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let select = compute("select_casters");
        let count = compute("count_casters");
        // Skinned casters (decision 036) bind their palette bounds instead of
        // candidates, poses and stillness: within eight storage buffers.
        let skinned_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/shadow-skinned-layout"),
            entries: &[
                cull_entry,
                storage_entry(2, true),
                storage_entry(4, true),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
                storage_entry(10, true),
                storage_entry(11, true),
            ],
        });
        let skinned_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/shadow-skinned-pipeline-layout"),
                bind_group_layouts: &[Some(&skinned_layout)],
                immediate_size: 0,
            });
        let select_skinned = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("select_skinned"),
            layout: Some(&skinned_pipeline_layout),
            module: &module,
            entry_point: Some("select_skinned"),
            compilation_options: Default::default(),
            cache: None,
        });
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/shadow-view-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        // Slot poses, and slot colors for masked casters.
        let storage = |binding, size| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };
        let pose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/shadow-pose-layout"),
            entries: &[storage(0, 48), storage(1, 16), storage(2, 48)],
        });
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let masked_vertex_attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: uv_offset,
                shader_location: 7,
            },
        ];
        let slot_attributes = wgpu::vertex_attr_array![2 => Uint32];
        let skin_layout = crate::browser::skin_layout();
        let model_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4
        ];
        // Columns and the packed color of public instance records.
        let masked_model_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Uint32x2
        ];
        let mesh_layout = wgpu::VertexBufferLayout {
            array_stride: vertex_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attributes,
        };
        let masked_mesh_layout = wgpu::VertexBufferLayout {
            array_stride: vertex_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &masked_vertex_attributes,
        };
        let pipeline = |label: &str,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        entry_point: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        clear: bool,
                        masked: bool| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    buffers,
                },
                // Both faces cast: open and single-sided meshes shadow too.
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(if clear {
                        wgpu::CompareFunction::Always
                    } else {
                        wgpu::CompareFunction::Less
                    }),
                    stencil: Default::default(),
                    // Receiver planes (decision 028) keep lit surfaces off
                    // their own depth; a unit of bias covers the sun shafts'
                    // plain comparisons.
                    bias: if clear {
                        Default::default()
                    } else {
                        wgpu::DepthBiasState {
                            constant: 1,
                            slope_scale: 0.0,
                            clamp: 0.0,
                        }
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                // Only masked casters test texels; the rest write depth alone.
                fragment: masked.then_some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_masked"),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let affine = pipeline(
            "engine/shadow-affine",
            &[Some(&view_layout), Some(&pose_layout)],
            "vs_affine",
            &[
                Some(mesh_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &slot_attributes,
                }),
            ],
            false,
            false,
        );
        let full = pipeline(
            "engine/shadow-full",
            &[Some(&view_layout)],
            "vs_full",
            &[
                Some(mesh_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: instance_stride,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &model_attributes,
                }),
            ],
            false,
            false,
        );
        let affine_masked = pipeline(
            "engine/shadow-affine-masked",
            &[
                Some(&view_layout),
                Some(&pose_layout),
                Some(material_layout),
            ],
            "vs_affine_masked",
            &[
                Some(masked_mesh_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &slot_attributes,
                }),
            ],
            false,
            true,
        );
        let full_masked = pipeline(
            "engine/shadow-full-masked",
            &[Some(&view_layout), None, Some(material_layout)],
            "vs_full_masked",
            &[
                Some(masked_mesh_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: instance_stride,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &masked_model_attributes,
                }),
            ],
            false,
            true,
        );
        let slot_layout = wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &slot_attributes,
        };
        let affine_skinned = pipeline(
            "engine/shadow-affine-skinned",
            &[Some(&view_layout), Some(&pose_layout)],
            "vs_affine_skinned",
            &[
                Some(mesh_layout.clone()),
                Some(slot_layout.clone()),
                skin_layout.clone(),
            ],
            false,
            false,
        );
        let affine_masked_skinned = pipeline(
            "engine/shadow-affine-masked-skinned",
            &[
                Some(&view_layout),
                Some(&pose_layout),
                Some(material_layout),
            ],
            "vs_affine_masked_skinned",
            &[Some(masked_mesh_layout), Some(slot_layout), skin_layout],
            false,
            true,
        );
        let clear = pipeline("engine/shadow-clear", &[], "vs_clear", &[], true, false);
        let view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/shadow-views"),
            size: FAR_VIEW + VIEW_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/shadow-view-group"),
            layout: &view_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &view_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });
        let cull = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/shadow-select-uniform"),
            size: size_of::<ShadowCullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            select,
            count,
            select_layout,
            select_skinned,
            skinned_layout,
            affine,
            full,
            affine_masked,
            full_masked,
            affine_skinned,
            affine_masked_skinned,
            clear,
            view_buffer,
            view_group,
            pose_layout,
            cull,
        }
    }
}

pub(crate) struct ShadowPass {
    pipelines: Option<ShadowPipelines>,
    // Mesh vertex and public instance record strides of the pipelines.
    // Vertex and instance strides and the texture coordinate offset, and the
    // material group layout of masked casters.
    layout: (u64, u64, u64),
    material_layout: wgpu::BindGroupLayout,
    // Pose bindings of both pose buffers, for the generation they belong to.
    pose_groups: Option<(u64, [Option<wgpu::BindGroup>; 2])>,
    uniform: wgpu::Buffer,
    written: Option<ShadowUniform>,
    sampler: wgpu::Sampler,
    // Reads the four texels around a receiver-plane lookup (decision 028).
    texel_sampler: wgpu::Sampler,
    _placeholder: wgpu::Texture,
    placeholder_view: wgpu::TextureView,
    placeholder_array: wgpu::TextureView,
    maps: Option<Maps>,
    // Advances when map views change: bindings of the main passes follow.
    generation: u64,
    // Per slot: pose key, stillness count and kept-map tiles.
    stillness: wgpu::Buffer,
    stillness_capacity: usize,
    ids: wgpu::Buffer,
    id_capacity: usize,
    args: wgpu::Buffer,
    args_template: wgpu::Buffer,
    arg_capacity: usize,
    dirty: wgpu::Buffer,
    // Stands in for presentation flags where frames have none.
    no_flags: wgpu::Buffer,
    readbacks: Vec<DirtyReadback>,
    // Dirty tiles read back and not yet taken.
    read_back: Rc<Cell<[u32; TILE_WORDS]>>,
    // The latest caster totals read back: near, moving and kept maps.
    casters: Rc<Cell<Option<[u32; 3]>>>,
    // Casters of CPU-prepared frames and one group record per mesh.
    listed: wgpu::Buffer,
    listed_groups: wgpu::Buffer,
    // Draw groups whose ID region starts fit after the counters of `dirty`.
    starts_capacity: usize,
    listed_capacity: usize,
    listed_staging: Vec<CandidateData>,
    listed_group_staging: Vec<BatchData>,
    // Selection bindings and the inputs they were built for.
    select_group: Option<(SelectKey, wgpu::BindGroup)>,
    // Skinned selection's bindings (decision 036) and the buffer and
    // presentation generations they were built for.
    skinned_group: Option<((u64, u64), wgpu::BindGroup)>,
    // Advances when this pass replaces a selection buffer.
    buffers_generation: u64,
    groups: Vec<ShadowGroup>,
    // Casters per map: every group's capacity.
    group_total: u32,
    // Group of each plan batch (GPU-presented frames and public snapshots).
    batch_groups: Vec<u32>,
    // Source kind and stamps, and whether virtual geometry draws.
    groups_key: Option<(u8, u64, u64, bool)>,
    groups_changed: bool,
    // The level of detail each map draws each group at (decision 047), as
    // the draw arguments' template holds them.
    template_levels: Vec<u8>,
    // The mesh arena layout the draw arguments' template was written for.
    mesh_layout: u64,
    // Whether the device draws indirectly from nonzero first instances, and
    // whether the template's first instances name the draws' ID regions.
    first_instance: bool,
    regions_in_args: bool,
    source: Source,
    listed_count: u32,
    // The kept map's basis and region, whether it holds the current casters
    // there, and its dirty tiles.
    kept_basis: Option<LightBasis>,
    kept_region: Option<MapRegion>,
    anchor: Option<Vec3>,
    kept_valid: bool,
    dirty_tiles: DirtyTiles,
    // Maps drawn this frame that select virtual casters' clusters.
    virtual_views: Vec<VirtualShadowView>,
}

impl ShadowPass {
    /// The bindings the main passes read; pipelines follow with the first
    /// frame that has shadows. `vertex_stride` and `instance_stride` are the
    /// renderer's mesh vertex and public instance record sizes.
    /// `layout`: the vertex and instance strides and the texture coordinate
    /// offset in a vertex; `material_layout`, the mesh material group.
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: (u64, u64, u64),
        material_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/shadow-uniform"),
            size: size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Comparisons filter four texels.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/shadow-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let texel_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/shadow-texel-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/shadow-placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let placeholder_view = placeholder.create_view(&Default::default());
        let placeholder_array = placeholder.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let empty = |label: &str, extra| storage(device, label, 0, extra);
        Self {
            pipelines: None,
            layout,
            material_layout: material_layout.clone(),
            pose_groups: None,
            uniform,
            written: None,
            sampler,
            texel_sampler,
            _placeholder: placeholder,
            placeholder_view,
            placeholder_array,
            maps: None,
            generation: 0,
            stillness: empty("engine/shadow-stillness", wgpu::BufferUsages::empty()),
            stillness_capacity: 0,
            ids: empty("engine/shadow-ids", wgpu::BufferUsages::VERTEX),
            id_capacity: 0,
            args: empty("engine/shadow-args", wgpu::BufferUsages::INDIRECT),
            args_template: empty("engine/shadow-args-template", wgpu::BufferUsages::COPY_SRC),
            arg_capacity: 0,
            dirty: storage(
                device,
                "engine/shadow-dirty",
                READBACK_BYTES,
                wgpu::BufferUsages::COPY_SRC,
            ),
            no_flags: empty("engine/shadow-no-flags", wgpu::BufferUsages::empty()),
            starts_capacity: 0,
            readbacks: Vec::new(),
            read_back: Rc::new(Cell::new([0; TILE_WORDS])),
            casters: Rc::new(Cell::new(None)),
            listed: empty("engine/shadow-casters", wgpu::BufferUsages::empty()),
            listed_groups: empty("engine/shadow-caster-groups", wgpu::BufferUsages::empty()),
            listed_capacity: 0,
            listed_staging: Vec::new(),
            listed_group_staging: Vec::new(),
            select_group: None,
            skinned_group: None,
            buffers_generation: 0,
            groups: Vec::new(),
            group_total: 0,
            batch_groups: Vec::new(),
            groups_key: None,
            groups_changed: false,
            template_levels: Vec::new(),
            mesh_layout: 0,
            first_instance: device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            regions_in_args: false,
            source: Source::Snapshot,
            listed_count: 0,
            kept_basis: None,
            kept_region: None,
            anchor: None,
            kept_valid: false,
            dirty_tiles: DirtyTiles::default(),
            virtual_views: Vec::new(),
        }
    }

    /// Advances when the map views bound by [`Self::entries`] change.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Bindings 2 to 7 of the main passes' first group: the lookup uniform,
    /// the near cascades, the kept and moving maps, the comparison sampler
    /// and the texel sampler of receiver planes.
    pub(crate) fn entries(&self) -> [wgpu::BindGroupEntry<'_>; 6] {
        let (near, kept, moving) = match &self.maps {
            Some(maps) => (&maps.near_array, &maps.kept, &maps.moving),
            None => (
                &self.placeholder_array,
                &self.placeholder_view,
                &self.placeholder_view,
            ),
        };
        [
            entry(2, &self.uniform),
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(near),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(kept),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(moving),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&self.texel_sampler),
            },
        ]
    }

    /// Meshes this frame's shadows draw; CPU-prepared frames may draw culled
    /// meshes the plan's batches lack.
    pub(crate) fn groups(&self) -> &[ShadowGroup] {
        &self.groups
    }

    /// Group of plan batch `index`, for presentation's batch records.
    pub(crate) fn batch_group(&self, index: usize) -> u32 {
        self.batch_groups.get(index).copied().unwrap_or(0)
    }

    /// Maps drawn this frame whose virtual casters' clusters virtual
    /// geometry selects (decision 044); empty without such casters.
    pub(crate) fn virtual_views(&self) -> &[VirtualShadowView] {
        &self.virtual_views
    }

    /// The selected casters' IDs (each map's region `map * capacity`, then
    /// each group's) and indirect arguments (five words per map and group,
    /// `map * groups + group`), the ID capacity per map, and a key that
    /// changes when the buffers do.
    pub(crate) fn caster_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u32, u64) {
        (
            &self.ids,
            &self.args,
            self.id_capacity as u32,
            self.buffers_generation,
        )
    }

    /// The view matrices of every map, at each map's view offset, once
    /// shadows ran.
    pub(crate) fn view_buffer(&self) -> Option<&wgpu::Buffer> {
        self.pipelines
            .as_ref()
            .map(|pipelines| &pipelines.view_buffer)
    }

    /// The latest caster totals of the near cascades together, the moving
    /// map and the kept map read back from GPU-selected frames, a few frames
    /// old; the kept total counts only frames that redrew part of it.
    pub(crate) fn caster_counts(&self) -> Option<[u32; 3]> {
        self.casters.get()
    }

    /// Bytes of shadow maps and selection buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let maps = self.maps.as_ref().map_or(0, |maps| {
            let far = u64::from(maps.resolution) * u64::from(maps.resolution) * 2;
            2 * far + u64::from(maps.cascades) * far / 4
        });
        maps + [
            &self.stillness,
            &self.ids,
            &self.args,
            &self.args_template,
            &self.listed,
            &self.listed_groups,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum::<u64>()
    }

    /// Plans this frame's draw groups, one per mesh: of the plan's batches
    /// (GPU-presented frames and public snapshots) or of `frame`'s casters
    /// (CPU-prepared frames). Call before mesh uploads, which must include
    /// every group's mesh.
    pub(crate) fn plan_groups(
        &mut self,
        snapshot: &RenderSnapshot,
        plan: &BatchPlan,
        frame: Option<&RenderFrame<'_>>,
        gpu: bool,
    ) {
        if !snapshot.shadows.enabled {
            self.groups.clear();
            self.batch_groups.clear();
            self.groups_key = None;
            return;
        }
        // Retained frames draw virtual geometry (decision 044) where it is
        // enabled.
        let clustered = frame.is_some() && snapshot.virtual_geometry.enabled;
        let (source, key) = match frame {
            Some(frame) if gpu => (
                Source::Presented,
                (0, frame.identity(), frame.source_revision(), clustered),
            ),
            // Both stamps only grow: a source change (such as a slot turning
            // transparent) regroups casters too.
            Some(frame) => (
                Source::Listed,
                (
                    1,
                    frame.identity(),
                    frame.caster_revision().max(frame.source_revision()),
                    clustered,
                ),
            ),
            None => (Source::Snapshot, (2, 0, 0, false)),
        };
        self.source = source;
        // Public snapshots may change their sources between any two frames.
        if self.groups_key == Some(key) && source != Source::Snapshot {
            return;
        }
        self.groups_key = Some(key);
        self.groups_changed = true;
        self.groups.clear();
        self.batch_groups.clear();
        let mut indices = GroupIndices::new();
        let mut counts = Vec::new();
        // Casters of alpha-masked materials test their texels (decision 031).
        let masked: HashSet<MaterialHandle> = snapshot
            .materials
            .iter()
            .filter(|source| matches!(source.material.alpha_mode(), AlphaMode::Mask { .. }))
            .map(|source| source.handle)
            .collect();
        let key = |mesh, material: Option<MaterialHandle>| {
            (mesh, material.filter(|material| masked.contains(material)))
        };
        if let Some(frame) = frame.filter(|_| !gpu) {
            let sources: HashMap<MeshHandle, usize> = snapshot
                .meshes
                .iter()
                .enumerate()
                .map(|(index, source)| (source.handle, index))
                .collect();
            let instances = &frame.snapshot().instances;
            self.listed_staging.clear();
            for &slot in frame.casters() {
                // Transparent surfaces cast no shadow (decision 026).
                if crate::batches::is_transparent(&instances[slot], &plan.transparent_materials) {
                    continue;
                }
                let mesh = instances[slot].mesh;
                let Some(&source_index) = sources.get(&mesh) else {
                    continue;
                };
                let group = add_to_group(
                    &mut self.groups,
                    &mut counts,
                    &mut indices,
                    key(mesh, instances[slot].material),
                    source_index,
                    1,
                );
                self.listed_staging.push(CandidateData {
                    slot: slot as u32,
                    node: 0,
                    batch: group,
                    position: 0,
                });
            }
            self.listed_count = self.listed_staging.len() as u32;
        } else {
            for batch in &plan.batches {
                // Transparent surfaces cast no shadow (decision 026).
                if batch.transparent {
                    self.batch_groups.push(u32::MAX);
                    continue;
                }
                let count = batch.instances.end - batch.instances.start;
                let group = add_to_group(
                    &mut self.groups,
                    &mut counts,
                    &mut indices,
                    key(batch.mesh, batch.material),
                    batch.source_index,
                    count,
                );
                self.batch_groups.push(group);
            }
        }
        let mut start = 0;
        self.listed_group_staging.clear();
        for (group, count) in self.groups.iter_mut().zip(&counts) {
            group.start = start;
            group.capacity = *count;
            start += count;
            let mesh = &snapshot.meshes[group.source_index].mesh;
            let bounds = mesh.bounds();
            // Meshes with joints (decision 036) cast wherever joints move them;
            // virtual geometry (decision 044, flag 2) casts clusters.
            let skinned = if mesh.joint_count() != 0 { 1.0 } else { 0.0 };
            group.virtual_geometry = clustered && mesh.virtual_geometry().is_some();
            self.listed_group_staging.push(BatchData {
                start: group.start,
                count: *count,
                index_count: mesh.indices().len() as u32,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
                first_plane: 0,
                lod_ranges: [0; 4],
                lod_errors: [0.0, 0.0, 1.0, 0.0],
                pad: self.listed_group_staging.len() as u32,
                bounds_min: bounds.min().extend(skinned).to_array(),
                bounds_max: bounds
                    .max()
                    .extend(if group.virtual_geometry { 2.0 } else { 0.0 })
                    .to_array(),
            });
        }
        self.group_total = start;
        if source == Source::Presented {
            // Removed casters leave their kept shadows: redraw every tile.
            self.dirty_tiles = DirtyTiles::ALL;
        }
    }

    /// Prepares this frame's shadow maps and lookups, or disables lookups
    /// when shadows are off. `presented` carries GPU presentation's buffers
    /// on GPU-presented frames; `poses`, the pose storage of retained frames.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &RenderSnapshot,
        presented: Option<Presented<'_>>,
        poses: Option<Poses<'_>>,
        slot_capacity: usize,
        mesh_layout: u64,
        placement: impl Fn(MeshHandle) -> CasterDraw,
    ) -> Option<ShadowFrame> {
        self.dirty_tiles
            .merge(self.read_back.replace([0; TILE_WORDS]));
        self.virtual_views.clear();
        let settings = snapshot.shadows.sanitized();
        if !settings.enabled || self.groups.is_empty() {
            if !settings.enabled && self.maps.take().is_some() {
                self.generation += 1;
            }
            self.kept_valid &= settings.enabled;
            // Shadows off: lookups see state.x 0.
            self.write_uniform(queue, bytemuck::Zeroable::zeroed());
            return None;
        }
        if self.pipelines.is_none() {
            self.pipelines = Some(ShadowPipelines::new(
                device,
                self.layout,
                &self.material_layout,
            ));
        }
        let resolution = settings.resolution;
        if self
            .maps
            .as_ref()
            .is_none_or(|maps| maps.resolution != resolution || maps.cascades != settings.cascades)
        {
            // New maps hold nothing: the kept one is redrawn whole.
            self.maps = Some(create_maps(device, resolution, settings.cascades));
            self.generation += 1;
            self.kept_valid = false;
        }
        let near_resolution = resolution / 2;
        let basis = LightBasis::new(snapshot.atmosphere.sanitized().sun_direction);
        if self.kept_basis != Some(basis) {
            self.kept_basis = Some(basis);
            self.kept_valid = false;
        }
        // The far region: the settings' bounds, or a cube around an anchor
        // that follows the camera a quarter of the distance at a time.
        let eye = snapshot.camera_position;
        let (min, max) = match settings.bounds {
            Some(bounds) => {
                self.anchor = None;
                bounds
            }
            None => {
                let distance = settings.distance;
                let anchor = match self.anchor {
                    Some(anchor) if anchor.distance(eye) <= distance * 0.25 => anchor,
                    _ => {
                        let grid = distance * 0.125;
                        (eye / grid).round() * grid
                    }
                };
                self.anchor = Some(anchor);
                (
                    anchor - Vec3::splat(distance),
                    anchor + Vec3::splat(distance),
                )
            }
        };
        let far = MapRegion::of_box(&basis, min, max, resolution, MARGIN_TEXELS);
        if self.kept_region != Some(far) {
            self.kept_region = Some(far);
            self.kept_valid = false;
        }
        // Near cascades (decision 028): the smallest sphere around the view
        // within each one's reach, nested. A reach inside the camera's near
        // plane covers nothing and has no cascade.
        let mut near = [(0.0, far); MAX_CASCADES];
        let mut cascades = 0;
        for reach in settings.cascade_reaches() {
            if let Some((center, radius)) = view_sphere(snapshot.view_projection, eye, reach) {
                let region = MapRegion::of_sphere(&basis, center, radius, near_resolution);
                near[cascades] = (reach, region);
                cascades += 1;
            }
        }
        let near = &near[..cascades];
        let classify = self.source == Source::Presented;
        if classify && self.stillness_capacity < slot_capacity {
            self.stillness_capacity = slot_capacity.next_power_of_two();
            self.stillness = storage(
                device,
                "engine/shadow-stillness",
                self.stillness_capacity as u64 * 8,
                wgpu::BufferUsages::empty(),
            );
            self.buffers_generation += 1;
            // Stillness starts over: the kept map must be redrawn whole.
            self.kept_valid = false;
        }
        let refresh = if !classify {
            None
        } else if !self.kept_valid {
            self.kept_valid = true;
            self.dirty_tiles = DirtyTiles::default();
            Some(Refresh::Whole)
        } else {
            self.dirty_tiles.take_region().map(Refresh::Tiles)
        };
        // Selection outputs: IDs and indirect arguments of every map.
        let groups = self.groups.len();
        let ids = (self.group_total as usize).max(1);
        if ids > self.id_capacity {
            self.id_capacity = ids.next_power_of_two();
            // Draw arguments name regions of the new capacity.
            self.groups_changed = true;
            self.ids = storage(
                device,
                "engine/shadow-ids",
                (MAPS * self.id_capacity * 4) as u64,
                wgpu::BufferUsages::VERTEX,
            );
            self.buffers_generation += 1;
        }
        if groups > self.arg_capacity {
            self.arg_capacity = groups.next_power_of_two();
            let bytes = (MAPS * self.arg_capacity) as u64 * DRAW_ARGS_BYTES;
            self.args = storage(
                device,
                "engine/shadow-args",
                bytes,
                wgpu::BufferUsages::INDIRECT,
            );
            self.args_template = storage(
                device,
                "engine/shadow-args-template",
                bytes,
                wgpu::BufferUsages::COPY_SRC,
            );
            self.buffers_generation += 1;
            self.groups_changed = true;
        }
        // The draw arguments' template names each group's mesh placement,
        // at the coarsest level of detail whose error stays within half a
        // texel of each map (decision 047): the rasterized depths then
        // differ from the mesh's by less than rounding. The moving map, whose
        // casters are drawn every frame, allows a whole texel: its outlines
        // may then move by one texel, and it draws half the geometry.
        self.groups_changed |= self.mesh_layout != mesh_layout;
        self.mesh_layout = mesh_layout;
        let draws: Vec<CasterDraw> = self
            .groups
            .iter()
            .map(|group| placement(group.mesh))
            .collect();
        let errors: [f32; MAPS] = std::array::from_fn(|map| match map {
            MOVING_MAP => far.texel(resolution),
            KEPT_MAP => far.texel(resolution) * 0.5,
            cascade => near
                .get(cascade)
                .map_or(0.0, |(_, region)| region.texel(near_resolution) * 0.5),
        });
        let levels: Vec<u8> = errors
            .iter()
            .flat_map(|&error| draws.iter().map(move |draw| draw.level_within(error)))
            .collect();
        if levels != self.template_levels {
            // Casters drawn at other levels than the kept map holds redraw
            // it whole, as a moved sun does (decision 047): its levels change
            // only with its texel and when levels of detail turn on or off.
            let kept = |levels: &[u8]| {
                levels
                    .get(KEPT_MAP * groups..(KEPT_MAP + 1) * groups)
                    .map(<[u8]>::to_vec)
            };
            if kept(&levels) != kept(&self.template_levels) {
                self.kept_valid = false;
            }
            self.groups_changed = true;
        }
        if self.groups_changed {
            self.template_levels = levels;
            self.groups_changed = false;
            if groups > self.starts_capacity {
                // Counters first, then every group's region start; a new
                // buffer drops unread counters, which only delays tiles.
                self.starts_capacity = groups.next_power_of_two();
                self.dirty = storage(
                    device,
                    "engine/shadow-dirty",
                    READBACK_BYTES + (self.starts_capacity * 4) as u64,
                    wgpu::BufferUsages::COPY_SRC,
                );
                self.dirty_tiles = DirtyTiles::ALL;
                self.buffers_generation += 1;
            }
            let starts: Vec<u32> = self.groups.iter().map(|group| group.start).collect();
            if !starts.is_empty() {
                queue.write_buffer(&self.dirty, READBACK_BYTES, bytemuck::cast_slice(&starts));
            }
            // Each map's draws name their ID regions as first instances
            // where the device draws indirectly from them.
            self.regions_in_args =
                self.first_instance && u32::try_from(MAPS * self.id_capacity).is_ok();
            let mut template: Vec<u32> = Vec::with_capacity(MAPS * groups * 5);
            for map in 0..MAPS {
                for (index, group) in self.groups.iter().enumerate() {
                    let (count, first_index, base_vertex) =
                        draws[index].range(self.template_levels[map * groups + index]);
                    let first_instance = if self.regions_in_args {
                        (map * self.id_capacity) as u32 + group.start
                    } else {
                        0
                    };
                    template.extend([count, 0, first_index, base_vertex as u32, first_instance]);
                }
            }
            if !template.is_empty() {
                queue.write_buffer(&self.args_template, 0, bytemuck::cast_slice(&template));
            }
            if self.source == Source::Listed {
                let casters = self.listed_staging.len().max(1);
                let group_bytes = (groups.max(1) * size_of::<BatchData>()) as u64;
                if casters > self.listed_capacity || group_bytes > self.listed_groups.size() {
                    self.listed_capacity = casters.next_power_of_two();
                    self.listed = storage(
                        device,
                        "engine/shadow-casters",
                        (self.listed_capacity * size_of::<CandidateData>()) as u64,
                        wgpu::BufferUsages::empty(),
                    );
                    self.listed_groups = storage(
                        device,
                        "engine/shadow-caster-groups",
                        group_bytes.next_power_of_two(),
                        wgpu::BufferUsages::empty(),
                    );
                    self.buffers_generation += 1;
                }
                if !self.listed_staging.is_empty() {
                    queue.write_buffer(&self.listed, 0, bytemuck::cast_slice(&self.listed_staging));
                }
                queue.write_buffer(
                    &self.listed_groups,
                    0,
                    bytemuck::cast_slice(&self.listed_group_staging),
                );
            }
        }
        // View matrices: each near cascade's, then both far maps'.
        let far_clip = far.clip(&basis);
        let mut near_clips = [far_clip; MAX_CASCADES];
        for (clip, (_, region)) in near_clips.iter_mut().zip(near) {
            *clip = region.clip(&basis);
        }
        let mut views = [0.0_f32; (MAX_CASCADES + 1) * VIEW_STRIDE as usize / 4];
        for (index, clip) in near_clips.iter().chain([&far_clip]).enumerate() {
            let start = index * VIEW_STRIDE as usize / 4;
            views[start..start + 16].copy_from_slice(&clip.to_cols_array());
        }
        queue.write_buffer(
            &self.pipelines.as_ref().expect("created above").view_buffer,
            0,
            bytemuck::cast_slice(&views),
        );
        let candidates = match (&presented, self.source) {
            (Some(presented), Source::Presented) => presented.count,
            (_, Source::Listed) => self.listed_count,
            _ => 0,
        };
        let skinned = match (&presented, self.source) {
            (Some(presented), Source::Presented) => presented.skinned_count,
            _ => 0,
        };
        if self.source != Source::Snapshot {
            let refresh_rect = match refresh {
                Some(Refresh::Whole) => far.rect,
                // A texel of margin keeps casters at the tiles' edges.
                Some(Refresh::Tiles(region)) => region.rect(far.rect, far.texel(resolution)),
                None => [0.0; 4],
            };
            let row = |axis: Vec3| axis.extend(0.0).to_array();
            // Casters matter only above receivers a map shadows: within the
            // view frustum up to how far the map reaches, widened by the
            // lookups' normal offset and texel gathers. Maps keep their
            // regions; selection skips casters beyond that part of them.
            let selection = |depth: f32, region: &MapRegion, texel: f32| {
                let margin = texel * (NORMAL_OFFSET_TEXELS + 2.0);
                match frustum_footprint(&basis, snapshot.view_projection, eye, depth) {
                    Some((low, high)) => (
                        [
                            math::max(region.rect[0], low.x - margin),
                            math::max(region.rect[1], low.y - margin),
                            math::min(region.rect[2], high.x + margin),
                            math::min(region.rect[3], high.y + margin),
                        ],
                        math::min(region.depth[1], high.z + margin),
                    ),
                    None => (region.rect, region.depth[1]),
                }
            };
            let mut near_rects = [[0.0; 4]; MAX_CASCADES];
            let mut near_depths = [0.0; 4];
            for (index, (reach, region)) in near.iter().enumerate() {
                (near_rects[index], near_depths[index]) =
                    selection(*reach, region, region.texel(near_resolution));
            }
            // The far maps shadow receivers up to the shadow distance, and
            // sun shafts (decision 024) sample them along view rays.
            let shafts = snapshot.sun_shafts.sanitized();
            let far_reach = if shafts.enabled() {
                math::max(settings.distance, shafts.distance)
            } else {
                settings.distance
            };
            let (moving_rect, moving_depth) = selection(far_reach, &far, far.texel(resolution));
            // Maps drawn every frame select virtual casters' clusters within
            // the same regions (decision 044). A cascade's receivers start
            // where the previous one hands over to it, across its last
            // tenth; the far maps' at the last cascade's reach.
            if self.groups.iter().any(|group| group.virtual_geometry) {
                // The selection region, tightened by the view slice swept
                // toward the sun.
                let planes = |rect: [f32; 4], depth: f32, reach: f32| {
                    let mut planes = [[0.0, 0.0, 0.0, 1.0]; VIRTUAL_SHADOW_PLANES];
                    let region = region_planes(&basis, rect, depth);
                    let swept = caster_planes(snapshot.view_projection, eye, reach, -basis.rows[2]);
                    for (slot, plane) in planes.iter_mut().zip(
                        region
                            .into_iter()
                            .chain(swept.iter().map(|plane| plane.to_array())),
                    ) {
                        *slot = plane;
                    }
                    planes
                };
                for (index, (reach, region)) in near.iter().enumerate() {
                    self.virtual_views.push(VirtualShadowView {
                        map: index,
                        planes: planes(near_rects[index], near_depths[index], *reach),
                        min_distance: index.checked_sub(1).map(|previous| near[previous].0 * 0.9),
                        texel: region.texel(near_resolution),
                        direction: basis.rows[2],
                        view_offset: (index as u64 * VIEW_STRIDE) as u32,
                    });
                }
                self.virtual_views.push(VirtualShadowView {
                    map: MOVING_MAP,
                    planes: planes(moving_rect, moving_depth, far_reach),
                    min_distance: near.last().map(|(reach, _)| *reach),
                    texel: far.texel(resolution),
                    direction: basis.rows[2],
                    view_offset: FAR_VIEW as u32,
                });
            }
            let cull = ShadowCullUniform {
                light: [row(basis.rows[0]), row(basis.rows[1]), row(basis.rows[2])],
                near: near_rects,
                far: far.rect,
                refresh: refresh_rect,
                moving: moving_rect,
                near_depths,
                depths: [0.0, far.depth[1], far.depth[1], moving_depth],
                candidates,
                groups: groups as u32,
                capacity: self.id_capacity as u32,
                mode: [(classify, STILLNESS), (refresh.is_some(), REFRESH)]
                    .iter()
                    .filter(|(set, _)| *set)
                    .fold(0, |mode, (_, bit)| mode | bit),
                tiles: [
                    far.rect[0],
                    far.rect[1],
                    (far.rect[2] - far.rect[0]) / TILES as f32,
                    (far.rect[3] - far.rect[1]) / TILES as f32,
                ],
                cascades: near.len() as u32,
                skinned,
                pad: [0; 2],
            };
            queue.write_buffer(
                &self.pipelines.as_ref().expect("created above").cull,
                0,
                bytemuck::bytes_of(&cull),
            );
        }
        // Cascades beyond those in use are never reached.
        let mut reach = [f32::MAX; 4];
        let mut near_offsets = [0.0; 4];
        let mut near_limits = [0.0; 4];
        for (index, (distance, region)) in near.iter().enumerate() {
            reach[index] = *distance;
            near_offsets[index] = region.texel(near_resolution) * NORMAL_OFFSET_TEXELS;
            near_limits[index] = slope_limit(region, near_resolution);
        }
        let far_texel = far.texel(resolution);
        self.write_uniform(
            queue,
            ShadowUniform {
                near: near_clips.map(|clip| clip.to_cols_array_2d()),
                far: far_clip.to_cols_array_2d(),
                state: [
                    1.0,
                    if classify { 1.0 } else { 0.0 },
                    near.len() as f32,
                    settings.distance * FADE_START,
                ],
                reach,
                near_offsets,
                near_limits,
                offsets: [
                    settings.distance,
                    far_texel * NORMAL_OFFSET_TEXELS,
                    slope_limit(&far, resolution),
                    0.0,
                ],
            },
        );
        // Bindings: poses of this frame, and the selection inputs.
        let pose_index = poses.as_ref().map_or(0, |poses| poses.index);
        if let Some(poses) = &poses {
            if self
                .pose_groups
                .as_ref()
                .is_none_or(|(generation, groups)| {
                    *generation != poses.generation || groups[poses.index].is_none()
                })
            {
                let group = |buffer: Option<&wgpu::Buffer>, palette: &wgpu::Buffer| {
                    buffer.map(|buffer| {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("engine/shadow-pose-group"),
                            layout: &self.pipelines.as_ref().expect("created above").pose_layout,
                            entries: &[entry(0, buffer), entry(1, poses.colors), entry(2, palette)],
                        })
                    })
                };
                self.pose_groups = Some((
                    poses.generation,
                    [
                        group(poses.buffers[0], poses.palettes[0]),
                        group(poses.buffers[1], poses.palettes[1]),
                    ],
                ));
            }
            if self.source != Source::Snapshot {
                let key = SelectKey {
                    source: self.source,
                    buffers: self.buffers_generation,
                    presented: presented
                        .as_ref()
                        .map_or(0, |presented| presented.generation),
                    poses: poses.generation,
                    pose_index,
                };
                if self
                    .select_group
                    .as_ref()
                    .is_none_or(|(built, _)| *built != key)
                {
                    let pose_buffer = poses.buffers[pose_index].expect("the frame's pose buffer");
                    let (candidate_buffer, batch_buffer, flag_buffer) = match &presented {
                        Some(presented) if self.source == Source::Presented => {
                            (presented.candidates, presented.batches, presented.flags)
                        }
                        _ => (&self.listed, &self.listed_groups, &self.no_flags),
                    };
                    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("engine/shadow-select-group"),
                        layout: &self
                            .pipelines
                            .as_ref()
                            .expect("created above")
                            .select_layout,
                        entries: &[
                            entry(0, &self.pipelines.as_ref().expect("created above").cull),
                            entry(1, candidate_buffer),
                            entry(2, batch_buffer),
                            entry(3, pose_buffer),
                            entry(4, flag_buffer),
                            entry(5, &self.stillness),
                            entry(6, &self.ids),
                            entry(7, &self.args),
                            entry(8, &self.dirty),
                        ],
                    });
                    self.select_group = Some((key, group));
                }
                if let Some(presented) = presented.as_ref().filter(|_| skinned != 0) {
                    let key = (self.buffers_generation, presented.generation);
                    if self
                        .skinned_group
                        .as_ref()
                        .is_none_or(|(built, _)| *built != key)
                    {
                        let pipelines = self.pipelines.as_ref().expect("created above");
                        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("engine/shadow-skinned-group"),
                            layout: &pipelines.skinned_layout,
                            entries: &[
                                entry(0, &pipelines.cull),
                                entry(2, presented.batches),
                                entry(4, presented.flags),
                                entry(6, &self.ids),
                                entry(7, &self.args),
                                entry(8, &self.dirty),
                                entry(10, presented.skinned),
                                entry(11, presented.skin_bounds),
                            ],
                        });
                        self.skinned_group = Some((key, group));
                    }
                }
            }
        }
        Some(ShadowFrame {
            source: self.source,
            candidates,
            skinned,
            cascades: near.len() as u32,
            refresh,
            pose_index,
            read_dirty: self.source != Source::Snapshot,
        })
    }

    fn write_uniform(&mut self, queue: &wgpu::Queue, uniform: ShadowUniform) {
        if self.written != Some(uniform) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            self.written = Some(uniform);
        }
    }

    /// Records caster selection of `frame`. With `timestamps` (a query set
    /// and its first index), the selection pass writes queries `first` and
    /// `first + 1`. Returns the profiler effects it timed.
    pub(crate) fn encode_selection(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ShadowFrame,
        timestamps: Option<(&wgpu::QuerySet, u32)>,
    ) -> u16 {
        if self.maps.is_none() || self.pipelines.is_none() {
            return 0;
        }
        let mut timed = 0;
        let groups = self.groups.len() as u64;
        if frame.source != Source::Snapshot {
            // Instance counts start at zero every frame.
            encoder.copy_buffer_to_buffer(
                &self.args_template,
                0,
                &self.args,
                0,
                MAPS as u64 * groups * DRAW_ARGS_BYTES,
            );
            if let Some((_, group)) = &self.select_group
                && frame.candidates != 0
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("engine/shadow-select"),
                    timestamp_writes: timestamps.map(|(query_set, first)| {
                        wgpu::ComputePassTimestampWrites {
                            query_set,
                            beginning_of_pass_write_index: Some(first),
                            end_of_pass_write_index: Some(first + 1),
                        }
                    }),
                });
                let pipelines = self.pipelines.as_ref().expect("checked above");
                pass.set_pipeline(&pipelines.select);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(frame.candidates.div_ceil(GROUP), 1, 1);
                // Skinned casters by their palette bounds (decision 036).
                if frame.skinned != 0
                    && let Some((_, skinned)) = &self.skinned_group
                {
                    pass.set_pipeline(&pipelines.select_skinned);
                    pass.set_bind_group(0, skinned, &[]);
                    pass.dispatch_workgroups(frame.skinned.div_ceil(SKINNED_GROUP), 1, 1);
                    pass.set_bind_group(0, group, &[]);
                }
                pass.set_pipeline(&pipelines.count);
                pass.dispatch_workgroups(1, 1, 1);
                timed |= u16::from(timestamps.is_some());
            }
        }
        if frame.read_dirty {
            let dirty = self.dirty.clone();
            if let Some(readback) = self.idle_readback(device) {
                encoder.copy_buffer_to_buffer(&dirty, 0, &readback.buffer, 0, READBACK_BYTES);
                readback.state.set(ReadbackState::Recorded);
                // Unread tiles stay until a readback copies them.
                encoder.clear_buffer(&dirty, 0, Some(TILE_WORDS as u64 * 4));
            }
        }
        timed
    }

    /// Records the depth passes of `frame`, after [`Self::encode_selection`]
    /// and virtual geometry's selection. `meshes` finds each group's
    /// uploaded vertex and index buffers and index count; public snapshots
    /// draw `plan`'s batches from `instances`; `virtual_draw` draws the
    /// clusters of a map's virtual casters (decision 044) last in its pass,
    /// returning its draws. With `timestamps` (a query set and its first
    /// index), the depth passes write their effects' queries.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_maps<'m>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &ShadowFrame,
        plan: &BatchPlan,
        instances: &'m wgpu::Buffer,
        meshes: impl Fn(MeshHandle) -> Option<(MeshDraw<'m>, bool)>,
        materials: impl Fn(MaterialHandle) -> Option<&'m wgpu::BindGroup>,
        virtual_draw: impl Fn(&mut wgpu::RenderPass<'_>, usize) -> u32,
        timestamps: Option<(&wgpu::QuerySet, u32)>,
    ) -> ShadowEncoded {
        if self.maps.is_none() || self.pipelines.is_none() {
            return ShadowEncoded::default();
        }
        let mut timed = u16::from(timestamps.is_some()) << 1;
        let draws = Cell::new(0_u32);
        let groups = self.groups.len() as u64;
        let maps = self.maps.as_ref().expect("checked above");
        let pipelines = self.pipelines.as_ref().expect("checked above");
        let pose_group = self
            .pose_groups
            .as_ref()
            .and_then(|(_, groups)| groups[frame.pose_index].as_ref());
        // Masked groups switch to the masked pipeline and their material.
        let bind_material = |pass: &mut wgpu::RenderPass<'_>,
                             masked: &mut bool,
                             material: Option<MaterialHandle>,
                             pipelines: [&wgpu::RenderPipeline; 2]| {
            let group = material.map(&materials);
            if group.is_some() != *masked {
                *masked = group.is_some();
                pass.set_pipeline(pipelines[usize::from(*masked)]);
            }
            if let Some(Some(group)) = group {
                pass.set_bind_group(2, group, &[]);
            }
            // A masked material without a binding draws nothing.
            group.is_none_or(|group| group.is_some())
        };
        // Plain, masked, skinned and masked skinned retained casters.
        let selected = [
            &pipelines.affine,
            &pipelines.affine_masked,
            &pipelines.affine_skinned,
            &pipelines.affine_masked_skinned,
        ];
        // Retained casters in pipeline, vertex buffer and material order,
        // which passes switch once per run: depth alone is written, so the
        // order changes no texel. Virtual geometry draws its casters'
        // clusters (decision 044), and masked materials without bindings
        // draw nothing.
        let mut casters: Vec<Caster<'m>> = if pose_group.is_some() {
            self.groups
                .iter()
                .enumerate()
                .filter(|(_, group)| !group.virtual_geometry)
                .filter_map(|(index, group)| {
                    let (draw, skinned) = meshes(group.mesh)?;
                    let material = match group.material.map(&materials) {
                        Some(None) => return None,
                        Some(Some(material)) => Some(material),
                        None => None,
                    };
                    let skin = draw.skin.filter(|_| skinned);
                    Some(Caster {
                        choice: usize::from(material.is_some()) + 2 * usize::from(skin.is_some()),
                        draw,
                        skin,
                        material,
                        index,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        casters.sort_unstable_by_key(|caster| {
            (
                caster.choice,
                std::ptr::from_ref(caster.draw.vertices) as usize,
                caster
                    .material
                    .map_or(0, |group| std::ptr::from_ref(group) as usize),
                caster.index,
            )
        });
        let draw_selected = |pass: &mut wgpu::RenderPass<'_>, map: usize| {
            let Some(pose_group) = pose_group else {
                return;
            };
            pass.set_pipeline(&pipelines.affine);
            pass.set_bind_group(1, pose_group, &[]);
            let mut bound = 0;
            let mut bindings = Bindings::default();
            for caster in &casters {
                if caster.choice != bound {
                    pass.set_pipeline(selected[caster.choice]);
                    bound = caster.choice;
                }
                if let Some(material) = caster.material {
                    bindings.group(pass, 2, material);
                }
                if let Some(skin) = caster.skin {
                    bindings.vertex(pass, 2, skin);
                }
                bindings.vertex(pass, 0, caster.draw.vertices);
                bindings.index(pass, caster.draw.indices);
                let ids = if self.regions_in_args {
                    Ids::Whole(&self.ids)
                } else {
                    let start = map * self.id_capacity + self.groups[caster.index].start as usize;
                    Ids::Region(self.ids.slice(start as u64 * 4..))
                };
                bindings.ids(pass, 1, ids);
                pass.draw_indexed_indirect(
                    &self.args,
                    (map as u64 * groups + caster.index as u64) * DRAW_ARGS_BYTES,
                );
                draws.set(draws.get() + 1);
            }
            draws.set(draws.get() + virtual_draw(pass, map));
        };
        let draw_snapshot = |pass: &mut wgpu::RenderPass<'_>| {
            pass.set_pipeline(&pipelines.full);
            pass.set_vertex_buffer(1, instances.slice(..));
            let mut masked = false;
            let mut bindings = Bindings::default();
            for (index, batch) in plan.batches.iter().enumerate() {
                // Transparent surfaces cast no shadow (decision 026).
                let Some(group) = self
                    .batch_groups
                    .get(index)
                    .and_then(|&group| self.groups.get(group as usize))
                else {
                    continue;
                };
                let Some((draw, _)) = meshes(batch.mesh) else {
                    continue;
                };
                if !bind_material(
                    pass,
                    &mut masked,
                    group.material,
                    [&pipelines.full, &pipelines.full_masked],
                ) {
                    continue;
                }
                bindings.vertex(pass, 0, draw.vertices);
                bindings.index(pass, draw.indices);
                pass.draw_indexed(
                    draw.index_range(),
                    draw.base_vertex,
                    batch.instances.clone(),
                );
                draws.set(draws.get() + 1);
            }
        };
        // Profiled frames time the near cascades together, the moving map and
        // the kept map (effects 11, 12 and 13 of the profiler).
        let last_cascade = (frame.cascades as usize).saturating_sub(1);
        let writes = |map: usize| {
            timestamps.and_then(|(query_set, first)| {
                let (effect, begin, end) = match map {
                    MOVING_MAP => (12, true, true),
                    KEPT_MAP => (13, true, true),
                    cascade => (11, cascade == 0, cascade == last_cascade),
                };
                let base = first + 2 * effect;
                (begin || end).then_some(wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: begin.then_some(base),
                    end_of_pass_write_index: end.then_some(base + 1),
                })
            })
        };
        if timestamps.is_some() {
            timed |= u16::from(frame.cascades > 0) << 11
                | 1 << 12
                | u16::from(frame.refresh.is_some()) << 13;
        }
        // Each near cascade in use (map and view matrix `cascade`), then the
        // moving map.
        let cascades = (0..frame.cascades as usize).map(|cascade| {
            (
                cascade,
                &maps.near_layers[cascade],
                cascade as u64 * VIEW_STRIDE,
            )
        });
        for (map, view, offset) in cascades.chain([(MOVING_MAP, &maps.moving, FAR_VIEW)]) {
            let mut pass = map_pass(encoder, view, wgpu::LoadOp::Clear(1.0), writes(map));
            pass.set_bind_group(0, &pipelines.view_group, &[offset as u32]);
            if frame.source == Source::Snapshot {
                draw_snapshot(&mut pass);
            } else {
                draw_selected(&mut pass, map);
            }
        }
        if let Some(refresh) = frame.refresh {
            let whole = refresh == Refresh::Whole;
            let mut pass = map_pass(
                encoder,
                &maps.kept,
                if whole {
                    wgpu::LoadOp::Clear(1.0)
                } else {
                    wgpu::LoadOp::Load
                },
                writes(KEPT_MAP),
            );
            if let Refresh::Tiles(region) = refresh {
                let [x, y, width, height] = region.scissor(maps.resolution);
                pass.set_scissor_rect(x, y, width, height);
                pass.set_pipeline(&pipelines.clear);
                pass.draw(0..3, 0..1);
            }
            pass.set_bind_group(0, &pipelines.view_group, &[FAR_VIEW as u32]);
            draw_selected(&mut pass, KEPT_MAP);
        }
        ShadowEncoded {
            timed,
            draws: draws.get(),
        }
    }

    fn idle_readback(&mut self, device: &wgpu::Device) -> Option<&DirtyReadback> {
        if let Some(index) = self
            .readbacks
            .iter()
            .position(|readback| readback.state.get() == ReadbackState::Idle)
        {
            return Some(&self.readbacks[index]);
        }
        if self.readbacks.len() == READBACKS {
            return None;
        }
        self.readbacks.push(DirtyReadback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/shadow-dirty-readback"),
                size: READBACK_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Rc::new(Cell::new(ReadbackState::Idle)),
        });
        self.readbacks.last()
    }

    /// Requests maps of dirty tiles recorded in the frame just submitted.
    pub(crate) fn after_submit(&mut self) {
        for readback in &self.readbacks {
            if readback.state.get() != ReadbackState::Recorded {
                continue;
            }
            readback.state.set(ReadbackState::Pending);
            let state = Rc::clone(&readback.state);
            let read_back = Rc::clone(&self.read_back);
            let casters = Rc::clone(&self.casters);
            let buffer = readback.buffer.clone();
            readback
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let word = |index: usize| {
                                let bytes = view.get(index * 4..index * 4 + 4);
                                bytes.map_or(0, |bytes| {
                                    u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
                                })
                            };
                            let mut tiles = read_back.get();
                            for (index, tile) in tiles.iter_mut().enumerate() {
                                *tile |= word(index);
                            }
                            read_back.set(tiles);
                            casters.set(Some([
                                word(TILE_WORDS),
                                word(TILE_WORDS + 1),
                                word(TILE_WORDS + 2),
                            ]));
                        }
                        buffer.unmap();
                    }
                    state.set(ReadbackState::Idle);
                });
        }
    }
}

/// Inward world planes of the light-space rectangle `rect` (minimum x and y,
/// maximum x and y) up to light-space depth `depth` in `basis`: where a
/// caster can darken receivers of a map's selection region.
fn region_planes(basis: &LightBasis, rect: [f32; 4], depth: f32) -> [[f32; 4]; 5] {
    let [x, y, z] = basis.rows;
    [
        x.extend(-rect[0]).to_array(),
        (-x).extend(rect[2]).to_array(),
        y.extend(-rect[1]).to_array(),
        (-y).extend(rect[3]).to_array(),
        (-z).extend(depth).to_array(),
    ]
}

/// Steepest receiver plane of `region` with `texels` texels a side, in its
/// depth (0 to 1 over the region's depths) per texel.
fn slope_limit(region: &MapRegion, texels: u32) -> f32 {
    SLOPE_LIMIT_TEXELS * region.texel(texels) / (region.depth[1] - region.depth[0])
}

fn map_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<f32>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("engine/shadow-map"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Maps of `resolution` texels (the near cascades half as many) with
/// `cascades` near layers.
fn create_maps(device: &wgpu::Device, resolution: u32, cascades: u32) -> Maps {
    let texture = |label, size, layers| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let textures = [
        texture("engine/shadow-near", resolution / 2, cascades),
        texture("engine/shadow-moving", resolution, 1),
        texture("engine/shadow-kept", resolution, 1),
    ];
    let near_array = textures[0].create_view(&wgpu::TextureViewDescriptor {
        label: Some("engine/shadow-near-cascades"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let near_layers = (0..cascades)
        .map(|layer| {
            textures[0].create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/shadow-near-cascade"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    let moving = textures[1].create_view(&Default::default());
    let kept = textures[2].create_view(&Default::default());
    Maps {
        resolution,
        cascades,
        textures,
        near_array,
        near_layers,
        moving,
        kept,
    }
}
