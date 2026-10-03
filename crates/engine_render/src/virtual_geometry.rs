//! Virtual geometry (decision 044): resident virtual meshes' hierarchies,
//! clusters and pages in one arena, and each frame's selection of the
//! clusters every view draws (`virtual_geometry.wgsl`). The camera's
//! clusters go to per-material bins the opaque pass draws 128 triangles a
//! cluster from (`meshes.wgsl`'s `vs_virtual`); each shadow map drawn every
//! frame (decision 020) selects its casters' clusters into a bin of its own,
//! drawn into the map by `vs_shadow`.
//!
//! Frames culling occluded instances (decision 040) retest the camera's
//! clusters the last frame's depth pyramid hid against this frame's, after
//! phase one, and draw those it does not hide in phase two from twin bins.
//!
//! Arena records carry absolute indices, rebased once at upload. Pages
//! stream (`virtual_streaming`): each lives in a slot of the page pool,
//! found through the pool's slot table, and selection draws a group only
//! while it is resident. Selection reports the groups it used and those it
//! would refine into; their pages load from the meshes' words or are
//! fetched from their files by byte range, coarser groups first, and pages
//! unused for a while make room under a budget or the device's limits.
//!
//! Selection runs on the GPU only, for all views in one pass; each bin's
//! region, the retest list and the node and group queues grow to what
//! frames read back.

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
    sync::Arc,
};

use engine_core::glam::{Mat4, Vec3};
use engine_core::{CLUSTER_WORDS, GROUP_WORDS, MeshHandle, NODE_WORDS, PAGE_WORDS, VirtualMesh};
use wasm_bindgen::JsCast;

use crate::gpu_data::VIRTUAL_SHADER;
use crate::shadows::{SHADOW_FORMAT, VIRTUAL_SHADOW_PLANES as PLANES};
use crate::virtual_streaming::{MeshLinks, PageState, Placement, Residency};

const NONE: u32 = u32::MAX;
const GROUP: u32 = 64;
/// Counters before the bins' (`virtual_geometry.wgsl`).
const COUNTERS: usize = 12;
/// Views one frame selects for: the camera's and those of the shadow maps
/// drawn every frame (up to four cascades and the moving map).
pub(crate) const MAX_VIEWS: usize = 8;
/// Floats of a view record: its planes, the eye, the error projection and
/// a direction.
const VIEW_FLOATS: usize = PLANES * 4 + 12;
/// Bytes of the uniform before its views: counts, two projections, the
/// viewport, the camera's tests and the feedback's locations.
const FRAME_BYTES: usize = 192;
const UNIFORM_BYTES: u64 = (FRAME_BYTES + MAX_VIEWS * VIEW_FLOATS * 4) as u64;
const READBACKS: usize = 3;
/// Starting capacities, grown from read-back counts.
const START_QUEUE: u64 = 1 << 16;
const START_BIN: u64 = 1 << 14;
const START_RETEST: u64 = 1 << 14;
/// Camera tests (`virtual_geometry.wgsl`).
const TEST_PREVIOUS: u32 = 1;
const TEST_SUBPIXEL: u32 = 2;
/// Bytes of an entry of the selected clusters: visible ID and cluster.
pub(crate) const VISIBLE_BYTES: u64 = 8;
/// ID sources of batches: the camera's, and the shadow maps' casters.
const CAMERA: usize = 0;
const SHADOWS: usize = 1;

/// A batch of instances of a virtual mesh in one view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VirtualBatch {
    /// Start of its IDs: visible IDs (GPU-presented frames) or slot IDs
    /// (CPU-prepared ones) for the camera, a map's casters for shadows.
    pub start: u32,
    /// Instances, when known; otherwise the instance count of draw
    /// arguments `draw` (GPU presentation's, or the shadow maps').
    pub count: Option<u32>,
    pub draw: u32,
    pub mesh: u32,
    pub bin: u32,
    pub double_sided: bool,
    pub view: u32,
}

/// How a view tests clusters' normal cones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Cone {
    /// From the eye (perspective views).
    Eye,
    /// Along a direction the view looks (orthographic views).
    Along(Vec3),
}

/// A view clusters are selected for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualView {
    /// Inward planes, normalized; unused ones hold everything.
    pub planes: [[f32; 4]; PLANES],
    /// The camera's eye, which errors project from.
    pub eye: Vec3,
    /// The smallest distance errors project from.
    pub min_distance: f32,
    /// cot(fovy / 2) / 2 of the camera.
    pub projection: f32,
    /// The largest projected error drawn, as a fraction of the viewport
    /// height.
    pub threshold: f32,
    /// The world error a shadow view accepts where the camera sees nothing
    /// (0 for the camera's own view).
    pub unseen_error: f32,
    pub cone: Cone,
}

/// Counts a frame read back: clusters and triangles the camera and the
/// shadow views selected, the camera's clusters occlusion culling left
/// undrawn, and entries dropped for want of room; and this frame's pages:
/// resident, loading, and loads dropped for want of room.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VirtualStats {
    pub clusters: u32,
    pub triangles: u32,
    pub shadow_clusters: u32,
    pub shadow_triangles: u32,
    pub occluded: u32,
    pub dropped: u32,
    pub resident_pages: u32,
    pub loading_pages: u32,
    pub missing_pages: u32,
    pub dropped_pages: u32,
}

/// The camera's cluster tests of a frame (decisions 040 and 041): the
/// projection it rasterizes, the viewport and depth pyramid levels, the
/// projection the last frame rasterized when its pyramid is this frame's
/// to reuse, the first phase-two bin when occlusion culling draws two
/// phases, and whether clusters covering no pixel center stay undrawn.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraTests {
    pub view_projection: Mat4,
    pub viewport: (u32, u32, u32),
    pub previous: Option<Mat4>,
    pub phase_two: Option<u32>,
    pub subpixel: bool,
}

/// The buffers one source of batches reads: IDs and draw arguments, with a
/// key that changes when either does.
#[derive(Clone, Copy)]
pub(crate) struct IdBuffers<'a> {
    pub ids: &'a wgpu::Buffer,
    pub draws: &'a wgpu::Buffer,
    pub key: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadbackState {
    Idle,
    Recorded,
    Pending,
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Rc<Cell<ReadbackState>>,
    // The frame it recorded, where its used and requested group bits start
    // and their words each.
    frame: u64,
    feedback: (usize, usize),
}

/// What a frame read back: its counts, dropped queue, group and cluster
/// entries and whether its retest list overflowed, and the groups it used
/// and would refine into (bits of global groups).
struct Feedback {
    stats: VirtualStats,
    drops: [u32; 4],
    used: Vec<u32>,
    requested: Vec<u32>,
    frame: u64,
}

/// A fetched byte range of a mesh's file: pages `first..first + count`,
/// starting at byte `start`, or the failure; for arena `generation`.
struct Arrival {
    generation: u64,
    mesh: u32,
    first: u32,
    count: u32,
    start: u64,
    bytes: Result<Vec<u8>, ()>,
}

struct Resident {
    handle: MeshHandle,
    geometry: Arc<VirtualMesh>,
}

struct ArenaBuffers {
    meshes: wgpu::Buffer,
    nodes: wgpu::Buffer,
    groups: wgpu::Buffer,
    clusters: wgpu::Buffer,
    pages: wgpu::Buffer,
}

struct Pipelines {
    seed: wgpu::ComputePipeline,
    level_args: [wgpu::ComputePipeline; 2],
    traverse: [wgpu::ComputePipeline; 2],
    work_args: wgpu::ComputePipeline,
    select: wgpu::ComputePipeline,
    bin_args: wgpu::ComputePipeline,
    retest_args: wgpu::ComputePipeline,
    retest: wgpu::ComputePipeline,
}

struct Groups {
    // Per ID source; shadows' only while shadow views select.
    seed: [Option<wgpu::BindGroup>; 2],
    level_args: [wgpu::BindGroup; 2],
    traverse: [wgpu::BindGroup; 2],
    work_args: wgpu::BindGroup,
    select: wgpu::BindGroup,
    bin_args: wgpu::BindGroup,
    retest_args: wgpu::BindGroup,
    retest: wgpu::BindGroup,
}

/// Depth-only drawing of shadow views' bins and its bindings, one per pose
/// buffer parity: the arena, the frame's poses and the shadow maps' views.
struct ShadowDrawing {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    groups: [Option<(ShadowGroupKey, wgpu::BindGroup)>; 2],
}

/// Generations of the arena, frame buffers, poses, camera IDs, shadow IDs
/// and the depth pyramid bind groups were built for.
type GroupsKey = (u64, u64, u64, u64, Option<u64>, Option<u64>);
/// The arena generation, the poses' key and the shadow views' buffer a
/// shadow drawing group binds.
type ShadowGroupKey = (u64, u64, wgpu::Buffer);

pub(crate) struct VirtualGeometryPass {
    // Built with the first arena (decision 049): scenes without virtual
    // meshes compile none of them.
    module: Option<wgpu::ShaderModule>,
    pipelines: Option<Pipelines>,
    resident: Vec<Resident>,
    index: HashMap<MeshHandle, u32>,
    dirty: bool,
    arena: Option<ArenaBuffers>,
    // Advances whenever arena buffers are replaced.
    arena_generation: u64,
    // Streaming: the resident pages and groups, each mesh's first global
    // group and page, the pool's slots, the words of its slot table (and of
    // the groups buffer before the residency bits), fetched ranges that
    // arrived, the frame count and this frame's dropped loads.
    residency: Residency,
    // Advances with every arena rebuild (not pool growth): fetches of an
    // earlier one are ignored.
    layout_generation: u64,
    bases: Vec<(u32, u32)>,
    // Each arena mesh's handle, in arena order.
    layout: Vec<MeshHandle>,
    pool_slots: usize,
    table_words: u64,
    bits_base: u64,
    inbox: Rc<RefCell<Vec<Arrival>>>,
    frame: u64,
    // The first frame of the current layout: earlier frames' read-back
    // groups are those of an older one.
    layout_frame: u64,
    // Pages the last read-back frame wanted and lacked, and those of them
    // (with arrivals since) that found no room.
    missing_pages: u32,
    dropped_pages: u32,
    // Nodes on the longest root-to-leaf path of any resident mesh.
    depth: u32,
    uniform: wgpu::Buffer,
    // Batches of each ID source, their capacities and counts, and the most
    // instances any of them has.
    batches: [wgpu::Buffer; 2],
    batch_capacity: [u64; 2],
    batch_counts: [u32; 2],
    max_instances: [u32; 2],
    bins: wgpu::Buffer,
    // Each bin's base and capacity, as uploaded.
    bin_regions: Vec<(u64, u64)>,
    bin_capacity: u64,
    queues: [wgpu::Buffer; 2],
    work: wgpu::Buffer,
    queue_capacity: u64,
    visible: wgpu::Buffer,
    visible_capacity: u64,
    counters: wgpu::Buffer,
    // Counters and bins; the used and requested group bits follow, this
    // many words each.
    counter_capacity: u64,
    feedback_words: u64,
    dispatch: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    // Clusters waiting for this frame's depth pyramid, and whether this
    // frame retests them.
    retest: wgpu::Buffer,
    retest_capacity: u64,
    retesting: bool,
    // Stands in for the depth pyramid on frames without one.
    _no_pyramid: wgpu::Texture,
    no_pyramid: wgpu::TextureView,
    // Advances whenever frame buffers are replaced.
    frame_generation: u64,
    // Bindings per pose buffer parity (temporal frames alternate them), and
    // this frame's parity.
    groups: [Option<(GroupsKey, Groups)>; 2],
    parity: usize,
    // This frame's poses and their key, for shadow drawing.
    poses: Option<(wgpu::Buffer, u64)>,
    shadow_drawing: Option<ShadowDrawing>,
    readbacks: Vec<Readback>,
    // Feedback mapped since the last frame took it.
    read: Rc<Cell<Option<Feedback>>>,
    last: Option<VirtualStats>,
    grow: [bool; 4],
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

fn storage_init(device: &wgpu::Device, label: &str, words: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let mut padded;
    let contents: &[u32] = if words.len() < 4 {
        padded = words.to_vec();
        padded.resize(4, 0);
        &padded
    } else {
        words
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// The page pool: a slot table of `table_words` words, then `slots` slots
/// of PAGE_WORDS words.
fn pool(device: &wgpu::Device, table_words: u64, slots: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/virtual-pages"),
        size: (table_words + slots.max(1) as u64 * PAGE_WORDS as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Fetches bytes `range` of `url` into `inbox` as `arrival` (a server
/// ignoring ranges sends the whole file, which is cut to the range).
fn fetch(
    url: &str,
    range: std::ops::Range<u64>,
    arrival: Arrival,
    inbox: Rc<RefCell<Vec<Arrival>>>,
) {
    let request = (|| {
        let init = web_sys::RequestInit::new();
        init.set_method("GET");
        let headers = web_sys::Headers::new().ok()?;
        headers
            .set("Range", &format!("bytes={}-{}", range.start, range.end - 1))
            .ok()?;
        init.set_headers(&headers);
        web_sys::Request::new_with_str_and_init(url, &init).ok()
    })();
    let promise = request
        .zip(web_sys::window())
        .map(|(request, window)| window.fetch_with_request(&request));
    wasm_bindgen_futures::spawn_local(async move {
        let bytes = async {
            let response: web_sys::Response =
                wasm_bindgen_futures::JsFuture::from(promise.ok_or(())?)
                    .await
                    .map_err(|_| ())?
                    .dyn_into()
                    .map_err(|_| ())?;
            if !response.ok() {
                return Err(());
            }
            let whole = response.status() == 200;
            let buffer =
                wasm_bindgen_futures::JsFuture::from(response.array_buffer().map_err(|_| ())?)
                    .await
                    .map_err(|_| ())?;
            let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
            if whole {
                bytes
                    .get(range.start as usize..range.end as usize)
                    .map(<[u8]>::to_vec)
                    .ok_or(())
            } else {
                Ok(bytes)
            }
        }
        .await;
        inbox.borrow_mut().push(Arrival { bytes, ..arrival });
    });
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

impl VirtualGeometryPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let none = wgpu::BufferUsages::empty();
        let no_pyramid = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/virtual-no-pyramid"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let no_pyramid_view = no_pyramid.create_view(&Default::default());
        Self {
            module: None,
            pipelines: None,
            resident: Vec::new(),
            index: HashMap::new(),
            dirty: false,
            arena: None,
            arena_generation: 0,
            residency: Residency::default(),
            layout_generation: 0,
            bases: Vec::new(),
            layout: Vec::new(),
            pool_slots: 0,
            table_words: 0,
            bits_base: 0,
            inbox: Rc::new(RefCell::new(Vec::new())),
            frame: 0,
            dropped_pages: 0,
            missing_pages: 0,
            layout_frame: 0,
            depth: 0,
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/virtual-frame"),
                size: UNIFORM_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            batches: [
                storage(device, "engine/virtual-camera-batches", 32, none),
                storage(device, "engine/virtual-shadow-batches", 32, none),
            ],
            batch_capacity: [1, 1],
            batch_counts: [0, 0],
            max_instances: [0, 0],
            bins: storage(device, "engine/virtual-bins", 16, none),
            bin_regions: Vec::new(),
            bin_capacity: START_BIN,
            queues: [
                storage(device, "engine/virtual-queue-a", START_QUEUE * 16, none),
                storage(device, "engine/virtual-queue-b", START_QUEUE * 16, none),
            ],
            work: storage(device, "engine/virtual-work", START_QUEUE * 16, none),
            queue_capacity: START_QUEUE,
            visible: storage(
                device,
                "engine/virtual-visible",
                VISIBLE_BYTES,
                wgpu::BufferUsages::VERTEX,
            ),
            visible_capacity: 1,
            counters: storage(
                device,
                "engine/virtual-counters",
                COUNTERS as u64 * 4,
                wgpu::BufferUsages::COPY_SRC,
            ),
            counter_capacity: COUNTERS as u64,
            feedback_words: 0,
            dispatch: storage(
                device,
                "engine/virtual-dispatch",
                24,
                wgpu::BufferUsages::INDIRECT,
            ),
            draw_args: storage(
                device,
                "engine/virtual-draws",
                16,
                wgpu::BufferUsages::INDIRECT,
            ),
            retest: storage(device, "engine/virtual-retest", START_RETEST * 16, none),
            retest_capacity: START_RETEST,
            retesting: false,
            _no_pyramid: no_pyramid,
            no_pyramid: no_pyramid_view,
            frame_generation: 0,
            groups: [None, None],
            parity: 0,
            poses: None,
            shadow_drawing: None,
            readbacks: Vec::new(),
            read: Rc::new(Cell::new(None)),
            last: None,
            grow: [false; 4],
        }
    }

    /// Makes `geometry` resident as mesh `handle`'s, once.
    pub(crate) fn insert(&mut self, handle: MeshHandle, geometry: &Arc<VirtualMesh>) {
        if !self.index.contains_key(&handle)
            && !self
                .resident
                .iter()
                .any(|resident| resident.handle == handle)
        {
            self.resident.push(Resident {
                handle,
                geometry: Arc::clone(geometry),
            });
            self.dirty = true;
        }
    }

    /// Releases every resident mesh `keep` rejects.
    pub(crate) fn retain(&mut self, keep: impl Fn(MeshHandle) -> bool) {
        let before = self.resident.len();
        self.resident.retain(|resident| keep(resident.handle));
        if self.resident.len() != before {
            self.dirty = true;
        }
    }

    /// The arena index of mesh `handle`'s geometry, once uploaded.
    pub(crate) fn mesh_index(&self, handle: MeshHandle) -> Option<u32> {
        self.index.get(&handle).copied()
    }

    /// Whether resident mesh `index` is closed (`VirtualMesh::closed`).
    pub(crate) fn closed(&self, index: u32) -> bool {
        self.resident
            .get(index as usize)
            .is_some_and(|resident| resident.geometry.closed())
    }

    /// Rebuilds the arena when resident meshes changed: every record
    /// rebased to absolute indices, and a new page pool holding the pinned
    /// pages (and, without `budget`, every page the meshes' words hold).
    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: Option<u32>,
    ) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        // The resident pages of the layout being replaced, by mesh and page
        // in it: meshes that stay keep them in the same slots.
        let kept: Vec<(MeshHandle, u32, u32)> = self
            .residency
            .resident_slots()
            .into_iter()
            .filter_map(|(page, slot)| {
                let (mesh, local, _) = self.residency.page(page);
                let handle = *self.layout.get(mesh as usize)?;
                Some((handle, local, slot))
            })
            .collect();
        let previous = self
            .arena
            .take()
            .map(|arena| (arena.pages, self.table_words, self.pool_slots));
        self.index.clear();
        self.depth = 0;
        let total_pages: usize = self
            .resident
            .iter()
            .map(|resident| resident.geometry.page_count())
            .sum();
        self.table_words = (total_pages as u64).next_multiple_of(4).max(4);
        let (mut meshes, mut nodes, mut groups, mut clusters) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut links = Vec::new();
        self.bases.clear();
        let mut page_base = 0u32;
        for (mesh_index, resident) in self.resident.iter().enumerate() {
            let geometry = &resident.geometry;
            self.index.insert(resident.handle, mesh_index as u32);
            let node_base = (nodes.len() / NODE_WORDS) as u32;
            let group_base = (groups.len() / GROUP_WORDS) as u32;
            let cluster_base = (clusters.len() / CLUSTER_WORDS) as u32;
            self.bases.push((group_base, page_base));
            links.push(MeshLinks::of(geometry));
            let (position_min, position_step, uv_min, uv_step) = geometry.quantization();
            // The last base is where the pool's slots start.
            meshes.extend([node_base, group_base, cluster_base, self.table_words as u32]);
            meshes.extend([geometry.levels(), geometry.depth(), 0, 0]);
            meshes.extend(position_min.map(f32::to_bits));
            meshes.push(0);
            meshes.extend(position_step.map(f32::to_bits));
            meshes.push(0);
            meshes.extend([uv_min[0], uv_min[1], uv_step[0], uv_step[1]].map(f32::to_bits));
            for words in geometry.node_section().as_chunks::<NODE_WORDS>().0 {
                let mut node = *words;
                if node[5] != NONE {
                    node[5] += group_base;
                } else {
                    node[6] += node_base;
                }
                nodes.extend(node);
            }
            for words in geometry.group_section().as_chunks::<GROUP_WORDS>().0 {
                let mut group = *words;
                group[5] += cluster_base;
                groups.extend(group);
            }
            // Offsets within the page, and the global page.
            for words in geometry.cluster_section().as_chunks::<CLUSTER_WORDS>().0 {
                let refined = if words[5] == NONE {
                    NONE
                } else {
                    words[5] + group_base
                };
                clusters.extend([
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                    words[4],
                    refined,
                    words[6] + group_base,
                    mesh_index as u32,
                    words[8],
                    words[9],
                    words[10],
                    page_base + words[7],
                ]);
            }
            page_base += geometry.page_count() as u32;
            self.depth = self.depth.max(geometry.depth());
        }
        // The groups' residency bits follow their records.
        self.bits_base = groups.len() as u64;
        groups.resize(groups.len() + (groups.len() / GROUP_WORDS).div_ceil(32), 0);
        self.residency.rebuild(&links, &self.bases);
        self.layout = self
            .resident
            .iter()
            .map(|resident| resident.handle)
            .collect();
        // The kept pages of meshes that stay, as pages of the new layout.
        let kept: Vec<(u32, u32)> = kept
            .into_iter()
            .filter_map(|(handle, local, slot)| {
                let mesh = *self.index.get(&handle)?;
                Some((self.bases[mesh as usize].1 + local, slot))
            })
            .collect();
        // Slots for the pinned pages (every held one without a budget) and
        // the kept pages where they are, as the device allows.
        let preload = if budget.is_some() {
            self.residency.pinned()
        } else {
            self.residency.held()
        };
        let kept_slots = kept
            .iter()
            .map(|&(_, slot)| slot as usize + 1)
            .max()
            .unwrap_or(0);
        self.pool_slots = preload
            .len()
            .max(kept_slots)
            .min(self.device_slots(device))
            .max(1);
        self.residency.add_slots(self.pool_slots);
        let pages = pool(device, self.table_words, self.pool_slots);
        // The kept pages' words move to the new pool's slots, submitted at
        // once so later writes land after them.
        if let Some((old, old_table_words, old_slots)) = previous {
            let slots = old_slots.min(self.pool_slots) as u64;
            if slots > 0 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/virtual-pool-rebuild"),
                });
                encoder.copy_buffer_to_buffer(
                    &old,
                    old_table_words * 4,
                    &pages,
                    self.table_words * 4,
                    slots * PAGE_WORDS as u64 * 4,
                );
                queue.submit([encoder.finish()]);
            }
        }
        let restored: Vec<(u32, u32)> = kept
            .into_iter()
            .filter(|&(_, slot)| (slot as usize) < self.pool_slots)
            .collect();
        self.residency.restore(&restored, self.frame);
        self.layout_generation += 1;
        self.layout_frame = self.frame + 1;
        (self.missing_pages, self.dropped_pages) = (0, 0);
        self.arena = Some(ArenaBuffers {
            meshes: storage_init(device, "engine/virtual-meshes", &meshes),
            nodes: storage_init(device, "engine/virtual-nodes", &nodes),
            groups: storage_init(device, "engine/virtual-groups", &groups),
            clusters: storage_init(device, "engine/virtual-clusters", &clusters),
            pages,
        });
        self.arena_generation += 1;
        self.groups = [None, None];
        for page in preload {
            if self.residency.begin(page, self.frame) {
                self.load_held(device, queue, page, usize::MAX);
            }
        }
        self.write_residency(queue);
    }

    /// The most pages the pool can hold on this device.
    fn device_slots(&self, device: &wgpu::Device) -> usize {
        let limits = device.limits();
        let bytes = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        ((bytes / 4).saturating_sub(self.table_words) / PAGE_WORDS as u64) as usize
    }

    /// Loads page `page` (loading) from its mesh's words, or fails it.
    fn load_held(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, page: u32, limit: usize) {
        let (mesh, local, _) = self.residency.page(page);
        let geometry = Arc::clone(&self.resident[mesh as usize].geometry);
        match geometry.page(local as usize) {
            Some(words) => self.place(device, queue, page, words, limit),
            None => self.residency.fail(page, self.frame, 0),
        }
    }

    /// Places loading page `page`'s `words` in a slot (growing the pool
    /// when allowed) and uploads them, or drops the load.
    fn place(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        page: u32,
        words: &[u32],
        limit: usize,
    ) {
        let limit = limit.min(self.device_slots(device));
        loop {
            match self.residency.place(page, self.frame, limit) {
                Placement::Slot(slot) => {
                    let arena = self.arena.as_ref().expect("an arena");
                    queue.write_buffer(
                        &arena.pages,
                        (self.table_words + slot as u64 * PAGE_WORDS as u64) * 4,
                        bytemuck::cast_slice(words),
                    );
                    return;
                }
                Placement::Grow => {
                    let slots = (self.pool_slots * 2).clamp(self.pool_slots + 1, limit.max(1));
                    if slots <= self.pool_slots {
                        self.residency.fail(page, self.frame, 0);
                        self.dropped_pages += 1;
                        return;
                    }
                    self.grow_pool(device, queue, slots);
                }
                Placement::Full => {
                    self.dropped_pages += 1;
                    return;
                }
            }
        }
    }

    /// Replaces the pool with one of `slots` slots holding the same pages.
    /// The copy is submitted at once, so later writes land after it.
    fn grow_pool(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, slots: usize) {
        let Some(arena) = self.arena.as_mut() else {
            return;
        };
        let grown = pool(device, self.table_words, slots);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/virtual-pool-growth"),
        });
        encoder.copy_buffer_to_buffer(&arena.pages, 0, &grown, 0, arena.pages.size());
        queue.submit([encoder.finish()]);
        arena.pages = grown;
        self.residency.add_slots(slots - self.pool_slots);
        self.pool_slots = slots;
        self.arena_generation += 1;
    }

    /// Writes the slot table and the groups' residency bits where they
    /// changed.
    fn write_residency(&mut self, queue: &wgpu::Queue) {
        let Some(arena) = &self.arena else {
            return;
        };
        if let Some(table) = self.residency.take_table() {
            queue.write_buffer(&arena.pages, 0, bytemuck::cast_slice(&table));
        }
        if let Some(bits) = self.residency.take_bits() {
            queue.write_buffer(
                &arena.groups,
                self.bits_base * 4,
                bytemuck::cast_slice(&bits),
            );
        }
    }

    /// Streams pages for this frame: loads the groups the last read-back
    /// frame requested (from the meshes' words, or fetched by byte range),
    /// places fetched pages that arrived, and keeps at most `budget` pages
    /// resident (the pinned ones always).
    fn stream(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: Option<u32>,
        feedback: Option<(Vec<u32>, Vec<u32>, u64)>,
    ) {
        self.frame += 1;
        if self.arena.is_none() {
            return;
        }
        let limit = budget.map_or(usize::MAX, |budget| {
            (budget as usize).max(self.residency.pinned().len())
        });
        if budget.is_some() {
            self.residency.shrink(limit);
        }
        if let Some((used, requested, frame)) = feedback {
            let requests = self.residency.feedback(
                &used,
                &requested,
                frame,
                limit.min(self.device_slots(device)),
            );
            self.missing_pages = requests.missing as u32;
            self.dropped_pages = requests.dropped as u32;
            // Fetched ones coalesce into byte ranges of consecutive pages.
            let mut ranges: Vec<(u32, u32, u32)> = Vec::new();
            for page in requests.begun {
                let (mesh, local, held) = self.residency.page(page);
                if held {
                    self.load_held(device, queue, page, limit);
                    continue;
                }
                match ranges.last_mut() {
                    Some((last_mesh, first, count))
                        if *last_mesh == mesh && *first + *count == local =>
                    {
                        *count += 1;
                    }
                    _ => ranges.push((mesh, local, 1)),
                }
            }
            for (mesh, first, count) in ranges {
                let geometry = &self.resident[mesh as usize].geometry;
                let (Some(url), Some(start), Some(end)) = (
                    geometry.source(),
                    geometry.page_bytes(first as usize),
                    geometry.page_bytes((first + count - 1) as usize),
                ) else {
                    for local in first..first + count {
                        let page = self.bases[mesh as usize].1 + local;
                        self.residency.fail(page, self.frame, 0);
                    }
                    continue;
                };
                fetch(
                    url,
                    start.start..end.end,
                    Arrival {
                        generation: self.layout_generation,
                        mesh,
                        first,
                        count,
                        start: start.start,
                        bytes: Err(()),
                    },
                    Rc::clone(&self.inbox),
                );
            }
        }
        let arrivals = std::mem::take(&mut *self.inbox.borrow_mut());
        for arrival in arrivals {
            if arrival.generation != self.layout_generation {
                continue;
            }
            let geometry = Arc::clone(&self.resident[arrival.mesh as usize].geometry);
            let page_base = self.bases[arrival.mesh as usize].1;
            for local in arrival.first..arrival.first + arrival.count {
                let page = page_base + local;
                if self.residency.state(page) != PageState::Loading {
                    continue;
                }
                let words = arrival.bytes.as_ref().ok().and_then(|bytes| {
                    let range = geometry.page_bytes(local as usize)?;
                    let bytes = bytes.get(
                        (range.start - arrival.start) as usize
                            ..(range.end - arrival.start) as usize,
                    )?;
                    let words: Vec<u32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| u32::from_le_bytes(*chunk))
                        .collect();
                    geometry
                        .validate_page(local as usize, &words)
                        .is_ok()
                        .then_some(words)
                });
                match words {
                    Some(words) => self.place(device, queue, page, &words, limit),
                    None => self.residency.fail(page, self.frame, 0),
                }
            }
        }
        self.write_residency(queue);
    }

    /// The arena buffers vertex shaders read (meshes, clusters, pages) and
    /// their generation.
    pub(crate) fn arena(&self) -> Option<([&wgpu::Buffer; 3], u64)> {
        self.arena.as_ref().map(|arena| {
            (
                [&arena.meshes, &arena.clusters, &arena.pages],
                self.arena_generation,
            )
        })
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let arena = self.arena.as_ref().map_or(0, |arena| {
            arena.meshes.size()
                + arena.nodes.size()
                + arena.groups.size()
                + arena.clusters.size()
                + arena.pages.size()
        });
        arena
            + self.queues[0].size()
            + self.queues[1].size()
            + self.work.size()
            + self.visible.size()
            + self.counters.size()
            + self.retest.size()
    }

    /// Streams pages (keeping at most `budget` resident) and uploads this
    /// frame's batches of both ID sources, bins, views and the camera's
    /// tests, growing the queues, bin regions and retest list to what frames
    /// read back needed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batches: [&[VirtualBatch]; 2],
        bins: usize,
        views: &[VirtualView],
        tests: &CameraTests,
        budget: Option<u32>,
    ) {
        let mut requests = None;
        if let Some(feedback) = self.read.take() {
            let [queue_drops, work_drops, cluster_drops, overflow] = feedback.drops;
            self.last = Some(feedback.stats);
            self.grow = [
                queue_drops > 0,
                work_drops > 0,
                cluster_drops > 0,
                overflow > 0,
            ];
            if feedback.frame >= self.layout_frame {
                requests = Some((feedback.used, feedback.requested, feedback.frame));
            }
        }
        self.stream(device, queue, budget, requests);
        let binding = device.limits().max_storage_buffer_binding_size;
        if self.grow[3] && self.retest_capacity * 32 <= binding {
            self.retest_capacity *= 2;
            self.retest = storage(
                device,
                "engine/virtual-retest",
                self.retest_capacity * 16,
                wgpu::BufferUsages::empty(),
            );
            self.frame_generation += 1;
        }
        if (self.grow[0] || self.grow[1]) && self.queue_capacity * 32 <= binding {
            self.queue_capacity *= 2;
            let none = wgpu::BufferUsages::empty();
            self.queues = [
                storage(
                    device,
                    "engine/virtual-queue-a",
                    self.queue_capacity * 16,
                    none,
                ),
                storage(
                    device,
                    "engine/virtual-queue-b",
                    self.queue_capacity * 16,
                    none,
                ),
            ];
            self.work = storage(
                device,
                "engine/virtual-work",
                self.queue_capacity * 16,
                none,
            );
            self.frame_generation += 1;
        }
        if self.grow[2] && self.bin_capacity * VISIBLE_BYTES * 2 * bins as u64 <= binding {
            self.bin_capacity *= 2;
        }
        self.grow = [false; 4];
        // Bins' regions: equal shares of the visible storage.
        let wanted = self.bin_capacity * bins.max(1) as u64;
        if wanted > self.visible_capacity {
            self.visible_capacity = wanted;
            self.visible = storage(
                device,
                "engine/virtual-visible",
                wanted * VISIBLE_BYTES,
                wgpu::BufferUsages::VERTEX,
            );
            self.frame_generation += 1;
        }
        self.bin_regions = (0..bins)
            .map(|bin| (bin as u64 * self.bin_capacity, self.bin_capacity))
            .collect();
        let bin_words: Vec<u32> = self
            .bin_regions
            .iter()
            .flat_map(|&(base, capacity)| [base as u32, capacity as u32, 0, 0])
            .collect();
        let feedback_words = (self.residency.group_count() as u64).div_ceil(32);
        if bins as u64 > self.counter_capacity - COUNTERS as u64
            || bin_words.len() * 4 > self.bins.size() as usize
            || feedback_words != self.feedback_words
        {
            let capacity = (bins.max(1) as u64).next_power_of_two();
            self.counter_capacity = COUNTERS as u64 + capacity;
            self.feedback_words = feedback_words;
            self.counters = storage(
                device,
                "engine/virtual-counters",
                (self.counter_capacity + 2 * feedback_words) * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            self.bins = storage(
                device,
                "engine/virtual-bins",
                capacity * 16,
                wgpu::BufferUsages::empty(),
            );
            self.draw_args = storage(
                device,
                "engine/virtual-draws",
                capacity * 16,
                wgpu::BufferUsages::INDIRECT,
            );
            self.readbacks.clear();
            self.frame_generation += 1;
        }
        if !bin_words.is_empty() {
            queue.write_buffer(&self.bins, 0, bytemuck::cast_slice(&bin_words));
        }
        for (source, batches) in batches.iter().enumerate() {
            if batches.len() as u64 > self.batch_capacity[source] {
                self.batch_capacity[source] = (batches.len() as u64).next_power_of_two();
                self.batches[source] = storage(
                    device,
                    if source == CAMERA {
                        "engine/virtual-camera-batches"
                    } else {
                        "engine/virtual-shadow-batches"
                    },
                    self.batch_capacity[source] * 32,
                    wgpu::BufferUsages::empty(),
                );
                self.frame_generation += 1;
            }
            let words: Vec<u32> = batches
                .iter()
                .flat_map(|batch| {
                    [
                        batch.start,
                        batch.count.unwrap_or(NONE),
                        batch.draw,
                        batch.mesh,
                        batch.bin,
                        u32::from(batch.double_sided),
                        batch.view,
                        0,
                    ]
                })
                .collect();
            if !words.is_empty() {
                queue.write_buffer(&self.batches[source], 0, bytemuck::cast_slice(&words));
            }
            self.batch_counts[source] = batches.len() as u32;
        }
        let counts = [
            self.queue_capacity as u32,
            self.queue_capacity as u32,
            bins as u32,
            views.len() as u32,
        ];
        let mut bytes: Vec<u8> = bytemuck::cast_slice(&counts).to_vec();
        self.retesting = tests.phase_two.is_some();
        let (width, height, levels) = tests.viewport;
        let flags = match (tests.previous, tests.phase_two) {
            (Some(_), Some(_)) => TEST_PREVIOUS,
            _ => 0,
        } | if tests.subpixel { TEST_SUBPIXEL } else { 0 };
        bytes.extend_from_slice(bytemuck::cast_slice(&tests.view_projection.to_cols_array()));
        bytes.extend_from_slice(bytemuck::cast_slice(
            &tests
                .previous
                .unwrap_or(tests.view_projection)
                .to_cols_array(),
        ));
        bytes.extend_from_slice(bytemuck::cast_slice(&[
            width as f32,
            height as f32,
            levels as f32,
            0.0,
        ]));
        bytes.extend_from_slice(bytemuck::cast_slice(&[
            flags,
            tests.phase_two.unwrap_or(0),
            self.retest_capacity as u32,
            0,
        ]));
        // Where selection marks used and requested groups, and where the
        // groups' residency bits start.
        bytes.extend_from_slice(bytemuck::cast_slice(&[
            self.counter_capacity as u32,
            (self.counter_capacity + self.feedback_words) as u32,
            self.bits_base as u32,
            0,
        ]));
        for view in views.iter().take(MAX_VIEWS) {
            let mut floats = Vec::with_capacity(VIEW_FLOATS);
            for plane in view.planes {
                floats.extend(plane);
            }
            floats.extend([view.eye.x, view.eye.y, view.eye.z, view.min_distance]);
            let (mode, direction) = match view.cone {
                Cone::Eye => (0.0, Vec3::ZERO),
                Cone::Along(direction) => (1.0, direction),
            };
            floats.extend([view.projection, view.threshold, mode, view.unseen_error]);
            floats.extend([direction.x, direction.y, direction.z, 0.0]);
            bytes.extend_from_slice(bytemuck::cast_slice(&floats));
        }
        queue.write_buffer(&self.uniform, 0, &bytes);
    }

    /// The selection's module and pipelines, built the first time an arena
    /// binds.
    fn ensure_pipelines(&mut self, device: &wgpu::Device) {
        if self.pipelines.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/virtual_geometry.wgsl"),
            source: wgpu::ShaderSource::Wgsl(VIRTUAL_SHADER.into()),
        });
        let pipeline = |entry: &str| {
            crate::pipelines::compute(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        self.pipelines = Some(Pipelines {
            seed: pipeline("seed"),
            level_args: [pipeline("level_args_even"), pipeline("level_args_odd")],
            traverse: [pipeline("traverse_even"), pipeline("traverse_odd")],
            work_args: pipeline("work_args"),
            select: pipeline("select_clusters"),
            bin_args: pipeline("bin_args"),
            retest_args: pipeline("retest_args"),
            retest: pipeline("retest_clusters"),
        });
        self.module = Some(module);
    }

    /// Binds this frame's slot poses (their key's lowest bit tells which of
    /// the two pose buffers they are), and the IDs and draw arguments of the
    /// camera's batches and of the shadow maps' casters, each with a key
    /// that changes when the buffers do. `max_instances` is the most
    /// instances a batch of each source has; `shadow_views`, the shadow
    /// maps' view matrices shadow bins are drawn with; `pyramid`, occlusion
    /// culling's depth pyramid and its generation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn bind(
        &mut self,
        device: &wgpu::Device,
        poses: (&wgpu::Buffer, u64),
        camera: IdBuffers<'_>,
        shadows: Option<IdBuffers<'_>>,
        max_instances: [u32; 2],
        shadow_views: Option<&wgpu::Buffer>,
        pyramid: Option<(&wgpu::TextureView, u64)>,
    ) {
        self.max_instances = max_instances;
        self.poses = Some((poses.0.clone(), poses.1));
        self.parity = (poses.1 & 1) as usize;
        if self.arena.is_none() {
            return;
        }
        self.ensure_pipelines(device);
        if let Some(views) = shadow_views {
            self.bind_shadow_drawing(device, views);
        }
        let key = (
            self.arena_generation,
            self.frame_generation,
            poses.1,
            camera.key,
            shadows.map(|shadows| shadows.key),
            pyramid.map(|(_, generation)| generation),
        );
        let pyramid_view = pyramid.map_or(&self.no_pyramid, |(view, _)| view);
        let texture = |binding| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(pyramid_view),
        };
        if self.groups[self.parity]
            .as_ref()
            .is_some_and(|(built, _)| *built == key)
        {
            return;
        }
        let arena = self.arena.as_ref().expect("checked above");
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry<'_>]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/virtual-group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })
        };
        let pipelines = self.pipelines.as_ref().expect("built above");
        let [queue_a, queue_b] = &self.queues;
        let level = |pipeline| {
            group(
                pipeline,
                &[
                    entry(0, &self.uniform),
                    entry(12, &self.counters),
                    entry(15, &self.dispatch),
                ],
            )
        };
        let traverse = |pipeline| {
            group(
                pipeline,
                &[
                    entry(0, &self.uniform),
                    entry(3, &arena.nodes),
                    entry(6, poses.0),
                    entry(9, queue_a),
                    entry(10, queue_b),
                    entry(11, &self.work),
                    entry(12, &self.counters),
                ],
            )
        };
        let seed = |source: usize, ids: &IdBuffers<'_>| {
            group(
                &pipelines.seed,
                &[
                    entry(0, &self.uniform),
                    entry(1, &self.batches[source]),
                    entry(2, &arena.meshes),
                    entry(7, ids.ids),
                    entry(8, ids.draws),
                    entry(9, queue_a),
                    entry(10, queue_b),
                    entry(12, &self.counters),
                ],
            )
        };
        let groups = Groups {
            seed: [
                Some(seed(CAMERA, &camera)),
                shadows.as_ref().map(|shadows| seed(SHADOWS, shadows)),
            ],
            level_args: [
                level(&pipelines.level_args[0]),
                level(&pipelines.level_args[1]),
            ],
            traverse: [
                traverse(&pipelines.traverse[0]),
                traverse(&pipelines.traverse[1]),
            ],
            work_args: level(&pipelines.work_args),
            select: group(
                &pipelines.select,
                &[
                    entry(0, &self.uniform),
                    entry(4, &arena.groups),
                    entry(5, &arena.clusters),
                    entry(6, poses.0),
                    entry(11, &self.work),
                    entry(12, &self.counters),
                    entry(13, &self.visible),
                    entry(14, &self.bins),
                    texture(19),
                    entry(20, &self.retest),
                ],
            ),
            bin_args: group(
                &pipelines.bin_args,
                &[
                    entry(0, &self.uniform),
                    entry(12, &self.counters),
                    entry(14, &self.bins),
                    entry(16, &self.draw_args),
                ],
            ),
            retest_args: level(&pipelines.retest_args),
            retest: group(
                &pipelines.retest,
                &[
                    entry(0, &self.uniform),
                    entry(5, &arena.clusters),
                    entry(6, poses.0),
                    entry(12, &self.counters),
                    entry(13, &self.visible),
                    entry(14, &self.bins),
                    texture(19),
                    entry(20, &self.retest),
                ],
            ),
        };
        self.groups[self.parity] = Some((key, groups));
    }

    /// The shadow drawing pipeline, once, and its bindings of the arena,
    /// this frame's poses and the shadow maps' views `views`.
    fn bind_shadow_drawing(&mut self, device: &wgpu::Device, views: &wgpu::Buffer) {
        let (Some(arena), Some((poses, poses_key))) = (&self.arena, &self.poses) else {
            return;
        };
        let module = self.module.as_ref().expect("built with the arena");
        let drawing = self.shadow_drawing.get_or_insert_with(|| {
            let storage = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("engine/virtual-shadow-layout"),
                entries: &[
                    storage(2),
                    storage(5),
                    storage(6),
                    storage(17),
                    wgpu::BindGroupLayoutEntry {
                        binding: 18,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(64),
                        },
                        count: None,
                    },
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/virtual-shadow-pipeline-layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let attributes = wgpu::vertex_attr_array![0 => Uint32, 1 => Uint32];
            let pipeline = crate::pipelines::render(
                device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some("engine/virtual-shadow"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vs_shadow"),
                        compilation_options: Default::default(),
                        buffers: &[Some(wgpu::VertexBufferLayout {
                            array_stride: VISIBLE_BYTES,
                            step_mode: wgpu::VertexStepMode::Instance,
                            attributes: &attributes,
                        })],
                    },
                    // Both faces cast, as the shadow maps' other casters.
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: SHADOW_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: Default::default(),
                        // The shadow maps' constant unit of bias (decision 028).
                        bias: wgpu::DepthBiasState {
                            constant: 1,
                            slope_scale: 0.0,
                            clamp: 0.0,
                        },
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: None,
                    multiview_mask: None,
                    cache: None,
                },
            );
            ShadowDrawing {
                layout,
                pipeline,
                groups: [None, None],
            }
        });
        let key = (self.arena_generation, *poses_key, views.clone());
        let parity = (*poses_key & 1) as usize;
        if drawing.groups[parity]
            .as_ref()
            .is_some_and(|(built, _)| *built == key)
        {
            return;
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/virtual-shadow-group"),
            layout: &drawing.layout,
            entries: &[
                entry(2, &arena.meshes),
                entry(5, &arena.clusters),
                entry(6, poses),
                entry(17, &arena.pages),
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: views,
                        offset: 0,
                        size: wgpu::BufferSize::new(64),
                    }),
                },
            ],
        });
        drawing.groups[parity] = Some((key, group));
    }

    /// Records this frame's selection: seeds of both ID sources, one
    /// traversal per hierarchy level, the groups' clusters and each bin's
    /// draw arguments, then the counters' copy for read-back. Nothing
    /// without bound batches.
    pub(crate) fn encode(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let Some((_, groups)) = &self.groups[self.parity] else {
            return;
        };
        if self.batch_counts == [0, 0] {
            return;
        }
        let pipelines = self.pipelines.as_ref().expect("bound groups build them");
        encoder.clear_buffer(&self.counters, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/virtual-selection"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.seed);
            for source in [CAMERA, SHADOWS] {
                if self.batch_counts[source] == 0 {
                    continue;
                }
                if let Some(group) = &groups.seed[source] {
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(
                        self.max_instances[source].div_ceil(GROUP).max(1),
                        self.batch_counts[source],
                        1,
                    );
                }
            }
            for level in 0..self.depth as usize {
                let parity = level % 2;
                pass.set_pipeline(&pipelines.level_args[parity]);
                pass.set_bind_group(0, &groups.level_args[parity], &[]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&pipelines.traverse[parity]);
                pass.set_bind_group(0, &groups.traverse[parity], &[]);
                pass.dispatch_workgroups_indirect(&self.dispatch, 0);
            }
            pass.set_pipeline(&pipelines.work_args);
            pass.set_bind_group(0, &groups.work_args, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&pipelines.select);
            pass.set_bind_group(0, &groups.select, &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch, 12);
            pass.set_pipeline(&pipelines.bin_args);
            pass.set_bind_group(0, &groups.bin_args, &[]);
            pass.dispatch_workgroups((self.bin_regions.len() as u32).div_ceil(GROUP).max(1), 1, 1);
        }
        // Frames retesting read their counters back after the retest.
        if !self.retesting {
            self.read_counters(device, encoder);
        }
    }

    /// Records a copy of this frame's counters for read-back, when a
    /// read-back buffer is free.
    fn read_counters(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let frame = self.frame;
        let feedback = (self.counter_capacity as usize, self.feedback_words as usize);
        if let Some(index) = self.idle_readback(device) {
            let readback = &mut self.readbacks[index];
            encoder.copy_buffer_to_buffer(
                &self.counters,
                0,
                &readback.buffer,
                0,
                readback.buffer.size(),
            );
            readback.state.set(ReadbackState::Recorded);
            readback.frame = frame;
            readback.feedback = feedback;
        }
    }

    /// Records the retest after phase one and this frame's depth pyramid
    /// (decision 040): the waiting clusters it does not hide go to phase
    /// two's bins, whose draw arguments follow. Nothing on frames drawing
    /// one phase.
    pub(crate) fn encode_retest(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let Some((_, groups)) = &self.groups[self.parity] else {
            return;
        };
        if !self.retesting || self.batch_counts == [0, 0] {
            return;
        }
        let pipelines = self.pipelines.as_ref().expect("bound groups build them");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/virtual-retest"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.retest_args);
            pass.set_bind_group(0, &groups.retest_args, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&pipelines.retest);
            pass.set_bind_group(0, &groups.retest, &[]);
            pass.dispatch_workgroups_indirect(&self.dispatch, 12);
            pass.set_pipeline(&pipelines.bin_args);
            pass.set_bind_group(0, &groups.bin_args, &[]);
            pass.dispatch_workgroups((self.bin_regions.len() as u32).div_ceil(GROUP).max(1), 1, 1);
        }
        self.read_counters(device, encoder);
    }

    /// Every bin's selected clusters (visible ID and cluster), and a
    /// generation that advances when the buffer is replaced.
    pub(crate) fn visible(&self) -> (&wgpu::Buffer, u64) {
        (&self.visible, self.frame_generation)
    }

    /// Bin `bin`'s first entry of [`Self::visible`].
    pub(crate) fn bin_base(&self, bin: usize) -> u32 {
        self.bin_regions
            .get(bin)
            .map_or(0, |&(base, _)| base as u32)
    }

    /// The selected clusters of bin `bin` (an instance-stepped vertex stream
    /// of visible ID and cluster) and its indirect draw arguments.
    pub(crate) fn bin_source(&self, bin: usize) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        let (base, capacity) = self.bin_regions[bin];
        (
            self.visible
                .slice(base * VISIBLE_BYTES..(base + capacity) * VISIBLE_BYTES),
            &self.draw_args,
            bin as u64 * 16,
        )
    }

    /// Draws shadow bin `bin` into the map of view matrix `view_offset`
    /// (an offset into the shadow maps' views). Returns whether it drew;
    /// the pass's pipeline and first group are this drawing's afterwards.
    pub(crate) fn draw_shadow(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        bin: usize,
        view_offset: u32,
    ) -> bool {
        let Some(ShadowDrawing {
            pipeline, groups, ..
        }) = &self.shadow_drawing
        else {
            return false;
        };
        let Some((_, group)) = &groups[self.parity] else {
            return false;
        };
        if bin >= self.bin_regions.len() {
            return false;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[view_offset]);
        let (stream, args, offset) = self.bin_source(bin);
        pass.set_vertex_buffer(0, stream);
        pass.draw_indirect(args, offset);
        true
    }

    /// Counts the last read-back frame selected (none before the first
    /// read-back), and this frame's pages.
    pub(crate) fn stats(&self) -> Option<VirtualStats> {
        self.arena.as_ref().map(|_| VirtualStats {
            resident_pages: self.residency.resident_pages() as u32,
            loading_pages: self.residency.loading_pages() as u32,
            missing_pages: self.missing_pages,
            dropped_pages: self.dropped_pages,
            ..self.last.unwrap_or_default()
        })
    }

    fn idle_readback(&mut self, device: &wgpu::Device) -> Option<usize> {
        if let Some(index) = self
            .readbacks
            .iter()
            .position(|readback| readback.state.get() == ReadbackState::Idle)
        {
            return Some(index);
        }
        if self.readbacks.len() == READBACKS {
            return None;
        }
        // Counters, bins and both feedback bitsets.
        self.readbacks.push(Readback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/virtual-counters-readback"),
                size: self.counters.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Rc::new(Cell::new(ReadbackState::Idle)),
            frame: 0,
            feedback: (0, 0),
        });
        Some(self.readbacks.len() - 1)
    }

    /// Requests maps of counters recorded in the frame just submitted.
    pub(crate) fn after_submit(&mut self) {
        for readback in &self.readbacks {
            if readback.state.get() != ReadbackState::Recorded {
                continue;
            }
            readback.state.set(ReadbackState::Pending);
            let state = Rc::clone(&readback.state);
            let read = Rc::clone(&self.read);
            let buffer = readback.buffer.clone();
            // The retest list's capacity during the frame just submitted.
            let capacity = self.retest_capacity as u32;
            let (frame, (feedback, feedback_words)) = (readback.frame, readback.feedback);
            readback
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let word = |index: usize| {
                                view.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
                                    u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
                                })
                            };
                            let drops = [word(3), word(4), word(5)];
                            let bits = |start: usize| -> Vec<u32> {
                                (start..start + feedback_words).map(word).collect()
                            };
                            read.set(Some(Feedback {
                                stats: VirtualStats {
                                    clusters: word(6),
                                    triangles: word(7),
                                    shadow_clusters: word(8),
                                    shadow_triangles: word(9),
                                    occluded: word(11),
                                    dropped: drops.iter().sum(),
                                    ..VirtualStats::default()
                                },
                                drops: [
                                    drops[0],
                                    drops[1],
                                    drops[2],
                                    u32::from(word(10) > capacity),
                                ],
                                used: bits(feedback),
                                requested: bits(feedback + feedback_words),
                                frame,
                            }));
                        }
                        buffer.unmap();
                    }
                    state.set(ReadbackState::Idle);
                });
        }
    }
}

/// Normalized inward planes of `view_projection`'s frustum (WebGPU depth).
pub(crate) fn planes(view_projection: engine_core::glam::Mat4) -> [[f32; 4]; 6] {
    let (planes, _) = crate::gpu_data::frustum_uniform(view_projection);
    planes.map(|plane| {
        let length = Vec3::new(plane[0], plane[1], plane[2]).length();
        if length > 0.0 && length.is_finite() {
            plane.map(|value| value / length)
        } else {
            [0.0, 0.0, 0.0, 1.0]
        }
    })
}
