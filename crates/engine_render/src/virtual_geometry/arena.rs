//! The arena of virtual geometry: resident virtual meshes'
//! hierarchies, groups and clusters in one arena with absolute indices,
//! rebased once at upload, and the page pool their pages stream into
//! (`virtual_streaming`). Each page lives in a slot found through the
//! pool's slot table, and selection draws a group only while it is
//! resident. Pages the read-back feedback asks for load from the meshes'
//! words or are fetched from their files by byte range, coarser groups
//! first, and pages unused for a while make room under a budget or the
//! device's limits.

use std::{cell::RefCell, rc::Rc, sync::Arc};

use engine_core::collections::HashMap;

use engine_core::{CLUSTER_WORDS, GROUP_WORDS, MeshHandle, NODE_WORDS, PAGE_WORDS, VirtualMesh};

use super::NONE;
use crate::gpu_layouts::virtual_geometry::MeshRecord;
use crate::virtual_streaming::{MeshLinks, PageState, Placement, Residency};

/// A fetched byte range of a mesh's file: pages `first..first + count`,
/// starting at byte `start`, or the failure; for arena `generation`.
struct Arrival {
    generation: u64,
    mesh: u32,
    first: u32,
    count: u32,
    start: u64,
    bytes: Result<Vec<u8>, String>,
}

struct Resident {
    handle: MeshHandle,
    geometry: Arc<VirtualMesh>,
}

pub(super) struct ArenaBuffers {
    pub(super) meshes: wgpu::Buffer,
    pub(super) nodes: wgpu::Buffer,
    pub(super) groups: wgpu::Buffer,
    pub(super) clusters: wgpu::Buffer,
    pub(super) pages: wgpu::Buffer,
}

pub(super) struct VirtualArena {
    resident: Vec<Resident>,
    index: HashMap<MeshHandle, u32>,
    // Resident meshes changed since the last upload.
    dirty: bool,
    buffers: Option<ArenaBuffers>,
    // Advances whenever arena buffers are replaced.
    generation: u64,
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
    // Page reads that failed or held no valid page since streaming started,
    // and why the latest did.
    failed_pages: u32,
    page_failure: Option<String>,
    // Nodes on the longest root-to-leaf path of any resident mesh.
    depth: u32,
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

/// Fetches bytes `range` of `url` into `inbox` as `arrival`.
fn fetch(
    url: &str,
    range: std::ops::Range<u64>,
    arrival: Arrival,
    inbox: Rc<RefCell<Vec<Arrival>>>,
) {
    crate::platform::read_range(url, range, move |bytes| {
        inbox.borrow_mut().push(Arrival { bytes, ..arrival });
    });
}

impl VirtualArena {
    pub(super) fn new() -> Self {
        Self {
            resident: Vec::new(),
            index: HashMap::default(),
            dirty: false,
            buffers: None,
            generation: 0,
            residency: Residency::default(),
            layout_generation: 0,
            bases: Vec::new(),
            layout: Vec::new(),
            pool_slots: 0,
            table_words: 0,
            bits_base: 0,
            inbox: Rc::new(RefCell::new(Vec::new())),
            frame: 0,
            layout_frame: 0,
            missing_pages: 0,
            dropped_pages: 0,
            failed_pages: 0,
            page_failure: None,
            depth: 0,
        }
    }

    /// The arena's buffers, once uploaded.
    pub(super) fn buffers(&self) -> Option<&ArenaBuffers> {
        self.buffers.as_ref()
    }

    /// Advances whenever arena buffers are replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// Nodes on the longest root-to-leaf path of any resident mesh: the
    /// traversal's levels.
    pub(super) fn depth(&self) -> u32 {
        self.depth
    }

    /// Where the groups' residency bits start in the groups buffer, in
    /// words.
    pub(super) fn bits_base(&self) -> u64 {
        self.bits_base
    }

    /// Groups of every resident mesh: the bits of each feedback set.
    pub(super) fn group_count(&self) -> usize {
        self.residency.group_count()
    }

    /// Frames streamed so far, and the first of the current layout:
    /// feedback of earlier frames names an older one's groups.
    pub(super) fn frames(&self) -> (u64, u64) {
        (self.frame, self.layout_frame)
    }

    /// This frame's resident and loading pages, the pages the last
    /// read-back frame wanted and lacked and those that found no room, and
    /// the page reads that failed since streaming started.
    pub(super) fn pages(&self) -> [u32; 5] {
        [
            self.residency.resident_pages() as u32,
            self.residency.loading_pages() as u32,
            self.missing_pages,
            self.dropped_pages,
            self.failed_pages,
        ]
    }

    /// Why the latest failed page read did, if any failed.
    pub(super) fn page_failure(&self) -> Option<&str> {
        self.page_failure.as_deref()
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, |arena| {
            arena.meshes.size()
                + arena.nodes.size()
                + arena.groups.size()
                + arena.clusters.size()
                + arena.pages.size()
        })
    }

    /// Makes `geometry` resident as mesh `handle`'s, once.
    pub(super) fn insert(&mut self, handle: MeshHandle, geometry: &Arc<VirtualMesh>) {
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
    pub(super) fn retain(&mut self, keep: impl Fn(MeshHandle) -> bool) {
        let before = self.resident.len();
        self.resident.retain(|resident| keep(resident.handle));
        if self.resident.len() != before {
            self.dirty = true;
        }
    }

    /// The arena index of mesh `handle`'s geometry, once uploaded.
    pub(super) fn mesh_index(&self, handle: MeshHandle) -> Option<u32> {
        self.index.get(&handle).copied()
    }

    /// Whether resident mesh `index` is closed (`VirtualMesh::closed`).
    pub(super) fn closed(&self, index: u32) -> bool {
        self.resident
            .get(index as usize)
            .is_some_and(|resident| resident.geometry.closed())
    }

    /// Rebuilds the arena when resident meshes changed: every record
    /// rebased to absolute indices, and a new page pool holding the pinned
    /// pages (and, without `budget`, every page the meshes' words hold).
    /// Returns whether it rebuilt.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: Option<u32>,
    ) -> bool {
        if !self.dirty {
            return false;
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
            .buffers
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
            meshes.push(MeshRecord {
                bases: [node_base, group_base, cluster_base, self.table_words as u32],
                info: [geometry.levels(), geometry.depth(), 0, 0],
                position_min: [position_min[0], position_min[1], position_min[2], 0.0],
                position_step: [position_step[0], position_step[1], position_step[2], 0.0],
                uv: [uv_min[0], uv_min[1], uv_step[0], uv_step[1]],
            });
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
        self.buffers = Some(ArenaBuffers {
            meshes: storage_init(
                device,
                "engine/virtual-meshes",
                bytemuck::cast_slice(&meshes),
            ),
            nodes: storage_init(device, "engine/virtual-nodes", &nodes),
            groups: storage_init(device, "engine/virtual-groups", &groups),
            clusters: storage_init(device, "engine/virtual-clusters", &clusters),
            pages,
        });
        self.generation += 1;
        for page in preload {
            if self.residency.begin(page, self.frame) {
                self.load_held(device, queue, page, usize::MAX);
            }
        }
        self.write_residency(queue);
        true
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
                    let arena = self.buffers.as_ref().expect("an arena");
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
        let Some(arena) = self.buffers.as_mut() else {
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
        self.generation += 1;
    }

    /// Writes the slot table and the groups' residency bits where they
    /// changed.
    fn write_residency(&mut self, queue: &wgpu::Queue) {
        let Some(arena) = &self.buffers else {
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
    pub(super) fn stream(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: Option<u32>,
        feedback: Option<(Vec<u32>, Vec<u32>, u64)>,
    ) {
        self.frame += 1;
        if self.buffers.is_none() {
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
                        bytes: Ok(Vec::new()),
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
                let words = arrival
                    .bytes
                    .as_ref()
                    .map_err(String::clone)
                    .and_then(|bytes| {
                        let range = geometry
                            .page_bytes(local as usize)
                            .ok_or_else(|| format!("page {local} lies outside its file"))?;
                        let bytes = bytes
                            .get(
                                (range.start - arrival.start) as usize
                                    ..(range.end - arrival.start) as usize,
                            )
                            .ok_or_else(|| format!("page {local} was read short"))?;
                        let words: Vec<u32> = bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|chunk| u32::from_le_bytes(*chunk))
                            .collect();
                        geometry
                            .validate_page(local as usize, &words)
                            .map_err(|error| format!("page {local}: {error}"))?;
                        Ok(words)
                    });
                match words {
                    Ok(words) => self.place(device, queue, page, &words, limit),
                    Err(reason) => {
                        // Read again later; the reason stays.
                        self.failed_pages += 1;
                        self.page_failure = Some(reason);
                        self.residency.fail(page, self.frame, 0);
                    }
                }
            }
        }
        self.write_residency(queue);
    }

    /// The arena buffers vertex shaders read (meshes, clusters, pages) and
    /// their generation.
    pub(super) fn vertex_buffers(&self) -> Option<([&wgpu::Buffer; 3], u64)> {
        self.buffers.as_ref().map(|arena| {
            (
                [&arena.meshes, &arena.clusters, &arena.pages],
                self.generation,
            )
        })
    }
}
