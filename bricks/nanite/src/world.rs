use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read as _, Seek as _, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use engine::prelude::*;
use engine_math::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use engine_meshlet::decompress_page_words;
use engine_meshlet::{
    BvhNode, ClusterHeader, MeshletManifest, NO_BVH_CHILD, PAGE_SIZE_BYTES, PAGE_WORDS, PageSpan,
};

use crate::types::{
    GpuBvhNode, GpuClusterHeader, MAX_HEAP_PAGES, NaniteInstance, UNMAPPED_PAGE,
};

#[derive(Clone, Debug)]
pub struct MeshEntry {
    pub name: String,
    pub source_tri_count: u32,
    pub bvh_root: u32,
    pub cluster_offset: u32,
    pub cluster_count: u32,
    pub page_offset: u32,
    pub page_count: u32,
    pub root_page_count: u32,
    pub page_spans: Vec<PageSpan>,
    pub bound_sphere: [f32; 4],
    pub shadow_c0_range: [u32; 2],
    pub shadow_c1_range: [u32; 2],
    #[cfg(not(target_arch = "wasm32"))]
    pub mlp_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct PageFetchRequest {
    pub mesh_id: u32,
    pub mesh_name: String,
    pub local_page_id: u32,
    pub global_page_id: u32,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub priority: u32,
}

#[derive(Clone, Debug)]
pub struct PageUpload {
    pub slot: u32,
    pub global_page_id: u32,
    pub words: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct SlotRecord {
    global_page_id: u32,
    pinned: bool,
    last_used_frame: u64,
}

#[derive(Resource, Default)]
pub struct NaniteWorld {
    pub meshes: Vec<MeshEntry>,
    pub clusters: Vec<GpuClusterHeader>,
    pub bvh: Vec<GpuBvhNode>,
    pub instances: Vec<NaniteInstance>,
    pub shadow_cut_clusters: Vec<u32>,
    pub shadow_mesh_ranges: Vec<[u32; 2]>,
    pub total_pages: u32,
    pub total_virtual_tris: u64,
    pub geometry_revision: u64,
    pub instance_revision: u64,
}

fn manifest_bound_sphere(manifest: &MeshletManifest) -> [f32; 4] {
    if let Some(root) = manifest.bvh.first() {
        return root.sphere;
    }
    let min = Vec3::from_array(manifest.bounds_min);
    let max = Vec3::from_array(manifest.bounds_max);
    let center = (min + max) * 0.5;
    let radius = (max - center).length();
    [center.x, center.y, center.z, radius]
}

fn rebase_cluster(
    src: &ClusterHeader,
    cluster_offset: u32,
    page_offset: u32,
) -> GpuClusterHeader {
    let mut page_and_fallback = src.page_and_fallback;
    page_and_fallback[0] = page_and_fallback[0].saturating_add(page_offset);
    let fb_count = src.fallback_count();
    if fb_count > 0 {
        let fb_start = src.fallback_start().saturating_add(cluster_offset);
        page_and_fallback[2] = ClusterHeader::pack_fallback(fb_start, fb_count);
    }
    GpuClusterHeader {
        bound_sphere: src.bound_sphere,
        lod_sphere: src.lod_sphere,
        parent_lod_sphere: src.parent_lod_sphere,
        errors_and_cone: src.errors_and_cone,
        page_and_fallback,
    }
}

fn rebase_bvh(src: &BvhNode, bvh_offset: u32, cluster_offset: u32) -> GpuBvhNode {
    if src.is_leaf() {
        let start = src.leaf_cluster_start().saturating_add(cluster_offset);
        let count = src.leaf_cluster_count();
        return GpuBvhNode {
            sphere: src.sphere,
            child_or_cluster: [start, count, 0, 0],
            max_parent_error: src.max_parent_error,
            min_self_error: [0.0; 4],
            info: [0, 1, start, count],
        };
    }
    let mut children = [NO_BVH_CHILD; 4];
    for (slot, &child) in src.child_or_cluster.iter().enumerate() {
        if child != NO_BVH_CHILD {
            children[slot] = child.saturating_add(bvh_offset);
        }
    }
    GpuBvhNode {
        sphere: src.sphere,
        child_or_cluster: children,
        max_parent_error: src.max_parent_error,
        min_self_error: [0.0; 4],
        info: src.meta,
    }
}

fn fill_bvh_min_self_error(
    manifest: &MeshletManifest,
    bvh_slice: &mut [GpuBvhNode],
    bvh_offset: u32,
) {
    let mut node_min = vec![0.0_f32; bvh_slice.len()];
    for i in (0..bvh_slice.len()).rev() {
        let src = &manifest.bvh[i];
        if src.is_leaf() {
            let start = src.leaf_cluster_start() as usize;
            let count = src.leaf_cluster_count() as usize;
            let mut m = f32::INFINITY;
            for c in manifest.clusters.iter().skip(start).take(count) {
                let se = c.self_error();
                if se < m {
                    m = se;
                }
            }
            node_min[i] = if m.is_finite() { m } else { 0.0 };
        } else {
            let mut m = f32::INFINITY;
            let mut child_min = [0.0_f32; 4];
            let child_cnt = (src.child_count() as usize).min(4);
            for (k, slot) in child_min.iter_mut().enumerate().take(child_cnt) {
                let c_global = bvh_slice[i].child_or_cluster[k];
                if c_global != NO_BVH_CHILD {
                    let local = (c_global - bvh_offset) as usize;
                    let cm = node_min.get(local).copied().unwrap_or(0.0);
                    *slot = cm;
                    if cm < m {
                        m = cm;
                    }
                }
            }
            bvh_slice[i].min_self_error = child_min;
            node_min[i] = if m.is_finite() { m } else { 0.0 };
        }
    }
}

fn extract_shadow_cut(
    clusters: &[ClusterHeader],
    cluster_offset: u32,
    max_clusters: usize,
    out: &mut Vec<u32>,
) -> [u32; 2] {
    let start = out.len() as u32;
    let thresholds = [
        0.002_f32, 0.004, 0.008, 0.015, 0.025, 0.04, 0.065, 0.09, 0.12, 0.20, 0.32, 0.50, 0.75,
        1.15, 1.75, 2.6, 3.8, 5.5, 8.0, 12.0, 18.0, 28.0, 48.0,
    ];
    let mut chosen: Option<f32> = None;
    for &thresh in &thresholds {
        let mut cnt = 0_usize;
        for c in clusters {
            if c.parent_error() > thresh && c.self_error() <= thresh {
                cnt += 1;
            }
        }
        if cnt > 0 && cnt <= max_clusters {
            chosen = Some(thresh);
            break;
        }
    }
    if let Some(thresh) = chosen {
        for (idx, c) in clusters.iter().enumerate() {
            if c.parent_error() > thresh && c.self_error() <= thresh {
                out.push(cluster_offset + idx as u32);
            }
        }
    }
    if out.len() as u32 == start {
        for (idx, c) in clusters.iter().enumerate() {
            if c.is_root() {
                out.push(cluster_offset + idx as u32);
            }
        }
    }
    let count = out.len() as u32 - start;
    [start, count]
}

impl NaniteWorld {
    pub fn clear(&mut self) {
        self.meshes.clear();
        self.clusters.clear();
        self.bvh.clear();
        self.instances.clear();
        self.shadow_cut_clusters.clear();
        self.shadow_mesh_ranges.clear();
        self.total_pages = 0;
        self.total_virtual_tris = 0;
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
        self.instance_revision = self.instance_revision.wrapping_add(1);
    }

    pub fn add_mesh(
        &mut self,
        name: impl Into<String>,
        manifest: &MeshletManifest,
        #[cfg(not(target_arch = "wasm32"))] mlp_path: Option<PathBuf>,
    ) -> u32 {
        let mesh_id = self.meshes.len() as u32;
        let cluster_offset = self.clusters.len() as u32;
        let bvh_offset = self.bvh.len() as u32;
        let page_offset = self.total_pages;

        self.clusters.reserve(manifest.clusters.len());
        for c in &manifest.clusters {
            self.clusters
                .push(rebase_cluster(c, cluster_offset, page_offset));
        }
        let bvh_start = self.bvh.len();
        self.bvh.reserve(manifest.bvh.len());
        for node in &manifest.bvh {
            self.bvh.push(rebase_bvh(node, bvh_offset, cluster_offset));
        }
        fill_bvh_min_self_error(manifest, &mut self.bvh[bvh_start..], bvh_offset);

        let c0_budget = if manifest.clusters.len() >= 1000 { 4096 } else { 512 };
        let c1_budget = if manifest.clusters.len() >= 1000 { 2048 } else { 256 };
        let shadow_c0_range = extract_shadow_cut(
            &manifest.clusters,
            cluster_offset,
            c0_budget,
            &mut self.shadow_cut_clusters,
        );
        let shadow_c1_range = extract_shadow_cut(
            &manifest.clusters,
            cluster_offset,
            c1_budget,
            &mut self.shadow_cut_clusters,
        );
        self.shadow_mesh_ranges.push(shadow_c0_range);
        self.shadow_mesh_ranges.push(shadow_c1_range);

        let bound_sphere = manifest_bound_sphere(manifest);
        self.total_pages = self.total_pages.saturating_add(manifest.page_count);
        self.meshes.push(MeshEntry {
            name: name.into(),
            source_tri_count: manifest.source_tri_count,
            bvh_root: bvh_offset,
            cluster_offset,
            cluster_count: manifest.clusters.len() as u32,
            page_offset,
            page_count: manifest.page_count,
            root_page_count: manifest.root_page_count,
            page_spans: manifest.page_spans.clone(),
            bound_sphere,
            shadow_c0_range,
            shadow_c1_range,
            #[cfg(not(target_arch = "wasm32"))]
            mlp_path,
        });
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
        mesh_id
    }

    pub fn add_instance(&mut self, mesh_id: u32, pos: Vec3, rot: Quat, scale: f32) {
        let Some(mesh) = self.meshes.get(mesh_id as usize) else {
            return;
        };
        let local_c = Vec3::new(
            mesh.bound_sphere[0],
            mesh.bound_sphere[1],
            mesh.bound_sphere[2],
        );
        let world_c = pos + rot * (local_c * scale);
        let world_r = mesh.bound_sphere[3] * scale;
        self.instances.push(NaniteInstance {
            pos_scale: [pos.x, pos.y, pos.z, scale],
            rot_quat: [rot.x, rot.y, rot.z, rot.w],
            bound_sphere: [world_c.x, world_c.y, world_c.z, world_r],
            info: [mesh.bvh_root, mesh.cluster_offset, mesh.cluster_count, mesh_id],
        });
        self.total_virtual_tris = self
            .total_virtual_tris
            .saturating_add(u64::from(mesh.source_tri_count));
        self.instance_revision = self.instance_revision.wrapping_add(1);
    }

    #[must_use]
    pub fn find_page_owner(&self, global_page_id: u32) -> Option<(u32, &MeshEntry, u32)> {
        for (idx, mesh) in self.meshes.iter().enumerate() {
            let end = mesh.page_offset.saturating_add(mesh.page_count);
            if global_page_id >= mesh.page_offset && global_page_id < end {
                return Some((idx as u32, mesh, global_page_id - mesh.page_offset));
            }
        }
        None
    }
}

#[derive(Resource)]
pub struct NaniteStreaming {
    page_table: Vec<u32>,
    slots: Vec<Option<SlotRecord>>,
    in_flight: Vec<bool>,
    pinned_mask: Vec<bool>,
    ready_mask: Vec<bool>,
    budget_pages: u32,
    frame_tick: u64,
    pending_requests: VecDeque<PageFetchRequest>,
    pinned_requests: HashMap<u32, PageFetchRequest>,
    ready_pages: VecDeque<(u32, Vec<u32>)>,
    pending_uploads: Vec<PageUpload>,
    table_dirty: bool,
    residency_version: u64,
    initialized_pages: u32,
    synced_mesh_count: usize,
    #[cfg(not(target_arch = "wasm32"))]
    open_files: Vec<Option<File>>,
}

impl Default for NaniteStreaming {
    fn default() -> Self {
        Self {
            page_table: Vec::new(),
            slots: vec![None; MAX_HEAP_PAGES as usize],
            in_flight: Vec::new(),
            pinned_mask: Vec::new(),
            ready_mask: Vec::new(),
            budget_pages: MAX_HEAP_PAGES,
            frame_tick: 0,
            pending_requests: VecDeque::new(),
            pinned_requests: HashMap::default(),
            ready_pages: VecDeque::new(),
            pending_uploads: Vec::new(),
            table_dirty: true,
            residency_version: 0,
            initialized_pages: 0,
            synced_mesh_count: 0,
            #[cfg(not(target_arch = "wasm32"))]
            open_files: Vec::new(),
        }
    }
}

fn make_fetch_request(
    mesh_id: u32,
    mesh: &MeshEntry,
    local_page_id: u32,
    global_page_id: u32,
    priority: u32,
) -> PageFetchRequest {
    let (byte_offset, byte_len) = match mesh.page_spans.get(local_page_id as usize) {
        Some(span) => (span.byte_offset, u64::from(span.byte_len)),
        None => (
            u64::from(local_page_id) * (PAGE_SIZE_BYTES as u64),
            PAGE_SIZE_BYTES as u64,
        ),
    };
    PageFetchRequest {
        mesh_id,
        mesh_name: mesh.name.clone(),
        local_page_id,
        global_page_id,
        byte_offset,
        byte_len,
        priority,
    }
}

impl NaniteStreaming {
    fn enqueue_mesh_range(&mut self, world: &NaniteWorld, start_mesh: usize) {
        for (mesh_id, mesh) in world.meshes.iter().enumerate().skip(start_mesh) {
            for local in 0..mesh.root_page_count {
                let global = mesh.page_offset + local;
                if let Some(flag) = self.pinned_mask.get_mut(global as usize) {
                    *flag = true;
                }
                self.pinned_requests.insert(
                    global,
                    make_fetch_request(mesh_id as u32, mesh, local, global, u32::MAX),
                );
                self.enqueue_fetch(mesh_id as u32, mesh, local, global, u32::MAX);
            }
        }
        for (mesh_id, mesh) in world.meshes.iter().enumerate().skip(start_mesh) {
            let warm_end = mesh.page_count.min(mesh.root_page_count.saturating_add(2));
            for local in mesh.root_page_count..warm_end {
                let global = mesh.page_offset + local;
                self.enqueue_fetch(mesh_id as u32, mesh, local, global, u32::MAX - 1);
            }
        }
    }

    pub fn sync_world(&mut self, world: &NaniteWorld) {
        if self.initialized_pages == world.total_pages
            && self.synced_mesh_count == world.meshes.len()
            && !self.page_table.is_empty()
        {
            return;
        }
        let total = world.total_pages as usize;
        let can_append = self.synced_mesh_count > 0
            && world.meshes.len() >= self.synced_mesh_count
            && world.total_pages >= self.initialized_pages
            && !self.page_table.is_empty();
        if can_append {
            let start_mesh = self.synced_mesh_count;
            self.page_table.resize(total.max(1), UNMAPPED_PAGE);
            self.in_flight.resize(total, false);
            self.pinned_mask.resize(total, false);
            self.ready_mask.resize(total, false);
            #[cfg(not(target_arch = "wasm32"))]
            {
                for mesh in world.meshes.iter().skip(start_mesh) {
                    self.open_files
                        .push(mesh.mlp_path.as_ref().and_then(|p| File::open(p).ok()));
                }
            }
            self.initialized_pages = world.total_pages;
            self.synced_mesh_count = world.meshes.len();
            self.table_dirty = true;
            self.residency_version = self.residency_version.wrapping_add(1);
            self.enqueue_mesh_range(world, start_mesh);
            #[cfg(not(target_arch = "wasm32"))]
            self.pump_native_io((world.meshes.len() - start_mesh).saturating_mul(4).max(8));
            return;
        }
        self.page_table = vec![UNMAPPED_PAGE; total.max(1)];
        self.slots.fill(None);
        self.in_flight = vec![false; total];
        self.pinned_mask = vec![false; total];
        self.ready_mask = vec![false; total];
        self.pending_requests.clear();
        self.pinned_requests.clear();
        self.ready_pages.clear();
        self.pending_uploads.clear();
        self.initialized_pages = world.total_pages;
        self.synced_mesh_count = world.meshes.len();
        self.table_dirty = true;
        self.residency_version = self.residency_version.wrapping_add(1);
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.open_files = world
                .meshes
                .iter()
                .map(|mesh| mesh.mlp_path.as_ref().and_then(|p| File::open(p).ok()))
                .collect();
        }
        self.enqueue_mesh_range(world, 0);
        #[cfg(not(target_arch = "wasm32"))]
        self.pump_native_io(world.meshes.len().saturating_mul(2).max(16));
    }

    pub fn set_budget(&mut self, budget_pages: u32) {
        let clamped = budget_pages.clamp(16, MAX_HEAP_PAGES);
        if self.budget_pages == clamped {
            return;
        }
        self.budget_pages = clamped;
        for slot_idx in (clamped as usize)..self.slots.len() {
            if let Some(rec) = self.slots[slot_idx]
                && !rec.pinned
            {
                if let Some(entry) = self.page_table.get_mut(rec.global_page_id as usize) {
                    *entry = UNMAPPED_PAGE;
                }
                self.slots[slot_idx] = None;
                self.table_dirty = true;
                self.residency_version = self.residency_version.wrapping_add(1);
            }
        }
    }

    pub fn deliver_page(&mut self, global_page_id: u32, words: Vec<u32>) {
        let idx = global_page_id as usize;
        if let Some(flag) = self.in_flight.get_mut(idx) {
            *flag = false;
        }
        if words.len() != PAGE_WORDS
            || idx >= self.page_table.len()
            || idx >= self.ready_mask.len()
            || self.page_table[idx] != UNMAPPED_PAGE
            || self.ready_mask[idx]
        {
            return;
        }
        self.ready_mask[idx] = true;
        self.ready_pages.push_back((global_page_id, words));
    }

    pub fn fail_page(&mut self, global_page_id: u32) {
        let idx = global_page_id as usize;
        if let Some(flag) = self.in_flight.get_mut(idx) {
            *flag = false;
        }
        let is_pinned = self.pinned_mask.get(idx).copied().unwrap_or(false);
        let is_unmapped = self.page_table.get(idx) == Some(&UNMAPPED_PAGE);
        let is_ready = self.ready_mask.get(idx).copied().unwrap_or(false);
        if !is_pinned || !is_unmapped || is_ready {
            return;
        }
        if self
            .pending_requests
            .iter()
            .any(|r| r.global_page_id == global_page_id)
        {
            return;
        }
        if let Some(req) = self.pinned_requests.get(&global_page_id) {
            if let Some(flag) = self.in_flight.get_mut(idx) {
                *flag = true;
            }
            let mut retry = req.clone();
            retry.priority = u32::MAX;
            self.pending_requests.push_back(retry);
        }
    }

    pub fn take_fetch_requests(&mut self, limit: usize) -> Vec<PageFetchRequest> {
        if self.pending_requests.is_empty() || limit == 0 {
            return Vec::new();
        }
        let slice = self.pending_requests.make_contiguous();
        slice.sort_unstable_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.mesh_id.cmp(&b.mesh_id))
                .then_with(|| a.local_page_id.cmp(&b.local_page_id))
        });
        let count = limit.min(self.pending_requests.len());
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(req) = self.pending_requests.pop_front() {
                out.push(req);
            }
        }
        out
    }

    pub fn requeue_fetch_requests(&mut self, reqs: Vec<PageFetchRequest>) {
        for req in reqs {
            let idx = req.global_page_id as usize;
            let mapped =
                self.page_table.get(idx).copied().unwrap_or(UNMAPPED_PAGE) != UNMAPPED_PAGE;
            let ready = self.ready_mask.get(idx).copied().unwrap_or(false);
            if mapped || ready {
                if let Some(flag) = self.in_flight.get_mut(idx) {
                    *flag = false;
                }
                continue;
            }
            if let Some(flag) = self.in_flight.get_mut(idx) {
                *flag = true;
            }
            self.pending_requests.push_back(req);
        }
    }

    pub fn ingest_gpu_requests(
        &mut self,
        world: &NaniteWorld,
        priorities: &[u32],
        max_new_requests: usize,
    ) {
        self.frame_tick = self.frame_tick.wrapping_add(1);
        let tick = self.frame_tick;
        let total = (world.total_pages as usize).min(priorities.len());
        let mut retained = VecDeque::with_capacity(self.pending_requests.len());
        while let Some(req) = self.pending_requests.pop_front() {
            let idx = req.global_page_id as usize;
            let mapped =
                self.page_table.get(idx).copied().unwrap_or(UNMAPPED_PAGE) != UNMAPPED_PAGE;
            let ready = self.ready_mask.get(idx).copied().unwrap_or(false);
            if req.priority >= u32::MAX - 1 && !mapped && !ready {
                retained.push_back(req);
            } else if let Some(flag) = self.in_flight.get_mut(idx) {
                *flag = false;
            }
        }
        self.pending_requests = retained;

        let mesh_roots_ready: Vec<bool> = world
            .meshes
            .iter()
            .map(|mesh| {
                (0..mesh.root_page_count).all(|local| {
                    let global = mesh.page_offset + local;
                    self.page_table
                        .get(global as usize)
                        .copied()
                        .unwrap_or(UNMAPPED_PAGE)
                        != UNMAPPED_PAGE
                })
            })
            .collect();

        let mut wanted: Vec<(u32, u32)> = Vec::new();
        for (page_id, &prio) in priorities.iter().enumerate().take(total) {
            if prio == 0 {
                continue;
            }
            let slot = self.page_table.get(page_id).copied().unwrap_or(UNMAPPED_PAGE);
            if slot != UNMAPPED_PAGE {
                if let Some(Some(rec)) = self.slots.get_mut(slot as usize) {
                    rec.last_used_frame = tick;
                }
                continue;
            }
            if self.in_flight.get(page_id).copied().unwrap_or(false)
                || self.ready_mask.get(page_id).copied().unwrap_or(false)
            {
                continue;
            }
            if let Some((mesh_id, _, _)) = world.find_page_owner(page_id as u32)
                && mesh_roots_ready.get(mesh_id as usize).copied().unwrap_or(false)
            {
                wanted.push((prio, page_id as u32));
            }
        }
        let evictable = self.count_evictable_slots();
        let in_flight_unmapped = self
            .in_flight
            .iter()
            .enumerate()
            .filter(|&(pid, &active)| {
                active
                    && self.page_table.get(pid).copied().unwrap_or(UNMAPPED_PAGE) == UNMAPPED_PAGE
                    && !self.ready_mask.get(pid).copied().unwrap_or(false)
            })
            .count();
        let pending_not_flagged = self
            .pending_requests
            .iter()
            .filter(|req| {
                let pid = req.global_page_id as usize;
                !self.in_flight.get(pid).copied().unwrap_or(false)
                    && self.page_table.get(pid).copied().unwrap_or(UNMAPPED_PAGE) == UNMAPPED_PAGE
                    && !self.ready_mask.get(pid).copied().unwrap_or(false)
            })
            .count();
        let outstanding = self
            .ready_pages
            .len()
            .saturating_add(in_flight_unmapped)
            .saturating_add(pending_not_flagged);
        let slot_headroom = evictable.saturating_sub(outstanding);
        let pipeline_headroom = max_new_requests.saturating_sub(outstanding);
        let queue_room = 256_usize.saturating_sub(self.pending_requests.len());
        let cap = max_new_requests
            .min(slot_headroom)
            .min(pipeline_headroom)
            .min(queue_room);
        if cap == 0 {
            return;
        }
        wanted.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (prio, global_page_id) in wanted.into_iter().take(cap) {
            if let Some((mesh_id, mesh, local_page_id)) = world.find_page_owner(global_page_id) {
                self.enqueue_fetch(
                    mesh_id,
                    mesh,
                    local_page_id,
                    global_page_id,
                    prio.min(u32::MAX - 2),
                );
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn pump_native_io(&mut self, max_pages: usize) {
        let mut raw = Vec::new();
        for req in self.take_fetch_requests(max_pages) {
            raw.resize(req.byte_len as usize, 0);
            let loaded = self
                .open_files
                .get_mut(req.mesh_id as usize)
                .and_then(Option::as_mut)
                .and_then(|file| {
                    file.seek(SeekFrom::Start(req.byte_offset)).ok()?;
                    file.read_exact(&mut raw).ok()?;
                    decompress_page_words(&raw).ok()
                });
            match loaded {
                Some(words) => self.deliver_page(req.global_page_id, words),
                None => self.fail_page(req.global_page_id),
            }
        }
    }

    pub fn commit_ready_pages(&mut self, max_uploads: usize) {
        let mut uploaded = 0_usize;
        while uploaded < max_uploads {
            let Some((global_page_id, words)) = self.ready_pages.pop_front() else {
                break;
            };
            let idx = global_page_id as usize;
            if let Some(flag) = self.ready_mask.get_mut(idx) {
                *flag = false;
            }
            if idx >= self.page_table.len() {
                continue;
            }
            if self.page_table[idx] != UNMAPPED_PAGE {
                continue;
            }
            let pinned = self.pinned_mask.get(idx).copied().unwrap_or(false);
            let Some(slot) = self.pick_slot(pinned) else {
                continue;
            };
            if let Some(old) = self.slots[slot as usize]
                && let Some(entry) = self.page_table.get_mut(old.global_page_id as usize)
            {
                *entry = UNMAPPED_PAGE;
            }
            self.slots[slot as usize] = Some(SlotRecord {
                global_page_id,
                pinned,
                last_used_frame: self.frame_tick,
            });
            self.page_table[idx] = slot;
            self.pending_uploads.push(PageUpload {
                slot,
                global_page_id,
                words,
            });
            self.table_dirty = true;
            self.residency_version = self.residency_version.wrapping_add(1);
            uploaded += 1;
        }
    }

    fn count_evictable_slots(&self) -> usize {
        let limit = (self.budget_pages as usize).min(self.slots.len());
        let stale_cutoff = self.frame_tick.saturating_sub(4);
        self.slots
            .iter()
            .take(limit)
            .filter(|slot| {
                slot.is_none_or(|rec| !rec.pinned && rec.last_used_frame < stale_cutoff)
            })
            .count()
    }

    fn pick_slot(&self, allow_active_evict: bool) -> Option<u32> {
        let limit = (self.budget_pages as usize).min(self.slots.len());
        for (idx, slot) in self.slots.iter().enumerate().take(limit) {
            if slot.is_none() {
                return Some(idx as u32);
            }
        }
        let stale_cutoff = self.frame_tick.saturating_sub(4);
        let mut best_idx = None;
        let mut best_age = u64::MAX;
        for (idx, slot) in self.slots.iter().enumerate().take(limit) {
            if let Some(rec) = slot
                && !rec.pinned
                && (allow_active_evict || rec.last_used_frame < stale_cutoff)
                && rec.last_used_frame < best_age
            {
                best_age = rec.last_used_frame;
                best_idx = Some(idx as u32);
            }
        }
        best_idx
    }

    fn enqueue_fetch(
        &mut self,
        mesh_id: u32,
        mesh: &MeshEntry,
        local_page_id: u32,
        global_page_id: u32,
        priority: u32,
    ) {
        let idx = global_page_id as usize;
        if self.page_table.get(idx).copied().unwrap_or(UNMAPPED_PAGE) != UNMAPPED_PAGE {
            return;
        }
        if self.ready_mask.get(idx).copied().unwrap_or(false) {
            return;
        }
        if let Some(flag) = self.in_flight.get_mut(idx) {
            if *flag {
                return;
            }
            *flag = true;
        }
        self.pending_requests.push_back(make_fetch_request(
            mesh_id,
            mesh,
            local_page_id,
            global_page_id,
            priority,
        ));
    }

    pub fn take_uploads(&mut self) -> Vec<PageUpload> {
        std::mem::take(&mut self.pending_uploads)
    }

    #[must_use]
    pub fn page_table(&self) -> &[u32] {
        &self.page_table
    }

    pub fn take_table_dirty(&mut self) -> bool {
        std::mem::take(&mut self.table_dirty)
    }

    #[must_use]
    pub fn residency_version(&self) -> u64 {
        self.residency_version
    }

    #[must_use]
    pub fn resident_count(&self) -> u32 {
        self.slots.iter().filter(|s| s.is_some()).count() as u32
    }

    #[must_use]
    pub fn pinned_count(&self) -> u32 {
        self.slots
            .iter()
            .filter(|s| s.is_some_and(|rec| rec.pinned))
            .count() as u32
    }

    #[must_use]
    pub fn pending_fetch_count(&self) -> u32 {
        self.in_flight.iter().filter(|&&v| v).count() as u32
    }

    #[must_use]
    pub fn pending_request_count(&self) -> u32 {
        self.pending_requests.len() as u32
    }

    #[must_use]
    pub fn ready_page_count(&self) -> u32 {
        self.ready_pages.len() as u32
    }

    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        !self.pending_requests.is_empty()
            || !self.ready_pages.is_empty()
            || self.in_flight.iter().any(|&v| v)
    }

    #[must_use]
    pub fn has_evictable_slots(&self) -> bool {
        self.count_evictable_slots() > 0
    }

    #[must_use]
    pub fn budget_pages(&self) -> u32 {
        self.budget_pages
    }
}

#[cfg(test)]
mod tests {
    use engine_meshlet::{BvhNode, ClusterHeader, MeshletManifest, PAGE_WORDS, PageSpan};

    use super::*;

    fn dummy_manifest(page_count: u32, root_page_count: u32) -> MeshletManifest {
        let mut clusters = Vec::new();
        for p in 0..page_count {
            clusters.push(ClusterHeader {
                bound_sphere: [0.0, 0.0, 0.0, 1.0],
                lod_sphere: [0.0, 0.0, 0.0, 1.0],
                parent_lod_sphere: [0.0, 0.0, 0.0, 1.0],
                errors_and_cone: [0, f32::INFINITY.to_bits(), 0, (3_u32) | (1_u32 << 8)],
                page_and_fallback: [p, 0, 0, 0],
            });
        }
        let spans = (0..page_count)
            .map(|p| PageSpan {
                byte_offset: u64::from(p) * 64,
                byte_len: 64,
            })
            .collect();
        MeshletManifest {
            source_tri_count: page_count,
            lod_levels: 1,
            page_count,
            root_page_count,
            bounds_min: [-1.0; 3],
            bounds_max: [1.0; 3],
            clusters,
            bvh: vec![BvhNode {
                sphere: [0.0, 0.0, 0.0, 1.0],
                child_or_cluster: [0, page_count, 0, 0],
                max_parent_error: [f32::INFINITY; 4],
                meta: [0, 1, 0, page_count],
            }],
            page_spans: spans,
        }
    }

    #[test]
    fn sync_world_appends_without_dropping_resident_pages() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(4, 2);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        streaming.sync_world(&world);
        streaming.deliver_page(0, vec![0_u32; PAGE_WORDS]);
        streaming.deliver_page(1, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(8);
        assert_eq!(streaming.resident_count(), 2);
        let slot0 = streaming.page_table()[0];
        let slot1 = streaming.page_table()[1];
        assert_ne!(slot0, UNMAPPED_PAGE);
        assert_ne!(slot1, UNMAPPED_PAGE);

        let m1 = dummy_manifest(3, 1);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_b", &m1, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_b", &m1);

        streaming.sync_world(&world);
        assert_eq!(streaming.page_table().len(), 7);
        assert_eq!(streaming.page_table()[0], slot0);
        assert_eq!(streaming.page_table()[1], slot1);
        assert_eq!(streaming.page_table()[4], UNMAPPED_PAGE);
        assert_eq!(streaming.resident_count(), 2);
        assert!(streaming.take_table_dirty());
    }

    #[test]
    fn root_pages_preempt_queued_fine_pages() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(8, 1);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        streaming.sync_world(&world);
        let _ = streaming.take_fetch_requests(16);
        let prios = vec![0, 0, 0, 50, 40, 30, 20, 10];
        streaming.ingest_gpu_requests(&world, &prios, 16);
        assert!(streaming.take_fetch_requests(16).is_empty());

        streaming.deliver_page(0, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(4);
        streaming.ingest_gpu_requests(&world, &prios, 16);

        let m1 = dummy_manifest(12, 10);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_b", &m1, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_b", &m1);
        streaming.sync_world(&world);

        let batch = streaming.take_fetch_requests(2);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].mesh_name, "chunk_b");
        assert_eq!(batch[0].priority, u32::MAX);
        assert_eq!(batch[1].mesh_name, "chunk_b");
        assert_eq!(batch[1].priority, u32::MAX);
    }

    #[test]
    fn ingest_gpu_requests_replaces_stale_pending_on_camera_cut() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(300, 1);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        streaming.sync_world(&world);
        let _ = streaming.take_fetch_requests(4);
        streaming.deliver_page(0, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(4);

        let mut old_prios = vec![0_u32; 300];
        for (p, slot) in old_prios.iter_mut().enumerate().take(280).skip(1) {
            *slot = 500 + p as u32;
        }
        streaming.ingest_gpu_requests(&world, &old_prios, 280);
        assert_eq!(streaming.pending_requests.len(), 256);

        let mut new_prios = vec![0_u32; 300];
        for (p, slot) in new_prios.iter_mut().enumerate().take(296).skip(280) {
            *slot = 9000 + p as u32;
        }
        streaming.ingest_gpu_requests(&world, &new_prios, 280);
        let batch = streaming.take_fetch_requests(32);
        assert_eq!(batch.len(), 16, "only current view pages should remain queued");
        for req in &batch {
            assert!(
                (280..296).contains(&req.global_page_id),
                "stale page {} must be evicted on camera cut",
                req.global_page_id
            );
        }
    }

    #[test]
    fn has_pending_work_tracks_pending_ready_and_in_flight() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(4, 1);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        assert!(!streaming.has_pending_work());
        assert_eq!(streaming.pending_request_count(), 0);
        assert_eq!(streaming.ready_page_count(), 0);

        streaming.sync_world(&world);
        let initial_pending = streaming.pending_request_count();
        assert!(initial_pending >= 1);
        assert_eq!(streaming.ready_page_count(), 0);
        assert!(streaming.has_pending_work());

        let taken = streaming.take_fetch_requests(16);
        assert_eq!(taken.len(), initial_pending as usize);
        assert_eq!(streaming.pending_request_count(), 0);
        assert_eq!(streaming.pending_fetch_count(), initial_pending);
        assert!(streaming.has_pending_work());

        for req in &taken {
            streaming.deliver_page(req.global_page_id, vec![0_u32; PAGE_WORDS]);
        }
        assert_eq!(streaming.pending_fetch_count(), 0);
        assert_eq!(streaming.ready_page_count(), initial_pending);
        assert!(streaming.has_pending_work());

        streaming.commit_ready_pages(8);
        assert_eq!(streaming.ready_page_count(), 0);
        assert_eq!(streaming.pending_request_count(), 0);
        assert_eq!(streaming.pending_fetch_count(), 0);
        assert!(!streaming.has_pending_work());

        let prios = vec![0, 500, 0, 0];
        streaming.ingest_gpu_requests(&world, &prios, 16);
        assert_eq!(streaming.pending_request_count(), 1);
        assert!(streaming.has_pending_work());
        let fine = streaming.take_fetch_requests(16);
        assert_eq!(fine.len(), 1);
        assert_eq!(streaming.pending_request_count(), 0);
        assert_eq!(streaming.pending_fetch_count(), 1);
        assert!(streaming.has_pending_work());
        streaming.deliver_page(1, vec![0_u32; PAGE_WORDS]);
        assert_eq!(streaming.ready_page_count(), 1);
        assert!(streaming.has_pending_work());
        streaming.commit_ready_pages(4);
        assert!(!streaming.has_pending_work());
    }

    #[test]
    fn failed_pinned_root_page_is_re_enqueued() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(4, 1);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        streaming.sync_world(&world);
        let taken = streaming.take_fetch_requests(16);
        assert!(taken.iter().any(|r| r.global_page_id == 0));

        streaming.fail_page(0);
        assert_eq!(streaming.pending_request_count(), 1);
        assert!(streaming.has_pending_work());

        let retry = streaming.take_fetch_requests(16);
        assert_eq!(retry.len(), 1);
        assert_eq!(retry[0].global_page_id, 0);
        assert_eq!(retry[0].priority, u32::MAX);

        streaming.deliver_page(0, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(4);
        streaming.fail_page(0);
        assert_eq!(streaming.pending_request_count(), 0);
        assert!(!streaming.has_pending_work());

        let prios = vec![0, 500, 0, 0];
        streaming.ingest_gpu_requests(&world, &prios, 16);
        let fine = streaming.take_fetch_requests(16);
        assert_eq!(fine.len(), 1);
        assert_eq!(fine[0].global_page_id, 1);
        streaming.fail_page(1);
        assert_eq!(streaming.pending_request_count(), 0);
        assert!(!streaming.has_pending_work());
    }

    #[test]
    fn unmapped_roots_on_mesh_b_do_not_starve_mesh_a_streaming() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(8, 1);
        let m1 = dummy_manifest(8, 2);
        #[cfg(not(target_arch = "wasm32"))]
        {
            world.add_mesh("chunk_a", &m0, None);
            world.add_mesh("chunk_b", &m1, None);
        }
        #[cfg(target_arch = "wasm32")]
        {
            world.add_mesh("chunk_a", &m0);
            world.add_mesh("chunk_b", &m1);
        }

        let mut streaming = NaniteStreaming::default();
        streaming.sync_world(&world);
        let _ = streaming.take_fetch_requests(32);

        streaming.deliver_page(0, vec![0_u32; PAGE_WORDS]);
        streaming.deliver_page(8, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(8);

        let mut prios = vec![0_u32; 16];
        prios[4] = 500;
        prios[5] = 400;
        prios[12] = 9000;
        prios[13] = 8000;
        streaming.ingest_gpu_requests(&world, &prios, 16);

        let batch = streaming.take_fetch_requests(16);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].mesh_name, "chunk_a");
        assert_eq!(batch[0].global_page_id, 4);
        assert_eq!(batch[1].mesh_name, "chunk_a");
        assert_eq!(batch[1].global_page_id, 5);

        streaming.deliver_page(9, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(4);
        streaming.ingest_gpu_requests(&world, &prios, 16);
        let batch_b = streaming.take_fetch_requests(16);
        assert_eq!(batch_b.len(), 2);
        assert_eq!(batch_b[0].mesh_name, "chunk_b");
        assert_eq!(batch_b[0].global_page_id, 12);
        assert_eq!(batch_b[1].mesh_name, "chunk_b");
        assert_eq!(batch_b[1].global_page_id, 13);
    }

    #[test]
    fn tight_heap_eviction_preserves_pinned_roots() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(32, 4);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        streaming.set_budget(16);
        streaming.sync_world(&world);
        let _ = streaming.take_fetch_requests(32);

        for p in 0..16 {
            streaming.deliver_page(p, vec![0_u32; PAGE_WORDS]);
        }
        streaming.commit_ready_pages(16);
        assert_eq!(streaming.resident_count(), 16);
        assert_eq!(streaming.pinned_count(), 4);

        let mut prios = vec![0_u32; 32];
        prios[20] = 1000;
        for _ in 0..5 {
            streaming.ingest_gpu_requests(&world, &prios, 16);
        }
        let batch = streaming.take_fetch_requests(16);
        assert!(!batch.is_empty());
        streaming.deliver_page(20, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(4);
        assert_eq!(streaming.resident_count(), 16);
        assert_eq!(streaming.pinned_count(), 4);
        for root in 0..4 {
            assert_ne!(streaming.page_table()[root], UNMAPPED_PAGE);
        }
        assert_ne!(streaming.page_table()[20], UNMAPPED_PAGE);
    }

    #[test]
    fn has_evictable_slots_tracks_empty_active_and_stale_slots() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(32, 4);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        streaming.set_budget(16);
        streaming.sync_world(&world);
        let _ = streaming.take_fetch_requests(32);
        assert!(streaming.has_evictable_slots());

        for p in 0..15 {
            streaming.deliver_page(p, vec![0_u32; PAGE_WORDS]);
        }
        streaming.commit_ready_pages(16);
        assert_eq!(streaming.resident_count(), 15);
        assert!(streaming.has_evictable_slots());

        streaming.deliver_page(15, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(4);
        assert_eq!(streaming.resident_count(), 16);
        assert!(!streaming.has_evictable_slots());

        let mut active_prios = vec![0_u32; 32];
        for slot in active_prios.iter_mut().take(16) {
            *slot = 500;
        }
        active_prios[20] = 1000;
        for _ in 0..6 {
            streaming.ingest_gpu_requests(&world, &active_prios, 16);
            assert!(!streaming.has_evictable_slots());
            assert_eq!(streaming.pending_request_count(), 0);
            assert!(!streaming.has_pending_work());
        }

        let mut cut_prios = vec![0_u32; 32];
        for slot in cut_prios.iter_mut().take(4) {
            *slot = 500;
        }
        cut_prios[20] = 1000;
        for _ in 0..4 {
            streaming.ingest_gpu_requests(&world, &cut_prios, 16);
            assert!(!streaming.has_evictable_slots());
            assert_eq!(streaming.pending_request_count(), 0);
        }

        streaming.ingest_gpu_requests(&world, &cut_prios, 16);
        assert!(streaming.has_evictable_slots());
        assert_eq!(streaming.pending_request_count(), 1);
    }

    #[test]
    fn ready_mask_dedups_deliveries_and_blocks_re_requests_under_async_burst() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(800, 16);
        let m1 = dummy_manifest(800, 16);
        #[cfg(not(target_arch = "wasm32"))]
        {
            world.add_mesh("chunk_a", &m0, None);
            world.add_mesh("chunk_b", &m1, None);
        }
        #[cfg(target_arch = "wasm32")]
        {
            world.add_mesh("chunk_a", &m0);
            world.add_mesh("chunk_b", &m1);
        }

        let mut streaming = NaniteStreaming::default();
        streaming.set_budget(1024);
        streaming.sync_world(&world);
        let init_reqs = streaming.take_fetch_requests(256);
        for req in init_reqs {
            streaming.deliver_page(req.global_page_id, vec![0_u32; PAGE_WORDS]);
        }
        streaming.commit_ready_pages(64);
        assert_eq!(streaming.ready_page_count(), 0);
        assert!(!streaming.has_pending_work());

        let mut prios = vec![0_u32; 1600];
        for (pid, slot) in prios.iter_mut().enumerate().take(950) {
            *slot = 1000 + pid as u32;
        }
        let mut prev_batch: Vec<PageFetchRequest> = Vec::new();
        for _ in 0..8 {
            for req in prev_batch.drain(..) {
                let before_ready = streaming.ready_page_count();
                streaming.deliver_page(req.global_page_id, vec![0_u32; PAGE_WORDS]);
                assert!(streaming.ready_page_count() <= before_ready + 1);
                let after_first = streaming.ready_page_count();
                streaming.deliver_page(req.global_page_id, vec![0_u32; PAGE_WORDS]);
                assert_eq!(streaming.ready_page_count(), after_first);
            }
            streaming.deliver_page(0, vec![0_u32; PAGE_WORDS]);
            streaming.commit_ready_pages(48);
            streaming.ingest_gpu_requests(&world, &prios, 192);
            let next_batch = streaming.take_fetch_requests(192);
            for req in &next_batch {
                let idx = req.global_page_id as usize;
                assert!(!streaming.ready_mask[idx]);
                assert_eq!(streaming.page_table()[idx], UNMAPPED_PAGE);
            }
            assert!(streaming.ready_page_count() <= 950);
            prev_batch = next_batch;
        }
        for req in prev_batch.drain(..) {
            streaming.deliver_page(req.global_page_id, vec![0_u32; PAGE_WORDS]);
        }
        let mut active_only = vec![0_u32; 1600];
        for (pid, slot) in active_only.iter_mut().enumerate() {
            if streaming.page_table()[pid] != UNMAPPED_PAGE {
                *slot = 500;
            }
        }
        for _ in 0..32 {
            streaming.commit_ready_pages(48);
            streaming.ingest_gpu_requests(&world, &active_only, 0);
            if !streaming.has_pending_work() {
                break;
            }
        }
        assert_eq!(streaming.ready_page_count(), 0);
        assert!(!streaming.has_pending_work());
        assert!(streaming.ready_mask.iter().all(|&m| !m));
    }

    #[test]
    fn commit_ready_pages_drains_without_burning_budget_on_skipped_entries() {
        let mut world = NaniteWorld::default();
        let m0 = dummy_manifest(128, 4);
        #[cfg(not(target_arch = "wasm32"))]
        world.add_mesh("chunk_a", &m0, None);
        #[cfg(target_arch = "wasm32")]
        world.add_mesh("chunk_a", &m0);

        let mut streaming = NaniteStreaming::default();
        streaming.sync_world(&world);
        let init_reqs = streaming.take_fetch_requests(32);
        let mut req_map = HashMap::default();
        for req in init_reqs {
            req_map.insert(req.global_page_id, req);
        }

        streaming.deliver_page(0, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(4);
        streaming.deliver_page(1, vec![0_u32; PAGE_WORDS]);
        assert!(streaming.ready_mask[1]);
        streaming.fail_page(1);
        assert_eq!(streaming.pending_request_count(), 0);
        assert!(!streaming.in_flight[1]);
        for pid in [2_u32, 3, 4, 5] {
            streaming.deliver_page(pid, vec![0_u32; PAGE_WORDS]);
        }
        streaming.commit_ready_pages(8);

        streaming.deliver_page(10, vec![0_u32; PAGE_WORDS]);
        let Some(r0) = req_map.remove(&0) else {
            panic!("missing root page 0 fetch request");
        };
        let mut r10 = r0.clone();
        r10.global_page_id = 10;
        r10.local_page_id = 10;
        let mut r11 = r0.clone();
        r11.global_page_id = 11;
        r11.local_page_id = 11;
        streaming.requeue_fetch_requests(vec![r0, r10, r11]);
        assert_eq!(streaming.pending_request_count(), 1);
        assert!(!streaming.in_flight[0]);
        assert!(!streaming.in_flight[10]);
        assert!(streaming.in_flight[11]);
        let _ = streaming.take_fetch_requests(8);
        streaming.deliver_page(11, vec![0_u32; PAGE_WORDS]);
        streaming.commit_ready_pages(8);
        let _ = streaming.take_uploads();

        for pid in 20..72_u32 {
            streaming.deliver_page(pid, vec![0_u32; PAGE_WORDS]);
        }
        assert_eq!(streaming.ready_page_count(), 52);
        streaming
            .ready_pages
            .push_front((20, vec![0_u32; PAGE_WORDS]));
        streaming
            .ready_pages
            .push_front((0, vec![0_u32; PAGE_WORDS]));
        streaming
            .ready_pages
            .push_front((999_999, vec![0_u32; PAGE_WORDS]));

        streaming.commit_ready_pages(48);
        let first_uploads = streaming.take_uploads();
        assert_eq!(first_uploads.len(), 48);
        assert_eq!(streaming.ready_page_count(), 4);
        assert_eq!(streaming.ready_mask.iter().filter(|&&m| m).count(), 4);

        streaming.commit_ready_pages(48);
        let second_uploads = streaming.take_uploads();
        assert_eq!(second_uploads.len(), 4);
        assert_eq!(streaming.ready_page_count(), 0);
        assert!(!streaming.has_pending_work());
        assert!(streaming.ready_mask.iter().all(|&m| !m));
    }

    fn run_async_burst_sim(budget: u32, step: usize, lat: usize) -> (u32, usize, usize, usize) {
        let mut world = NaniteWorld::default();
        for i in 0..8 {
            let m = dummy_manifest(300, 22);
            #[cfg(not(target_arch = "wasm32"))]
            world.add_mesh(format!("chunk_{i}"), &m, None);
            #[cfg(target_arch = "wasm32")]
            world.add_mesh(format!("chunk_{i}"), &m);
        }
        let mut prios = vec![0_u32; world.total_pages as usize];
        for mesh in &world.meshes {
            for local in 0..mesh.root_page_count {
                prios[(mesh.page_offset + local) as usize] = 5000;
            }
            for local in (mesh.root_page_count..mesh.page_count).step_by(step) {
                prios[(mesh.page_offset + local) as usize] = 2000 - local;
            }
        }
        let mut streaming = NaniteStreaming::default();
        streaming.set_budget(budget);
        streaming.sync_world(&world);
        let mut seen = vec![false; world.total_pages as usize];
        let mut queue: VecDeque<(usize, Vec<PageFetchRequest>)> = VecDeque::new();
        let (mut max_ready, mut delivered, mut dup, mut discarded) = (0_u32, 0, 0, 0);
        for frame in 0..120_usize {
            while queue.front().is_some_and(|(f, _)| *f <= frame)
                && let Some((_, batch)) = queue.pop_front()
            {
                for req in batch {
                    let pid = req.global_page_id as usize;
                    dup += usize::from(seen[pid]);
                    seen[pid] = true;
                    delivered += 1;
                    streaming.deliver_page(req.global_page_id, vec![0_u32; PAGE_WORDS]);
                }
            }
            max_ready = max_ready.max(streaming.ready_page_count());
            let before = streaming.ready_page_count() as usize;
            streaming.commit_ready_pages(48);
            let popped = before - streaming.ready_page_count() as usize;
            discarded += popped.saturating_sub(streaming.take_uploads().len());
            streaming.ingest_gpu_requests(&world, &prios, 192);
            let next = streaming.take_fetch_requests(192);
            if !next.is_empty() {
                queue.push_back((frame + lat, next));
            }
        }
        assert_eq!(streaming.resident_count(), budget);
        assert_eq!(streaming.ready_page_count(), 0);
        assert!(!streaming.has_pending_work());
        (max_ready, delivered, dup, discarded)
    }

    #[test]
    fn ingest_backpressure_caps_ready_pages_and_prevents_overfetch_under_burst() {
        for (budget, lat) in [(1024_u32, 1_usize), (1024, 2), (256, 1)] {
            let (max_ready, delivered, dup, discarded) = run_async_burst_sim(budget, 1, lat);
            assert!(max_ready <= 192);
            assert!(delivered <= budget as usize);
            assert_eq!((dup, discarded), (0, 0));
        }
    }

    #[test]
    fn ingest_slot_headroom_prevents_discard_and_dup_under_sparse_eviction() {
        let (max_ready, _delivered, dup, discarded) = run_async_burst_sim(256, 3, 1);
        assert!(max_ready <= 192);
        assert_eq!((dup, discarded), (0, 0));
    }
}
