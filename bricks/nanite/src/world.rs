use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read as _, Seek as _, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use engine::prelude::*;
use engine_meshlet::{
    BvhNode, ClusterHeader, MeshletManifest, NO_BVH_CHILD, NO_FALLBACK_CLUSTER, PAGE_SIZE_BYTES,
    PAGE_WORDS,
};
#[cfg(not(target_arch = "wasm32"))]
use engine_meshlet::page_bytes_as_words;

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
    let mut cone_and_meta = src.cone_and_meta;
    cone_and_meta[2] = cone_and_meta[2].saturating_add(page_offset);
    let mut streaming_meta = src.streaming_meta;
    if streaming_meta[1] != NO_FALLBACK_CLUSTER {
        streaming_meta[1] = streaming_meta[1].saturating_add(cluster_offset);
    }
    GpuClusterHeader {
        bound_sphere: src.bound_sphere,
        lod_sphere: src.lod_sphere,
        aabb_min: src.aabb_min,
        aabb_extent: src.aabb_extent,
        parent_lod_sphere: src.parent_lod_sphere,
        cone_and_meta,
        streaming_meta,
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
    world_err_thresh: f32,
    out: &mut Vec<u32>,
) -> [u32; 2] {
    let start = out.len() as u32;
    for (idx, c) in clusters.iter().enumerate() {
        let pe = c.parent_error();
        let se = c.self_error();
        if pe > world_err_thresh && se <= world_err_thresh {
            out.push(cluster_offset + idx as u32);
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

        let shadow_c0_range =
            extract_shadow_cut(&manifest.clusters, cluster_offset, 0.85, &mut self.shadow_cut_clusters);
        let shadow_c1_range =
            extract_shadow_cut(&manifest.clusters, cluster_offset, 1.85, &mut self.shadow_cut_clusters);
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
    budget_pages: u32,
    frame_tick: u64,
    pending_requests: VecDeque<PageFetchRequest>,
    ready_pages: VecDeque<(u32, Vec<u32>)>,
    pending_uploads: Vec<PageUpload>,
    table_dirty: bool,
    residency_version: u64,
    initialized_pages: u32,
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
            budget_pages: 512,
            frame_tick: 0,
            pending_requests: VecDeque::new(),
            ready_pages: VecDeque::new(),
            pending_uploads: Vec::new(),
            table_dirty: true,
            residency_version: 0,
            initialized_pages: 0,
            #[cfg(not(target_arch = "wasm32"))]
            open_files: Vec::new(),
        }
    }
}

impl NaniteStreaming {
    pub fn sync_world(&mut self, world: &NaniteWorld) {
        if self.initialized_pages == world.total_pages && !self.page_table.is_empty() {
            return;
        }
        let total = world.total_pages as usize;
        self.page_table = vec![UNMAPPED_PAGE; total.max(1)];
        self.slots.fill(None);
        self.in_flight = vec![false; total];
        self.pinned_mask = vec![false; total];
        self.pending_requests.clear();
        self.ready_pages.clear();
        self.pending_uploads.clear();
        self.initialized_pages = world.total_pages;
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
        for (mesh_id, mesh) in world.meshes.iter().enumerate() {
            for local in 0..mesh.root_page_count {
                let global = mesh.page_offset + local;
                if let Some(flag) = self.pinned_mask.get_mut(global as usize) {
                    *flag = true;
                }
                self.enqueue_fetch(mesh_id as u32, mesh, local, global, u32::MAX);
            }
        }
        for (mesh_id, mesh) in world.meshes.iter().enumerate() {
            let warm_end = mesh.page_count.min(mesh.root_page_count.saturating_add(24));
            for local in mesh.root_page_count..warm_end {
                let global = mesh.page_offset + local;
                self.enqueue_fetch(mesh_id as u32, mesh, local, global, u32::MAX - 1);
            }
        }
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
        if let Some(flag) = self.in_flight.get_mut(global_page_id as usize) {
            *flag = false;
        }
        if words.len() == PAGE_WORDS {
            self.ready_pages.push_back((global_page_id, words));
        }
    }

    pub fn fail_page(&mut self, global_page_id: u32) {
        if let Some(flag) = self.in_flight.get_mut(global_page_id as usize) {
            *flag = false;
        }
    }

    pub fn take_fetch_requests(&mut self, limit: usize) -> Vec<PageFetchRequest> {
        let count = limit.min(self.pending_requests.len());
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(req) = self.pending_requests.pop_front() {
                out.push(req);
            }
        }
        out
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
            if self.in_flight.get(page_id).copied().unwrap_or(false) {
                continue;
            }
            wanted.push((prio, page_id as u32));
        }
        wanted.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (prio, global_page_id) in wanted.into_iter().take(max_new_requests) {
            if let Some((mesh_id, mesh, local_page_id)) = world.find_page_owner(global_page_id) {
                self.enqueue_fetch(mesh_id, mesh, local_page_id, global_page_id, prio);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn pump_native_io(&mut self, max_pages: usize) {
        let mut raw = vec![0_u8; PAGE_SIZE_BYTES];
        for _ in 0..max_pages {
            let Some(req) = self.pending_requests.pop_front() else {
                break;
            };
            let loaded = self
                .open_files
                .get_mut(req.mesh_id as usize)
                .and_then(Option::as_mut)
                .and_then(|file| {
                    file.seek(SeekFrom::Start(req.byte_offset)).ok()?;
                    file.read_exact(&mut raw).ok()?;
                    page_bytes_as_words(&raw).ok()
                });
            match loaded {
                Some(words) => self.deliver_page(req.global_page_id, words),
                None => self.fail_page(req.global_page_id),
            }
        }
    }

    pub fn commit_ready_pages(&mut self, max_uploads: usize) {
        for _ in 0..max_uploads {
            let Some((global_page_id, words)) = self.ready_pages.pop_front() else {
                break;
            };
            if (global_page_id as usize) >= self.page_table.len() {
                continue;
            }
            if self.page_table[global_page_id as usize] != UNMAPPED_PAGE {
                continue;
            }
            let pinned = self
                .pinned_mask
                .get(global_page_id as usize)
                .copied()
                .unwrap_or(false);
            let Some(slot) = self.pick_slot() else {
                break;
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
            self.page_table[global_page_id as usize] = slot;
            self.pending_uploads.push(PageUpload {
                slot,
                global_page_id,
                words,
            });
            self.table_dirty = true;
            self.residency_version = self.residency_version.wrapping_add(1);
        }
    }

    fn pick_slot(&self) -> Option<u32> {
        let limit = (self.budget_pages as usize).min(self.slots.len());
        for (idx, slot) in self.slots.iter().enumerate().take(limit) {
            if slot.is_none() {
                return Some(idx as u32);
            }
        }
        let mut best_idx = None;
        let mut best_age = u64::MAX;
        for (idx, slot) in self.slots.iter().enumerate().take(limit) {
            if let Some(rec) = slot
                && !rec.pinned
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
        if let Some(flag) = self.in_flight.get_mut(global_page_id as usize) {
            if *flag {
                return;
            }
            *flag = true;
        }
        let byte_offset = u64::from(local_page_id) * (PAGE_SIZE_BYTES as u64);
        self.pending_requests.push_back(PageFetchRequest {
            mesh_id,
            mesh_name: mesh.name.clone(),
            local_page_id,
            global_page_id,
            byte_offset,
            byte_len: PAGE_SIZE_BYTES as u64,
            priority,
        });
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
    pub fn budget_pages(&self) -> u32 {
        self.budget_pages
    }
}
