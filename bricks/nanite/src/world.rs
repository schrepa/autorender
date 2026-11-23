use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read as _, Seek as _, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use engine::prelude::*;
use engine_math::collections::HashMap;
use engine_meshlet::{
    BvhNode, ClusterHeader, MeshletManifest, NO_BVH_CHILD, PAGE_SIZE_BYTES, PAGE_WORDS, PageSpan,
    decompress_page_words,
};

use crate::types::{
    GpuBvhNode, GpuClusterHeader, MAX_HEAP_PAGES, NaniteInstance, UNMAPPED_PAGE,
};

const DEFAULT_CACHE_BUDGET_BYTES: usize = 192 * 1024 * 1024;

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

#[derive(Clone, Copy, Debug)]
struct PrefetchState {
    last_pos: Vec3,
    last_res_ver: u64,
    last_tick: u64,
    last_lod_bits: u64,
    has_walked: bool,
    prefetch_only: bool,
}

impl Default for PrefetchState {
    fn default() -> Self {
        Self {
            last_pos: Vec3::splat(f32::INFINITY),
            last_res_ver: u64::MAX,
            last_tick: 0,
            last_lod_bits: 0,
            has_walked: false,
            prefetch_only: false,
        }
    }
}

#[derive(Clone, Copy)]
struct PrefetchParams {
    cam_pos: Vec3,
    near_z: f32,
    lod_scale: f32,
}

#[derive(Clone, Copy)]
struct InstXform {
    pos: Vec3,
    rot: Quat,
    scale: f32,
    err_factor: f32,
}

fn bvh_child_needed(
    node: &GpuBvhNode,
    k: usize,
    child: &GpuBvhNode,
    xform: InstXform,
    params: PrefetchParams,
    thresh: f32,
) -> bool {
    let max_pe = node.max_parent_error[k];
    let min_se = node.min_self_error[k];
    if max_pe < 1.0e29 || min_se > 0.0 {
        let child_c = Vec3::new(child.sphere[0], child.sphere[1], child.sphere[2]);
        let world_c = xform.pos + xform.rot * (child_c * xform.scale);
        let world_r = child.sphere[3] * xform.scale;
        let dist_c = (params.cam_pos - world_c).length();
        if max_pe < 1.0e29
            && max_pe * xform.err_factor
                <= thresh * engine::math::max(dist_c - world_r, params.near_z)
        {
            return false;
        }
        if min_se > 0.0
            && min_se * xform.err_factor
                > thresh * engine::math::max(dist_c + world_r, params.near_z)
        {
            return false;
        }
    }
    true
}

fn eval_prefetch_cluster(
    c: &GpuClusterHeader,
    xform: InstXform,
    params: PrefetchParams,
    thresh: f32,
) -> Option<f32> {
    let pe = f32::from_bits(c.errors_and_cone[1]);
    let p_proj = if pe < 1.0e29 {
        let pls = c.parent_lod_sphere;
        let pc = xform.pos + xform.rot * (Vec3::new(pls[0], pls[1], pls[2]) * xform.scale);
        let p_dist = engine::math::max(
            (params.cam_pos - pc).length() - pls[3] * xform.scale,
            params.near_z,
        );
        let proj = (pe * xform.err_factor) / p_dist;
        if proj <= thresh {
            return None;
        }
        proj
    } else {
        1024.0
    };
    let se = f32::from_bits(c.errors_and_cone[0]);
    if se > 0.0 {
        let ls = c.lod_sphere;
        let sc = xform.pos + xform.rot * (Vec3::new(ls[0], ls[1], ls[2]) * xform.scale);
        let s_dist = engine::math::max(
            (params.cam_pos - sc).length() - ls[3] * xform.scale,
            params.near_z,
        );
        if se * xform.err_factor > thresh * s_dist {
            return None;
        }
    }
    let bs = c.bound_sphere;
    let world_c = xform.pos + xform.rot * (Vec3::new(bs[0], bs[1], bs[2]) * xform.scale);
    let world_r = bs[3] * xform.scale;
    let dist = engine::math::max(
        (params.cam_pos - world_c).length() - world_r,
        params.near_z,
    );
    let boost = engine::math::clamp(12.0 / engine::math::max(dist, 1.5), 1.0, 8.0);
    Some(engine::math::min(p_proj * boost, 65535.0))
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
    pending_decode: VecDeque<u32>,
    compressed_cache: HashMap<u32, Vec<u8>>,
    cache_touch: HashMap<u32, u64>,
    cache_lru: VecDeque<(u32, u64)>,
    cache_epoch: u64,
    cache_bytes: usize,
    cache_budget_bytes: usize,
    pending_uploads: Vec<PageUpload>,
    table_dirty: bool,
    residency_version: u64,
    initialized_pages: u32,
    synced_mesh_count: usize,
    prefetch: PrefetchState,
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
            pending_decode: VecDeque::new(),
            compressed_cache: HashMap::default(),
            cache_touch: HashMap::default(),
            cache_lru: VecDeque::new(),
            cache_epoch: 0,
            cache_bytes: 0,
            cache_budget_bytes: DEFAULT_CACHE_BUDGET_BYTES,
            pending_uploads: Vec::new(),
            table_dirty: true,
            residency_version: 0,
            initialized_pages: 0,
            synced_mesh_count: 0,
            prefetch: PrefetchState::default(),
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
            self.prefetch = PrefetchState::default();
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
        self.pending_decode.clear();
        self.compressed_cache.clear();
        self.cache_touch.clear();
        self.cache_lru.clear();
        self.cache_epoch = 0;
        self.cache_bytes = 0;
        self.pending_uploads.clear();
        self.initialized_pages = world.total_pages;
        self.synced_mesh_count = world.meshes.len();
        self.table_dirty = true;
        self.residency_version = self.residency_version.wrapping_add(1);
        self.prefetch = PrefetchState::default();
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

    fn compact_cache_lru(&mut self) {
        let mut kept = VecDeque::with_capacity(self.compressed_cache.len());
        while let Some((pid, epoch)) = self.cache_lru.pop_front() {
            if self.cache_touch.get(&pid).copied() == Some(epoch) {
                kept.push_back((pid, epoch));
            }
        }
        self.cache_lru = kept;
    }

    fn touch_cache(&mut self, page_id: u32) {
        self.cache_epoch = self.cache_epoch.wrapping_add(1);
        let epoch = self.cache_epoch;
        self.cache_touch.insert(page_id, epoch);
        self.cache_lru.push_back((page_id, epoch));
        let max_queue = self.compressed_cache.len().saturating_mul(4).max(1024);
        if self.cache_lru.len() > max_queue {
            self.compact_cache_lru();
        }
    }

    fn remove_from_cache(&mut self, page_id: u32) {
        self.cache_touch.remove(&page_id);
        if let Some(prev) = self.compressed_cache.remove(&page_id) {
            self.cache_bytes = self.cache_bytes.saturating_sub(prev.len());
        }
    }

    fn evict_compressed_cache(&mut self) {
        if self.cache_bytes <= self.cache_budget_bytes {
            return;
        }
        let mut deferred = Vec::new();
        while self.cache_bytes > self.cache_budget_bytes {
            let Some((pid, epoch)) = self.cache_lru.pop_front() else {
                break;
            };
            if self.cache_touch.get(&pid).copied() != Some(epoch) {
                continue;
            }
            if self.ready_mask.get(pid as usize).copied().unwrap_or(false) {
                deferred.push((pid, epoch));
                continue;
            }
            self.cache_touch.remove(&pid);
            if let Some(bytes) = self.compressed_cache.remove(&pid) {
                self.cache_bytes = self.cache_bytes.saturating_sub(bytes.len());
            }
        }
        for item in deferred.into_iter().rev() {
            self.cache_lru.push_front(item);
        }
    }

    fn insert_compressed_cache(&mut self, page_id: u32, compressed: Vec<u8>) {
        let new_len = compressed.len();
        if let Some(prev) = self.compressed_cache.insert(page_id, compressed) {
            self.cache_bytes = self.cache_bytes.saturating_sub(prev.len());
        }
        self.cache_bytes = self.cache_bytes.saturating_add(new_len);
        self.touch_cache(page_id);
        self.evict_compressed_cache();
    }

    pub fn deliver_compressed_page(
        &mut self,
        global_page_id: u32,
        compressed: Vec<u8>,
        requested: bool,
    ) {
        let idx = global_page_id as usize;
        if requested && let Some(flag) = self.in_flight.get_mut(idx) {
            *flag = false;
        }
        if compressed.is_empty()
            || compressed.len() > self.cache_budget_bytes
            || idx >= self.page_table.len()
            || idx >= self.ready_mask.len()
        {
            return;
        }
        let queue_decode =
            requested && self.page_table[idx] == UNMAPPED_PAGE && !self.ready_mask[idx];
        if queue_decode {
            self.ready_mask[idx] = true;
            self.pending_decode.push_back(global_page_id);
        }
        self.insert_compressed_cache(global_page_id, compressed);
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

    fn retain_pinned_pending(&mut self) {
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
    }

    fn prune_completed_pending(&mut self) {
        let mut retained = VecDeque::with_capacity(self.pending_requests.len());
        while let Some(req) = self.pending_requests.pop_front() {
            let idx = req.global_page_id as usize;
            let mapped =
                self.page_table.get(idx).copied().unwrap_or(UNMAPPED_PAGE) != UNMAPPED_PAGE;
            let ready = self.ready_mask.get(idx).copied().unwrap_or(false);
            if !mapped && !ready {
                retained.push_back(req);
            } else if let Some(flag) = self.in_flight.get_mut(idx) {
                *flag = false;
            }
        }
        self.pending_requests = retained;
    }

    fn mesh_roots_ready(&self, world: &NaniteWorld) -> Vec<bool> {
        world
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
            .collect()
    }

    fn count_outstanding(&self) -> usize {
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
        self.ready_pages
            .len()
            .saturating_add(self.pending_decode.len())
            .saturating_add(in_flight_unmapped)
            .saturating_add(pending_not_flagged)
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
        let mesh_roots_ready = self.mesh_roots_ready(world);

        let mut has_unmapped_fault = false;
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
            if !has_unmapped_fault
                && !self.ready_mask.get(page_id).copied().unwrap_or(false)
                && let Some((mesh_id, _, _)) = world.find_page_owner(page_id as u32)
                && mesh_roots_ready.get(mesh_id as usize).copied().unwrap_or(false)
            {
                has_unmapped_fault = true;
            }
        }

        if !has_unmapped_fault {
            if self.prefetch.prefetch_only {
                self.prune_completed_pending();
                if !self.has_pending_work() {
                    self.prefetch.prefetch_only = false;
                }
            } else {
                self.retain_pinned_pending();
            }
            return;
        }

        self.prefetch.prefetch_only = false;
        self.retain_pinned_pending();

        let mut wanted: Vec<(u32, u32)> = Vec::new();
        for (page_id, &prio) in priorities.iter().enumerate().take(total) {
            if prio == 0 {
                continue;
            }
            if self.page_table.get(page_id).copied().unwrap_or(UNMAPPED_PAGE) != UNMAPPED_PAGE
                || self.in_flight.get(page_id).copied().unwrap_or(false)
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
        let outstanding = self.count_outstanding();
        let slot_headroom = evictable.saturating_sub(outstanding);
        let pipeline_headroom = max_new_requests.saturating_sub(outstanding);
        let mut remaining_slots = max_new_requests.min(slot_headroom).min(pipeline_headroom);
        if remaining_slots == 0 {
            return;
        }
        let mut remaining_queue = 1024_usize.saturating_sub(self.pending_requests.len());
        wanted.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (prio, global_page_id) in wanted {
            if remaining_slots == 0 {
                break;
            }
            let idx = global_page_id as usize;
            if self.compressed_cache.contains_key(&global_page_id) {
                if idx < self.ready_mask.len() && !self.ready_mask[idx] {
                    self.ready_mask[idx] = true;
                    self.pending_decode.push_back(global_page_id);
                    self.touch_cache(global_page_id);
                    remaining_slots -= 1;
                }
                continue;
            }
            if remaining_queue == 0 {
                continue;
            }
            if let Some((mesh_id, mesh, local_page_id)) = world.find_page_owner(global_page_id) {
                self.enqueue_fetch(
                    mesh_id,
                    mesh,
                    local_page_id,
                    global_page_id,
                    prio.min(u32::MAX - 2),
                );
                remaining_slots -= 1;
                remaining_queue -= 1;
            }
        }
    }

    fn collect_prefetch_tier(
        &self,
        world: &NaniteWorld,
        roots_ready: &[bool],
        params: PrefetchParams,
        thresh: f32,
    ) -> Vec<(u32, u32, f32)> {
        let mut scores: HashMap<u32, (u32, f32)> = HashMap::default();
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        for inst in &world.instances {
            let mesh_id = inst.info[3];
            if !roots_ready.get(mesh_id as usize).copied().unwrap_or(false) {
                continue;
            }
            let root_node = inst.info[0];
            if (root_node as usize) >= world.bvh.len() {
                continue;
            }
            let scale = inst.pos_scale[3];
            let xform = InstXform {
                pos: Vec3::new(inst.pos_scale[0], inst.pos_scale[1], inst.pos_scale[2]),
                rot: Quat::from_xyzw(
                    inst.rot_quat[0],
                    inst.rot_quat[1],
                    inst.rot_quat[2],
                    inst.rot_quat[3],
                ),
                scale,
                err_factor: scale * params.lod_scale,
            };
            stack.clear();
            stack.push(root_node);
            while let Some(node_id) = stack.pop() {
                let Some(node) = world.bvh.get(node_id as usize) else {
                    continue;
                };
                if node.info[1] != 0 {
                    let start = node.info[2] as usize;
                    let end = start
                        .saturating_add(node.info[3] as usize)
                        .min(world.clusters.len());
                    for c in &world.clusters[start..end] {
                        let pid = c.page_and_fallback[0];
                        let idx = pid as usize;
                        if idx >= self.page_table.len()
                            || self.page_table[idx] != UNMAPPED_PAGE
                            || self.in_flight.get(idx).copied().unwrap_or(false)
                            || self.ready_mask.get(idx).copied().unwrap_or(false)
                        {
                            continue;
                        }
                        if let Some(score) = eval_prefetch_cluster(c, xform, params, thresh) {
                            scores
                                .entry(pid)
                                .and_modify(|e| {
                                    if score > e.1 {
                                        *e = (mesh_id, score);
                                    }
                                })
                                .or_insert((mesh_id, score));
                        }
                    }
                    continue;
                }
                let child_cnt = (node.info[0] as usize).min(4);
                for k in 0..child_cnt {
                    let child_id = node.child_or_cluster[k];
                    if child_id == NO_BVH_CHILD {
                        continue;
                    }
                    if let Some(child) = world.bvh.get(child_id as usize)
                        && bvh_child_needed(node, k, child, xform, params, thresh)
                    {
                        stack.push(child_id);
                    }
                }
            }
        }
        scores
            .into_iter()
            .map(|(pid, (mesh_id, score))| (pid, mesh_id, score))
            .collect()
    }

    pub fn enqueue_prefetch_360(
        &mut self,
        world: &NaniteWorld,
        cam_pos: Vec3,
        near_z: f32,
        lod_scale: f32,
        lod_thresh: f32,
        max_pages: usize,
    ) -> usize {
        if world.instances.is_empty()
            || max_pages == 0
            || lod_scale <= 0.0
            || self.has_pending_work()
        {
            return 0;
        }
        let cap_pages = (self.budget_pages as usize * 3) / 4;
        let resident = self.resident_count() as usize;
        if resident >= cap_pages {
            return 0;
        }
        let lod_bits = (u64::from(lod_scale.to_bits()) << 32) | u64::from(lod_thresh.to_bits());
        let moved = (cam_pos - self.prefetch.last_pos).length_squared() > 1.0;
        let res_changed = self.residency_version != self.prefetch.last_res_ver;
        let tick_due = self.frame_tick.saturating_sub(self.prefetch.last_tick) >= 8;
        let lod_changed = lod_bits != self.prefetch.last_lod_bits;
        if self.prefetch.has_walked && !moved && !res_changed && !tick_due && !lod_changed {
            return 0;
        }
        self.prefetch.last_pos = cam_pos;
        self.prefetch.last_res_ver = self.residency_version;
        self.prefetch.last_tick = self.frame_tick;
        self.prefetch.last_lod_bits = lod_bits;
        self.prefetch.has_walked = true;

        let roots_ready = self.mesh_roots_ready(world);
        let params = PrefetchParams {
            cam_pos,
            near_z,
            lod_scale,
        };
        let mut candidates =
            self.collect_prefetch_tier(world, &roots_ready, params, lod_thresh * 2.0);
        if candidates.is_empty() {
            candidates = self.collect_prefetch_tier(world, &roots_ready, params, lod_thresh);
        }
        if candidates.is_empty() {
            return 0;
        }
        candidates.sort_unstable_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

        let budget = max_pages.min(cap_pages.saturating_sub(resident));
        let mut queued = 0_usize;
        for (pid, mesh_id, score) in candidates {
            if queued >= budget {
                break;
            }
            let idx = pid as usize;
            if self.compressed_cache.contains_key(&pid) {
                if idx < self.ready_mask.len() && !self.ready_mask[idx] {
                    self.ready_mask[idx] = true;
                    self.pending_decode.push_back(pid);
                    self.touch_cache(pid);
                    queued += 1;
                }
                continue;
            }
            if let Some(mesh) = world.meshes.get(mesh_id as usize) {
                let local_pid = pid.saturating_sub(mesh.page_offset);
                let prio = ((score * 16.0) as u32).clamp(1, 0x0FFF_FFFF);
                self.enqueue_fetch(mesh_id, mesh, local_pid, pid, prio);
                queued += 1;
            }
        }
        if queued > 0 {
            self.prefetch.prefetch_only = true;
        }
        queued
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn pump_native_io(&mut self, max_pages: usize) {
        for req in self.take_fetch_requests(max_pages) {
            let mut raw = vec![0_u8; req.byte_len as usize];
            let ok = self
                .open_files
                .get_mut(req.mesh_id as usize)
                .and_then(Option::as_mut)
                .is_some_and(|file| {
                    file.seek(SeekFrom::Start(req.byte_offset)).is_ok()
                        && file.read_exact(&mut raw).is_ok()
                });
            if ok {
                self.deliver_compressed_page(req.global_page_id, raw, true);
            } else {
                self.fail_page(req.global_page_id);
            }
        }
    }

    fn install_page_in_slot(
        &mut self,
        global_page_id: u32,
        slot: u32,
        pinned: bool,
        words: Vec<u32>,
    ) {
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

    pub fn commit_ready_pages(&mut self, max_uploads: usize) {
        let mut uploaded = 0_usize;
        while uploaded < max_uploads {
            if let Some((global_page_id, words)) = self.ready_pages.pop_front() {
                let idx = global_page_id as usize;
                if let Some(flag) = self.ready_mask.get_mut(idx) {
                    *flag = false;
                }
                if idx >= self.page_table.len() || self.page_table[idx] != UNMAPPED_PAGE {
                    continue;
                }
                let pinned = self.pinned_mask.get(idx).copied().unwrap_or(false);
                let Some(slot) = self.pick_slot(pinned) else {
                    continue;
                };
                self.install_page_in_slot(global_page_id, slot, pinned, words);
                uploaded += 1;
                continue;
            }
            let Some(global_page_id) = self.pending_decode.pop_front() else {
                break;
            };
            let idx = global_page_id as usize;
            if let Some(flag) = self.ready_mask.get_mut(idx) {
                *flag = false;
            }
            if idx >= self.page_table.len() || self.page_table[idx] != UNMAPPED_PAGE {
                continue;
            }
            let pinned = self.pinned_mask.get(idx).copied().unwrap_or(false);
            let Some(slot) = self.pick_slot(pinned) else {
                continue;
            };
            let decoded = self
                .compressed_cache
                .get(&global_page_id)
                .and_then(|bytes| decompress_page_words(bytes).ok())
                .filter(|w| w.len() == PAGE_WORDS);
            let Some(words) = decoded else {
                self.remove_from_cache(global_page_id);
                self.fail_page(global_page_id);
                continue;
            };
            self.touch_cache(global_page_id);
            self.install_page_in_slot(global_page_id, slot, pinned, words);
            uploaded += 1;
        }
    }

    fn count_evictable_slots(&self) -> usize {
        let limit = (self.budget_pages as usize).min(self.slots.len());
        let stale_cutoff = self.frame_tick.saturating_sub(1);
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
        let stale_cutoff = self.frame_tick.saturating_sub(1);
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
        self.ready_pages
            .len()
            .saturating_add(self.pending_decode.len()) as u32
    }

    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        !self.pending_requests.is_empty()
            || !self.ready_pages.is_empty()
            || !self.pending_decode.is_empty()
            || self.in_flight.iter().any(|&v| v)
    }

    #[must_use]
    pub fn is_prefetch_only(&self) -> bool {
        self.prefetch.prefetch_only
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
