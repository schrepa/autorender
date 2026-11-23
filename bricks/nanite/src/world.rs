use engine::prelude::*;
use engine_meshlet::{
    BvhNode, ClusterHeader, FLAG_DOUBLE_SIDED, MeshletManifest, NO_BVH_CHILD, PageSpan,
};

use crate::types::{GpuBvhNode, GpuClusterHeader, NaniteInstance};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceHandle {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug)]
struct InstanceSlot {
    dense_idx: Option<usize>,
    generation: u32,
}

#[derive(Clone, Debug)]
pub struct MeshEntry {
    pub name: String,
    pub alive: bool,
    pub source_tri_count: u32,
    pub bvh_root: u32,
    pub bvh_count: u32,
    pub cluster_offset: u32,
    pub cluster_count: u32,
    pub page_offset: u32,
    pub page_count: u32,
    pub root_page_count: u32,
    pub page_spans: Vec<PageSpan>,
    pub bound_sphere: [f32; 4],
    pub shadow_c0_range: [u32; 2],
    pub shadow_c1_range: [u32; 2],
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
    pub compact_epoch: u64,
    pub clear_epoch: u64,
    instance_slots: Vec<InstanceSlot>,
    dense_to_slot: Vec<u32>,
    free_slots: Vec<u32>,
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

const SHADOW_THRESHOLDS: [f32; 23] = [
    0.002_f32, 0.004, 0.008, 0.015, 0.025, 0.04, 0.065, 0.09, 0.12, 0.20, 0.32, 0.50, 0.75, 1.15,
    1.75, 2.6, 3.8, 5.5, 8.0, 12.0, 18.0, 28.0, 48.0,
];

fn count_shadow_cut_thresholds(clusters: &[ClusterHeader]) -> [usize; SHADOW_THRESHOLDS.len()] {
    let mut diff = [0_isize; SHADOW_THRESHOLDS.len() + 1];
    for c in clusters {
        let se = c.self_error();
        let pe = c.parent_error();
        if !se.is_finite() || !pe.is_finite() || se >= pe {
            continue;
        }
        let lo = SHADOW_THRESHOLDS.partition_point(|&t| t < se);
        let hi = SHADOW_THRESHOLDS.partition_point(|&t| t < pe);
        if lo < hi {
            diff[lo] += 1;
            diff[hi] -= 1;
        }
    }
    let mut counts = [0_usize; SHADOW_THRESHOLDS.len()];
    let mut running = 0_isize;
    for (slot, &delta) in counts.iter_mut().zip(diff.iter()) {
        running += delta;
        *slot = running.max(0) as usize;
    }
    counts
}

fn extract_shadow_cut(
    clusters: &[ClusterHeader],
    counts: &[usize; SHADOW_THRESHOLDS.len()],
    cluster_offset: u32,
    max_clusters: usize,
    out: &mut Vec<u32>,
) -> [u32; 2] {
    let start = out.len() as u32;
    let chosen = SHADOW_THRESHOLDS
        .iter()
        .zip(counts.iter())
        .find_map(|(&thresh, &cnt)| (cnt > 0 && cnt <= max_clusters).then_some(thresh));
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

fn compute_world_sphere(bound_sphere: [f32; 4], pos: Vec3, rot: Quat, scale: f32) -> [f32; 4] {
    let local_c = Vec3::new(bound_sphere[0], bound_sphere[1], bound_sphere[2]);
    let world_c = pos + rot * (local_c * scale);
    let world_r = bound_sphere[3] * scale.abs();
    [world_c.x, world_c.y, world_c.z, world_r]
}

fn array4_bits_eq(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0].to_bits() == b[0].to_bits()
        && a[1].to_bits() == b[1].to_bits()
        && a[2].to_bits() == b[2].to_bits()
        && a[3].to_bits() == b[3].to_bits()
}

impl NaniteWorld {
    pub fn clear(&mut self) {
        self.meshes.clear();
        self.clusters.clear();
        self.bvh.clear();
        self.instances.clear();
        self.instance_slots.clear();
        self.dense_to_slot.clear();
        self.free_slots.clear();
        self.shadow_cut_clusters.clear();
        self.shadow_mesh_ranges.clear();
        self.total_pages = 0;
        self.total_virtual_tris = 0;
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
        self.instance_revision = self.instance_revision.wrapping_add(1);
        self.compact_epoch = self.compact_epoch.wrapping_add(1);
        self.clear_epoch = self.clear_epoch.wrapping_add(1);
    }

    pub fn clear_instances(&mut self) {
        if self.instances.is_empty() && self.instance_slots.is_empty() {
            return;
        }
        self.instances.clear();
        self.dense_to_slot.clear();
        self.instance_slots.clear();
        self.free_slots.clear();
        self.total_virtual_tris = 0;
        self.instance_revision = self.instance_revision.wrapping_add(1);
        self.clear_epoch = self.clear_epoch.wrapping_add(1);
    }

    pub fn add_mesh(&mut self, name: impl Into<String>, manifest: &MeshletManifest) -> u32 {
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
        let shadow_counts = count_shadow_cut_thresholds(&manifest.clusters);
        let shadow_c0_range = extract_shadow_cut(
            &manifest.clusters,
            &shadow_counts,
            cluster_offset,
            c0_budget,
            &mut self.shadow_cut_clusters,
        );
        let shadow_c1_range = extract_shadow_cut(
            &manifest.clusters,
            &shadow_counts,
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
            alive: true,
            source_tri_count: manifest.source_tri_count,
            bvh_root: bvh_offset,
            bvh_count: manifest.bvh.len() as u32,
            cluster_offset,
            cluster_count: manifest.clusters.len() as u32,
            page_offset,
            page_count: manifest.page_count,
            root_page_count: manifest.root_page_count,
            page_spans: manifest.page_spans.clone(),
            bound_sphere,
            shadow_c0_range,
            shadow_c1_range,
        });
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
        mesh_id
    }

    pub fn add_instance(
        &mut self,
        mesh_id: u32,
        pos: Vec3,
        rot: Quat,
        scale: f32,
    ) -> Option<InstanceHandle> {
        self.add_instance_custom(mesh_id, pos, rot, scale, 0)
    }

    pub fn add_instance_custom(
        &mut self,
        mesh_id: u32,
        pos: Vec3,
        rot: Quat,
        scale: f32,
        custom_id: u32,
    ) -> Option<InstanceHandle> {
        let mesh = self.meshes.get(mesh_id as usize)?;
        if !mesh.alive {
            return None;
        }
        let world_bs = compute_world_sphere(mesh.bound_sphere, pos, rot, scale);
        let bvh_root = mesh.bvh_root;
        let tris = u64::from(mesh.source_tri_count);
        let dense_idx = self.instances.len();
        self.instances.push(NaniteInstance {
            pos_scale: [pos.x, pos.y, pos.z, scale],
            rot_quat: [rot.x, rot.y, rot.z, rot.w],
            bound_sphere: world_bs,
            info: [bvh_root, custom_id, 0, mesh_id],
        });
        let (slot_idx, generation) = if let Some(free) = self.free_slots.pop() {
            let slot = &mut self.instance_slots[free as usize];
            slot.dense_idx = Some(dense_idx);
            (free, slot.generation)
        } else {
            let idx = self.instance_slots.len() as u32;
            self.instance_slots.push(InstanceSlot {
                dense_idx: Some(dense_idx),
                generation: 1,
            });
            (idx, 1)
        };
        self.dense_to_slot.push(slot_idx);
        self.total_virtual_tris = self.total_virtual_tris.saturating_add(tris);
        self.instance_revision = self.instance_revision.wrapping_add(1);
        Some(InstanceHandle {
            index: slot_idx,
            generation,
        })
    }

    fn resolve_dense(&self, handle: InstanceHandle) -> Option<usize> {
        let slot = self.instance_slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.dense_idx
    }

    pub fn set_instance_transform(
        &mut self,
        handle: InstanceHandle,
        pos: Vec3,
        rot: Quat,
        scale: f32,
    ) -> bool {
        let Some(dense) = self.resolve_dense(handle) else {
            return false;
        };
        let pos_scale = [pos.x, pos.y, pos.z, scale];
        let rot_quat = [rot.x, rot.y, rot.z, rot.w];
        let cur = self.instances[dense];
        if array4_bits_eq(cur.pos_scale, pos_scale) && array4_bits_eq(cur.rot_quat, rot_quat) {
            return true;
        }
        let mesh_id = cur.info[3] as usize;
        let Some(mesh) = self.meshes.get(mesh_id) else {
            return false;
        };
        let world_bs = compute_world_sphere(mesh.bound_sphere, pos, rot, scale);
        self.instances[dense].pos_scale = pos_scale;
        self.instances[dense].rot_quat = rot_quat;
        self.instances[dense].bound_sphere = world_bs;
        self.instance_revision = self.instance_revision.wrapping_add(1);
        true
    }

    pub fn set_instance_custom_id(&mut self, handle: InstanceHandle, custom_id: u32) -> bool {
        let Some(dense) = self.resolve_dense(handle) else {
            return false;
        };
        if self.instances[dense].info[1] != custom_id {
            self.instances[dense].info[1] = custom_id;
            self.instance_revision = self.instance_revision.wrapping_add(1);
        }
        true
    }

    pub fn remove_instance(&mut self, handle: InstanceHandle) -> bool {
        let Some(dense) = self.resolve_dense(handle) else {
            return false;
        };
        let removed = self.instances.swap_remove(dense);
        let _ = self.dense_to_slot.swap_remove(dense);
        if dense < self.dense_to_slot.len() {
            let moved_slot = self.dense_to_slot[dense] as usize;
            if let Some(slot) = self.instance_slots.get_mut(moved_slot) {
                slot.dense_idx = Some(dense);
            }
        }
        if let Some(slot) = self.instance_slots.get_mut(handle.index as usize) {
            slot.dense_idx = None;
            slot.generation = slot.generation.wrapping_add(1).max(1);
            self.free_slots.push(handle.index);
        }
        if let Some(mesh) = self.meshes.get(removed.info[3] as usize) {
            self.total_virtual_tris = self
                .total_virtual_tris
                .saturating_sub(u64::from(mesh.source_tri_count));
        }
        self.instance_revision = self.instance_revision.wrapping_add(1);
        true
    }

    pub fn remove_mesh(&mut self, mesh_id: u32) -> bool {
        let Some(mesh) = self.meshes.get_mut(mesh_id as usize) else {
            return false;
        };
        if !mesh.alive {
            return false;
        }
        mesh.alive = false;
        let to_remove: Vec<InstanceHandle> = self
            .instances
            .iter()
            .enumerate()
            .filter_map(|(dense, inst)| {
                if inst.info[3] != mesh_id {
                    return None;
                }
                let slot_idx = *self.dense_to_slot.get(dense)?;
                let generation = self.instance_slots.get(slot_idx as usize)?.generation;
                Some(InstanceHandle {
                    index: slot_idx,
                    generation,
                })
            })
            .collect();
        for handle in to_remove {
            let _ = self.remove_instance(handle);
        }
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
        true
    }

    pub fn compact(&mut self) {
        if self.meshes.iter().all(|m| m.alive) {
            return;
        }
        let mut new_clusters = Vec::new();
        let mut new_bvh = Vec::new();
        let mut new_shadow_cuts = Vec::new();
        let mut new_shadow_ranges = Vec::with_capacity(self.meshes.len() * 2);
        let mut new_total_pages = 0_u32;

        for mesh in &mut self.meshes {
            if !mesh.alive {
                mesh.bvh_root = 0;
                mesh.bvh_count = 0;
                mesh.cluster_offset = 0;
                mesh.cluster_count = 0;
                mesh.page_offset = 0;
                mesh.page_count = 0;
                mesh.root_page_count = 0;
                mesh.page_spans.clear();
                mesh.shadow_c0_range = [0, 0];
                mesh.shadow_c1_range = [0, 0];
                new_shadow_ranges.push([0, 0]);
                new_shadow_ranges.push([0, 0]);
                continue;
            }
            let old_cl = mesh.cluster_offset;
            let old_bvh = mesh.bvh_root;
            let old_page = mesh.page_offset;
            let new_cl = new_clusters.len() as u32;
            let new_bvh_off = new_bvh.len() as u32;
            let new_page = new_total_pages;

            let cl_start = old_cl as usize;
            let cl_end = cl_start
                .saturating_add(mesh.cluster_count as usize)
                .min(self.clusters.len());
            for &(mut c) in &self.clusters[cl_start..cl_end] {
                c.page_and_fallback[0] = c.page_and_fallback[0]
                    .saturating_sub(old_page)
                    .saturating_add(new_page);
                let fb_count = c.page_and_fallback[2] >> 24;
                if fb_count > 0 {
                    let fb_start = (c.page_and_fallback[2] & 0x00FF_FFFF)
                        .saturating_sub(old_cl)
                        .saturating_add(new_cl);
                    c.page_and_fallback[2] = ClusterHeader::pack_fallback(fb_start, fb_count);
                }
                new_clusters.push(c);
            }

            let bvh_start = old_bvh as usize;
            let bvh_end = bvh_start
                .saturating_add(mesh.bvh_count as usize)
                .min(self.bvh.len());
            for &(mut node) in &self.bvh[bvh_start..bvh_end] {
                if node.info[1] != 0 {
                    let s = node.info[2].saturating_sub(old_cl).saturating_add(new_cl);
                    node.child_or_cluster[0] = s;
                    node.info[2] = s;
                } else {
                    for slot in &mut node.child_or_cluster {
                        if *slot != NO_BVH_CHILD {
                            *slot = slot.saturating_sub(old_bvh).saturating_add(new_bvh_off);
                        }
                    }
                }
                new_bvh.push(node);
            }

            let mut remap_cut = |old_range: [u32; 2]| -> [u32; 2] {
                let dst_start = new_shadow_cuts.len() as u32;
                let src_s = old_range[0] as usize;
                let src_e = src_s
                    .saturating_add(old_range[1] as usize)
                    .min(self.shadow_cut_clusters.len());
                for &cid in &self.shadow_cut_clusters[src_s..src_e] {
                    new_shadow_cuts.push(cid.saturating_sub(old_cl).saturating_add(new_cl));
                }
                [dst_start, (new_shadow_cuts.len() as u32).saturating_sub(dst_start)]
            };
            mesh.shadow_c0_range = remap_cut(mesh.shadow_c0_range);
            mesh.shadow_c1_range = remap_cut(mesh.shadow_c1_range);
            new_shadow_ranges.push(mesh.shadow_c0_range);
            new_shadow_ranges.push(mesh.shadow_c1_range);

            mesh.cluster_offset = new_cl;
            mesh.bvh_root = new_bvh_off;
            mesh.page_offset = new_page;
            new_total_pages = new_total_pages.saturating_add(mesh.page_count);
        }

        self.clusters = new_clusters;
        self.bvh = new_bvh;
        self.shadow_cut_clusters = new_shadow_cuts;
        self.shadow_mesh_ranges = new_shadow_ranges;
        self.total_pages = new_total_pages;

        for inst in &mut self.instances {
            if let Some(mesh) = self.meshes.get(inst.info[3] as usize) {
                inst.info[0] = mesh.bvh_root;
            }
        }
        self.compact_epoch = self.compact_epoch.wrapping_add(1);
        self.geometry_revision = self.geometry_revision.wrapping_add(1);
        self.instance_revision = self.instance_revision.wrapping_add(1);
    }

    pub fn set_mesh_double_sided(&mut self, mesh_id: u32, double_sided: bool) {
        let Some(mesh) = self.meshes.get(mesh_id as usize) else {
            return;
        };
        if !mesh.alive {
            return;
        }
        let start = mesh.cluster_offset as usize;
        let end = start
            .saturating_add(mesh.cluster_count as usize)
            .min(self.clusters.len());
        let bit = FLAG_DOUBLE_SIDED << 16;
        let mut changed = false;
        for c in &mut self.clusters[start..end] {
            let prev = c.page_and_fallback[1];
            let next = if double_sided { prev | bit } else { prev & !bit };
            if next != prev {
                c.page_and_fallback[1] = next;
                changed = true;
            }
        }
        if changed {
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
        }
    }

    pub fn set_mesh_double_sided_materials(&mut self, mesh_id: u32, materials: &[u8]) {
        let Some(mesh) = self.meshes.get(mesh_id as usize) else {
            return;
        };
        if !mesh.alive {
            return;
        }
        let start = mesh.cluster_offset as usize;
        let end = start
            .saturating_add(mesh.cluster_count as usize)
            .min(self.clusters.len());
        let bit = FLAG_DOUBLE_SIDED << 16;
        let mut changed = false;
        for c in &mut self.clusters[start..end] {
            let mat_id = ((c.errors_and_cone[3] >> 24) & 0xFF) as u8;
            if materials.contains(&mat_id) && (c.page_and_fallback[1] & bit) == 0 {
                c.page_and_fallback[1] |= bit;
                changed = true;
            }
        }
        if changed {
            self.geometry_revision = self.geometry_revision.wrapping_add(1);
        }
    }

    #[must_use]
    pub fn find_page_owner(&self, global_page_id: u32) -> Option<(u32, &MeshEntry, u32)> {
        for (idx, mesh) in self.meshes.iter().enumerate() {
            if !mesh.alive {
                continue;
            }
            let end = mesh.page_offset.saturating_add(mesh.page_count);
            if global_page_id >= mesh.page_offset && global_page_id < end {
                return Some((idx as u32, mesh, global_page_id - mesh.page_offset));
            }
        }
        None
    }
}
