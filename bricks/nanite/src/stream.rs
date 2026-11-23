use engine::prelude::*;
use engine_math::collections::HashMap;
use engine_meshlet::{FLAG_DOUBLE_SIDED, NO_BVH_CHILD};

use crate::types::{GpuBvhNode, GpuClusterHeader, UNMAPPED_PAGE};
use crate::world::{NaniteWorld, PageFetchRequest};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoalesceConfig {
    pub max_run_pages: usize,
    pub max_run_bytes: u64,
    pub max_gap_bytes: u64,
}

impl Default for CoalesceConfig {
    fn default() -> Self {
        Self {
            max_run_pages: 24,
            max_run_bytes: 256 * 1024,
            max_gap_bytes: 12 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PageRunItem {
    pub req: Option<PageFetchRequest>,
    pub global_page_id: u32,
    pub rel_offset: usize,
    pub byte_len: usize,
    pub is_requested: bool,
}

#[derive(Clone, Debug)]
pub struct PageRun {
    pub mesh_id: u32,
    pub mesh_name: String,
    pub first_local_page: u32,
    pub last_local_page: u32,
    pub byte_offset: u64,
    pub total_bytes: u64,
    pub max_priority: u32,
    pub items: Vec<PageRunItem>,
}

impl PageRun {
    pub fn into_requested(self, out: &mut Vec<PageFetchRequest>) {
        for item in self.items {
            if item.is_requested
                && let Some(req) = item.req
            {
                out.push(req);
            }
        }
    }
}

#[must_use]
pub fn coalesce_page_runs(
    mut reqs: Vec<PageFetchRequest>,
    world: &NaniteWorld,
    cfg: CoalesceConfig,
) -> Vec<PageRun> {
    reqs.sort_unstable_by(|a, b| {
        a.mesh_id
            .cmp(&b.mesh_id)
            .then_with(|| a.local_page_id.cmp(&b.local_page_id))
    });
    let mut runs: Vec<PageRun> = Vec::new();
    for req in reqs {
        if let Some(last) = runs.last_mut() {
            let expected_end = last.byte_offset.saturating_add(last.total_bytes);
            if last.mesh_id == req.mesh_id
                && req.local_page_id > last.last_local_page
                && req.byte_offset >= expected_end
            {
                let gap = req.byte_offset - expected_end;
                let next_total = req
                    .byte_offset
                    .saturating_add(req.byte_len)
                    .saturating_sub(last.byte_offset);
                let page_span = req
                    .local_page_id
                    .saturating_sub(last.first_local_page)
                    .saturating_add(1) as usize;
                if gap <= cfg.max_gap_bytes
                    && next_total <= cfg.max_run_bytes
                    && page_span <= cfg.max_run_pages
                {
                    if gap > 0
                        && let Some(mesh) = world.meshes.get(req.mesh_id as usize)
                    {
                        for local_pid in (last.last_local_page + 1)..req.local_page_id {
                            if let Some(span) = mesh.page_spans.get(local_pid as usize)
                                && span.byte_len > 0
                                && span.byte_offset >= last.byte_offset
                                && span.byte_offset.saturating_add(u64::from(span.byte_len))
                                    <= req.byte_offset
                            {
                                last.items.push(PageRunItem {
                                    req: None,
                                    global_page_id: mesh.page_offset + local_pid,
                                    rel_offset: (span.byte_offset - last.byte_offset) as usize,
                                    byte_len: span.byte_len as usize,
                                    is_requested: false,
                                });
                            }
                        }
                    }
                    let rel_offset = (req.byte_offset - last.byte_offset) as usize;
                    let global_page_id = req.global_page_id;
                    let byte_len = req.byte_len as usize;
                    last.max_priority = last.max_priority.max(req.priority);
                    last.total_bytes = next_total;
                    last.last_local_page = req.local_page_id;
                    last.items.push(PageRunItem {
                        req: Some(req),
                        global_page_id,
                        rel_offset,
                        byte_len,
                        is_requested: true,
                    });
                    continue;
                }
            }
        }
        let mesh_id = req.mesh_id;
        let mesh_name = req.mesh_name.clone();
        let local_page_id = req.local_page_id;
        let global_page_id = req.global_page_id;
        let byte_offset = req.byte_offset;
        let total_bytes = req.byte_len;
        let max_priority = req.priority;
        runs.push(PageRun {
            mesh_id,
            mesh_name,
            first_local_page: local_page_id,
            last_local_page: local_page_id,
            byte_offset,
            total_bytes,
            max_priority,
            items: vec![PageRunItem {
                req: Some(req),
                global_page_id,
                rel_offset: 0,
                byte_len: total_bytes as usize,
                is_requested: true,
            }],
        });
    }
    runs.sort_unstable_by(|a, b| {
        b.max_priority
            .cmp(&a.max_priority)
            .then_with(|| a.mesh_id.cmp(&b.mesh_id))
            .then_with(|| a.byte_offset.cmp(&b.byte_offset))
    });
    runs
}

#[derive(Clone, Copy, Debug)]
pub struct VisibleCutParams {
    pub cam_pos: Vec3,
    pub near_z: f32,
    pub lod_scale: f32,
    pub lod_thresh: f32,
    pub frustum_planes: [[f32; 4]; 6],
}

#[derive(Clone, Copy)]
pub(crate) struct CutQuery<'a> {
    pub cam_pos: Vec3,
    pub near_z: f32,
    pub lod_scale: f32,
    pub thresh: f32,
    pub frustum_planes: Option<&'a [[f32; 4]; 6]>,
    pub start_mesh_id: u32,
    pub roots_ready: Option<&'a [bool]>,
}

#[derive(Clone, Copy)]
struct InstXform {
    pos: Vec3,
    rot: Quat,
    scale: f32,
    err_factor: f32,
}

fn frustum_visible(planes: &[[f32; 4]; 6], c: Vec3, r: f32) -> bool {
    for p in planes {
        if p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] < -r {
            return false;
        }
    }
    true
}

fn unpack_i8(word: u32, shift: u32) -> i32 {
    let raw = ((word >> shift) & 0xFF) as i32;
    if raw >= 128 { raw - 256 } else { raw }
}

fn cone_culled(packed: u32, rot: Quat, world_c: Vec3, world_r: f32, cam_pos: Vec3) -> bool {
    let cutoff = unpack_i8(packed, 24);
    if cutoff >= 127 {
        return false;
    }
    let to_c = world_c - cam_pos;
    let dist = to_c.length();
    if dist <= world_r {
        return false;
    }
    let ax = Vec3::new(
        unpack_i8(packed, 0) as f32,
        unpack_i8(packed, 8) as f32,
        unpack_i8(packed, 16) as f32,
    );
    let ax_l2 = ax.length_squared();
    if ax_l2 <= 1.0e-4 {
        return false;
    }
    let world_axis = rot * (ax * (1.0 / ax_l2.sqrt()));
    let sin_cone = cutoff as f32 * (1.0 / 127.0);
    to_c.dot(world_axis) >= sin_cone * dist + world_r
}

fn bvh_child_needed(
    node: &GpuBvhNode,
    k: usize,
    child: &GpuBvhNode,
    xform: InstXform,
    query: CutQuery<'_>,
) -> bool {
    let child_c = Vec3::new(child.sphere[0], child.sphere[1], child.sphere[2]);
    let world_c = xform.pos + xform.rot * (child_c * xform.scale);
    let world_r = child.sphere[3] * xform.scale;
    if let Some(planes) = query.frustum_planes
        && !frustum_visible(planes, world_c, world_r)
    {
        return false;
    }
    let max_pe = node.max_parent_error[k];
    let min_se = node.min_self_error[k];
    if max_pe < 1.0e29 || min_se > 0.0 {
        let dist_c = (query.cam_pos - world_c).length();
        if max_pe < 1.0e29
            && max_pe * xform.err_factor
                <= query.thresh * engine::math::max(dist_c - world_r, query.near_z)
        {
            return false;
        }
        if min_se > 0.0
            && min_se * xform.err_factor
                > query.thresh * engine::math::max(dist_c + world_r, query.near_z)
        {
            return false;
        }
    }
    true
}

fn eval_cut_cluster(
    c: &GpuClusterHeader,
    xform: InstXform,
    query: CutQuery<'_>,
) -> Option<f32> {
    let pe = f32::from_bits(c.errors_and_cone[1]);
    let p_proj = if pe < 1.0e29 {
        let pls = c.parent_lod_sphere;
        let pc = xform.pos + xform.rot * (Vec3::new(pls[0], pls[1], pls[2]) * xform.scale);
        let p_dist = engine::math::max(
            (query.cam_pos - pc).length() - pls[3] * xform.scale,
            query.near_z,
        );
        let proj = (pe * xform.err_factor) / p_dist;
        if proj <= query.thresh {
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
            (query.cam_pos - sc).length() - ls[3] * xform.scale,
            query.near_z,
        );
        if se * xform.err_factor > query.thresh * s_dist {
            return None;
        }
    }
    let bs = c.bound_sphere;
    let world_c = xform.pos + xform.rot * (Vec3::new(bs[0], bs[1], bs[2]) * xform.scale);
    let world_r = bs[3] * xform.scale;
    if let Some(planes) = query.frustum_planes {
        if !frustum_visible(planes, world_c, world_r) {
            return None;
        }
        let is_ds = (c.page_and_fallback[1] & (FLAG_DOUBLE_SIDED << 16)) != 0;
        if !is_ds
            && cone_culled(c.errors_and_cone[2], xform.rot, world_c, world_r, query.cam_pos)
        {
            return None;
        }
    }
    let dist = engine::math::max(
        (query.cam_pos - world_c).length() - world_r,
        query.near_z,
    );
    let boost = engine::math::clamp(12.0 / engine::math::max(dist, 1.5), 1.0, 8.0);
    Some(engine::math::min(p_proj * boost, 65535.0))
}

pub(crate) fn collect_cut_pages(
    world: &NaniteWorld,
    page_table: &[u32],
    in_flight: &[bool],
    ready_mask: &[bool],
    query: CutQuery<'_>,
) -> Vec<(u32, u32, f32)> {
    let mut scores: HashMap<u32, (u32, f32)> = HashMap::default();
    let mut stack: Vec<u32> = Vec::with_capacity(64);
    for inst in &world.instances {
        let mesh_id = inst.info[3];
        if mesh_id < query.start_mesh_id {
            continue;
        }
        if let Some(roots) = query.roots_ready
            && !roots.get(mesh_id as usize).copied().unwrap_or(false)
        {
            continue;
        }
        if let Some(planes) = query.frustum_planes {
            let ib = inst.bound_sphere;
            if !frustum_visible(planes, Vec3::new(ib[0], ib[1], ib[2]), ib[3]) {
                continue;
            }
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
            err_factor: scale * query.lod_scale,
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
                    if idx >= page_table.len()
                        || page_table[idx] != UNMAPPED_PAGE
                        || in_flight.get(idx).copied().unwrap_or(false)
                        || ready_mask.get(idx).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    if let Some(score) = eval_cut_cluster(c, xform, query) {
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
                    && bvh_child_needed(node, k, child, xform, query)
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
