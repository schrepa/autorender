use std::cmp::Reverse;
use std::collections::BTreeMap;

use engine_math::max;
use glam::Vec3;

use crate::bounds::merge_spheres;
use crate::format::{BVH_BRANCHING, BvhNode, ClusterHeader, NO_BVH_CHILD};

const MAX_LEAF_CLUSTERS: usize = 8;

enum BvhTree {
    Leaf {
        start: u32,
        count: u32,
        sphere: [f32; 4],
        max_parent_error: f32,
    },
    Internal {
        children: Vec<BvhTree>,
        sphere: [f32; 4],
        max_parent_error: f32,
    },
}

impl BvhTree {
    fn sphere(&self) -> [f32; 4] {
        match self {
            BvhTree::Leaf { sphere, .. } | BvhTree::Internal { sphere, .. } => *sphere,
        }
    }

    fn max_parent_error(&self) -> f32 {
        match self {
            BvhTree::Leaf {
                max_parent_error, ..
            }
            | BvhTree::Internal {
                max_parent_error, ..
            } => *max_parent_error,
        }
    }
}

fn run_centroid(run: &[usize], headers: &[ClusterHeader]) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for &idx in run {
        let s = headers[idx].bound_sphere;
        sum += Vec3::new(s[0], s[1], s[2]);
    }
    sum / (run.len().max(1) as f32)
}

fn longest_run_axis(run_ids: &[usize], centroids: &[Vec3]) -> usize {
    let mut min_c = Vec3::splat(f32::INFINITY);
    let mut max_c = Vec3::splat(f32::NEG_INFINITY);
    for &rid in run_ids {
        let c = centroids[rid];
        min_c = min_c.min(c);
        max_c = max_c.max(c);
    }
    let ext = max_c - min_c;
    if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    }
}

fn split_runs_half<'a>(
    run_ids: &'a mut [usize],
    centroids: &[Vec3],
) -> (&'a mut [usize], &'a mut [usize]) {
    let axis = longest_run_axis(run_ids, centroids);
    run_ids.sort_unstable_by(|&a, &b| {
        centroids[a][axis]
            .total_cmp(&centroids[b][axis])
            .then_with(|| a.cmp(&b))
    });
    let mid = run_ids.len() / 2;
    run_ids.split_at_mut(mid)
}

fn build_tier_tree_runs(
    run_ids: &mut [usize],
    runs: &[Vec<usize>],
    centroids: &[Vec3],
    headers: &[ClusterHeader],
    ordered: &mut Vec<usize>,
    run_ranges: &mut [(u32, u32)],
) -> BvhTree {
    let total_clusters: usize = run_ids.iter().map(|&rid| runs[rid].len()).sum();
    if run_ids.len() <= 1 || total_clusters <= MAX_LEAF_CLUSTERS {
        let leaf_start = ordered.len() as u32;
        let mut spheres = Vec::with_capacity(total_clusters);
        let mut max_pe = 0.0_f32;
        for &rid in run_ids.iter() {
            let r_start = ordered.len() as u32;
            for &idx in &runs[rid] {
                ordered.push(idx);
                spheres.push(headers[idx].parent_lod_sphere);
                max_pe = max(max_pe, headers[idx].parent_error());
            }
            run_ranges[rid] = (r_start, runs[rid].len() as u32);
        }
        return BvhTree::Leaf {
            start: leaf_start,
            count: total_clusters as u32,
            sphere: merge_spheres(&spheres),
            max_parent_error: max_pe,
        };
    }
    let (left, right) = split_runs_half(run_ids, centroids);
    let mut children = Vec::with_capacity(BVH_BRANCHING);
    if left.len() >= 2 && right.len() >= 2 && total_clusters > MAX_LEAF_CLUSTERS * 2 {
        let (ll, lr) = split_runs_half(left, centroids);
        for part in [ll, lr] {
            if !part.is_empty() {
                children.push(build_tier_tree_runs(
                    part, runs, centroids, headers, ordered, run_ranges,
                ));
            }
        }
        let (rl, rr) = split_runs_half(right, centroids);
        for part in [rl, rr] {
            if !part.is_empty() {
                children.push(build_tier_tree_runs(
                    part, runs, centroids, headers, ordered, run_ranges,
                ));
            }
        }
    } else {
        for part in [left, right] {
            if !part.is_empty() {
                children.push(build_tier_tree_runs(
                    part, runs, centroids, headers, ordered, run_ranges,
                ));
            }
        }
    }
    let child_spheres: Vec<[f32; 4]> = children.iter().map(BvhTree::sphere).collect();
    let mut max_pe = 0.0_f32;
    for child in &children {
        max_pe = max(max_pe, child.max_parent_error());
    }
    BvhTree::Internal {
        children,
        sphere: merge_spheres(&child_spheres),
        max_parent_error: max_pe,
    }
}

fn combine_trees(mut trees: Vec<BvhTree>) -> BvhTree {
    while trees.len() > 1 {
        let mut next = Vec::with_capacity(trees.len().div_ceil(BVH_BRANCHING));
        let mut iter = trees.into_iter();
        loop {
            let mut chunk = Vec::with_capacity(BVH_BRANCHING);
            for _ in 0..BVH_BRANCHING {
                match iter.next() {
                    Some(tree) => chunk.push(tree),
                    None => break,
                }
            }
            if chunk.is_empty() {
                break;
            }
            if chunk.len() == 1 {
                next.push(chunk.remove(0));
            } else {
                let spheres: Vec<[f32; 4]> = chunk.iter().map(BvhTree::sphere).collect();
                let mut max_pe = 0.0_f32;
                for c in &chunk {
                    max_pe = max(max_pe, c.max_parent_error());
                }
                next.push(BvhTree::Internal {
                    children: chunk,
                    sphere: merge_spheres(&spheres),
                    max_parent_error: max_pe,
                });
            }
        }
        trees = next;
    }
    trees.remove(0)
}

fn emit_preorder(tree: &BvhTree, out: &mut Vec<BvhNode>) -> u32 {
    let idx = out.len() as u32;
    out.push(BvhNode::default());
    match tree {
        BvhTree::Leaf {
            start,
            count,
            sphere,
            max_parent_error,
        } => {
            out[idx as usize] = BvhNode {
                sphere: *sphere,
                child_or_cluster: [*start, *count, 0, 0],
                max_parent_error: [*max_parent_error, 0.0, 0.0, 0.0],
                meta: [0, 1, *start, *count],
            };
        }
        BvhTree::Internal {
            children, sphere, ..
        } => {
            let mut child_indices = [NO_BVH_CHILD; BVH_BRANCHING];
            let mut child_errors = [0.0_f32; BVH_BRANCHING];
            for (slot, child) in children.iter().take(BVH_BRANCHING).enumerate() {
                child_indices[slot] = emit_preorder(child, out);
                child_errors[slot] = child.max_parent_error();
            }
            out[idx as usize] = BvhNode {
                sphere: *sphere,
                child_or_cluster: child_indices,
                max_parent_error: child_errors,
                meta: [children.len() as u32, 0, 0, 0],
            };
        }
    }
    idx
}

fn compute_run_heights_and_root_closure(
    runs: &[Vec<usize>],
    headers: &[ClusterHeader],
    parent_runs: &[Option<usize>],
) -> (Vec<u32>, Vec<bool>) {
    let n = runs.len();
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut root_closure = vec![false; n];
    for (rid, run) in runs.iter().enumerate() {
        for &ci in run {
            if headers[ci].is_root() || headers[ci].is_pinned_coarse() {
                root_closure[rid] = true;
            }
            if let Some(pr) = parent_runs[ci]
                && pr != rid
                && !deps[rid].contains(&pr)
            {
                deps[rid].push(pr);
                children[pr].push(rid);
            }
        }
    }
    let mut stack: Vec<usize> = (0..n).filter(|&r| root_closure[r]).collect();
    while let Some(r) = stack.pop() {
        for &pr in &deps[r] {
            if !root_closure[pr] {
                root_closure[pr] = true;
                stack.push(pr);
            }
        }
    }
    let mut height = vec![u32::MAX; n];
    fn resolve(r: usize, children: &[Vec<usize>], height: &mut [u32]) -> u32 {
        if height[r] != u32::MAX {
            return height[r];
        }
        let mut h = 0_u32;
        for &ch in &children[r] {
            if ch != r {
                h = h.max(resolve(ch, children, height) + 1);
            }
        }
        height[r] = h;
        h
    }
    for r in 0..n {
        resolve(r, &children, &mut height);
    }
    (height, root_closure)
}

pub fn build_bvh_and_order(
    headers: &mut Vec<ClusterHeader>,
    payloads: &mut Vec<Vec<u32>>,
    runs: &[Vec<usize>],
    parent_runs: &[Option<usize>],
) -> (Vec<BvhNode>, usize) {
    if headers.is_empty() {
        return (Vec::new(), 0);
    }
    let centroids: Vec<Vec3> = runs.iter().map(|r| run_centroid(r, headers)).collect();
    let (heights, root_closure) = compute_run_heights_and_root_closure(runs, headers, parent_runs);
    let mut buckets: BTreeMap<(u8, Reverse<u32>), Vec<usize>> = BTreeMap::new();
    for rid in 0..runs.len() {
        if runs[rid].is_empty() {
            continue;
        }
        let key = if root_closure[rid] {
            (0_u8, Reverse(heights[rid]))
        } else {
            (1_u8, Reverse(heights[rid]))
        };
        buckets.entry(key).or_default().push(rid);
    }
    let mut ordered = Vec::with_capacity(headers.len());
    let mut run_ranges = vec![(0_u32, 0_u32); runs.len()];
    let mut tier_trees = Vec::with_capacity(buckets.len());
    let mut root_prefix_len = 0_usize;
    for (key, mut bucket) in buckets {
        tier_trees.push(build_tier_tree_runs(
            &mut bucket,
            runs,
            &centroids,
            headers,
            &mut ordered,
            &mut run_ranges,
        ));
        if key.0 == 0 {
            root_prefix_len = ordered.len();
        }
    }
    let root_tree = combine_trees(tier_trees);
    let mut bvh = Vec::new();
    emit_preorder(&root_tree, &mut bvh);
    let mut new_headers = Vec::with_capacity(headers.len());
    let mut new_payloads = Vec::with_capacity(payloads.len());
    let old_payloads = std::mem::take(payloads);
    let mut payload_slots: Vec<Option<Vec<u32>>> = old_payloads.into_iter().map(Some).collect();
    for &old_idx in &ordered {
        let mut header = headers[old_idx];
        if let Some(pr) = parent_runs[old_idx] {
            let (start, count) = run_ranges[pr];
            header.set_fallback(start, count);
        } else {
            header.set_fallback(0, 0);
        }
        new_headers.push(header);
        new_payloads.push(payload_slots[old_idx].take().unwrap_or_default());
    }
    *headers = new_headers;
    *payloads = new_payloads;
    (bvh, root_prefix_len)
}
