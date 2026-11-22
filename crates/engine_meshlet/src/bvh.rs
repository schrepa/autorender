use std::collections::BTreeMap;

use engine_math::max;
use glam::Vec3;

use crate::bounds::merge_spheres;
use crate::format::{BVH_BRANCHING, BvhNode, ClusterHeader, NO_BVH_CHILD, NO_FALLBACK_CLUSTER};

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

fn longest_axis(indices: &[usize], headers: &[ClusterHeader]) -> usize {
    let mut min_c = Vec3::splat(f32::INFINITY);
    let mut max_c = Vec3::splat(f32::NEG_INFINITY);
    for &idx in indices {
        let s = headers[idx].bound_sphere;
        let c = Vec3::new(s[0], s[1], s[2]);
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

fn split_half<'a>(
    indices: &'a mut [usize],
    headers: &[ClusterHeader],
) -> (&'a mut [usize], &'a mut [usize]) {
    let axis = longest_axis(indices, headers);
    indices.sort_unstable_by(|&a, &b| {
        headers[a].bound_sphere[axis]
            .total_cmp(&headers[b].bound_sphere[axis])
            .then_with(|| a.cmp(&b))
    });
    let mid = indices.len() / 2;
    indices.split_at_mut(mid)
}

fn build_tier_tree(
    indices: &mut [usize],
    headers: &[ClusterHeader],
    ordered: &mut Vec<usize>,
) -> BvhTree {
    if indices.len() <= MAX_LEAF_CLUSTERS {
        let start = ordered.len() as u32;
        let count = indices.len() as u32;
        let mut spheres = Vec::with_capacity(indices.len());
        let mut max_pe = 0.0_f32;
        for &idx in indices.iter() {
            ordered.push(idx);
            spheres.push(headers[idx].parent_lod_sphere);
            max_pe = max(max_pe, headers[idx].parent_error());
        }
        return BvhTree::Leaf {
            start,
            count,
            sphere: merge_spheres(&spheres),
            max_parent_error: max_pe,
        };
    }
    let (left, right) = split_half(indices, headers);
    let mut children = Vec::with_capacity(BVH_BRANCHING);
    if left.len() + right.len() > MAX_LEAF_CLUSTERS * 2 {
        let (ll, lr) = split_half(left, headers);
        if !ll.is_empty() {
            children.push(build_tier_tree(ll, headers, ordered));
        }
        if !lr.is_empty() {
            children.push(build_tier_tree(lr, headers, ordered));
        }
        let (rl, rr) = split_half(right, headers);
        if !rl.is_empty() {
            children.push(build_tier_tree(rl, headers, ordered));
        }
        if !rr.is_empty() {
            children.push(build_tier_tree(rr, headers, ordered));
        }
    } else {
        if !left.is_empty() {
            children.push(build_tier_tree(left, headers, ordered));
        }
        if !right.is_empty() {
            children.push(build_tier_tree(right, headers, ordered));
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

pub fn build_bvh_and_order(
    headers: &mut Vec<ClusterHeader>,
    payloads: &mut Vec<Vec<u32>>,
) -> Vec<BvhNode> {
    if headers.is_empty() {
        return Vec::new();
    }
    let mut roots = Vec::new();
    let mut by_lod: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (idx, header) in headers.iter().enumerate() {
        if header.is_root() {
            roots.push(idx);
        } else {
            by_lod.entry(header.lod_level()).or_default().push(idx);
        }
    }
    let mut ordered = Vec::with_capacity(headers.len());
    let mut tier_trees = Vec::new();
    if !roots.is_empty() {
        tier_trees.push(build_tier_tree(&mut roots, headers, &mut ordered));
    }
    for (_lod, mut bucket) in by_lod.into_iter().rev() {
        if !bucket.is_empty() {
            tier_trees.push(build_tier_tree(&mut bucket, headers, &mut ordered));
        }
    }
    let root_tree = combine_trees(tier_trees);
    let mut bvh = Vec::new();
    emit_preorder(&root_tree, &mut bvh);
    let mut old_to_new = vec![0_u32; headers.len()];
    for (new_idx, &old_idx) in ordered.iter().enumerate() {
        old_to_new[old_idx] = new_idx as u32;
    }
    let mut new_headers = Vec::with_capacity(headers.len());
    let mut new_payloads = Vec::with_capacity(payloads.len());
    let old_payloads = std::mem::take(payloads);
    let mut payload_slots: Vec<Option<Vec<u32>>> = old_payloads.into_iter().map(Some).collect();
    for &old_idx in &ordered {
        let mut header = headers[old_idx];
        let fb = header.streaming_meta[1];
        if fb != NO_FALLBACK_CLUSTER {
            header.streaming_meta[1] = old_to_new[fb as usize];
        }
        new_headers.push(header);
        new_payloads.push(payload_slots[old_idx].take().unwrap_or_default());
    }
    *headers = new_headers;
    *payloads = new_payloads;
    bvh
}
