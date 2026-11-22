use engine_math::collections::{FixedState, HashMap};
use engine_math::max;
use glam::Vec3;

use crate::bounds::{aabb_of_points, morton_3d};

pub struct GroupInput<'a> {
    pub pos_ids: &'a [u32],
    pub center: Vec3,
    pub radius: f32,
}

struct ClusterAdjacency {
    adj: Vec<Vec<(usize, u32)>>,
    total_boundary: Vec<u32>,
    morton_order: Vec<usize>,
    morton_rank: Vec<usize>,
}

fn build_adjacency(inputs: &[GroupInput<'_>]) -> ClusterAdjacency {
    let n = inputs.len();
    let mut pos_to_clusters: HashMap<u32, Vec<usize>> =
        HashMap::with_capacity_and_hasher(n * 16, FixedState::default());
    for (ci, input) in inputs.iter().enumerate() {
        let mut uniq = input.pos_ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        for p in uniq {
            pos_to_clusters.entry(p).or_default().push(ci);
        }
    }
    let mut pair_weights: Vec<HashMap<usize, u32>> =
        (0..n).map(|_| HashMap::with_hasher(FixedState::default())).collect();
    let mut total_boundary = vec![0_u32; n];
    for clusters in pos_to_clusters.values() {
        if clusters.len() < 2 {
            continue;
        }
        for (idx_a, &a) in clusters.iter().enumerate() {
            total_boundary[a] += 1;
            for &b in clusters.iter().skip(idx_a + 1) {
                *pair_weights[a].entry(b).or_insert(0) += 1;
                *pair_weights[b].entry(a).or_insert(0) += 1;
            }
        }
    }
    let adj: Vec<Vec<(usize, u32)>> = pair_weights
        .into_iter()
        .map(|map| {
            let mut list: Vec<(usize, u32)> = map.into_iter().collect();
            list.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            list
        })
        .collect();
    let centers: Vec<Vec3> = inputs.iter().map(|i| i.center).collect();
    let (c_min, c_ext) = aabb_of_points(&centers);
    let mut pairs: Vec<(u32, usize)> = centers
        .iter()
        .enumerate()
        .map(|(idx, &c)| (morton_3d(c, c_min, c_ext), idx))
        .collect();
    pairs.sort_unstable();
    let morton_order: Vec<usize> = pairs.into_iter().map(|(_, idx)| idx).collect();
    let mut morton_rank = vec![0_usize; n];
    for (rank, &ci) in morton_order.iter().enumerate() {
        morton_rank[ci] = rank;
    }
    ClusterAdjacency {
        adj,
        total_boundary,
        morton_order,
        morton_rank,
    }
}

fn pick_neighbor_for_group(
    group: &[usize],
    inputs: &[GroupInput<'_>],
    graph: &ClusterAdjacency,
    assigned: &[bool],
) -> Option<usize> {
    let mut shared_map: HashMap<usize, u32> = HashMap::with_hasher(FixedState::default());
    let mut center_sum = Vec3::ZERO;
    let mut radius_sum = 0.0_f32;
    for &c in group {
        center_sum += inputs[c].center;
        radius_sum += inputs[c].radius;
        for &(nb, weight) in &graph.adj[c] {
            if !assigned[nb] {
                *shared_map.entry(nb).or_insert(0) += weight;
            }
        }
    }
    if shared_map.is_empty() {
        return None;
    }
    let g_center = center_sum / (group.len() as f32);
    let g_scale = max(radius_sum / (group.len() as f32), 1.0e-4);
    let mut best_nb = None;
    let mut best_score = f32::NEG_INFINITY;
    let mut candidates: Vec<(usize, u32)> = shared_map.into_iter().collect();
    candidates.sort_unstable_by_key(|&(nb, _)| nb);
    for (nb, shared) in candidates {
        let external = graph.total_boundary[nb].saturating_sub(shared) as f32;
        let dist = (inputs[nb].center - g_center).length() / g_scale;
        let score = (shared as f32) * 4.0 - external * 0.5 - dist * 0.75;
        if score > best_score {
            best_score = score;
            best_nb = Some(nb);
        }
    }
    best_nb
}

fn pick_spatial_fallback(
    seed: usize,
    group_center: Vec3,
    inputs: &[GroupInput<'_>],
    graph: &ClusterAdjacency,
    assigned: &[bool],
) -> Option<usize> {
    let rank = graph.morton_rank[seed];
    let n = graph.morton_order.len();
    let window = 64_usize;
    let start = rank.saturating_sub(window);
    let end = (rank + window + 1).min(n);
    let mut best = None;
    let mut best_d2 = f32::INFINITY;
    for &ci in &graph.morton_order[start..end] {
        if assigned[ci] {
            continue;
        }
        let d2 = (inputs[ci].center - group_center).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(ci);
        }
    }
    if best.is_some() {
        return best;
    }
    for &ci in &graph.morton_order {
        if assigned[ci] {
            continue;
        }
        let d2 = (inputs[ci].center - group_center).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best = Some(ci);
        }
    }
    best
}

fn absorb_orphans(
    groups: &mut Vec<Vec<usize>>,
    inputs: &[GroupInput<'_>],
    graph: &ClusterAdjacency,
    max_group: usize,
) {
    let mut group_of = vec![0_usize; inputs.len()];
    for (gi, g) in groups.iter().enumerate() {
        for &c in g {
            group_of[c] = gi;
        }
    }
    let mut idx = 0_usize;
    while idx < groups.len() {
        if groups[idx].len() != 1 || groups.len() <= 1 {
            idx += 1;
            continue;
        }
        let orphan = groups[idx][0];
        let mut best_target = None;
        let mut best_score = f32::NEG_INFINITY;
        for &(nb, w) in &graph.adj[orphan] {
            let tg = group_of[nb];
            if tg == idx || groups[tg].is_empty() {
                continue;
            }
            let size_bonus = if groups[tg].len() < max_group {
                100.0
            } else {
                0.0
            };
            let score = size_bonus + (w as f32);
            if score > best_score {
                best_score = score;
                best_target = Some(tg);
            }
        }
        let target = best_target.unwrap_or_else(|| {
            let mut near_tg = usize::MAX;
            let mut near_d2 = f32::INFINITY;
            for (tg, g) in groups.iter().enumerate() {
                if tg == idx || g.is_empty() {
                    continue;
                }
                let center = inputs[g[0]].center;
                let d2 = (center - inputs[orphan].center).length_squared();
                let biased = if g.len() < max_group { d2 * 0.25 } else { d2 };
                if biased < near_d2 {
                    near_d2 = biased;
                    near_tg = tg;
                }
            }
            near_tg
        });
        if target < groups.len() {
            groups[idx].clear();
            groups[target].push(orphan);
            group_of[orphan] = target;
            if groups[target].len() > max_group {
                let split_at = (groups[target].len() / 2).max(2);
                let split_off = groups[target].split_off(split_at);
                let new_gi = groups.len();
                for &c in &split_off {
                    group_of[c] = new_gi;
                }
                groups.push(split_off);
            }
        }
        idx += 1;
    }
    groups.retain(|g| !g.is_empty());
}

pub fn group_clusters(inputs: &[GroupInput<'_>], max_group: usize, pass: u32) -> Vec<Vec<usize>> {
    let n = inputs.len();
    let limit = max_group.max(2);
    if n == 0 {
        return Vec::new();
    }
    if n <= limit {
        return vec![(0..n).collect()];
    }
    let graph = build_adjacency(inputs);
    let mix = (pass.wrapping_mul(2_654_435_761)) as usize;
    let mut seed_order: Vec<usize> = (0..n).collect();
    seed_order.sort_unstable_by(|&a, &b| {
        let ra = graph.morton_rank[a].wrapping_add(mix) ^ mix;
        let rb = graph.morton_rank[b].wrapping_add(mix) ^ mix;
        graph.adj[a]
            .len()
            .cmp(&graph.adj[b].len())
            .then_with(|| ra.cmp(&rb))
    });
    let mut assigned = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(n.div_ceil(limit));
    for seed in seed_order {
        if assigned[seed] {
            continue;
        }
        assigned[seed] = true;
        let mut group = vec![seed];
        while group.len() < limit {
            let next = pick_neighbor_for_group(&group, inputs, &graph, &assigned).or_else(|| {
                let mut sum = Vec3::ZERO;
                for &c in &group {
                    sum += inputs[c].center;
                }
                let center = sum / (group.len() as f32);
                pick_spatial_fallback(seed, center, inputs, &graph, &assigned)
            });
            let Some(chosen) = next else {
                break;
            };
            assigned[chosen] = true;
            group.push(chosen);
        }
        groups.push(group);
    }
    absorb_orphans(&mut groups, inputs, &graph, limit);
    groups
}
