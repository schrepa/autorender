use glam::Vec3;

use super::shapes::{least, most};

const LEAF_TRIANGLES: usize = 4;
const BINS: usize = 12;
const MAX_DEPTH: u32 = 48;
const STACK: usize = 64;
const TRAVERSAL_COST: f32 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Node {
    pub low: Vec3,
    pub high: Vec3,
    pub first: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Probe {
    pub origin: Vec3,
    pub along: Vec3,
    pub grow_low: Vec3,
    pub grow_high: Vec3,
}

#[derive(Clone, Copy)]
struct Bin {
    count: usize,
    low: Vec3,
    high: Vec3,
}

const EMPTY_BIN: Bin = Bin {
    count: 0,
    low: Vec3::INFINITY,
    high: Vec3::NEG_INFINITY,
};

fn area(low: Vec3, high: Vec3) -> f32 {
    let size = (high - low).max(Vec3::ZERO);
    2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
}

fn span(bounds: &[(Vec3, Vec3)], order: &[u32]) -> (Vec3, Vec3) {
    order
        .iter()
        .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(low, high), &k| {
            let (l, h) = bounds[k as usize];
            (low.min(l), high.max(h))
        })
}

fn centre(bounds: &[(Vec3, Vec3)], k: u32) -> Vec3 {
    let (low, high) = bounds[k as usize];
    (low + high) * 0.5
}

fn bin_of(value: f32, low: f32, scale: f32) -> usize {
    (((value - low) * scale) as usize).min(BINS - 1)
}

fn split(bounds: &[(Vec3, Vec3)], order: &mut [u32], node_area: f32) -> Option<usize> {
    let (centre_low, centre_high) =
        order
            .iter()
            .fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(low, high), &k| {
                let c = centre(bounds, k);
                (low.min(c), high.max(c))
            });
    let mut best: Option<(f32, usize, usize)> = None;
    for axis in 0..3 {
        let extent = centre_high[axis] - centre_low[axis];
        if extent <= 0.0 {
            continue;
        }
        let scale = BINS as f32 / extent;
        let mut bins = [EMPTY_BIN; BINS];
        for &k in order.iter() {
            let bin = &mut bins[bin_of(centre(bounds, k)[axis], centre_low[axis], scale)];
            let (low, high) = bounds[k as usize];
            bin.count += 1;
            bin.low = bin.low.min(low);
            bin.high = bin.high.max(high);
        }
        let mut left_cost = [0.0f32; BINS];
        let mut running = EMPTY_BIN;
        for (i, bin) in bins.iter().enumerate().take(BINS - 1) {
            running = merge(running, *bin);
            left_cost[i] = running.count as f32 * area(running.low, running.high);
        }
        let mut running = EMPTY_BIN;
        for i in (1..BINS).rev() {
            running = merge(running, bins[i]);
            let left_count: usize = bins[..i].iter().map(|bin| bin.count).sum();
            if left_count == 0 || running.count == 0 {
                continue;
            }
            let cost = left_cost[i - 1] + running.count as f32 * area(running.low, running.high);
            if best.is_none_or(|(held, _, _)| cost < held) {
                best = Some((cost, axis, i));
            }
        }
    }
    let count = order.len();
    let (cost, axis, cut) = best?;
    let split_cost = TRAVERSAL_COST + cost / most(node_area, f32::MIN_POSITIVE);
    if count <= LEAF_TRIANGLES * 2 && split_cost >= count as f32 {
        return None;
    }
    let scale = BINS as f32 / (centre_high[axis] - centre_low[axis]);
    let mut left = 0;
    for i in 0..count {
        if bin_of(centre(bounds, order[i])[axis], centre_low[axis], scale) < cut {
            order.swap(i, left);
            left += 1;
        }
    }
    (left > 0 && left < count).then_some(left)
}

fn merge(a: Bin, b: Bin) -> Bin {
    Bin {
        count: a.count + b.count,
        low: a.low.min(b.low),
        high: a.high.max(b.high),
    }
}

pub(super) fn build(bounds: &[(Vec3, Vec3)]) -> (Vec<Node>, Vec<u32>) {
    let mut order: Vec<u32> = (0..bounds.len() as u32).collect();
    if order.is_empty() {
        return (Vec::new(), order);
    }
    let (low, high) = span(bounds, &order);
    let mut nodes = vec![Node {
        low,
        high,
        first: 0,
        count: order.len() as u32,
    }];
    let mut work = vec![(0usize, 0usize, order.len(), 0u32)];
    while let Some((index, start, end, depth)) = work.pop() {
        let count = end - start;
        if count <= LEAF_TRIANGLES || depth >= MAX_DEPTH {
            continue;
        }
        let node_area = area(nodes[index].low, nodes[index].high);
        let cut = match split(bounds, &mut order[start..end], node_area) {
            Some(cut) => start + cut,
            None if count > LEAF_TRIANGLES * 2 => start + count / 2,
            None => continue,
        };
        let left = nodes.len();
        for (from, to) in [(start, cut), (cut, end)] {
            let (low, high) = span(bounds, &order[from..to]);
            nodes.push(Node {
                low,
                high,
                first: from as u32,
                count: (to - from) as u32,
            });
        }
        nodes[index].first = left as u32;
        nodes[index].count = 0;
        work.push((left + 1, cut, end, depth + 1));
        work.push((left, start, cut, depth + 1));
    }
    (nodes, order)
}

pub(super) fn slab(origin: Vec3, along: Vec3, low: Vec3, high: Vec3, limit: f32) -> Option<f32> {
    let mut near = 0.0f32;
    let mut far = limit;
    for axis in 0..3 {
        let start = origin[axis];
        let step = along[axis];
        if step == 0.0 {
            if start < low[axis] || start > high[axis] {
                return None;
            }
            continue;
        }
        let inverse = 1.0 / step;
        let a = (low[axis] - start) * inverse;
        let b = (high[axis] - start) * inverse;
        near = most(near, least(a, b));
        far = least(far, most(a, b));
        if near > far {
            return None;
        }
    }
    Some(near)
}

fn enter(node: &Node, probe: &Probe, limit: f32) -> Option<f32> {
    slab(
        probe.origin,
        probe.along,
        node.low - probe.grow_low,
        node.high + probe.grow_high,
        limit,
    )
}

pub(super) fn nearest(
    nodes: &[Node],
    probe: &Probe,
    limit: f32,
    first: bool,
    mut leaf: impl FnMut(usize, f32) -> Option<f32>,
) -> f32 {
    let mut limit = limit;
    let Some(root) = nodes.first() else {
        return limit;
    };
    let mut stack = [(0u32, 0.0f32); STACK];
    let mut len = 0;
    if let Some(entry) = enter(root, probe, limit) {
        stack[0] = (0, entry);
        len = 1;
    }
    while len > 0 {
        len -= 1;
        let (index, entry) = stack[len];
        if entry > limit {
            continue;
        }
        let node = nodes[index as usize];
        if node.count > 0 {
            for slot in node.first..node.first + node.count {
                if let Some(found) = leaf(slot as usize, limit) {
                    limit = found;
                    if first {
                        return limit;
                    }
                }
            }
            continue;
        }
        let left = node.first;
        let right = left + 1;
        let near_left = enter(&nodes[left as usize], probe, limit);
        let near_right = enter(&nodes[right as usize], probe, limit);
        let mut push = |child: u32, entry: f32| {
            stack[len] = (child, entry);
            len += 1;
        };
        match (near_left, near_right) {
            (Some(l), Some(r)) if l <= r => {
                push(right, r);
                push(left, l);
            }
            (Some(l), Some(r)) => {
                push(left, l);
                push(right, r);
            }
            (Some(l), None) => push(left, l),
            (None, Some(r)) => push(right, r),
            (None, None) => {}
        }
    }
    limit
}

pub(super) fn overlapping(nodes: &[Node], low: Vec3, high: Vec3, mut visit: impl FnMut(usize)) {
    if nodes.is_empty() {
        return;
    }
    let mut stack = [0u32; STACK];
    let mut len = 1;
    while len > 0 {
        len -= 1;
        let node = nodes[stack[len] as usize];
        if node.low.cmpgt(high).any() || node.high.cmplt(low).any() {
            continue;
        }
        if node.count > 0 {
            for slot in node.first..node.first + node.count {
                visit(slot as usize);
            }
            continue;
        }
        stack[len] = node.first + 1;
        stack[len + 1] = node.first;
        len += 2;
    }
}
