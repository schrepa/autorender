//! How far a level of detail's surface strays from its mesh's, both
//! ways: the largest distance from a point of either surface to the nearest
//! point of the other, at points sampled over each (every vertex, edge
//! middle and triangle center, and across triangles wider than the
//! sampling step a grid of points). Pruned parts, folded silhouettes and
//! collapsed ridges all show in it, as the error the simplifier estimates
//! from its quadrics need not. Nearest points come from a bounding volume
//! hierarchy over the other surface's triangles. Every step is plain f32
//! arithmetic in a fixed order, so every host measures the same bits.

use engine_math::{max, min};
use glam::Vec3;

/// Triangles a leaf of the hierarchy holds at most.
const LEAF: usize = 4;
/// Grid points across a triangle's longest edge at most.
const MOST_STEPS: u32 = 64;

/// The largest distance between the surfaces of triangles `a` and `b`
/// (indices into `positions`), both ways, sampled every `step` units.
pub(crate) fn surface_distance(positions: &[Vec3], a: &[u32], b: &[u32], step: f32) -> f32 {
    let (towards_b, towards_a) = (Hierarchy::new(positions, b), Hierarchy::new(positions, a));
    let one_way = |from: &[u32], to: &Hierarchy| {
        let mut farthest = 0.0_f32;
        samples(positions, from, step, &mut |point| {
            farthest = max(farthest, to.distance_squared(point));
        });
        farthest
    };
    let farthest = max(one_way(a, &towards_b), one_way(b, &towards_a));
    farthest.sqrt()
}

/// The mean length of `triangles`' edges.
pub(crate) fn mean_edge(positions: &[Vec3], triangles: &[u32]) -> f32 {
    let mut total = 0.0_f64;
    let mut edges = 0_u64;
    for triangle in triangles.as_chunks::<3>().0 {
        let [a, b, c] = triangle.map(|index| positions[index as usize]);
        total += f64::from(a.distance(b)) + f64::from(b.distance(c)) + f64::from(c.distance(a));
        edges += 3;
    }
    if edges == 0 {
        return 0.0;
    }
    (total / edges as f64) as f32
}

/// Calls `visit` with every point sampled over `triangles`: their
/// corners, edge middles and centers, and where a triangle's longest edge
/// is longer than `step`, a grid of points across it that many steps a
/// side (at most [`MOST_STEPS`]).
fn samples(positions: &[Vec3], triangles: &[u32], step: f32, visit: &mut impl FnMut(Vec3)) {
    for triangle in triangles.as_chunks::<3>().0 {
        let [a, b, c] = triangle.map(|index| positions[index as usize]);
        let longest = max(a.distance(b), max(b.distance(c), c.distance(a)));
        let steps = if step > 0.0 && longest > step {
            min((longest / step).ceil(), MOST_STEPS as f32) as u32
        } else {
            2
        };
        let steps = steps.max(2);
        // Barycentric grid points, corners and edge middles among them.
        for i in 0..=steps {
            for j in 0..=steps - i {
                let (u, v) = (i as f32 / steps as f32, j as f32 / steps as f32);
                visit(a + (b - a) * u + (c - a) * v);
            }
        }
        visit((a + b + c) / 3.0);
    }
}

/// A node of the hierarchy: its box, and either its two children or its
/// triangles (a range of the ordered triangles).
#[derive(Clone, Copy)]
struct Node {
    low: Vec3,
    high: Vec3,
    // Leaves: the first triangle and the count; inner nodes: the first
    // child (the second follows it) and a count of 0.
    first: u32,
    count: u32,
}

/// A bounding volume hierarchy over a surface's triangles.
struct Hierarchy {
    nodes: Vec<Node>,
    triangles: Vec<[Vec3; 3]>,
}

impl Hierarchy {
    fn new(positions: &[Vec3], indices: &[u32]) -> Self {
        let mut triangles: Vec<[Vec3; 3]> = indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|triangle| triangle.map(|index| positions[index as usize]))
            .collect();
        let mut nodes = Vec::with_capacity(triangles.len() / LEAF * 2 + 1);
        if !triangles.is_empty() {
            nodes.push(Node {
                low: Vec3::ZERO,
                high: Vec3::ZERO,
                first: 0,
                count: 0,
            });
            let count = triangles.len();
            build(&mut nodes, &mut triangles, 0, (0, count));
        }
        Self { nodes, triangles }
    }

    /// The squared distance from `point` to the nearest of its triangles:
    /// infinite without any.
    fn distance_squared(&self, point: Vec3) -> f32 {
        let mut best = f32::INFINITY;
        if self.nodes.is_empty() {
            return best;
        }
        let mut stack = vec![0_u32];
        while let Some(node) = stack.pop() {
            let node = self.nodes[node as usize];
            if box_distance_squared(point, node.low, node.high) >= best {
                continue;
            }
            if node.count > 0 {
                for triangle in
                    &self.triangles[node.first as usize..(node.first + node.count) as usize]
                {
                    let [a, b, c] = *triangle;
                    best = min(best, closest_on_triangle(point, a, b, c).distance_squared(point));
                }
                continue;
            }
            // The nearer child is searched first.
            let (left, right) = (node.first, node.first + 1);
            let near = |child: u32| {
                let child = self.nodes[child as usize];
                box_distance_squared(point, child.low, child.high)
            };
            if near(left) <= near(right) {
                stack.extend([right, left]);
            } else {
                stack.extend([left, right]);
            }
        }
        best
    }
}

/// Fills node `at` over `triangles[range]`: a leaf of at most [`LEAF`], or
/// two children split at the middle of the triangles ordered along the
/// longest side of their centers' box.
fn build(nodes: &mut Vec<Node>, triangles: &mut [[Vec3; 3]], at: usize, range: (usize, usize)) {
    let (first, count) = range;
    let slice = &mut triangles[first..first + count];
    let (mut low, mut high) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let (mut center_low, mut center_high) = (low, high);
    for triangle in slice.iter() {
        for &corner in triangle {
            low = low.min(corner);
            high = high.max(corner);
        }
        let center = (triangle[0] + triangle[1] + triangle[2]) / 3.0;
        center_low = center_low.min(center);
        center_high = center_high.max(center);
    }
    if count <= LEAF {
        nodes[at] = Node {
            low,
            high,
            first: first as u32,
            count: count as u32,
        };
        return;
    }
    let span = center_high - center_low;
    let axis = if span.x >= span.y && span.x >= span.z {
        0
    } else if span.y >= span.z {
        1
    } else {
        2
    };
    // A total order, so every host builds the same hierarchy.
    slice.sort_by(|a, b| {
        let key = |triangle: &[Vec3; 3]| triangle[0][axis] + triangle[1][axis] + triangle[2][axis];
        key(a).total_cmp(&key(b)).then_with(|| {
            let bits = |triangle: &[Vec3; 3]| triangle.map(|corner| corner.to_array().map(f32::to_bits));
            bits(a).cmp(&bits(b))
        })
    });
    let children = nodes.len();
    nodes[at] = Node {
        low,
        high,
        first: children as u32,
        count: 0,
    };
    let empty = Node {
        low: Vec3::ZERO,
        high: Vec3::ZERO,
        first: 0,
        count: 0,
    };
    nodes.extend([empty, empty]);
    let half = count / 2;
    build(nodes, triangles, children, (first, half));
    build(nodes, triangles, children + 1, (first + half, count - half));
}

/// The squared distance from `point` to the box `low` to `high`.
fn box_distance_squared(point: Vec3, low: Vec3, high: Vec3) -> f32 {
    let outside = (low - point).max(point - high).max(Vec3::ZERO);
    outside.dot(outside)
}

/// The point of triangle `a`, `b`, `c` nearest `point` (Ericson, Real-Time
/// Collision Detection, 5.1.5); of its edges where it has no area.
fn closest_on_triangle(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, point - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = point - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let sum = va + vb + vc;
    if sum.is_nan() || sum <= 0.0 {
        // No area: the nearest of its edges.
        let edges = [(a, b), (b, c), (c, a)].map(|(from, to)| closest_on_segment(point, from, to));
        return edges
            .into_iter()
            .fold((edges[0], f32::INFINITY), |(best, distance), candidate| {
                let candidate_distance = candidate.distance_squared(point);
                if candidate_distance < distance {
                    (candidate, candidate_distance)
                } else {
                    (best, distance)
                }
            })
            .0;
    }
    let denominator = 1.0 / sum;
    a + ab * (vb * denominator) + ac * (vc * denominator)
}

/// The point of the segment `from` to `to` nearest `point`.
fn closest_on_segment(point: Vec3, from: Vec3, to: Vec3) -> Vec3 {
    let along = to - from;
    let length = along.dot(along);
    if length.is_nan() || length <= 0.0 {
        return from;
    }
    let t = engine_math::clamp((point - from).dot(along) / length, 0.0, 1.0);
    from + along * t
}
