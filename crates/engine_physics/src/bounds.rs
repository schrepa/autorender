use glam::Vec3;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn overlaps(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }

    pub fn expanded(self, margin: f32) -> Self {
        Self {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }

    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn surface_area(self) -> f32 {
        let extent = self.max - self.min;
        2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
    }
}

#[derive(Clone, Copy)]
struct Node {
    bounds: Bounds,
    left: usize,
    right: usize,
    leaf: usize,
}

const NONE: usize = usize::MAX;
const REFIT_LIMIT: f32 = 1.5;

#[derive(Default)]
pub(crate) struct Bvh {
    nodes: Vec<Node>,
    indices: Vec<usize>,
    built_area: f32,
}

impl Bvh {
    pub fn rebuild(&mut self, bounds: &[Bounds]) {
        self.nodes.clear();
        self.indices.clear();
        self.indices.reserve_exact(bounds.len());
        self.indices.extend(0..bounds.len());
        if bounds.is_empty() {
            return;
        }
        self.nodes.reserve_exact(bounds.len() * 2 - 1);
        build(&mut self.nodes, &mut self.indices, bounds);
        self.built_area = self
            .nodes
            .iter()
            .filter(|node| node.left != NONE)
            .map(|node| node.bounds.surface_area())
            .sum();
    }

    pub fn update(&mut self, bounds: &[Bounds], same_leaves: bool) {
        if !same_leaves || bounds.is_empty() || self.nodes.len() != bounds.len() * 2 - 1 {
            self.rebuild(bounds);
            return;
        }
        let mut area = 0.0;
        for index in (0..self.nodes.len()).rev() {
            let node = self.nodes[index];
            self.nodes[index].bounds = if node.left == NONE {
                bounds[node.leaf]
            } else {
                let merged = self.nodes[node.left]
                    .bounds
                    .union(self.nodes[node.right].bounds);
                area += merged.surface_area();
                merged
            };
        }
        if area <= self.built_area * REFIT_LIMIT {
            return;
        }
        self.rebuild(bounds);
    }

    pub fn query<'a>(
        &'a self,
        predicate: impl Fn(Bounds) -> bool + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        Traversal {
            bvh: self,
            stack: [0; 32],
            len: usize::from(!self.nodes.is_empty()),
            predicate,
        }
    }
}

fn build(nodes: &mut Vec<Node>, indices: &mut [usize], bounds: &[Bounds]) -> usize {
    let mut merged = bounds[indices[0]];
    let mut centers = Bounds {
        min: merged.center(),
        max: merged.center(),
    };
    for &index in &indices[1..] {
        merged = merged.union(bounds[index]);
        let center = bounds[index].center();
        centers = centers.union(Bounds {
            min: center,
            max: center,
        });
    }
    let index = nodes.len();
    nodes.push(Node {
        bounds: merged,
        left: NONE,
        right: NONE,
        leaf: indices[0],
    });
    if indices.len() > 1 {
        let extent = centers.max - centers.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        let middle = indices.len() / 2;
        indices.select_nth_unstable_by(middle, |&a, &b| {
            bounds[a].center()[axis]
                .total_cmp(&bounds[b].center()[axis])
                .then(a.cmp(&b))
        });
        let (left, right) = indices.split_at_mut(middle);
        nodes[index].left = build(nodes, left, bounds);
        nodes[index].right = build(nodes, right, bounds);
    }
    index
}

struct Traversal<'a, F> {
    bvh: &'a Bvh,
    stack: [usize; 32],
    len: usize,
    predicate: F,
}

impl<F: Fn(Bounds) -> bool> Iterator for Traversal<'_, F> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        while self.len > 0 {
            self.len -= 1;
            let node = self.bvh.nodes[self.stack[self.len]];
            if !(self.predicate)(node.bounds) {
                continue;
            }
            if node.left == NONE {
                return Some(node.leaf);
            }
            self.stack[self.len] = node.right;
            self.stack[self.len + 1] = node.left;
            self.len += 2;
        }
        None
    }
}
