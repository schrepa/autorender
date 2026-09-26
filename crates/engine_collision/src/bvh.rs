use crate::geometry::Bounds;

#[derive(Clone, Copy)]
struct Node {
    bounds: Bounds,
    left: usize,
    right: usize,
    leaf: usize,
}
const NONE: usize = usize::MAX;

/// Balanced flat BVH. Rebuilds only after collider snapshot changes. Median
/// partitions reuse one index vector; traversal uses a fixed stack.
#[derive(Default)]
pub(crate) struct Bvh {
    nodes: Vec<Node>,
    indices: Vec<usize>,
    // Summed internal surface area when last built.
    built_area: f32,
    #[cfg(test)]
    pub(crate) refits: usize,
}

/// Moved leaves refit while the summed internal surface area stays within
/// this factor of the last build; beyond it they rebuild, bounding traversal.
const REFIT_LIMIT: f32 = 1.5;
impl Bvh {
    pub fn rebuild(&mut self, bounds: &[Bounds]) {
        let _profile = engine_profile::span("collision", "bvh_rebuild");
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

    /// Updates the tree for moved leaves. With `same_leaves` (the same
    /// entries in the same order as the last update) it refits every node's
    /// bounds bottom-up instead of rebuilding. Internal bounds stay exact
    /// unions of their children, so traversal visits the same leaves as a
    /// rebuilt tree would; only its internal order differs, which queries
    /// already make irrelevant by sorting or identity tie-breaks.
    pub fn update(&mut self, bounds: &[Bounds], same_leaves: bool) {
        if !same_leaves || bounds.is_empty() || self.nodes.len() != bounds.len() * 2 - 1 {
            self.rebuild(bounds);
            return;
        }
        let _profile = engine_profile::span("collision", "bvh_refit");
        let mut area = 0.0;
        // Pre-order storage puts children after their parent.
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
        // A degraded (or NaN) refit rebuilds instead.
        if area <= self.built_area * REFIT_LIMIT {
            #[cfg(test)]
            {
                self.refits += 1;
            }
            return;
        }
        self.rebuild(bounds);
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<Node>() + self.indices.capacity() * size_of::<usize>()
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

    pub fn query_profiled<'a>(
        &'a self,
        predicate: impl Fn(Bounds) -> bool + 'a,
        timing: &'a mut engine_profile::Aggregate,
    ) -> impl Iterator<Item = usize> + 'a {
        ProfiledTraversal {
            inner: self.query(predicate),
            timing,
        }
    }
}

struct ProfiledTraversal<'a, I> {
    inner: I,
    timing: &'a mut engine_profile::Aggregate,
}
impl<I: Iterator> Iterator for ProfiledTraversal<'_, I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.timing.measure(|| self.inner.next())
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
