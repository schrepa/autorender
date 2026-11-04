use std::sync::{Arc, OnceLock};

use bevy_ecs::prelude::{Component, Entity};
use engine_math::max;
use glam::{Mat3, Mat4, Vec3};
use parry3d::shape::{HeightField, HeightFieldCellStatus};
use parry3d::utils::Array2;

use crate::ground::{TILE_QUADS, TILE_SAMPLES, TileKey};
use crate::physics::{ColliderShape, CollisionRay};

pub trait FieldSamples: Send + Sync + 'static {
    fn height(&self, x: i32, z: i32) -> f32;

    fn hole(&self, x: u32, z: u32) -> bool;
}

pub struct FieldTile {
    pub owner: Entity,
    pub key: TileKey,
    pub center: Vec3,
    pub min: Vec3,
    pub max: Vec3,
    spacing: f32,
    samples: Box<dyn FieldSamples>,
    shape: OnceLock<Arc<HeightField>>,
}

impl FieldTile {
    pub fn new(
        owner: Entity,
        key: TileKey,
        origin: Vec3,
        spacing: f32,
        [low, high]: [f32; 2],
        samples: Box<dyn FieldSamples>,
    ) -> Self {
        let size = TILE_QUADS as f32 * spacing;
        let corner = Vec3::new(
            origin.x + f32::from(key.x) * size,
            origin.y,
            origin.z + f32::from(key.z) * size,
        );
        Self {
            owner,
            key,
            center: corner + Vec3::new(size * 0.5, 0.0, size * 0.5),
            min: Vec3::new(corner.x, origin.y + low, corner.z),
            max: Vec3::new(corner.x + size, origin.y + high, corner.z + size),
            spacing,
            samples,
            shape: OnceLock::new(),
        }
    }

    pub fn bounds(&self) -> Bounds {
        Bounds {
            min: self.min,
            max: self.max,
        }
    }

    pub fn shape(&self) -> &Arc<HeightField> {
        self.shape.get_or_init(|| Arc::new(self.build()))
    }

    fn build(&self) -> HeightField {
        let samples = TILE_SAMPLES as usize;
        let mut heights = Vec::with_capacity(samples * samples);
        for x in 0..samples as i32 {
            for z in 0..samples as i32 {
                heights.push(self.samples.height(x, z));
            }
        }
        let size = TILE_QUADS as f32 * self.spacing;
        let mut shape = HeightField::new(
            Array2::new(samples, samples, heights),
            Vec3::new(size, 1.0, size),
        );
        for z in 0..TILE_QUADS {
            for x in 0..TILE_QUADS {
                if self.samples.hole(x, z) {
                    shape.set_cell_status(
                        z as usize,
                        x as usize,
                        HeightFieldCellStatus::CELL_REMOVED,
                    );
                }
            }
        }
        shape
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    pub fn of_shape(shape: ColliderShape, center: Vec3, axes: Mat3) -> Self {
        let half = match shape {
            ColliderShape::Cuboid { .. } => {
                axes.x_axis.abs() + axes.y_axis.abs() + axes.z_axis.abs()
            }
            ColliderShape::Sphere { .. } => {
                let squared = axes.x_axis * axes.x_axis
                    + axes.y_axis * axes.y_axis
                    + axes.z_axis * axes.z_axis;
                Vec3::new(squared.x.sqrt(), squared.y.sqrt(), squared.z.sqrt())
            }
        };
        let margin = 16.0 * f32::EPSILON * max((center.abs() + half).max_element(), 1.0);
        Self {
            min: center - half,
            max: center + half,
        }
        .expanded(margin)
    }

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

    pub fn swept(self, displacement: Vec3) -> Self {
        self.union(Self {
            min: self.min + displacement,
            max: self.max + displacement,
        })
    }

    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn surface_area(self) -> f32 {
        let extent = self.max - self.min;
        2.0 * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x)
    }

    pub fn ray(self, ray: CollisionRay) -> bool {
        let mut near = 0.0_f64;
        let mut far = f64::from(ray.max_distance);
        for axis in 0..3 {
            let origin = f64::from(ray.origin[axis]);
            let direction = f64::from(ray.direction[axis]);
            if direction == 0.0 {
                if origin < f64::from(self.min[axis]) || origin > f64::from(self.max[axis]) {
                    return false;
                }
            } else {
                let a = (f64::from(self.min[axis]) - origin) / direction;
                let b = (f64::from(self.max[axis]) - origin) / direction;
                near = near.max(a.min(b));
                far = far.min(a.max(b));
                if near > far {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Clone, Copy)]
struct Node {
    bounds: Bounds,
    parent: usize,
    left: usize,
    right: usize,
    leaf: usize,
}

const NONE: usize = usize::MAX;
const REFIT_LIMIT: f32 = 1.5;

#[derive(Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    indices: Vec<usize>,
    leaves: Vec<usize>,
    built_area: f32,
    area: f32,
}

impl Bvh {
    pub fn rebuild(&mut self, bounds: &[Bounds]) {
        self.nodes.clear();
        self.indices.clear();
        self.indices.reserve_exact(bounds.len());
        self.indices.extend(0..bounds.len());
        self.leaves.clear();
        self.leaves.resize(bounds.len(), NONE);
        self.built_area = 0.0;
        self.area = 0.0;
        if bounds.is_empty() {
            return;
        }
        self.nodes.reserve_exact(bounds.len() * 2 - 1);
        build(
            &mut self.nodes,
            &mut self.leaves,
            &mut self.indices,
            bounds,
            NONE,
        );
        self.built_area = self
            .nodes
            .iter()
            .filter(|node| node.left != NONE)
            .map(|node| node.bounds.surface_area())
            .sum();
        self.area = self.built_area;
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
        self.area = area;
        if self.degraded() {
            self.rebuild(bounds);
        }
    }

    pub fn move_leaf(&mut self, leaf: usize, bounds: Bounds) {
        let mut index = self.leaves[leaf];
        self.nodes[index].bounds = bounds;
        index = self.nodes[index].parent;
        while index != NONE {
            let node = self.nodes[index];
            let merged = self.nodes[node.left]
                .bounds
                .union(self.nodes[node.right].bounds);
            self.area += merged.surface_area() - node.bounds.surface_area();
            self.nodes[index].bounds = merged;
            index = node.parent;
        }
    }

    pub fn degraded(&self) -> bool {
        self.area > self.built_area * REFIT_LIMIT
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<Node>()
            + (self.indices.capacity() + self.leaves.capacity()) * size_of::<usize>()
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

fn build(
    nodes: &mut Vec<Node>,
    leaves: &mut [usize],
    indices: &mut [usize],
    bounds: &[Bounds],
    parent: usize,
) -> usize {
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
        parent,
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
        nodes[index].left = build(nodes, leaves, left, bounds, index);
        nodes[index].right = build(nodes, leaves, right, bounds, index);
    } else {
        leaves[indices[0]] = index;
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSolid {
    pub owner: Entity,
    pub layer: u16,
    pub asset: u16,
    pub shape: ColliderShape,
    pub model: Mat4,
    pub memberships: u32,
    pub filter: u32,
}

impl WorldSolid {
    fn bounds(&self) -> Bounds {
        let axes = Mat3::from_mat4(self.model) * Mat3::from_diagonal(self.shape.dimensions());
        Bounds::of_shape(self.shape, self.model.w_axis.truncate(), axes)
    }
}

#[derive(Default)]
pub struct SolidTree {
    solids: Vec<WorldSolid>,
    bvh: Bvh,
}

impl SolidTree {
    pub fn new(mut solids: Vec<WorldSolid>) -> Self {
        let mut bounds = Vec::with_capacity(solids.len());
        solids.retain(|solid| {
            let found = solid.bounds();
            let finite = found.min.is_finite() && found.max.is_finite();
            if finite {
                bounds.push(found);
            }
            finite
        });
        let mut bvh = Bvh::default();
        bvh.rebuild(&bounds);
        Self { solids, bvh }
    }

    pub fn solids(&self) -> &[WorldSolid] {
        &self.solids
    }

    pub fn query<'a>(
        &'a self,
        test: impl Fn(Bounds) -> bool + 'a,
    ) -> impl Iterator<Item = &'a WorldSolid> + 'a {
        self.bvh.query(test).map(move |index| &self.solids[index])
    }
}

#[derive(Default)]
pub struct SolidField {
    pub fields: Vec<Arc<FieldTile>>,
    pub solids: Arc<SolidTree>,
}

#[derive(Component, Clone, Default)]
pub struct SolidSource {
    pub field: Arc<SolidField>,
}
