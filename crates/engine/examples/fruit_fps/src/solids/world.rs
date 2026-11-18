use std::ops::Range;

use glam::Vec3;

use super::shapes::{self, Boxed, Swept, Triangle};
use super::tree::{self, Node, Probe};
use super::{Block, Capsule, Dropped, Ground, Hit, Kind, Solid};

const SWEEP_TOLERANCE: f32 = 1e-5;
const BOUNDS_MARGIN: f32 = 1e-4;
const PUSH_PASSES: usize = 4;
const PUSH_EXTRA: f32 = 1e-4;
const OVERLAP_TOLERANCE: f32 = 1e-4;
const TINY: f32 = 1e-12;

#[derive(Clone, Debug)]
struct Placed {
    triangles: Vec<Triangle>,
    low: Vec3,
    high: Vec3,
    shape: Boxed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    Static(usize),
    Block(usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Support {
    pub normal: Vec3,
    pub ground: Ground,
}

pub(super) struct Footing {
    pub bottom: Vec3,
    pub radius: f32,
    pub reach: f32,
    pub cos_limit: f32,
    pub highest: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Work {
    pub sweeps: u32,
    pub triangles: u32,
}

#[derive(Clone, Debug)]
pub struct Solids {
    triangles: Vec<Triangle>,
    sources: Vec<u32>,
    surfaces: Vec<Ground>,
    nodes: Vec<Node>,
    blocks: Vec<Block>,
    placed: Vec<Placed>,
    dropped: Dropped,
}

fn place(block: &Block) -> Placed {
    let axes = [block.axes.x_axis, block.axes.y_axis, block.axes.z_axis];
    let half = block.half_extents.to_array();
    let mut triangles = Vec::with_capacity(12);
    let mut low = Vec3::INFINITY;
    let mut high = Vec3::NEG_INFINITY;
    for face in 0..6 {
        let axis = face / 2;
        let sign = if face % 2 == 0 { 1.0 } else { -1.0 };
        let out = axes[axis] * (sign * half[axis]);
        let u = axes[(axis + 1) % 3] * half[(axis + 1) % 3];
        let v = axes[(axis + 2) % 3] * half[(axis + 2) % 3];
        let middle = block.centre + out;
        let corners = [
            middle - u - v,
            middle + u - v,
            middle + u + v,
            middle - u + v,
        ];
        for corner in corners {
            low = low.min(corner);
            high = high.max(corner);
        }
        for (a, b, c) in [(0, 1, 2), (0, 2, 3)] {
            if let Some(triangle) = Triangle::new(corners[a], corners[b], corners[c]) {
                triangles.push(triangle);
            }
        }
    }
    Placed {
        triangles,
        low,
        high,
        shape: Boxed {
            centre: block.centre,
            half: block.half_extents,
            axes,
        },
    }
}

impl Solids {
    pub fn build(positions: &[Vec3], indices: &[u32], surfaces: &[(Range<u32>, Ground)]) -> Solids {
        let total = indices.len() / 3;
        let mut labels: Vec<Option<Ground>> = vec![None; total];
        for (range, surface) in surfaces {
            let end = (range.end as usize).min(total);
            let start = (range.start as usize).min(end);
            for label in &mut labels[start..end] {
                *label = Some(*surface);
            }
        }
        let mut dropped = Dropped {
            out_of_range: u32::from(!indices.len().is_multiple_of(3)),
            ..Dropped::default()
        };
        let mut found = Vec::with_capacity(total);
        for (k, corner) in indices.as_chunks::<3>().0.iter().enumerate() {
            let fetch = |i: u32| positions.get(i as usize).copied();
            let (Some(a), Some(b), Some(c)) =
                (fetch(corner[0]), fetch(corner[1]), fetch(corner[2]))
            else {
                dropped.out_of_range += 1;
                continue;
            };
            let Some(triangle) = Triangle::new(a, b, c) else {
                dropped.degenerate += 1;
                continue;
            };
            let Some(surface) = labels[k] else {
                dropped.unlabelled += 1;
                continue;
            };
            found.push((triangle, k as u32, surface));
        }
        let bounds: Vec<(Vec3, Vec3)> = found
            .iter()
            .map(|(triangle, _, _)| (triangle.low(), triangle.high()))
            .collect();
        let (nodes, order) = tree::build(&bounds);
        Solids {
            triangles: order.iter().map(|&k| found[k as usize].0).collect(),
            sources: order.iter().map(|&k| found[k as usize].1).collect(),
            surfaces: order.iter().map(|&k| found[k as usize].2).collect(),
            nodes,
            blocks: Vec::new(),
            placed: Vec::new(),
            dropped,
        }
    }

    pub fn dropped(&self) -> Dropped {
        self.dropped
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn relabel(&mut self, from: Kind, to: Kind) {
        for ground in &mut self.surfaces {
            if ground.kind == from {
                ground.kind = to;
            }
        }
        for block in &mut self.blocks {
            if block.kind == from {
                block.kind = to;
            }
        }
    }

    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn add_block(&mut self, block: Block) {
        let placed = place(&block);
        match self.blocks.iter().position(|held| held.id == block.id) {
            Some(index) => {
                self.blocks[index] = block;
                self.placed[index] = placed;
            }
            None => {
                self.blocks.push(block);
                self.placed.push(placed);
            }
        }
    }

    pub fn remove_block(&mut self, id: u32) -> bool {
        let Some(index) = self.blocks.iter().position(|held| held.id == id) else {
            return false;
        };
        self.blocks.remove(index);
        self.placed.remove(index);
        true
    }

    fn triangle(&self, owner: Owner) -> &Triangle {
        match owner {
            Owner::Static(slot) => &self.triangles[slot],
            Owner::Block(index, k) => &self.placed[index].triangles[k],
        }
    }

    fn label(&self, owner: Owner) -> (Ground, Solid) {
        match owner {
            Owner::Static(slot) => (self.surfaces[slot], Solid::Triangle(self.sources[slot])),
            Owner::Block(index, _) => {
                let block = &self.blocks[index];
                (Ground { kind: block.kind }, Solid::Block(block.id))
            }
        }
    }

    fn nearest(
        &self,
        probe: &Probe,
        limit: f32,
        first: bool,
        mut test: impl FnMut(&Triangle, f32) -> Option<(f32, Vec3)>,
    ) -> Option<(f32, Vec3, Owner)> {
        let mut best = None;
        let mut limit = tree::nearest(&self.nodes, probe, limit, first, |slot, limit| {
            let (t, normal) = test(&self.triangles[slot], limit).filter(|(t, _)| *t < limit)?;
            best = Some((t, normal, Owner::Static(slot)));
            Some(t)
        });
        if first && best.is_some() {
            return best;
        }
        for (index, placed) in self.placed.iter().enumerate() {
            let low = placed.low - probe.grow_low;
            let high = placed.high + probe.grow_high;
            if tree::slab(probe.origin, probe.along, low, high, limit).is_none() {
                continue;
            }
            for (k, triangle) in placed.triangles.iter().enumerate() {
                if let Some((t, normal)) = test(triangle, limit).filter(|(t, _)| *t < limit) {
                    limit = t;
                    best = Some((t, normal, Owner::Block(index, k)));
                    if first {
                        return best;
                    }
                }
            }
        }
        best
    }

    fn near(&self, low: Vec3, high: Vec3, mut visit: impl FnMut(&Triangle, Owner)) {
        tree::overlapping(&self.nodes, low, high, |slot| {
            visit(&self.triangles[slot], Owner::Static(slot));
        });
        for (index, placed) in self.placed.iter().enumerate() {
            if placed.low.cmpgt(high).any() || placed.high.cmplt(low).any() {
                continue;
            }
            for (k, triangle) in placed.triangles.iter().enumerate() {
                visit(triangle, Owner::Block(index, k));
            }
        }
    }

    fn cast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max: f32,
        first: bool,
    ) -> Option<(f32, Vec3, Owner)> {
        let length = direction.length();
        if !(length.is_finite() && length > 0.0 && max > 0.0) {
            return None;
        }
        let along = direction / length;
        let probe = Probe {
            origin,
            along,
            grow_low: Vec3::splat(BOUNDS_MARGIN),
            grow_high: Vec3::splat(BOUNDS_MARGIN),
        };
        let limit = max.next_up();
        let (t, _, owner) = self.nearest(&probe, limit, first, |triangle, _| {
            shapes::ray_triangle(origin, along, triangle)
                .filter(|t| *t >= 0.0)
                .map(|t| (t, Vec3::ZERO))
        })?;
        Some((t, along, owner))
    }

    pub fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<Hit> {
        let (t, along, owner) = self.cast(origin, direction, max, false)?;
        let normal = shapes::facing(self.triangle(owner).normal, -along);
        let (ground, solid) = self.label(owner);
        Some(Hit {
            distance: t,
            point: origin + along * t,
            normal,
            face_normal: normal,
            kind: ground.kind,
            solid,
        })
    }

    pub fn blocked(&self, a: Vec3, b: Vec3) -> bool {
        self.cast(a, b - a, a.distance(b), true).is_some()
    }

    pub fn ground_under(&self, point: Vec3, max_drop: f32) -> Option<Hit> {
        self.ray(point, Vec3::NEG_Y, max_drop)
    }

    fn sweep(&self, shape: Swept, motion: Vec3, work: &mut Work) -> Option<Hit> {
        work.sweeps += 1;
        let length = motion.length();
        if !(length.is_finite() && length > 0.0) {
            return None;
        }
        let lower = -SWEEP_TOLERANCE / length;
        let reach = Vec3::splat(shape.radius + BOUNDS_MARGIN);
        let probe = Probe {
            origin: shape.bottom,
            along: motion,
            grow_low: reach + Vec3::Y * shape.lift,
            grow_high: reach,
        };
        let (fraction, normal, owner) =
            self.nearest(&probe, 1.0f32.next_up(), false, |triangle, _| {
                work.triangles += 1;
                shapes::sweep(triangle, &shape, motion, lower)
            })?;
        let at = shape.bottom + motion * fraction;
        let triangle = self.triangle(owner);
        let (_, point) = shapes::segment_triangle(at, at + Vec3::Y * shape.lift, triangle);
        let (ground, solid) = self.label(owner);
        Some(Hit {
            distance: fraction * length,
            point,
            normal,
            face_normal: shapes::facing(triangle.normal, normal),
            kind: ground.kind,
            solid,
        })
    }

    pub fn sphere_sweep(&self, centre: Vec3, radius: f32, motion: Vec3) -> Option<Hit> {
        self.sweep(
            Swept {
                bottom: centre,
                lift: 0.0,
                radius,
            },
            motion,
            &mut Work::default(),
        )
    }

    pub fn capsule_sweep(&self, capsule: Capsule, motion: Vec3) -> Option<Hit> {
        self.sweep(swept(capsule), motion, &mut Work::default())
    }

    pub(super) fn capsule_sweep_counted(
        &self,
        capsule: Capsule,
        motion: Vec3,
        work: &mut Work,
    ) -> Option<Hit> {
        self.sweep(swept(capsule), motion, work)
    }

    pub fn capsule_push_out(&self, capsule: Capsule) -> Vec3 {
        self.capsule_push_out_counted(capsule, &mut Work::default())
    }

    pub(super) fn capsule_push_out_counted(&self, capsule: Capsule, work: &mut Work) -> Vec3 {
        let mut total = Vec3::ZERO;
        let reach = Vec3::splat(capsule.radius + BOUNDS_MARGIN);
        for _ in 0..PUSH_PASSES {
            let (bottom, top) = axis(capsule, total);
            let mut moved = false;
            self.near(bottom - reach, top + reach, |triangle, _| {
                work.triangles += 1;
                let (bottom, top) = axis(capsule, total);
                if let Some((depth, normal)) =
                    shapes::penetration(triangle, bottom, top, capsule.radius)
                {
                    total += normal * (depth + PUSH_EXTRA);
                    moved = true;
                }
            });
            for placed in &self.placed {
                work.triangles += 1;
                let (bottom, top) = axis(capsule, total);
                if let Some((depth, normal)) = placed.shape.penetration(bottom, top, capsule.radius)
                {
                    total += normal * (depth + PUSH_EXTRA);
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        total
    }

    pub fn capsule_overlaps(&self, capsule: Capsule) -> bool {
        self.capsule_overlaps_counted(capsule, &mut Work::default())
    }

    pub(super) fn capsule_overlaps_counted(&self, capsule: Capsule, work: &mut Work) -> bool {
        let (bottom, top) = axis(capsule, Vec3::ZERO);
        let reach = Vec3::splat(capsule.radius + BOUNDS_MARGIN);
        let mut overlaps = false;
        self.near(bottom - reach, top + reach, |triangle, _| {
            work.triangles += 1;
            overlaps = overlaps
                || shapes::penetration(triangle, bottom, top, capsule.radius)
                    .is_some_and(|(depth, _)| depth > OVERLAP_TOLERANCE);
        });
        work.triangles += self.placed.len() as u32;
        overlaps
            || self.placed.iter().any(|placed| {
                placed
                    .shape
                    .penetration(bottom, top, capsule.radius)
                    .is_some_and(|(depth, _)| depth > OVERLAP_TOLERANCE)
            })
    }

    pub(super) fn support(&self, footing: &Footing, work: &mut Work) -> Option<Support> {
        let span = Vec3::splat(footing.radius + footing.reach + BOUNDS_MARGIN);
        let mut best: Option<Support> = None;
        self.near(
            footing.bottom - span,
            footing.bottom + span,
            |triangle, owner| {
                work.triangles += 1;
                let point = shapes::closest_on_triangle(footing.bottom, triangle);
                let gap = footing.bottom - point;
                let distance = gap.length();
                if distance > footing.radius + footing.reach {
                    return;
                }
                let normal = if distance * distance > TINY {
                    gap / distance
                } else {
                    shapes::facing(triangle.normal, Vec3::Y)
                };
                if normal.y <= 0.0 {
                    return;
                }
                let face = shapes::facing(triangle.normal, normal);
                let stands = normal.y >= footing.cos_limit
                    || (face.y >= footing.cos_limit && point.y <= footing.highest);
                if stands && best.is_none_or(|held| normal.y > held.normal.y) {
                    best = Some(Support {
                        normal,
                        ground: self.label(owner).0,
                    });
                }
            },
        );
        best
    }
}

fn swept(capsule: Capsule) -> Swept {
    Swept {
        bottom: capsule.feet + Vec3::Y * capsule.radius,
        lift: shapes::most(capsule.height - 2.0 * capsule.radius, 0.0),
        radius: capsule.radius,
    }
}

fn axis(capsule: Capsule, offset: Vec3) -> (Vec3, Vec3) {
    let shape = swept(capsule);
    let bottom = shape.bottom + offset;
    (bottom, bottom + Vec3::Y * shape.lift)
}
