use std::ops::Range;

use bevy_ecs::prelude::Entity;
use engine_open_world_kit::ground::{TerrainEntry, TerrainTiles};
use engine_render::{Allocation, Prepare};
use glam::{Mat4, Vec3};

use super::part::{Handles, TerrainGpu, TerrainViews};
use super::{SLOT, SURFACE_SLOTS};
use crate::gpu::{
    SUBDIVISION_ARGS, SUBDIVISION_COUNTED, SUBDIVISION_DRAW, SUBDIVISION_HOLED, SUBDIVISION_WHOLE,
    SURFACE_MESHLET_INDICES, SURFACE_MESHLET_LEVEL, SURFACE_MESHLET_VERTICES, SubdivisionUniform,
};

const LEAF_BYTES: u64 = 24;
const NODE_BYTES: u64 = 4;
const ARGS_BYTES: u64 = SUBDIVISION_ARGS as u64 * 4;
const SHALLOWEST: u32 = SUBDIVISION_COUNTED + 2;
const _: () = assert!(SURFACE_MESHLET_VERTICES <= 1 << 16, "meshlet_fits_u16");

#[derive(Clone, Copy, PartialEq)]
struct Tree {
    owner: Entity,
    depth: u32,
    side: f32,
    room: u32,
}

#[derive(Clone, Copy)]
struct Subdivided {
    slot: usize,
    first: u32,
    finest: f32,
    side: f32,
    watered: bool,
}

#[derive(Default)]
pub(crate) struct SurfaceTrees {
    trees: [Option<Tree>; SURFACE_SLOTS],
    frame: Vec<Option<Subdivided>>,
    uniforms: Vec<SubdivisionUniform>,
    inputs: Vec<u32>,
    fresh: bool,
    targets: [f32; 2],
    unseen: f32,
    leaves_bytes: u64,
    indexed: bool,
    parity: u32,
    shallow: u32,
    dropped: u32,
}

impl SurfaceTrees {
    fn subdivided(&self, index: usize) -> Option<Subdivided> {
        self.frame.get(index).copied().flatten()
    }

    pub(super) fn of(&self, index: usize) -> (u32, f32, f32) {
        self.subdivided(index).map_or((0, 0.0, 0.0), |subdivided| {
            (subdivided.first, subdivided.finest, subdivided.side)
        })
    }

    pub(super) fn water_slot(&self, index: usize) -> Option<u32> {
        self.subdivided(index)
            .filter(|subdivided| subdivided.watered)
            .map(|subdivided| subdivided.slot as u32)
    }

    pub(super) fn any(&self) -> bool {
        self.frame.iter().any(Option::is_some)
    }

    pub(super) fn dropped(&self) -> u32 {
        self.dropped
    }

    pub(super) fn counter_bytes(&self) -> Range<u64> {
        0..SURFACE_SLOTS as u64 * ARGS_BYTES
    }

    pub(super) fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.frame
            .iter()
            .flatten()
            .map(|subdivided| subdivided.slot)
    }

    pub(super) fn args_offset(&self, slot: usize) -> u64 {
        slot as u64 * ARGS_BYTES
    }

    pub(super) fn draw_offset(&self, index: usize) -> Option<u64> {
        self.record_offset(index, SUBDIVISION_DRAW)
    }

    pub(super) fn split_offsets(&self, index: usize) -> Option<[u64; 2]> {
        Some([
            self.record_offset(index, SUBDIVISION_WHOLE)?,
            self.record_offset(index, SUBDIVISION_HOLED)?,
        ])
    }

    fn record_offset(&self, index: usize, record: u32) -> Option<u64> {
        self.subdivided(index)
            .map(|subdivided| subdivided.slot as u64 * ARGS_BYTES + u64::from(record) * 4)
    }

    pub(super) fn rest(&mut self) {
        self.frame.clear();
        self.inputs.clear();
        self.shallow = 0;
    }

    pub(super) fn settle(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (mut inputs, moved): (Vec<u32>, bool),
    ) {
        inputs.extend_from_slice(bytemuck::cast_slice(&self.uniforms));
        if moved || self.fresh || inputs != self.inputs {
            for subdivided in self.frame.iter().flatten() {
                let Some(tree) = self.trees[subdivided.slot] else {
                    continue;
                };
                let (_, buffer) = handles.slots[subdivided.slot];
                cx.write(buffer, state_bytes(tree.depth), &[0_u32]);
            }
        }
        self.inputs = inputs;
    }

    pub(super) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        (tiles, views, room, drawn): (&TerrainTiles, &TerrainViews<'_>, u64, &[bool]),
        subdivided: bool,
    ) {
        if !self.indexed {
            let indices = meshlet_indices();
            let bytes = std::mem::size_of_val(indices.as_slice()) as u64;
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(handles.meshlet, bytes);
            cx.write(handles.meshlet, 0, &indices);
            self.indexed = true;
        }
        self.parity ^= u32::from(subdivided);
        self.shallow = 0;
        self.dropped = 0;
        self.frame.clear();
        self.uniforms.clear();
        let (settings, water) = (views.settings, views.water);
        let count = tiles.terrains.len();
        let share = room / 16 * 16 / LEAF_BYTES / (count as u64).max(1);
        let capacity = share.min(u64::from(settings.subdivision_leaves)) as u32;
        let leaves = count as u64 * u64::from(capacity);
        self.leaves_bytes = self.leaves_bytes.max(leaves * LEAF_BYTES);
        let sizes = [
            (handles.leaves, self.leaves_bytes.max(LEAF_BYTES)),
            (handles.nodes, (leaves * NODE_BYTES).max(NODE_BYTES)),
            (handles.watered, (leaves * NODE_BYTES).max(NODE_BYTES)),
            (handles.counters, SURFACE_SLOTS as u64 * ARGS_BYTES),
            (handles.args, SURFACE_SLOTS as u64 * ARGS_BYTES),
        ];
        self.fresh = false;
        for (buffer, bytes) in sizes {
            self.fresh |= cx.allocate_buffer(buffer, bytes) == Allocation::Fresh;
        }
        self.targets = [settings.subdivision_error, water.subdivision_error];
        self.unseen = settings.subdivision_unseen;
        let planes = world_planes(views.culling, views.eye);
        let depth = settings.subdivision_depth;
        for (index, terrain) in tiles.terrains.iter().enumerate() {
            if !drawn[index] {
                self.frame.push(None);
                continue;
            }
            let leaves = (index as u32 * capacity, capacity);
            let tree = (terrain, index, water.watered.contains(&terrain.terrain));
            self.lay_out(cx, handles, (tree, room), (depth, leaves), planes);
        }
        let held: Vec<Entity> = self
            .frame
            .iter()
            .flatten()
            .filter_map(|subdivided| self.trees[subdivided.slot].map(|tree| tree.owner))
            .collect();
        for tree in &mut self.trees {
            if tree.is_some_and(|tree| !held.contains(&tree.owner)) {
                *tree = None;
            }
        }
    }

    fn slot_of(&mut self, owner: Entity) -> Option<usize> {
        self.trees
            .iter()
            .position(|tree| tree.is_some_and(|tree| tree.owner == owner))
            .or_else(|| {
                let taken: Vec<usize> = self
                    .frame
                    .iter()
                    .flatten()
                    .map(|subdivided| subdivided.slot)
                    .collect();
                (0..SURFACE_SLOTS).find(|slot| self.trees[*slot].is_none() && !taken.contains(slot))
            })
    }

    fn lay_out(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        ((terrain, index, watered), room): ((&TerrainEntry, usize, bool), u64),
        (depth, (first, capacity)): (u32, (u32, u32)),
        planes: [[f32; 4]; 6],
    ) {
        let Some(slot) = self.slot_of(terrain.terrain) else {
            self.dropped += 1;
            self.frame.push(None);
            return;
        };
        let (side, depth) = self.tree(cx, handles, slot, terrain, (depth, capacity, room));
        let uniform = SubdivisionUniform {
            square: [side, self.targets[0], self.targets[1], self.unseen],
            tree: [depth, counts_words(depth), first, capacity],
            frame: [self.parity, slot as u32, index as u32, u32::from(watered)],
            extent: {
                let extent = terrain.pyramid.extent();
                [0.0, 0.0, extent.x, extent.y]
            },
            cutout: terrain
                .cutout
                .map_or([0.0; 4], |(low, high)| [low.x, low.y, high.x, high.y]),
            planes,
        };
        let (buffer, _) = handles.slots[slot];
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, SLOT);
        cx.write(buffer, 0, &[uniform]);
        self.uniforms.push(SubdivisionUniform {
            frame: [0, uniform.frame[1], uniform.frame[2], uniform.frame[3]],
            ..uniform
        });
        self.frame.push(Some(Subdivided {
            slot,
            first,
            finest: finest_spacing(side, depth),
            side,
            watered,
        }));
    }

    fn tree(
        &mut self,
        cx: &mut Prepare<'_, TerrainGpu>,
        handles: &Handles,
        slot: usize,
        terrain: &TerrainEntry,
        (wanted, capacity, room): (u32, u32, u64),
    ) -> (f32, u32) {
        let wanted = wanted.max(SHALLOWEST);
        let mut depth = wanted;
        while depth > SHALLOWEST && tree_bytes(depth) > room {
            depth -= 1;
        }
        self.shallow += u32::from(depth < wanted);
        let pyramid = &terrain.pyramid;
        let [tiles_x, tiles_z] = pyramid.tiles(0);
        let side = pyramid.tile_size(0) * tiles_x.max(tiles_z).next_power_of_two() as f32;
        let owner = terrain.terrain;
        match &mut self.trees[slot] {
            Some(tree)
                if tree.owner == owner
                    && tree.depth == depth
                    && tree.side == side
                    && tree.room <= capacity =>
            {
                tree.room = capacity;
            }
            held => {
                let (_, buffer) = handles.slots[slot];
                let (Allocation::Fresh | Allocation::Kept) =
                    cx.allocate_buffer(buffer, tree_bytes(depth));
                cx.write(buffer, 0, &vec![0_u32; (tree_bytes(depth) / 4) as usize]);
                seed(cx, buffer, depth);
                *held = Some(Tree {
                    owner,
                    depth,
                    side,
                    room: capacity,
                });
            }
        }
        (side, depth)
    }

    pub(super) fn shallow(&self) -> u32 {
        self.shallow
    }
}

fn seed(cx: &mut Prepare<'_, TerrainGpu>, buffer: engine_render::Buffer<TerrainGpu>, depth: u32) {
    let mut word = |at: u64, value: u32| cx.write(buffer, at * 4, &[value]);
    word(1, 2);
    for level in 1..=depth - SUBDIVISION_COUNTED {
        word(1 << level, 1);
        word((1 << level) + (1 << (level - 1)), 1);
    }
    let bitfield = u64::from(counts_words(depth));
    word(bitfield, 1);
    word(bitfield + (1 << (depth - 6)), 1);
}

fn counts_words(depth: u32) -> u32 {
    1 << (depth + 1 - SUBDIVISION_COUNTED)
}

fn state_bytes(depth: u32) -> u64 {
    (u64::from(counts_words(depth)) + (1_u64 << (depth - 5))) * 4
}

fn tree_bytes(depth: u32) -> u64 {
    state_bytes(depth) + 8
}

fn finest_spacing(side: f32, depth: u32) -> f32 {
    let halvings = (depth - 1) / 2;
    let leg = side / (1_u64 << halvings) as f32;
    let leg = if (depth - 1).is_multiple_of(2) {
        leg
    } else {
        leg * std::f32::consts::FRAC_1_SQRT_2
    };
    leg / (1 << SURFACE_MESHLET_LEVEL) as f32
}

pub(crate) fn meshlet_indices() -> Vec<u32> {
    let parts = 1_u32 << SURFACE_MESHLET_LEVEL;
    (0..SURFACE_MESHLET_INDICES / 3)
        .flat_map(|piece| {
            let mut corners = [[parts, 0], [0, parts], [0, 0]];
            for bit in (0..2 * SURFACE_MESHLET_LEVEL).rev() {
                let [a, b, c] = corners;
                let middle = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
                corners = if ((piece >> bit) & 1) == 0 {
                    [c, a, middle]
                } else {
                    [b, c, middle]
                };
            }
            corners.map(|[i, j]| lattice_vertex(i, j))
        })
        .collect()
}

fn lattice_vertex(i: u32, j: u32) -> u32 {
    let row = i + j;
    row * (row + 1) / 2 + j
}

fn world_planes(culling: Mat4, eye: Vec3) -> [[f32; 4]; 6] {
    let mut world = [[0.0; 4]; 6];
    for (plane, held) in engine_core::frustum_planes(culling).iter().zip(&mut world) {
        let normal = plane.truncate().as_dvec3();
        let offset = f64::from(plane.w) - normal.dot(eye.as_dvec3());
        *held = [plane.x, plane.y, plane.z, offset as f32];
    }
    world
}
