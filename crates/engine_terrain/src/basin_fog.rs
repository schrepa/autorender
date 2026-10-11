//! Finite-volume fills of a fixed terrain grid. The lower-star depression
//! tree retains true elevations; equal heights are ordered by cell index.

use engine_math as math;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::{Layers, Pyramid};

const NONE: u32 = u32::MAX;

/// An immutable fixed-level medium, in terrain-local meters.
#[derive(Clone, Debug)]
pub struct BasinGrid {
    pub level: u32,
    pub size: [u32; 2],
    pub spacing: f32,
    pub extent: [f32; 2],
    pub floors: Vec<f32>,
    pub tops: Vec<f32>,
    pub pools: Vec<u32>,
    pub valid: Vec<bool>,
    pub input_m3: f64,
    pub retained_m3: f64,
    pub drained_m3: f64,
    pub published_retained_m3: f64,
}

impl BasinGrid {
    pub fn bytes(&self) -> u64 {
        (self.floors.capacity() + self.tops.capacity() + self.pools.capacity()) as u64 * 4
            + self.valid.capacity() as u64
    }
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    height: f32,
    index: u32,
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.height.to_bits() == other.height.to_bits() && self.index == other.index
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.height
            .total_cmp(&other.height)
            .then(self.index.cmp(&other.index))
    }
}

#[derive(Debug)]
struct Depression {
    children: Vec<u32>,
    cells: Vec<u32>,
    area: f64,
    floor_volume: f64,
    spill: f32,
    input: f64,
    stored: f64,
    overflow: f64,
    id: u32,
}

impl Depression {
    fn capacity(&self) -> f64 {
        (f64::from(self.spill) * self.area - self.floor_volume).max(0.0)
    }
}

/// Cancellable, sliced preparation. Dropping this value cancels all work.
/// A cell is the unit of sampling and topology work; the final tree pass
/// uses explicit stacks, including on deeply nested terrain.
#[derive(Debug)]
pub struct BasinJob {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    budget: f64,
    water: bool,
    sea: Option<f32>,
    grid: BasinGrid,
    areas: Vec<f64>,
    heap: BinaryHeap<Reverse<Cell>>,
    parent: Vec<u32>,
    component: Vec<u32>,
    route: Vec<u32>,
    drains: Vec<bool>,
    nodes: Vec<Depression>,
    roots: Vec<u32>,
    sampled: usize,
    finished: bool,
    direct_drain: f64,
}

impl BasinJob {
    /// Checked workspace bound, including the output and an upload copy.
    /// It is an admission estimate, never a private grid-size limit.
    pub fn requested_bytes(pyramid: &Pyramid, level: u32) -> Option<u64> {
        let level = level.min(pyramid.levels().checked_sub(1)?);
        let size = pyramid
            .base_tiles()
            .map(|tiles| (tiles * crate::TILE_QUADS).div_ceil(1 << level));
        u64::from(size[0])
            .checked_mul(u64::from(size[1]))?
            .checked_mul(1024)
    }

    pub fn new(
        layers: Arc<Layers>,
        pyramid: Pyramid,
        level: u32,
        budget_m: f32,
        water: bool,
        sea: Option<f32>,
    ) -> Option<Self> {
        if !budget_m.is_finite() || budget_m < 0.0 || sea.is_some_and(|v| !v.is_finite()) {
            return None;
        }
        let level = level.min(pyramid.levels().checked_sub(1)?);
        let size = pyramid
            .base_tiles()
            .map(|tiles| (tiles * crate::TILE_QUADS).div_ceil(1 << level));
        let count = size[0].checked_mul(size[1])?;
        if count == NONE {
            return None;
        }
        let count = usize::try_from(count).ok()?;
        let mut floors = Vec::new();
        floors.try_reserve_exact(count).ok()?;
        floors.resize(count, 0.0);
        let grid = BasinGrid {
            level,
            size,
            spacing: pyramid.spacing(level),
            extent: pyramid.extent().to_array(),
            floors,
            tops: allocated(count, 0.0)?,
            pools: allocated(count, 0)?,
            valid: allocated(count, false)?,
            input_m3: 0.0,
            retained_m3: 0.0,
            drained_m3: 0.0,
            published_retained_m3: 0.0,
        };
        Some(Self {
            layers,
            pyramid,
            budget: f64::from(budget_m),
            water,
            sea,
            grid,
            areas: allocated(count, 0.0)?,
            heap: BinaryHeap::new(),
            parent: allocated(count, NONE)?,
            component: allocated(count, NONE)?,
            route: allocated(count, NONE)?,
            drains: allocated(count, false)?,
            nodes: Vec::new(),
            roots: Vec::new(),
            sampled: 0,
            finished: false,
            direct_drain: 0.0,
        })
    }

    pub fn step(&mut self, work: u64) -> bool {
        if self.finished {
            return true;
        }
        for _ in 0..work {
            if self.sampled < self.grid.floors.len() {
                self.sample(self.sampled);
                self.sampled += 1;
            } else if let Some(Reverse(cell)) = self.heap.pop() {
                self.activate(cell.index);
            } else {
                self.finish();
                self.finished = true;
                return true;
            }
        }
        false
    }

    /// Current owned map and solver allocations, excluding shared terrain layers.
    pub fn bytes(&self) -> u64 {
        self.grid.bytes()
            + (self.areas.capacity() * size_of::<f64>()
                + self.heap.capacity() * size_of::<Reverse<Cell>>()
                + (self.parent.capacity()
                    + self.component.capacity()
                    + self.route.capacity()
                    + self.roots.capacity())
                    * 4
                + self.drains.capacity()
                + self.nodes.capacity() * size_of::<Depression>()
                + self
                    .nodes
                    .iter()
                    .map(|n| (n.children.capacity() + n.cells.capacity()) * 4)
                    .sum::<usize>()) as u64
    }

    pub fn into_grid(self) -> Option<BasinGrid> {
        self.finished.then_some(self.grid)
    }

    pub fn grid(&self) -> Option<&BasinGrid> {
        self.finished.then_some(&self.grid)
    }

    /// Fills the retained topology again without sampling terrain or
    /// rebuilding its depression tree. Zero budget is handled by the owner
    /// releasing the job and all its maps.
    pub fn refill(&mut self, budget_m: f32) -> bool {
        if !self.finished || self.budget <= 0.0 || !budget_m.is_finite() || budget_m <= 0.0 {
            return false;
        }
        let budget = f64::from(budget_m);
        let ratio = budget / self.budget;
        self.grid.input_m3 *= ratio;
        self.direct_drain *= ratio;
        for node in &mut self.nodes {
            node.input *= ratio;
            node.stored = 0.0;
            node.overflow = 0.0;
        }
        self.grid.tops.clone_from(&self.grid.floors);
        self.grid.pools.fill(0);
        self.grid.published_retained_m3 = 0.0;
        self.budget = budget;
        self.finish();
        true
    }

    fn sample(&mut self, index: usize) {
        let [nx, nz] = self.grid.size;
        let x = index as u32 % nx;
        let z = index as u32 / nx;
        let spacing = self.grid.spacing;
        let width = f64::from(math::min(spacing, self.grid.extent[0] - x as f32 * spacing));
        let depth = f64::from(math::min(spacing, self.grid.extent[1] - z as f32 * spacing));
        let px = x as f32 * spacing;
        let pz = z as f32 * spacing;
        if self
            .layers
            .hole_at(px + width as f32 * 0.5, pz + depth as f32 * 0.5)
        {
            return;
        }
        let h10 = self.layers.sample_height(
            &self.pyramid,
            self.grid.level,
            [i64::from(x) + 1, i64::from(z)],
        );
        let h01 = self.layers.sample_height(
            &self.pyramid,
            self.grid.level,
            [i64::from(x), i64::from(z) + 1],
        );
        let mut floor = h10 * 0.5 + h01 * 0.5;
        if width < f64::from(spacing) || depth < f64::from(spacing) {
            let h00 = self.layers.sample_height(
                &self.pyramid,
                self.grid.level,
                [i64::from(x), i64::from(z)],
            );
            floor = h00
                + (h10 - h00) * (width as f32 / spacing * 0.5)
                + (h01 - h00) * (depth as f32 / spacing * 0.5);
        }
        if self.water
            && let Some((surface, _)) = self
                .layers
                .hydrology()
                .and_then(|h| h.water_at(px + spacing * 0.5, pz + spacing * 0.5))
        {
            floor = math::max(floor, surface);
        }
        if let Some(sea) = self.sea {
            floor = math::max(floor, sea);
        }
        if !floor.is_finite() || width <= 0.0 || depth <= 0.0 {
            return;
        }
        self.grid.valid[index] = true;
        self.grid.floors[index] = floor;
        self.grid.tops[index] = floor;
        self.areas[index] = width * depth;
        self.grid.input_m3 += self.budget * self.areas[index];
        self.drains[index] = x == 0 || z == 0 || x + 1 == nx || z + 1 == nz;
        self.heap.push(Reverse(Cell {
            height: floor,
            index: index as u32,
        }));
    }

    fn neighbors(&self, index: u32) -> [Option<u32>; 4] {
        let [nx, nz] = self.grid.size;
        let x = index % nx;
        let z = index / nx;
        [
            x.checked_sub(1).map(|_| index - 1),
            (x + 1 < nx).then_some(index + 1),
            z.checked_sub(1).map(|_| index - nx),
            (z + 1 < nz).then_some(index + nx),
        ]
    }

    fn root(&mut self, mut cell: u32) -> u32 {
        let first = cell;
        while self.parent[cell as usize] != cell {
            cell = self.parent[cell as usize];
        }
        let mut next = first;
        while self.parent[next as usize] != cell {
            let parent = self.parent[next as usize];
            self.parent[next as usize] = cell;
            next = parent;
        }
        cell
    }

    fn activate(&mut self, index: u32) {
        let at = index as usize;
        let floor = self.grid.floors[at];
        let mut roots = Vec::with_capacity(4);
        let mut downhill: Option<Cell> = None;
        let mut drains = self.drains[at];
        for neighbor in self.neighbors(index).into_iter().flatten() {
            let n = neighbor as usize;
            if !self.grid.valid[n] {
                drains = true;
            }
            if self.parent[n] == NONE {
                continue;
            }
            let candidate = Cell {
                height: self.grid.floors[n],
                index: neighbor,
            };
            if downhill.is_none_or(|old| candidate < old) {
                downhill = Some(candidate);
            }
            let root = self.root(neighbor);
            drains |= self.drains[root as usize];
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        self.parent[at] = index;
        let mut children: Vec<u32> = roots
            .iter()
            .filter_map(|&r| {
                let node = self.component[r as usize];
                (node != NONE).then_some(node)
            })
            .collect();
        children.sort_unstable_by_key(|&n| self.nodes[n as usize].id);
        let node = if drains {
            for child in children {
                self.nodes[child as usize].spill = floor;
                self.roots.push(child);
            }
            NONE
        } else if children.len() == 1 {
            children[0]
        } else {
            let mut area = 0.0;
            let mut floor_volume = 0.0;
            let mut id = index + 1;
            for &child in &children {
                let child = &mut self.nodes[child as usize];
                child.spill = floor;
                area += child.area;
                floor_volume += child.floor_volume;
                id = id.min(child.id);
            }
            let node = self.nodes.len() as u32;
            self.nodes.push(Depression {
                children,
                cells: Vec::new(),
                area,
                floor_volume,
                spill: floor,
                input: 0.0,
                stored: 0.0,
                overflow: 0.0,
                id,
            });
            node
        };
        if node != NONE {
            let entry = &mut self.nodes[node as usize];
            entry.cells.push(index);
            entry.area += self.areas[at];
            entry.floor_volume += f64::from(floor) * self.areas[at];
        }
        let route = if self.drains[at] {
            NONE
        } else {
            downhill.map_or(node, |cell| self.route[cell.index as usize])
        };
        self.route[at] = route;
        let volume = self.budget * self.areas[at];
        if route == NONE {
            self.direct_drain += volume;
        } else {
            self.nodes[route as usize].input += volume;
        }
        self.component[at] = node;
        self.drains[at] = drains;
        for root in roots {
            self.parent[root as usize] = index;
        }
    }

    fn add(&mut self, root: u32, amount: f64) -> f64 {
        let accepted = amount.min(
            (self.nodes[root as usize].capacity() - self.nodes[root as usize].stored).max(0.0),
        );
        let mut pending = vec![(root, accepted)];
        while let Some((node, amount)) = pending.pop() {
            let at = node as usize;
            self.nodes[at].stored += amount;
            let mut rest = amount;
            for child in self.nodes[at].children.clone() {
                let child_node = &self.nodes[child as usize];
                let part = rest.min((child_node.capacity() - child_node.stored).max(0.0));
                if part > 0.0 {
                    pending.push((child, part));
                    rest -= part;
                }
            }
        }
        amount - accepted
    }

    fn finish(&mut self) {
        for node in 0..self.nodes.len() {
            let children = self.nodes[node].children.clone();
            self.nodes[node].stored = children
                .iter()
                .map(|&c| self.nodes[c as usize].stored)
                .sum();
            let volume = self.nodes[node].input
                + children
                    .iter()
                    .map(|&c| self.nodes[c as usize].overflow)
                    .sum::<f64>();
            self.nodes[node].overflow = self.add(node as u32, volume);
        }
        self.grid.drained_m3 = self.direct_drain
            + self
                .roots
                .iter()
                .map(|&r| self.nodes[r as usize].overflow)
                .sum::<f64>();
        self.grid.retained_m3 = self
            .roots
            .iter()
            .map(|&r| self.nodes[r as usize].stored)
            .sum();
        let mut pending = self.roots.clone();
        while let Some(node) = pending.pop() {
            let node_ref = &self.nodes[node as usize];
            if node_ref.stored <= 0.0 {
                continue;
            }
            let capacity: f64 = node_ref
                .children
                .iter()
                .map(|&c| self.nodes[c as usize].capacity())
                .sum();
            if !node_ref.children.is_empty() && node_ref.stored < capacity {
                pending.extend_from_slice(&node_ref.children);
                continue;
            }
            let mut cells = Vec::new();
            let mut descendants = vec![node];
            while let Some(child) = descendants.pop() {
                let child = &self.nodes[child as usize];
                cells.extend_from_slice(&child.cells);
                descendants.extend_from_slice(&child.children);
            }
            cells.sort_unstable_by(|&a, &b| {
                self.grid.floors[a as usize]
                    .total_cmp(&self.grid.floors[b as usize])
                    .then(a.cmp(&b))
            });
            let mut volume = node_ref.stored;
            let mut area = 0.0;
            let mut height = f64::from(self.grid.floors[cells[0] as usize]);
            for &cell in &cells {
                let next = f64::from(self.grid.floors[cell as usize]);
                let need = (next - height) * area;
                if area > 0.0 && volume <= need {
                    break;
                }
                volume -= need;
                height = next;
                area += self.areas[cell as usize];
            }
            let top = (height + volume / area).min(f64::from(node_ref.spill)) as f32;
            for cell in cells {
                let at = cell as usize;
                let floor = self.grid.floors[at];
                if top > floor {
                    self.grid.tops[at] = top;
                    self.grid.pools[at] = node_ref.id;
                    self.grid.published_retained_m3 +=
                        (f64::from(top) - f64::from(floor)) * self.areas[at];
                }
            }
        }
    }
}

fn allocated<T: Clone>(count: usize, value: T) -> Option<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).ok()?;
    values.resize(count, value);
    Some(values)
}
