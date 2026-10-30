use engine_core::{
    Cut,
    glam::{Mat4, Vec3, Vec4},
    math,
};

use crate::gpu_layouts::{SHADOW_APRON, SHADOW_REGIONS};

const LOW_SUN: f32 = 0.1;

pub(crate) const LEVEL_TEXELS: u32 = 8192;

pub(crate) const REGION_TEXELS: u32 = 512;

const MARKED_FRAMES: u32 = 8;

const MOVING_FRAMES: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LightBasis {
    pub rows: [Vec3; 3],
}

impl LightBasis {
    pub(crate) fn new(toward_sun: Vec3) -> Self {
        let away = -toward_sun.normalize();
        let reference = if away.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = away.cross(reference).normalize();
        let up = right.cross(away);
        Self {
            rows: [right, up, away],
        }
    }

    pub(crate) fn apply(&self, point: Vec3) -> Vec3 {
        Vec3::new(
            self.rows[0].dot(point),
            self.rows[1].dot(point),
            self.rows[2].dot(point),
        )
    }

    pub(crate) fn sine(&self) -> f32 {
        math::max(-self.rows[2].y, LOW_SUN)
    }

    pub(crate) fn bounds(&self, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
        let center = self.apply((min + max) * 0.5);
        let half = (max - min) * 0.5;
        let extent = Vec3::new(
            self.rows[0].abs().dot(half),
            self.rows[1].abs().dot(half),
            self.rows[2].abs().dot(half),
        );
        (center - extent, center + extent)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MapRegion {
    pub rect: [f32; 4],
    pub depth: [f32; 2],
}

impl MapRegion {
    pub(crate) fn clip(&self, basis: &LightBasis) -> Mat4 {
        let [x0, y0, x1, y1] = self.rect;
        let [z0, z1] = self.depth;
        let sx = 2.0 / (x1 - x0);
        let sy = 2.0 / (y1 - y0);
        let sz = 1.0 / (z1 - z0);
        let row = |axis: Vec3, scale: f32, offset: f32| (axis * scale).extend(-offset * scale);
        Mat4::from_cols(
            row(basis.rows[0], sx, (x0 + x1) * 0.5),
            row(basis.rows[1], sy, (y0 + y1) * 0.5),
            row(basis.rows[2], sz, z0),
            Vec4::W,
        )
        .transpose()
    }
}

pub(crate) fn region_tiles(tile_texels: u32) -> u32 {
    (REGION_TEXELS / tile_texels).max(1)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ClipSettings {
    pub levels: u32,
    pub finest_texel: f32,
    pub tile_texels: u32,
    pub slots: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    slot: Option<u16>,
    drawn: bool,
    dirty: bool,
    marked: u32,
    moving: u32,
}

impl Cell {
    fn marked_at(&self, frame: u32) -> bool {
        self.marked != 0 && self.marked + MARKED_FRAMES > frame
    }

    fn moving_at(&self, frame: u32) -> bool {
        self.moving != 0 && self.moving + MOVING_FRAMES > frame
    }
}

#[derive(Debug)]
struct ClipLevel {
    texel: f32,
    tile: f32,
    origin: [i32; 2],
    depth: f32,
    half_depth: f32,
    cells: Vec<Cell>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Region {
    pub level: usize,
    pub first: [i32; 2],
    pub rect: [f32; 4],
    pub depth: [f32; 2],
    pub texel: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Compose {
    pub slot: u16,
    pub region: u8,
    pub at: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MovingTile {
    pub level: usize,
    pub tile: [i32; 2],
    pub index: u16,
}

#[derive(Debug, Default)]
pub(crate) struct ClipPlan {
    pub regions: Vec<Region>,
    pub composes: Vec<Compose>,
    pub moving: Vec<MovingTile>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ClipStats {
    pub marked: u32,
    pub resident: u32,
    pub drawn: u32,
    pub reclaimed: u32,
    pub moving: u32,
    pub tile_cut: Cut,
    pub moving_cut: Cut,
}

#[derive(Debug, Default)]
pub(crate) struct Clipmap {
    basis: Option<LightBasis>,
    settings: Option<ClipSettings>,
    side: u32,
    levels: Vec<ClipLevel>,
    owners: Vec<Option<(u8, u16)>>,
    free: Vec<u16>,
    table: Vec<u32>,
    changed: Vec<bool>,
    moving: Vec<(usize, usize)>,
    frame: u32,
    epoch: u64,
    stats: ClipStats,
}

fn entry(slot: Option<u16>, moving: Option<u32>) -> u32 {
    slot.map_or(0, |slot| u32::from(slot) + 1) | moving.map_or(0, |slot| (slot + 1) << 16)
}

impl Clipmap {
    pub(crate) fn place(&mut self, basis: LightBasis, eye: Vec3, settings: ClipSettings) -> bool {
        let restart = self.basis != Some(basis) || self.settings != Some(settings);
        if restart {
            self.basis = Some(basis);
            self.settings = Some(settings);
            self.side = (LEVEL_TEXELS / settings.tile_texels).max(1);
            let cells = (self.side * self.side) as usize;
            self.levels.clear();
            self.owners = vec![None; settings.slots as usize];
            self.free = (0..settings.slots as u16).rev().collect();
            self.table = vec![0; settings.levels as usize * cells];
            self.changed = vec![true; settings.levels as usize];
            self.moving.clear();
            self.epoch += 1;
        }
        self.frame += 1;
        let light = basis.apply(eye);
        let rise = basis.rows[1].y.abs();
        let sine = basis.sine();
        let side = self.side as i32;
        let cells = (self.side * self.side) as usize;
        for index in 0..settings.levels as usize {
            let texel = settings.finest_texel * 2.0_f32.powi(index as i32);
            let tile = texel * settings.tile_texels as f32;
            let width = tile * side as f32;
            let target = [
                (light.x / tile).floor() as i32 - side / 2,
                (light.y / tile).floor() as i32 - side / 2,
            ];
            let half_depth = 0.5 * width * ((1.0 + rise) / sine + 1.0);
            let fresh = || ClipLevel {
                texel,
                tile,
                origin: target,
                depth: light.z,
                half_depth,
                cells: vec![Cell::default(); cells],
            };
            match self.levels.get(index) {
                Some(level) if (light.z - level.depth).abs() <= 0.5 * width => {
                    self.scroll(index, target);
                }
                Some(_) => {
                    for cell in 0..cells {
                        self.release(index, cell);
                    }
                    self.levels[index] = fresh();
                }
                None => self.levels.push(fresh()),
            }
        }
        restart
    }

    fn scroll(&mut self, index: usize, target: [i32; 2]) {
        let origin = self.levels[index].origin;
        let shift = [target[0] - origin[0], target[1] - origin[1]];
        if shift.iter().all(|step| step.abs() <= 1) {
            return;
        }
        let side = self.side as i32;
        let entering = |origin: i32, shift: i32| {
            if shift >= 0 {
                origin + side..origin + side + shift.min(side)
            } else {
                origin + shift.max(-side)..origin
            }
        };
        let mask = side - 1;
        for column in entering(origin[0], shift[0]) {
            for row in 0..side {
                self.release(index, (row * side + (column & mask)) as usize);
            }
        }
        for row in entering(origin[1], shift[1]) {
            for column in 0..side {
                self.release(index, ((row & mask) * side + column) as usize);
            }
        }
        self.levels[index].origin = target;
    }

    fn release(&mut self, level: usize, cell: usize) {
        if let Some(slot) = self.vacate(level, cell) {
            self.free.push(slot);
        }
    }

    fn vacate(&mut self, level: usize, cell: usize) -> Option<u16> {
        let held = std::mem::take(&mut self.levels[level].cells[cell]);
        if let Some(slot) = held.slot {
            self.owners[slot as usize] = None;
        }
        self.set_entry(level, cell, 0);
        held.slot
    }

    fn set_entry(&mut self, level: usize, cell: usize, value: u32) {
        let at = level * (self.side * self.side) as usize + cell;
        if self.table[at] != value {
            self.table[at] = value;
            self.changed[level] = true;
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.levels.len()
    }

    pub(crate) fn side(&self) -> u32 {
        self.side
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    pub(crate) fn mask_words(&self) -> usize {
        (self.levels.len() * (self.side * self.side) as usize).div_ceil(32)
    }

    pub(crate) fn stats(&self) -> ClipStats {
        self.stats
    }

    fn marked_cells(&self, words: &[u32]) -> Vec<(usize, usize)> {
        let cells = (self.side * self.side) as usize;
        let mut found = Vec::new();
        for (index, &word) in words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let at = index * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if at / cells < self.levels.len() {
                    found.push((at / cells, at % cells));
                }
            }
        }
        found
    }

    pub(crate) fn merge_marks(&mut self, words: &[u32]) {
        let frame = self.frame;
        for (level, cell) in self.marked_cells(words) {
            self.levels[level].cells[cell].marked = frame;
        }
    }

    pub(crate) fn merge_dirty(&mut self, words: &[u32]) {
        for (level, cell) in self.marked_cells(words) {
            let cell = &mut self.levels[level].cells[cell];
            cell.dirty |= cell.drawn;
        }
    }

    pub(crate) fn merge_moving(&mut self, words: &[u32]) {
        let (frame, side) = (self.frame, self.side as usize);
        for (level, cell) in self.marked_cells(words) {
            let (column, row) = (cell % side, cell / side);
            for dy in [side - 1, 0, 1] {
                for dx in [side - 1, 0, 1] {
                    let beside = (row + dy) % side * side + (column + dx) % side;
                    self.levels[level].cells[beside].moving = frame;
                }
            }
        }
    }

    pub(crate) fn dirty_box(&mut self, min: Vec3, max: Vec3) {
        for level in 0..self.levels.len() {
            self.dirty_level_box(level, min, max);
        }
    }

    pub(crate) fn dirty_level_box(&mut self, level: usize, min: Vec3, max: Vec3) {
        let (Some(basis), Some(clip)) = (self.basis, self.levels.get_mut(level)) else {
            return;
        };
        let (low, high) = basis.bounds(min, max);
        let side = self.side as i32;
        let at = |value: f32| (value / clip.tile).floor() as i32;
        let columns = at(low.x).max(clip.origin[0])..=at(high.x).min(clip.origin[0] + side - 1);
        let rows = at(low.y).max(clip.origin[1])..=at(high.y).min(clip.origin[1] + side - 1);
        for row in rows {
            for column in columns.clone() {
                let cell =
                    &mut clip.cells[((row & (side - 1)) * side + (column & (side - 1))) as usize];
                cell.dirty |= cell.drawn;
            }
        }
    }

    pub(crate) fn dirty_all(&mut self) {
        for cell in self.levels.iter_mut().flat_map(|level| &mut level.cells) {
            cell.dirty |= cell.drawn;
        }
    }

    pub(crate) fn invalidate(&mut self) {
        let cells = (self.side * self.side) as usize;
        for level in 0..self.levels.len() {
            for cell in 0..cells {
                let held = &mut self.levels[level].cells[cell];
                held.drawn = false;
                held.dirty = false;
                let moving = self.table[level * cells + cell] & 0xffff_0000;
                self.set_entry(level, cell, moving);
            }
        }
    }

    pub(crate) fn plan(
        &mut self,
        (budget, region): (u32, u32),
        moving_budget: u32,
        out: &mut ClipPlan,
    ) {
        out.regions.clear();
        out.composes.clear();
        out.moving.clear();
        self.stats = ClipStats::default();
        self.allocate();
        self.redraw(budget, region, out);
        self.draw_moving(moving_budget, out);
        let frame = self.frame;
        let (mut slotted, mut moving) = (0, 0);
        let cells = self.levels.iter().flat_map(|level| &level.cells);
        for cell in cells {
            let marked = cell.marked_at(frame);
            self.stats.marked += u32::from(marked);
            slotted += u32::from(marked && cell.slot.is_some());
            moving += u32::from(cell.drawn && marked && cell.moving_at(frame));
        }
        self.stats.resident = self.owners.iter().flatten().count() as u32;
        self.stats.drawn = out.composes.len() as u32;
        self.stats.moving = out.moving.len() as u32;
        self.stats.tile_cut = Cut::of(self.stats.marked, slotted);
        self.stats.moving_cut = Cut::of(moving, self.stats.moving);
    }

    fn allocate(&mut self) {
        let frame = self.frame;
        let cells = (self.side * self.side) as usize;
        let wanted = self
            .levels
            .iter()
            .flat_map(|level| &level.cells)
            .filter(|cell| cell.marked_at(frame) && cell.slot.is_none())
            .count();
        let mut reclaimable: Vec<(u32, u16)> = Vec::new();
        if wanted > self.free.len() {
            for (slot, owner) in self.owners.iter().enumerate() {
                let Some((level, cell)) = *owner else {
                    continue;
                };
                let held = &self.levels[level as usize].cells[cell as usize];
                if !held.marked_at(frame) {
                    reclaimable.push((held.marked, slot as u16));
                }
            }
            reclaimable.sort_unstable_by(|a, b| b.cmp(a));
        }
        for level in (0..self.levels.len()).rev() {
            for cell in 0..cells {
                let held = self.levels[level].cells[cell];
                if !held.marked_at(frame) || held.slot.is_some() {
                    continue;
                }
                let slot = match self.free.pop() {
                    Some(slot) => slot,
                    None => match reclaimable.pop() {
                        Some((_, slot)) => {
                            if let Some((owner, owned)) = self.owners[slot as usize] {
                                self.vacate(owner as usize, owned as usize);
                            }
                            self.stats.reclaimed += 1;
                            slot
                        }
                        None => return,
                    },
                };
                self.owners[slot as usize] = Some((level as u8, cell as u16));
                self.levels[level].cells[cell].slot = Some(slot);
            }
        }
    }

    fn redraw(&mut self, budget: u32, region: u32, out: &mut ClipPlan) {
        let frame = self.frame;
        let side = self.side as i32;
        let square = region as i32;
        let mut left = budget;
        for level in (0..self.levels.len()).rev() {
            for redrawing in [false, true] {
                for row in 0..side {
                    for column in 0..side {
                        if left == 0 {
                            return;
                        }
                        let clip = &self.levels[level];
                        let tile = [clip.origin[0] + column, clip.origin[1] + row];
                        let cell =
                            ((tile[1] & (side - 1)) * side + (tile[0] & (side - 1))) as usize;
                        let held = clip.cells[cell];
                        let Some(slot) = held.slot.filter(|_| held.marked_at(frame)) else {
                            continue;
                        };
                        if held.drawn != redrawing || (redrawing && !held.dirty) {
                            continue;
                        }
                        let first = tile.map(|at| at.div_euclid(square) * square);
                        let index = match out
                            .regions
                            .iter()
                            .position(|region| region.level == level && region.first == first)
                        {
                            Some(index) => index,
                            None if out.regions.len() < SHADOW_REGIONS => {
                                out.regions.push(self.region(level, first, square));
                                out.regions.len() - 1
                            }
                            None => continue,
                        };
                        out.composes.push(Compose {
                            slot,
                            region: index as u8,
                            at: [(tile[0] - first[0]) as u32, (tile[1] - first[1]) as u32],
                        });
                        let held = &mut self.levels[level].cells[cell];
                        held.drawn = true;
                        held.dirty = false;
                        let moving =
                            self.table[level * (side * side) as usize + cell] & 0xffff_0000;
                        self.set_entry(level, cell, moving | entry(Some(slot), None));
                        left -= 1;
                    }
                }
            }
        }
    }

    fn region(&self, level: usize, first: [i32; 2], square: i32) -> Region {
        let clip = &self.levels[level];
        let apron = SHADOW_APRON as f32 * clip.texel;
        let corner = first.map(|at| at as f32 * clip.tile);
        let width = square as f32 * clip.tile;
        Region {
            level,
            first,
            rect: [
                corner[0] - apron,
                corner[1] - apron,
                corner[0] + width + apron,
                corner[1] + width + apron,
            ],
            depth: self.band(level),
            texel: clip.texel,
        }
    }

    fn draw_moving(&mut self, budget: u32, out: &mut ClipPlan) {
        let cells = (self.side * self.side) as usize;
        for (level, cell) in std::mem::take(&mut self.moving) {
            let still = self.table[level * cells + cell] & 0xffff;
            self.set_entry(level, cell, still);
        }
        let frame = self.frame;
        let side = self.side as i32;
        for level in (0..self.levels.len()).rev() {
            for row in 0..side {
                for column in 0..side {
                    if out.moving.len() >= budget as usize {
                        return;
                    }
                    let clip = &self.levels[level];
                    let tile = [clip.origin[0] + column, clip.origin[1] + row];
                    let cell = ((tile[1] & (side - 1)) * side + (tile[0] & (side - 1))) as usize;
                    let held = clip.cells[cell];
                    if !(held.drawn && held.marked_at(frame) && held.moving_at(frame)) {
                        continue;
                    }
                    let index = out.moving.len() as u16;
                    out.moving.push(MovingTile { level, tile, index });
                    self.moving.push((level, cell));
                    let slot = self.owners.len() as u32 + u32::from(index);
                    self.set_entry(level, cell, entry(held.slot, Some(slot)));
                }
            }
        }
    }

    pub(crate) fn take_table(&mut self) -> (&[u32], Vec<usize>) {
        let changed = self
            .changed
            .iter_mut()
            .enumerate()
            .filter_map(|(level, changed)| std::mem::take(changed).then_some(level))
            .collect();
        (&self.table, changed)
    }

    pub(crate) fn band(&self, level: usize) -> [f32; 2] {
        self.levels.get(level).map_or([0.0; 2], |level| {
            [
                level.depth - level.half_depth,
                level.depth + level.half_depth,
            ]
        })
    }

    pub(crate) fn window(&self, level: usize) -> [f32; 4] {
        self.levels.get(level).map_or([0.0; 4], |level| {
            [
                level.origin[0] as f32 * level.tile,
                level.origin[1] as f32 * level.tile,
                level.tile,
                level.texel,
            ]
        })
    }

    pub(crate) fn tile_rect(&self, level: usize, tile: [i32; 2]) -> [f32; 4] {
        let clip = &self.levels[level];
        let apron = SHADOW_APRON as f32 * clip.texel;
        let corner = tile.map(|at| at as f32 * clip.tile);
        [
            corner[0] - apron,
            corner[1] - apron,
            corner[0] + clip.tile + apron,
            corner[1] + clip.tile + apron,
        ]
    }

    pub(crate) fn texel(&self, level: usize) -> f32 {
        self.levels.get(level).map_or(0.0, |level| level.texel)
    }
}
