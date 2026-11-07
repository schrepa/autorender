use engine_core::{
    Cut, GpuBits,
    glam::{Mat4, Vec3, Vec4},
    math,
};
use engine_places_rendering::shadows::SHADOW_REGIONS;

use crate::constants::SHADOW_APRON;
use crate::gpu_layouts::TileEntry;

const LOW_SUN: f32 = 0.1;

pub(crate) const LEVEL_TEXELS: u32 = 8192;

pub(crate) const REGION_TEXELS: u32 = 512;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LightBasis {
    pub rows: [Vec3; 3],
    pub shift: Vec3,
}

impl LightBasis {
    pub(crate) fn toward(sun: Vec3) -> Self {
        let away = -sun;
        let reference = if away.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = away.cross(reference).normalize();
        let up = right.cross(away);
        Self {
            rows: [right, up, away],
            shift: Vec3::ZERO,
        }
    }

    pub(crate) fn apply(&self, point: Vec3) -> Vec3 {
        self.shift
            + Vec3::new(
                self.rows[0].dot(point),
                self.rows[1].dot(point),
                self.rows[2].dot(point),
            )
    }

    pub(crate) fn words(&self) -> [[f32; 4]; 3] {
        [0, 1, 2].map(|row| self.rows[row].extend(self.shift[row]).to_array())
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
            row(basis.rows[0], sx, (x0 + x1) * 0.5 - basis.shift.x),
            row(basis.rows[1], sy, (y0 + y1) * 0.5 - basis.shift.y),
            row(basis.rows[2], sz, z0 - basis.shift.z),
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
    listed: bool,
    marked: u32,
    moving: u32,
}

impl Cell {
    fn marked_in(&self, marks: u32) -> bool {
        self.marked != 0 && self.marked == marks
    }

    fn moving_in(&self, masks: u32) -> bool {
        self.moving != 0 && self.moving == masks
    }
}

#[derive(Debug)]
struct ClipLevel {
    texel: f32,
    tile: f32,
    origin: [i32; 2],
    depth: f32,
    half_depth: f32,
    before: [f32; 2],
    cells: Vec<Cell>,
    live: Vec<u16>,
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

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct TileState {
    pub level: usize,
    pub tile: [i32; 2],
    pub slot: Option<u16>,
    pub drawn: bool,
    pub dirty: bool,
    pub marked: bool,
    pub moving: bool,
}

#[derive(Clone, Copy, Debug, Default)]
enum Turning {
    #[default]
    Still,
    Turned {
        from: LightBasis,
    },
}

pub(crate) type DrawnTile = (usize, usize, u16);

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
    pub waiting: u32,
    pub moving: u32,
    pub tile_cut: Cut,
    pub moving_cut: Cut,
}

#[derive(Debug, Default)]
pub(crate) struct Clipmap {
    basis: Option<LightBasis>,
    turning: Turning,
    settings: Option<ClipSettings>,
    side: u32,
    levels: Vec<ClipLevel>,
    owners: Vec<Option<(u32, u16)>>,
    free: Vec<u16>,
    table: Vec<u32>,
    changed: Vec<bool>,
    moving: Vec<(usize, usize)>,
    order: Vec<(i32, u16)>,
    drawing: Vec<DrawnTile>,
    marks: u32,
    masks: u32,
    stats: ClipStats,
}

fn tile_of(word: u32) -> TileEntry {
    TileEntry::unpack(word).unwrap_or_default()
}

impl Clipmap {
    pub(crate) fn place(
        &mut self,
        sun: LightBasis,
        eye: Vec3,
        settings: ClipSettings,
    ) -> (LightBasis, bool) {
        let restart = self.settings != Some(settings);
        if restart {
            self.basis = Some(sun);
            self.turning = Turning::Still;
            self.settings = Some(settings);
            self.side = (LEVEL_TEXELS / settings.tile_texels).max(1);
            let cells = (self.side * self.side) as usize;
            self.levels.clear();
            self.owners = vec![None; settings.slots as usize];
            self.free = (0..settings.slots as u16).rev().collect();
            self.table = vec![0; settings.levels as usize * cells];
            self.changed = vec![true; settings.levels as usize];
            self.moving.clear();
        }
        let basis = self.follow(sun, eye);
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
                before: [light.z - half_depth, light.z + half_depth],
                cells: vec![Cell::default(); cells],
                live: Vec::new(),
            };
            match self.levels.get(index) {
                Some(level) if (light.z - level.depth).abs() <= 0.5 * width => {
                    self.levels[index].half_depth = half_depth;
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
        (basis, restart)
    }

    fn follow(&mut self, sun: LightBasis, eye: Vec3) -> LightBasis {
        let held = self.basis.unwrap_or(sun);
        if let Turning::Turned { .. } = self.turning {
            if self.old_sun_wanted() {
                return held;
            }
            if !self.old_sun_held() {
                self.turning = Turning::Still;
            }
        }
        if held.rows == sun.rows {
            return held;
        }
        self.turn(held, sun, eye)
    }

    fn old_sun_wanted(&self) -> bool {
        let (marks, cells) = (self.marks, (self.side * self.side) as usize);
        self.levels.iter().enumerate().any(|(level, clip)| {
            clip.live.iter().map(|&cell| usize::from(cell)).any(|cell| {
                clip.cells[cell].marked_in(marks)
                    && tile_of(self.table[level * cells + cell]).old_sun
            })
        })
    }

    fn old_sun_held(&self) -> bool {
        let cells = (self.side * self.side) as usize;
        self.owners.iter().flatten().any(|&(level, cell)| {
            tile_of(self.table[level as usize * cells + usize::from(cell)]).old_sun
        })
    }

    fn tile(&self, level: usize, cell: usize) -> TileEntry {
        tile_of(self.table[level * (self.side * self.side) as usize + cell])
    }

    fn turn(&mut self, held: LightBasis, sun: LightBasis, eye: Vec3) -> LightBasis {
        let basis = LightBasis {
            shift: held.apply(eye) - sun.apply(eye),
            ..sun
        };
        self.basis = Some(basis);
        self.turning = Turning::Turned { from: held };
        for level in &mut self.levels {
            level.before = [
                level.depth - level.half_depth,
                level.depth + level.half_depth,
            ];
        }
        let marks = self.marks;
        for slot in 0..self.owners.len() {
            let Some((level, cell)) = self.owners[slot] else {
                continue;
            };
            let (level, cell) = (level as usize, cell as usize);
            let tile = &mut self.levels[level].cells[cell];
            if !tile.marked_in(marks) {
                self.release(level, cell);
            } else if tile.drawn {
                tile.dirty = true;
                let kept = TileEntry {
                    old_sun: true,
                    ..self.tile(level, cell)
                };
                self.set_entry(level, cell, kept.pack());
            }
        }
        basis
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

    pub(crate) fn mask_words(&self) -> usize {
        (self.levels.len() * (self.side * self.side) as usize).div_ceil(32)
    }

    pub(crate) fn stats(&self) -> ClipStats {
        self.stats
    }

    pub(crate) fn idle(&mut self) {
        self.stats = ClipStats::default();
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
        self.marks += 1;
        let marks = self.marks;
        for (level, cell) in self.marked_cells(words) {
            let clip = &mut self.levels[level];
            let held = &mut clip.cells[cell];
            held.marked = marks;
            if !held.listed {
                held.listed = true;
                clip.live.push(cell as u16);
            }
        }
    }

    pub(crate) fn merge_dirty(&mut self, words: &[u32]) {
        for (level, cell) in self.marked_cells(words) {
            let cell = &mut self.levels[level].cells[cell];
            cell.dirty |= cell.drawn;
        }
    }

    pub(crate) fn merge_moving(&mut self, words: &[u32]) {
        self.masks += 1;
        let (masks, side) = (self.masks, self.side as usize);
        for (level, cell) in self.marked_cells(words) {
            let (column, row) = (cell % side, cell / side);
            for dy in [side - 1, 0, 1] {
                for dx in [side - 1, 0, 1] {
                    let beside = (row + dy) % side * side + (column + dx) % side;
                    self.levels[level].cells[beside].moving = masks;
                }
            }
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

    pub(crate) fn invalidate(&mut self, kept: &[DrawnTile]) {
        let cells = (self.side * self.side) as usize;
        self.drawing.clear();
        for level in 0..self.levels.len() {
            for cell in 0..cells {
                let held = self.levels[level].cells[cell];
                let drawn = held.slot.filter(|_| held.drawn);
                if drawn.is_some_and(|slot| !kept.contains(&(level, cell, slot))) {
                    self.undraw_cell(level, cell);
                }
            }
        }
    }

    fn undraw_cell(&mut self, level: usize, cell: usize) {
        let held = &mut self.levels[level].cells[cell];
        held.drawn = false;
        held.dirty = false;
        let moving = TileEntry {
            moving: self.tile(level, cell).moving,
            ..TileEntry::default()
        };
        self.set_entry(level, cell, moving.pack());
    }

    pub(crate) fn take_drawing(&mut self) -> Vec<DrawnTile> {
        std::mem::take(&mut self.drawing)
    }

    pub(crate) fn undraw(&mut self, tiles: &[DrawnTile]) {
        for &(level, cell, slot) in tiles {
            let held = self
                .levels
                .get(level)
                .and_then(|clip| clip.cells.get(cell))
                .is_some_and(|held| held.drawn && held.slot == Some(slot));
            if held {
                self.undraw_cell(level, cell);
            }
        }
    }

    pub(crate) fn plan<F: FnMut() -> bool>(
        &mut self,
        (afford, region): (&mut F, u32),
        moving_slots: u32,
        out: &mut ClipPlan,
    ) {
        out.regions.clear();
        out.composes.clear();
        out.moving.clear();
        self.drawing.clear();
        self.stats = ClipStats::default();
        self.prune_live();
        self.allocate();
        self.redraw(afford, region, out);
        self.draw_moving(moving_slots, out);
        let masks = self.masks;
        let (mut slotted, mut moving) = (0, 0);
        for clip in &self.levels {
            for &cell in &clip.live {
                let cell = &clip.cells[cell as usize];
                self.stats.marked += 1;
                slotted += u32::from(cell.slot.is_some());
                self.stats.waiting += u32::from(cell.slot.is_some() && (!cell.drawn || cell.dirty));
                moving += u32::from(cell.drawn && cell.moving_in(masks));
            }
        }
        self.stats.resident = self.owners.iter().flatten().count() as u32;
        self.stats.drawn = out.composes.len() as u32;
        self.stats.moving = out.moving.len() as u32;
        self.stats.tile_cut = Cut::of(self.stats.marked, slotted);
        self.stats.moving_cut = Cut::of(moving, self.stats.moving);
    }

    fn prune_live(&mut self) {
        let marks = self.marks;
        for clip in &mut self.levels {
            let ClipLevel { cells, live, .. } = clip;
            live.sort_unstable();
            live.dedup();
            live.retain(|&cell| {
                let held = &mut cells[cell as usize];
                let kept = held.marked_in(marks);
                held.listed = kept;
                kept
            });
        }
    }

    fn scan_order(&self, level: usize, order: &mut Vec<(i32, u16)>) {
        let clip = &self.levels[level];
        let side = self.side as i32;
        let mask = side - 1;
        order.clear();
        order.extend(clip.live.iter().map(|&cell| {
            let (row, column) = (i32::from(cell) / side, i32::from(cell) % side);
            let across = (row - clip.origin[1]) & mask;
            let along = (column - clip.origin[0]) & mask;
            (across * side + along, cell)
        }));
        order.sort_unstable();
    }

    fn allocate(&mut self) {
        let marks = self.marks;
        let wanted = self
            .levels
            .iter()
            .flat_map(|level| level.live.iter().map(|&cell| &level.cells[cell as usize]))
            .filter(|cell| cell.slot.is_none())
            .count();
        let mut reclaimable: Vec<(u32, u16)> = Vec::new();
        if wanted > self.free.len() {
            for (slot, owner) in self.owners.iter().enumerate() {
                let Some((level, cell)) = *owner else {
                    continue;
                };
                let held = &self.levels[level as usize].cells[cell as usize];
                if !held.marked_in(marks) {
                    reclaimable.push((held.marked, slot as u16));
                }
            }
            reclaimable.sort_unstable_by(|a, b| b.cmp(a));
        }
        for level in (0..self.levels.len()).rev() {
            for at in 0..self.levels[level].live.len() {
                let cell = usize::from(self.levels[level].live[at]);
                if self.levels[level].cells[cell].slot.is_some() {
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
                self.owners[slot as usize] = Some((level as u32, cell as u16));
                self.levels[level].cells[cell].slot = Some(slot);
            }
        }
    }

    fn redraw<F: FnMut() -> bool>(&mut self, afford: &mut F, region: u32, out: &mut ClipPlan) {
        let side = self.side as i32;
        let square = region as i32;
        let mut order = std::mem::take(&mut self.order);
        'levels: for level in (0..self.levels.len()).rev() {
            self.scan_order(level, &mut order);
            for redrawing in [false, true] {
                for &(place, cell) in &order {
                    let cell = usize::from(cell);
                    let clip = &self.levels[level];
                    let tile = [clip.origin[0] + place % side, clip.origin[1] + place / side];
                    let held = clip.cells[cell];
                    let Some(slot) = held.slot else {
                        continue;
                    };
                    if held.drawn != redrawing || (redrawing && !held.dirty) {
                        continue;
                    }
                    let first = tile.map(|at| at.div_euclid(square) * square);
                    let found = out
                        .regions
                        .iter()
                        .position(|region| region.level == level && region.first == first);
                    if found.is_none() && out.regions.len() >= SHADOW_REGIONS as usize {
                        continue;
                    }
                    if !afford() {
                        break 'levels;
                    }
                    let index = found.unwrap_or_else(|| {
                        out.regions.push(self.region(level, first, square));
                        out.regions.len() - 1
                    });
                    out.composes.push(Compose {
                        slot,
                        region: index as u8,
                        at: [(tile[0] - first[0]) as u32, (tile[1] - first[1]) as u32],
                    });
                    let held = &mut self.levels[level].cells[cell];
                    held.drawn = true;
                    held.dirty = false;
                    let drawn = TileEntry {
                        still: slot + 1,
                        old_sun: false,
                        moving: self.tile(level, cell).moving,
                    };
                    self.set_entry(level, cell, drawn.pack());
                    self.drawing.push((level, cell, slot));
                }
            }
        }
        self.order = order;
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

    fn draw_moving(&mut self, slots: u32, out: &mut ClipPlan) {
        for (level, cell) in std::mem::take(&mut self.moving) {
            let still = TileEntry {
                moving: 0,
                ..self.tile(level, cell)
            };
            self.set_entry(level, cell, still.pack());
        }
        let masks = self.masks;
        let side = self.side as i32;
        let mut order = std::mem::take(&mut self.order);
        'levels: for level in (0..self.levels.len()).rev() {
            self.scan_order(level, &mut order);
            for &(place, cell) in &order {
                if out.moving.len() >= slots as usize {
                    break 'levels;
                }
                let cell = usize::from(cell);
                let clip = &self.levels[level];
                let tile = [clip.origin[0] + place % side, clip.origin[1] + place / side];
                let held = clip.cells[cell];
                if !(held.drawn && held.moving_in(masks)) {
                    continue;
                }
                let index = out.moving.len() as u16;
                out.moving.push(MovingTile { level, tile, index });
                self.moving.push((level, cell));
                let slot = self.owners.len() + usize::from(index) + 1;
                let routed = TileEntry {
                    moving: u16::try_from(slot).unwrap_or(0),
                    ..self.tile(level, cell)
                };
                self.set_entry(level, cell, routed.pack());
            }
        }
        self.order = order;
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

    pub(crate) fn band_before(&self, level: usize) -> [f32; 2] {
        self.levels
            .get(level)
            .map_or([0.0; 2], |level| level.before)
    }

    pub(crate) fn before(&self) -> Option<LightBasis> {
        match self.turning {
            Turning::Still => None,
            Turning::Turned { from } => Some(from),
        }
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

    pub(crate) fn inspect(&self, point: Vec3) -> Vec<TileState> {
        let Some(basis) = self.basis else {
            return Vec::new();
        };
        let light = basis.apply(point);
        let side = self.side as i32;
        let (marks, masks) = (self.marks, self.masks);
        self.levels
            .iter()
            .enumerate()
            .filter_map(|(level, clip)| {
                let tile = [
                    (light.x / clip.tile).floor() as i32,
                    (light.y / clip.tile).floor() as i32,
                ];
                let inside = (0..2).all(|axis| {
                    tile[axis] >= clip.origin[axis] && tile[axis] < clip.origin[axis] + side
                });
                inside.then(|| {
                    let cell = (tile[1] & (side - 1)) * side + (tile[0] & (side - 1));
                    let held = clip.cells[cell as usize];
                    TileState {
                        level,
                        tile,
                        slot: held.slot,
                        drawn: held.drawn,
                        dirty: held.dirty,
                        marked: held.marked_in(marks),
                        moving: held.moving_in(masks),
                    }
                })
            })
            .collect()
    }
}
