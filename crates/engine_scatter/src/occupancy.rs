use bevy_ecs::prelude::Resource;
use glam::Vec3;

use crate::layouts::{
    CELLS, CanopyPiece, Cover, ListedDraw, MAX_OCCUPANCY_LEVELS, OCCUPANCY_GROUP, QUAD, Splat,
};
use crate::settings::{FoliageSettings, OccupancySettings};

const SIDE: i32 = CELLS as i32;

const CELL_WORDS: usize = (CELLS * CELLS / 32) as usize;

const _: () = assert!(CELLS == 16, "word_holds_two_rows");

const REACH_SHARE: f32 = 3.0 / 8.0;

impl Cover {
    pub(crate) fn of(min: Vec3, max: Vec3) -> Self {
        let half = (max - min) * 0.5;
        let footprint = (half.x + half.z) * 0.5 * 0.85;
        let height = max.y - min.y;
        let bottom = if height > 2.5 * footprint {
            min.y + 0.2 * height
        } else {
            min.y
        };
        Self {
            footprint,
            bottom,
            top: max.y,
            density: 0.9,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Lit {
    pub(crate) occupancy: [f32; 4],
    pub(crate) maps: [f32; 4],
    pub(crate) centers: [[f32; 4]; MAX_OCCUPANCY_LEVELS / 2],
}

pub(crate) fn texel_share(settings: &OccupancySettings) -> f32 {
    let (texel, reach) = finest(settings);
    texel / reach
}

fn finest(settings: &OccupancySettings) -> (f32, f32) {
    let texel = settings.radius / (REACH_SHARE * side_texels(settings) as f32);
    (texel, settings.radius - 0.5 * texel)
}

fn side_texels(settings: &OccupancySettings) -> u32 {
    settings.resolution.next_multiple_of(2 * CELLS)
}

#[derive(Clone, Copy, Debug, Default)]
struct Cells([u32; CELL_WORDS]);

impl Cells {
    const ALL: Self = Self([u32::MAX; CELL_WORDS]);

    fn set(&mut self, column: u32, row: u32) {
        let bit = row * CELLS + column;
        self.0[(bit / 32) as usize] |= 1 << (bit % 32);
    }

    fn get(&self, column: u32, row: u32) -> bool {
        let bit = row * CELLS + column;
        self.0[(bit / 32) as usize] & (1 << (bit % 32)) != 0
    }

    fn set_column(&mut self, column: u32) {
        for row in 0..CELLS {
            self.set(column, row);
        }
    }

    fn set_row(&mut self, row: u32) {
        for column in 0..CELLS {
            self.set(column, row);
        }
    }

    fn any(&self) -> bool {
        self.0.iter().any(|&word| word != 0)
    }
}

#[derive(Clone, Copy, Debug)]
struct Window {
    cell: f32,
    origin: [i32; 2],
}

impl Window {
    fn scroll(&mut self, target: [i32; 2], dirty: &mut Cells) {
        let shift = [target[0] - self.origin[0], target[1] - self.origin[1]];
        if shift.iter().all(|step| step.abs() <= 1) {
            return;
        }
        let entering = |origin: i32, shift: i32| {
            if shift >= 0 {
                origin + SIDE..origin + SIDE + shift.min(SIDE)
            } else {
                origin + shift.max(-SIDE)..origin
            }
        };
        for column in entering(self.origin[0], shift[0]) {
            dirty.set_column(column.rem_euclid(SIDE) as u32);
        }
        for row in entering(self.origin[1], shift[1]) {
            dirty.set_row(row.rem_euclid(SIDE) as u32);
        }
        self.origin = target;
    }

    fn holds(&self, cell: [i32; 2]) -> bool {
        (0..2).all(|axis| (self.origin[axis]..self.origin[axis] + SIDE).contains(&cell[axis]))
    }

    fn holds_box(&self, min: Vec3, max: Vec3) -> bool {
        let corner = self.corner();
        let side = self.cell * CELLS as f32;
        min.x >= corner[0]
            && min.z >= corner[1]
            && max.x < corner[0] + side
            && max.z < corner[1] + side
    }

    fn corner(&self) -> [f32; 2] {
        self.origin.map(|cell| cell as f32 * self.cell)
    }
}

fn wrapped(cell: [i32; 2]) -> [u32; 2] {
    cell.map(|axis| axis.rem_euclid(SIDE) as u32)
}

fn cells(window: Window) -> impl Iterator<Item = [i32; 2]> {
    (0..SIDE).flat_map(move |row| {
        (0..SIDE).map(move |column| [window.origin[0] + column, window.origin[1] + row])
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MapFrame {
    pub(crate) splat: Splat,
    pub(crate) redraw: bool,
    pub(crate) average: bool,
}

#[derive(Resource, Default)]
pub(crate) struct OccupancyWindows {
    pub(crate) resolution: u32,
    pub(crate) maps: usize,
    windows: Vec<Option<Window>>,
    reference: f32,
    dirty: Vec<Cells>,
    pub(crate) covers: Vec<Cover>,
    pub(crate) covers_revision: u64,
    pub(crate) frames: Vec<MapFrame>,
    pub(crate) workgroups: u32,
    pub(crate) slots: u32,
    pub(crate) runs: bool,
    pub(crate) redraws: u32,
    pub(crate) averages: u32,
}

pub(crate) const LISTED_RESET: ListedDraw = ListedDraw {
    vertices: QUAD,
    instances: 0,
    first_vertex: 0,
    first_instance: 0,
};

impl OccupancyWindows {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn prepare_windows(
        &mut self,
        settings: &OccupancySettings,
        eye: Vec3,
        (covers, slots): (&[Cover], u32),
    ) -> Option<Lit> {
        for frame in &mut self.frames {
            frame.redraw = false;
            frame.average = false;
        }
        self.runs = settings.runs();
        if !self.runs {
            self.release();
            return None;
        }
        if slots == 0 || covers.is_empty() {
            self.windows.clear();
            return None;
        }
        let maps = settings.levels.clamp(1, MAX_OCCUPANCY_LEVELS as u32) as usize;
        let resolution = side_texels(settings);
        if self.resolution != resolution || self.maps != maps {
            self.resolution = resolution;
            self.maps = maps;
            self.windows.clear();
        }
        self.size(maps);
        if covers != self.covers.as_slice() {
            self.covers = covers.to_vec();
            self.covers_revision += 1;
            self.windows.clear();
        }
        self.slots = slots;
        self.workgroups = slots.div_ceil(OCCUPANCY_GROUP);
        if self.windows.first().copied().flatten().is_none()
            || (eye.y - self.reference).abs() > settings.radius
        {
            self.windows.clear();
            self.reference = eye.y;
        }
        self.windows.resize(maps, None);
        let before = self.windows.clone();
        for (map, (window, dirty)) in self.windows.iter_mut().zip(&mut self.dirty).enumerate() {
            let cell = settings.radius * (1_u32 << map) as f32 / REACH_SHARE / CELLS as f32;
            let target = [
                (eye.x / cell).floor() as i32 - SIDE / 2,
                (eye.z / cell).floor() as i32 - SIDE / 2,
            ];
            match window {
                Some(window) if window.cell == cell => window.scroll(target, dirty),
                window => {
                    *dirty = Cells::ALL;
                    *window = Some(Window {
                        cell,
                        origin: target,
                    });
                }
            }
        }
        for map in 1..maps {
            let (Some(left), Some(finer), Some(window)) =
                (before[map - 1], self.windows[map - 1], self.windows[map])
            else {
                continue;
            };
            for cell in cells(left).filter(|&cell| !finer.holds(cell)) {
                let under = cell.map(|axis| axis.div_euclid(2));
                if window.holds(under) {
                    let [column, row] = wrapped(under);
                    self.dirty[map].set(column, row);
                }
            }
        }
        let texels = resolution / CELLS;
        let mut lit = Lit {
            occupancy: [0.0; 4],
            maps: [maps as f32, self.reference, 0.0, 0.0],
            centers: [[0.0; 4]; MAX_OCCUPANCY_LEVELS / 2],
        };
        let mut changed = Cells::default();
        for map in 0..maps {
            let Some(window) = self.windows[map] else {
                continue;
            };
            let dirty = std::mem::take(&mut self.dirty[map]);
            let finer = map.checked_sub(1).and_then(|finer| self.windows[finer]);
            let mut means = Cells::default();
            let mut now = dirty;
            if let Some(finer) = finer {
                for (index, cell) in cells(finer).enumerate() {
                    let [x, y] = wrapped(cell);
                    let [under_x, under_y] = wrapped(cell.map(|axis| axis.div_euclid(2)));
                    if changed.get(x, y) || dirty.get(under_x, under_y) {
                        means.set(index as u32 % CELLS, index as u32 / CELLS);
                        now.set(under_x, under_y);
                    }
                }
            }
            changed = now;
            let size = window.cell * CELLS as f32;
            let corner = window.corner();
            let center = [corner[0] + size * 0.5, corner[1] + size * 0.5];
            lit.centers[map / 2][2 * (map % 2)] = center[0];
            lit.centers[map / 2][2 * (map % 2) + 1] = center[1];
            let (redraw, average) = (dirty.any(), means.any());
            if !redraw && !average {
                continue;
            }
            self.frames[map] = MapFrame {
                splat: Splat {
                    area: [corner[0], corner[1], size, self.reference],
                    counts: [slots, covers.len() as u32, 0, resolution],
                    place: [
                        window.origin[0].rem_euclid(SIDE) as u32 * texels,
                        window.origin[1].rem_euclid(SIDE) as u32 * texels,
                        texels,
                        0,
                    ],
                    dirty: bytemuck::cast(dirty.0),
                    finer: finer.map_or([0; 4], |finer| [finer.origin[0], finer.origin[1], 1, 0]),
                    means: bytemuck::cast(means.0),
                },
                redraw,
                average,
            };
        }
        let (_, reach) = finest(settings);
        let finest_side = self.windows[0].map_or(1.0, |window| window.cell * CELLS as f32);
        lit.occupancy = [
            1.0 / finest_side,
            resolution as f32,
            reach,
            settings.strength,
        ];
        Some(lit)
    }

    fn size(&mut self, maps: usize) {
        self.dirty.resize(maps, Cells::default());
        self.frames.resize(maps, MapFrame::default());
    }

    fn dirty_box(&mut self, min: Vec3, max: Vec3) {
        let resolution = self.resolution.max(1) as f32;
        for map in 0..self.windows.len() {
            let Some(window) = self.windows[map] else {
                continue;
            };
            let margin = window.cell * CELLS as f32 / resolution;
            let finer = map.checked_sub(1).and_then(|finer| self.windows[finer]);
            if finer.is_some_and(|finer| {
                let margin = Vec3::splat(finer.cell * CELLS as f32 / resolution);
                finer.holds_box(min - margin, max + margin)
            }) {
                continue;
            }
            let span = |low: f32, high: f32, origin: i32| {
                let cell = |value: f32| (value / window.cell).floor() as i32;
                cell(low - margin).max(origin)..=cell(high + margin).min(origin + SIDE - 1)
            };
            let rows = span(min.z, max.z, window.origin[1]);
            for column in span(min.x, max.x, window.origin[0]) {
                for row in rows.clone() {
                    let [column, row] = wrapped([column, row]);
                    self.dirty[map].set(column, row);
                }
            }
        }
    }

    fn release(&mut self) {
        self.windows.clear();
        if self.frames.is_empty() {
            return;
        }
        self.dirty.clear();
        self.frames.clear();
        self.covers.clear();
        self.covers_revision += 1;
        self.resolution = 1;
        self.maps = 1;
    }

    pub(crate) fn place(
        &mut self,
        settings: &OccupancySettings,
        eye: Vec3,
        placed: (&[Cover], u32),
    ) -> Option<Lit> {
        let lit = self.prepare_windows(settings, eye, placed);
        for frame in &self.frames {
            self.redraws += u32::from(frame.redraw);
            self.averages += u32::from(frame.average);
        }
        lit
    }

    pub(crate) fn moved(&mut self, moved: &[(Vec3, Vec3)]) {
        for &(min, max) in moved {
            self.dirty_box(min, max);
        }
    }

    pub(crate) fn drawn(&self) -> bool {
        self.frames
            .iter()
            .any(|frame| frame.redraw || frame.average)
    }
}

pub(crate) fn canopy(lit: Option<Lit>, foliage: &FoliageSettings, temporal: bool) -> CanopyPiece {
    let mut canopy = CanopyPiece {
        occupancy: [0.0; 4],
        occupancy_maps: [0.0; 4],
        occupancy_centers: [[0.0; 4]; MAX_OCCUPANCY_LEVELS / 2],
        foliage: [
            foliage.transmission,
            foliage.variation,
            if foliage.hashed_cutoff && temporal {
                1.0
            } else {
                0.0
            },
            foliage.ground_blend,
        ],
    };
    if let Some(maps) = lit {
        canopy.occupancy = maps.occupancy;
        canopy.occupancy_maps = maps.maps;
        canopy.occupancy_centers = maps.centers;
    }
    canopy
}
