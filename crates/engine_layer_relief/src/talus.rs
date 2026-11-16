use std::any::Any;
use std::sync::Arc;

use engine_codec::{Expected, Reader, Writer};
use engine_core::{Error, fields, parse_value};
use engine_heightfield::layer::{
    Below, ChangeMap, LayerFiles, LayerKind, Made, Place, Published, Rests, Row, TerrainLayer,
    image_row, maps, no_files, no_points, same_as,
};
use engine_heightfield::noise::smooth;
use engine_heightfield::tile::{producer, read_map, unknown_map};
use engine_heightfield::{Gather, INVALID_TERRAIN, Layers, Pyramid, TileHeights};
use engine_math::{clamp, max, min};
use glam::Vec2;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::relief::{blobs, calm};
use crate::site::{Grid, Sheet, sheet, tangent};

pub const FALL: &str = "fall";
pub const DEBRIS: &str = "talus";

const WALL: &str = "cliff";
const MAX_LEVEL: u32 = 12;
const RESOLVED: f32 = 2.0;
const TRACE: f32 = 0.05;
const MAX_RUNNELS: usize = 2048;
const BUCKET: f32 = 16.0;
const STRIDE: f32 = 0.5;
const TALUS_BLOCKS: u32 = 0x2f1b_d4a7;
const AROUND: [(isize, isize, f32); 8] = [
    (-1, -1, std::f32::consts::SQRT_2),
    (0, -1, 1.0),
    (1, -1, std::f32::consts::SQRT_2),
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (-1, 1, std::f32::consts::SQRT_2),
    (0, 1, 1.0),
    (1, 1, std::f32::consts::SQRT_2),
];

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct TalusLayer {
        pub level: u32 = 4 => range(0, MAX_LEVEL),
        pub seed: u32 = 0 => any,
        pub reach: f32 = 500.0 => range(1.0, 1.0e5),
        pub sag: f32 = 0.0 => range(0.0, 1.0),
        pub relief: f32 = 0.0 => range(0.0, 100.0),
        pub block: f32 = 3.0 => range(0.1, 1000.0),
        pub runnel: f32 = 0.0 => range(0.0, 100.0),
        pub levee: f32 = 0.0 => range(0.0, 100.0),
        pub width: f32 = 3.0 => range(0.1, 1000.0),
        pub chute: f32 = 1.0e12 => range(1.0, 1.0e12),
        pub apart: f32 = 40.0 => range(1.0, 1.0e5),
        pub slope: [f32; 2] = [15.0, 25.0] => range(0.0, 90.0),
        pub rest: [f32; 2] = [90.0, 90.0] => range(0.0, 90.0),
        pub gather: f32 = 0.0 => range(0.0, 1.0e4),
        pub clear: f32 = 16.0 => range(0.001, 1.0e5),
    }
}

engine_codec::record!(TalusLayer {
    level,
    seed,
    reach,
    sag,
    relief,
    block,
    runnel,
    levee,
    width,
    chute,
    apart,
    slope,
    rest,
    gather,
    clear,
});

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Runnels {
    pub points: Vec<[f32; 4]>,
    pub across: u32,
    pub starts: Vec<u32>,
    pub listed: Vec<u32>,
}

engine_codec::record!(Runnels {
    points,
    across,
    starts,
    listed,
});

#[derive(Clone, Debug, PartialEq)]
pub struct Talus {
    pub spacing: f32,
    pub fall: Sheet,
    pub share: Sheet,
    pub sag: Sheet,
    pub runnels: Runnels,
}

engine_codec::record!(Talus {
    spacing,
    fall,
    share,
    sag,
    runnels,
});

#[derive(Clone, Debug, PartialEq)]
pub struct TalusStage {
    pub layer: TalusLayer,
    pub baked: Option<Arc<Talus>>,
}

engine_codec::record!(TalusStage { layer, baked });

pub static TALUS: LayerKind = LayerKind {
    name: "talus",
    source: crate::SOURCE.hash(),
    from_json,
    read,
    files: no_files,
    schema: engine_heightfield::schema::<TalusLayer>,
    points: no_points,
};

fn made(stage: TalusStage) -> Made {
    (stage.layer.validate()).map_err(|invalid| invalid.error(INVALID_TERRAIN, "$"))?;
    for (path, [low, high]) in [("$.rest", stage.layer.rest), ("$.slope", stage.layer.slope)] {
        if low > high {
            return Err(Error::new(INVALID_TERRAIN, path)
                .expected(Expected::Sorted)
                .got([low, high]));
        }
    }
    Ok(Arc::new(stage))
}

fn from_json(value: &Value, _files: &dyn LayerFiles) -> Made {
    made(TalusStage {
        layer: parse_value(value.clone())?,
        baked: None,
    })
}

fn read(input: &mut Reader<'_>) -> Made {
    made(input.get()?)
}

impl Runnels {
    fn nearest(&self, x: f32, z: f32) -> Option<(f32, f32, f32)> {
        if self.across == 0 || x < 0.0 || z < 0.0 {
            return None;
        }
        let (column, row) = ((x / BUCKET) as usize, (z / BUCKET) as usize);
        let bucket = row * self.across as usize + column;
        if column >= self.across as usize || bucket + 1 >= self.starts.len() {
            return None;
        }
        let listed = &self.listed[self.starts[bucket] as usize..self.starts[bucket + 1] as usize];
        let mut nearest: Option<(f32, f32, f32)> = None;
        for &first in listed {
            let (from, to) = (self.points[first as usize], self.points[first as usize + 1]);
            let (ax, az) = (to[0] - from[0], to[1] - from[1]);
            let length = max(ax * ax + az * az, 1.0e-12);
            let t = clamp(((x - from[0]) * ax + (z - from[1]) * az) / length, 0.0, 1.0);
            let (dx, dz) = (x - from[0] - ax * t, z - from[1] - az * t);
            let away = (dx * dx + dz * dz).sqrt();
            if nearest.is_none_or(|(known, _, _)| away < known) {
                nearest = Some((
                    away,
                    from[2] + (to[2] - from[2]) * t,
                    from[3] + (to[3] - from[3]) * t,
                ));
            }
        }
        nearest
    }

    fn indexed(points: Vec<[f32; 4]>, ends: &[usize], (width, depth): (f32, f32), far: f32) -> Self {
        let (across, down) = (
            (width / BUCKET) as usize + 1,
            (depth / BUCKET) as usize + 1,
        );
        let mut held: Vec<Vec<u32>> = vec![Vec::new(); across * down];
        let mut first = 0;
        for &end in ends {
            for index in first..end.saturating_sub(1) {
                let (from, to) = (points[index], points[index + 1]);
                let span = |a: f32, b: f32, last: usize| {
                    let low = (max(min(a, b) - far, 0.0) / BUCKET) as usize;
                    let high = (max(max(a, b) + far, 0.0) / BUCKET) as usize;
                    low.min(last)..=high.min(last)
                };
                for row in span(from[1], to[1], down - 1) {
                    for column in span(from[0], to[0], across - 1) {
                        held[row * across + column].push(index as u32);
                    }
                }
            }
            first = end;
        }
        let mut starts = Vec::with_capacity(held.len() + 1);
        let mut listed = Vec::new();
        for bucket in &held {
            starts.push(listed.len() as u32);
            listed.extend_from_slice(bucket);
        }
        starts.push(listed.len() as u32);
        Self {
            points,
            across: across as u32,
            starts,
            listed,
        }
    }
}

impl TalusStage {
    fn carved(&self, runnels: &Runnels, x: f32, z: f32) -> f32 {
        let layer = &self.layer;
        let Some((away, along, left)) = runnels.nearest(x, z) else {
            return 0.0;
        };
        let half = 0.5 * layer.width;
        let head = smooth(along / (2.0 * layer.width));
        let toe = smooth(left / (2.0 * layer.width));
        let channel = 1.0 - smooth(away / half);
        let rim = clamp((away - half) / layer.width, 0.0, 1.0);
        let bank = 16.0 * rim * rim * (1.0 - rim) * (1.0 - rim);
        head * (layer.levee * bank - layer.runnel * channel * toe)
    }
}

impl TerrainLayer for TalusStage {
    fn kind(&self) -> &'static LayerKind {
        &TALUS
    }

    fn write(&self, out: &mut Writer) {
        out.put(self);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(self.layer).expect("talus_settings_serialize")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn bakes(&self) -> bool {
        true
    }

    fn outputs(&self) -> Vec<String> {
        vec![FALL.to_owned(), DEBRIS.to_owned()]
    }

    fn images(&self) -> Vec<String> {
        self.outputs()
    }

    fn check(&self, place: Place<'_>) -> Result<(), Error> {
        if producer(place.below(), WALL).is_some() {
            return Ok(());
        }
        let published = (place.below().iter())
            .flat_map(|layer| layer.outputs())
            .collect();
        Err(unknown_map("$", WALL, published))
    }

    fn rests(&self, name: &str) -> Option<Rests> {
        (name == maps::DEPOSIT).then(|| Rests {
            slope: self.layer.rest.map(tangent),
            gather: self.layer.gather,
        })
    }

    fn row_cost(&self) -> u64 {
        3
    }

    fn apply_row(&self, below: Below<'_>, at: &Row<'_>, row: &mut [f32]) {
        let Some(baked) = self.baked.as_deref() else {
            return;
        };
        let (layer, Row { xs, z, spacing, .. }) = (&self.layer, *at);
        let own = (baked.spacing, spacing);
        let mut moved = vec![0.0_f32; xs.len()];
        if layer.sag > 0.0 {
            let sag = &baked.sag;
            image_row(&sag.image, sag.range, own, xs.iter().copied(), z, |index, sag| {
                moved[index] -= sag;
            });
        }
        let lumpy = layer.relief > 0.0 && 2.0 * layer.block >= RESOLVED * spacing;
        let seen = smooth(2.0 * layer.width / (RESOLVED * spacing) - 1.0);
        let cut = seen > 0.0 && !baked.runnels.listed.is_empty();
        if lumpy || cut {
            let mut share = vec![0.0_f32; xs.len()];
            let held = &baked.share;
            image_row(&held.image, held.range, own, xs.iter().copied(), z, |index, value| {
                share[index] = value;
            });
            for ((moved, &share), &x) in moved.iter_mut().zip(&share).zip(xs) {
                if share <= 0.0 {
                    continue;
                }
                if lumpy {
                    let lumps = blobs(x, z, layer.block, layer.seed ^ TALUS_BLOCKS);
                    *moved += layer.relief * share * lumps;
                }
                if cut {
                    *moved += self.carved(&baked.runnels, x, z) * seen * share;
                }
            }
        }
        let water = below.water();
        for ((height, &moved), &x) in row.iter_mut().zip(&moved).zip(xs) {
            if moved != 0.0 {
                let dry = water.map_or(1.0, |water| calm(water, (x, z, spacing), layer.clear));
                *height += moved * dry * below.shown(x, z);
            }
        }
    }

    fn needs_bake(&self) -> bool {
        self.baked.is_none()
    }

    fn bake(
        &self,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        tiles: TileHeights,
    ) -> Option<Arc<dyn TerrainLayer>> {
        self.needs_bake().then(|| {
            let gather = Gather::new(self.layer.level, below, pyramid, tiles);
            let grid = Grid {
                heights: &gather.heights,
                width: gather.width,
                rows: gather.heights.len() / gather.width,
                spacing: pyramid.spacing(gather.level),
            };
            let (wall, debris) = held(&grid, below);
            Arc::new(Self {
                layer: self.layer,
                baked: Some(Arc::new(Slopes::new(&grid, &wall, &debris).talus(&self.layer))),
            }) as Arc<dyn TerrainLayer>
        })
    }

    fn unbaked(&self) -> Option<Arc<dyn TerrainLayer>> {
        Some(Arc::new(Self {
            layer: self.layer,
            baked: None,
        }))
    }

    fn published(&self) -> Vec<Published<'_>> {
        let Some(baked) = self.baked.as_deref() else {
            return Vec::new();
        };
        [(FALL, &baked.fall), (DEBRIS, &baked.share)]
            .into_iter()
            .map(|(name, sheet)| Published {
                name,
                map: ChangeMap {
                    spacing: baked.spacing,
                    image: &sheet.image,
                    range: sheet.range,
                },
                rests: None,
            })
            .collect()
    }

    fn map_row(
        &self,
        name: &str,
        below: Below<'_>,
        row: &Row<'_>,
        _heights: &[f32],
        out: &mut [f32],
    ) -> bool {
        let Some(baked) = self.baked.as_deref() else {
            return false;
        };
        let sheet = match name {
            FALL => &baked.fall,
            DEBRIS => &baked.share,
            _ => return false,
        };
        let count = out.len().min(row.xs.len());
        image_row(
            &sheet.image,
            sheet.range,
            (baked.spacing, row.spacing),
            row.xs[..count].iter().copied(),
            row.z,
            |index, value| out[index] = value * below.shown(row.xs[index], row.z),
        );
        true
    }
}

fn held(grid: &Grid<'_>, below: &Layers) -> (Vec<f32>, Vec<f32>) {
    let cells = grid.heights.len();
    let (mut wall, mut debris) = (vec![0.0_f32; cells], vec![0.0_f32; cells]);
    let xs: Vec<f32> = (0..grid.width).map(|x| x as f32 * grid.spacing).collect();
    let cliffs = producer(below.layers(), WALL);
    for z in 0..grid.rows {
        let span = z * grid.width..(z + 1) * grid.width;
        let at = z as f32 * grid.spacing;
        let row = Row {
            xs: &xs,
            z: at,
            spacing: grid.spacing,
            low: Vec2::new(0.0, at),
            high: Vec2::new(xs[xs.len() - 1], at),
        };
        let heights = &grid.heights[span.clone()];
        read_map(below, cliffs, WALL, &row, heights, &mut wall[span.clone()]);
        let debris = &mut debris[span];
        for change in below.changes() {
            change.change_row(xs.iter().copied(), at, grid.spacing, |index, value| {
                debris[index] += value;
            });
        }
    }
    for depth in &mut debris {
        *depth = max(*depth, 0.0);
    }
    (wall, debris)
}

struct Slopes<'a> {
    grid: &'a Grid<'a>,
    wall: &'a [f32],
    debris: &'a [f32],
    order: Vec<u32>,
}

impl<'a> Slopes<'a> {
    fn new(grid: &'a Grid<'a>, wall: &'a [f32], debris: &'a [f32]) -> Self {
        let mut order: Vec<u32> = (0..grid.heights.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            grid.heights[b as usize].total_cmp(&grid.heights[a as usize])
        });
        Self {
            grid,
            wall,
            debris,
            order,
        }
    }

    fn around(&self, cell: usize) -> impl Iterator<Item = (usize, f32, f32)> + '_ {
        let grid = self.grid;
        let (x, z) = ((cell % grid.width) as isize, (cell / grid.width) as isize);
        let own = grid.heights[cell];
        AROUND.iter().filter_map(move |&(dx, dz, steps)| {
            let (nx, nz) = (x + dx, z + dz);
            if nx < 0 || nz < 0 || nx >= grid.width as isize || nz >= grid.rows as isize {
                return None;
            }
            let other = nz as usize * grid.width + nx as usize;
            let run = steps * grid.spacing;
            Some((other, (grid.heights[other] - own) / run, run))
        })
    }

    fn travelled(&self, upward: bool, start: impl Fn(usize, Option<f32>) -> f32) -> Vec<f32> {
        let mut far = vec![0.0_f32; self.grid.heights.len()];
        let mut visit = |cell: usize| {
            let (mut weight, mut sum) = (0.0_f32, 0.0_f32);
            for (other, rise, run) in self.around(cell) {
                let pull = if upward { rise } else { -rise };
                if pull > 0.0 {
                    weight += pull;
                    sum += pull * (far[other] + run);
                }
            }
            far[cell] = start(cell, (weight > 0.0).then(|| sum / weight));
        };
        if upward {
            self.order.iter().for_each(|&cell| visit(cell as usize));
        } else {
            self.order.iter().rev().for_each(|&cell| visit(cell as usize));
        }
        far
    }

    fn shares(&self, layer: &TalusLayer) -> Vec<f32> {
        let grid = self.grid;
        let [low, high] = layer.slope.map(tangent);
        let mut fed = vec![0.0_f32; grid.heights.len()];
        let mut share = vec![0.0_f32; grid.heights.len()];
        for &cell in &self.order {
            let cell = cell as usize;
            let (mut weight, mut sum) = (0.0_f32, 0.0_f32);
            for (other, rise, _) in self.around(cell) {
                if rise > 0.0 {
                    weight += rise;
                    sum += rise * fed[other];
                }
            }
            let at = ((cell % grid.width) as isize, (cell / grid.width) as isize);
            let [gx, gz] = grid.gradient(at.0, at.1);
            let steep = (gx * gx + gz * gz).sqrt();
            let rests = if high > low {
                smooth((steep - low) / (high - low))
            } else if steep >= low {
                1.0
            } else {
                0.0
            };
            let loose = clamp(self.debris[cell] / TRACE - 1.0, 0.0, 1.0);
            let wall = clamp(self.wall[cell], 0.0, 1.0);
            let reached = if weight > 0.0 && rests > 0.0 {
                sum / weight
            } else {
                0.0
            };
            fed[cell] = wall + (1.0 - wall) * reached;
            share[cell] = (1.0 - wall) * reached * rests * loose;
        }
        share
    }

    fn heads(&self, layer: &TalusLayer, share: &[f32]) -> Vec<usize> {
        let grid = self.grid;
        let area = grid.spacing * grid.spacing;
        let mut inflow = vec![0.0_f32; grid.heights.len()];
        let mut found: Vec<(f32, usize)> = Vec::new();
        for &cell in &self.order {
            let cell = cell as usize;
            let wall = clamp(self.wall[cell], 0.0, 1.0);
            if inflow[cell] * (1.0 - wall) * share[cell] >= layer.chute {
                found.push((inflow[cell], cell));
            }
            let carried = (inflow[cell] + area) * wall;
            let lowest = (self.around(cell))
                .filter(|&(_, rise, _)| rise < 0.0)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let (true, Some((other, _, _))) = (carried > 0.0, lowest) {
                inflow[other] += carried;
            }
        }
        found.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let place = |cell: usize| {
            Vec2::new((cell % grid.width) as f32, (cell / grid.width) as f32) * grid.spacing
        };
        let mut heads: Vec<usize> = Vec::new();
        for (_, cell) in found {
            if heads.len() >= MAX_RUNNELS {
                break;
            }
            let near = |&head: &usize| place(head).distance(place(cell)) < layer.apart;
            if !heads.iter().any(near) {
                heads.push(cell);
            }
        }
        heads
    }

    fn runnels(&self, layer: &TalusLayer, share: &[f32]) -> Runnels {
        let grid = self.grid;
        let (mut points, mut ends) = (Vec::new(), Vec::new());
        if max(layer.runnel, layer.levee) <= 0.0 {
            return Runnels::default();
        }
        for head in self.heads(layer, share) {
            let first = points.len();
            let (mut x, mut z) = ((head % grid.width) as f32, (head / grid.width) as f32);
            let mut along = 0.0_f32;
            loop {
                points.push([x * grid.spacing, z * grid.spacing, along, 0.0]);
                let fall = [
                    grid.smooth(x + 0.5, z) - grid.smooth(x - 0.5, z),
                    grid.smooth(x, z + 0.5) - grid.smooth(x, z - 0.5),
                ];
                let steep = (fall[0] * fall[0] + fall[1] * fall[1]).sqrt();
                if steep < 1.0e-4 * grid.spacing || along >= layer.reach {
                    break;
                }
                (x, z) = (x - fall[0] / steep * STRIDE, z - fall[1] / steep * STRIDE);
                along += STRIDE * grid.spacing;
                let inside = x >= 0.0
                    && z >= 0.0
                    && x <= (grid.width - 1) as f32
                    && z <= (grid.rows - 1) as f32;
                if !inside || grid.sample(share, x, z) < 0.5 {
                    break;
                }
            }
            if along < 4.0 * layer.width {
                points.truncate(first);
                continue;
            }
            for point in &mut points[first..] {
                point[3] = along - point[2];
            }
            ends.push(points.len());
        }
        let size = (
            (grid.width - 1) as f32 * grid.spacing,
            (grid.rows - 1) as f32 * grid.spacing,
        );
        Runnels::indexed(points, &ends, size, 1.5 * layer.width)
    }

    fn talus(&self, layer: &TalusLayer) -> Talus {
        let fall = self.travelled(true, |cell, above| {
            (1.0 - clamp(self.wall[cell], 0.0, 1.0)) * above.map_or(layer.reach, |far| min(far, layer.reach))
        });
        let share = self.shares(layer);
        let on_talus = |cell: usize, far: Option<f32>| match far {
            Some(far) if share[cell] > 0.0 => far,
            Some(_) | None => 0.0,
        };
        let above = self.travelled(true, on_talus);
        let under = self.travelled(false, on_talus);
        let sag: Vec<f32> = (0..self.grid.heights.len())
            .map(|cell| {
                let along = above[cell] / max(above[cell] + under[cell], 1.0e-6);
                layer.sag * self.debris[cell] * share[cell] * 4.0 * along * (1.0 - along)
            })
            .collect();
        Talus {
            spacing: self.grid.spacing,
            fall: sheet(&fall, self.grid.width),
            sag: sheet(&sag, self.grid.width),
            runnels: self.runnels(layer, &share),
            share: sheet(&share, self.grid.width),
        }
    }
}
