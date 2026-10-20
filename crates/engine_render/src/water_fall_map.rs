//! Falling water's map (`meshes/water.wgsl`): one tiling texture of the
//! grain waterfalls' sheets and white water carry, drawn once from fixed
//! noise with its levels. Water reads it [`FALL_MAP_STRETCH`] times longer
//! along the water than across it, so what it holds is drawn that many
//! times shorter along it than it stands. Its red holds strands, broad
//! bands of denser water wandering down it; its green the fibers they draw
//! out into, noise integrated along lines that run down the water and
//! wander across it (line integral convolution, Cabral and Leedom), so its
//! grain streaks as moving water's does; both spread alike about a middle
//! of a half. Its blue holds the walls between foam's bubbles and its alpha
//! the bubbles, from 0 to 1 (cellular noise: points scattered one a cell,
//! the second nearest's distance less the nearest's, and the nearest's,
//! Worley).

use engine_core::math::{clamp, max, min};

use crate::gpu_layouts::constants::FALL_MAP_STRETCH;
use crate::water_waves::hash;

/// Texels along each side of the map, and its levels down to one texel.
const TEXELS: usize = 256;
const LEVELS: u32 = TEXELS.trailing_zeros() + 1;
/// The texels across and along the cells of the strands' two octaves, and
/// the finer's share.
const STRAND_CELLS: [[usize; 2]; 2] = [[32, 64], [16, 32]];
const STRAND_FINE: f32 = 0.35;
/// The texels either way along a fiber's line its noise is integrated
/// over, and the most texels its lines wander across, over cells
/// MEANDER_CELLS texels across and along.
const FIBER_REACH: i32 = 16;
const MEANDER: f32 = 16.0;
const MEANDER_CELLS: [usize; 2] = [64, 64];
/// The texels across foam's bubbles' cells (as many meters along as
/// across), and the share of a cell their walls span.
const BUBBLE_CELLS: usize = 16;
const BUBBLE_ALONG: usize = BUBBLE_CELLS / FALL_MAP_STRETCH as usize;
const BUBBLE_WALL: f32 = 0.2;
/// How far the strands and fibers spread about their middle (their
/// standard deviation).
const SPREAD: f32 = 0.17;
/// The patterns' seeds: the strands' octaves, the fibers' noise, their
/// meander and the bubbles'.
const SEEDS: [u32; 5] = [
    0x9e37_79b9,
    0x85eb_ca6b,
    0xc2b2_ae35,
    0x27d4_eb2f,
    0x1656_67b1,
];

const _: () = assert!(
    TEXELS.is_multiple_of(STRAND_CELLS[0][0])
        && TEXELS.is_multiple_of(STRAND_CELLS[0][1])
        && TEXELS.is_multiple_of(STRAND_CELLS[1][0])
        && TEXELS.is_multiple_of(STRAND_CELLS[1][1])
        && TEXELS.is_multiple_of(MEANDER_CELLS[0])
        && TEXELS.is_multiple_of(MEANDER_CELLS[1])
        && TEXELS.is_multiple_of(BUBBLE_CELLS)
        && BUBBLE_ALONG * FALL_MAP_STRETCH as usize == BUBBLE_CELLS,
    "the map's cells tile it"
);

/// The map, sampled with water's repeating sampler.
pub(crate) struct FallMap {
    view: wgpu::TextureView,
}

impl FallMap {
    /// The map drawn and uploaded with its levels.
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let side = TEXELS as u32;
        let texture = crate::targets::texture_2d(
            device,
            "engine/water-fall-map",
            (side, side),
            1,
            LEVELS,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        for (level, texels) in levels(map()).iter().enumerate() {
            let side = side >> level;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(texels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * 4),
                    rows_per_image: Some(side),
                },
                wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: 1,
                },
            );
        }
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Bytes of the map with its levels.
    pub(crate) fn capacity_bytes() -> u64 {
        (0..LEVELS)
            .map(|level| ((TEXELS >> level) * (TEXELS >> level) * 4) as u64)
            .sum()
    }
}

/// The map at full size, its rows along the water: strands, fibers,
/// bubbles' walls and bubbles.
fn map() -> Vec<[f32; 4]> {
    let count = TEXELS * TEXELS;
    // How far across each texel's line has wandered.
    let meander: Vec<f32> = (0..count)
        .map(|index| {
            let (x, y) = (index % TEXELS, index / TEXELS);
            MEANDER * (2.0 * value_noise([x as f32, y as f32], MEANDER_CELLS, SEEDS[3]) - 1.0)
        })
        .collect();
    let strands = spread(
        (0..count)
            .map(|index| {
                let at = [
                    (index % TEXELS) as f32 - meander[index],
                    (index / TEXELS) as f32,
                ];
                (1.0 - STRAND_FINE) * value_noise(at, STRAND_CELLS[0], SEEDS[0])
                    + STRAND_FINE * value_noise(at, STRAND_CELLS[1], SEEDS[1])
            })
            .collect(),
    );
    let fibers = spread(fibers(&meander));
    let bubbles = bubbles();
    (0..count)
        .map(|index| {
            [
                strands[index],
                fibers[index],
                bubbles[index][0],
                bubbles[index][1],
            ]
        })
        .collect()
}

/// A number in [0, 1) hashed from `bits`.
fn unit(bits: u32) -> f32 {
    (hash(bits) >> 8) as f32 / 16_777_216.0
}

/// Tiling value noise at texel `at` over cells `cells` texels across and
/// along: smooth between values hashed at its cells' corners.
fn value_noise(at: [f32; 2], cells: [usize; 2], seed: u32) -> f32 {
    let (u, v) = (at[0] / cells[0] as f32, at[1] / cells[1] as f32);
    let (column, row) = (u.floor(), v.floor());
    let periods = [(TEXELS / cells[0]) as i32, (TEXELS / cells[1]) as i32];
    let corner = |dx: i32, dy: i32| {
        let i = (column as i32 + dx).rem_euclid(periods[0]) as u32;
        let j = (row as i32 + dy).rem_euclid(periods[1]) as u32;
        unit(i ^ hash(j ^ seed))
    };
    let fade = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (fade(u - column), fade(v - row));
    let top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * sx;
    let bottom = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * sx;
    top + (bottom - top) * sy
}

/// Each texel's fiber: noise smoothed across three texels, integrated over
/// FIBER_REACH texels either way along its line (each texel's line as far
/// across as `meander` says it has wandered), the nearer the more (a
/// biweight).
fn fibers(meander: &[f32]) -> Vec<f32> {
    let count = TEXELS * TEXELS;
    let noise: Vec<f32> = (0..count)
        .map(|index| unit(index as u32 ^ SEEDS[2]))
        .collect();
    let smoothed: Vec<f32> = (0..count)
        .map(|index| {
            let (x, y) = (index % TEXELS, index / TEXELS);
            let at = |step: usize| noise[y * TEXELS + (x + step) % TEXELS];
            0.25 * at(TEXELS - 1) + 0.5 * at(0) + 0.25 * at(1)
        })
        .collect();
    let across = |x: f32, row: usize| {
        let column = x.floor();
        let share = x - column;
        let left = (column as i32).rem_euclid(TEXELS as i32) as usize;
        let right = (left + 1) % TEXELS;
        smoothed[row * TEXELS + left] * (1.0 - share) + smoothed[row * TEXELS + right] * share
    };
    let weights: Vec<f32> = (-FIBER_REACH..=FIBER_REACH)
        .map(|step| {
            let t = step as f32 / (FIBER_REACH + 1) as f32;
            (1.0 - t * t) * (1.0 - t * t)
        })
        .collect();
    let total: f32 = weights.iter().sum();
    (0..count)
        .map(|index| {
            let (x, y) = (index % TEXELS, index / TEXELS);
            let mut sum = 0.0;
            for (step, weight) in (-FIBER_REACH..=FIBER_REACH).zip(&weights) {
                let row = (y as i32 + step).rem_euclid(TEXELS as i32) as usize;
                let wandered = meander[row * TEXELS + x] - meander[index];
                sum += weight * across(x as f32 + wandered, row);
            }
            sum / total
        })
        .collect()
}

/// Each texel's bubbles' walls and bubble: in cells BUBBLE_CELLS texels
/// across and BUBBLE_ALONG along, each holding a point, 1 where the
/// nearest two points' distances meet (within BUBBLE_WALL of a cell) and
/// fading out from the nearest point.
fn bubbles() -> Vec<[f32; 2]> {
    let periods = [
        (TEXELS / BUBBLE_CELLS) as i32,
        (TEXELS / BUBBLE_ALONG) as i32,
    ];
    let point = |i: i32, j: i32| {
        let key = hash(
            i.rem_euclid(periods[0]) as u32 ^ hash(j.rem_euclid(periods[1]) as u32 ^ SEEDS[4]),
        );
        [
            i as f32 + 0.15 + 0.7 * unit(key),
            j as f32 + 0.15 + 0.7 * unit(key ^ 0x68e3_1da4),
        ]
    };
    (0..TEXELS * TEXELS)
        .map(|index| {
            let u = ((index % TEXELS) as f32 + 0.5) / BUBBLE_CELLS as f32;
            let v = ((index / TEXELS) as f32 + 0.5) / BUBBLE_ALONG as f32;
            let (i, j) = (u.floor() as i32, v.floor() as i32);
            let (mut nearest, mut second) = (f32::INFINITY, f32::INFINITY);
            for dj in -1..=1 {
                for di in -1..=1 {
                    let [x, y] = point(i + di, j + dj);
                    let distance = ((u - x) * (u - x) + (v - y) * (v - y)).sqrt();
                    if distance < nearest {
                        second = nearest;
                        nearest = distance;
                    } else if distance < second {
                        second = distance;
                    }
                }
            }
            let wall = clamp(1.0 - (second - nearest) / BUBBLE_WALL, 0.0, 1.0);
            [
                wall * wall * (3.0 - 2.0 * wall),
                max(1.0 - nearest / 0.75, 0.0),
            ]
        })
        .collect()
}

/// `values` spread SPREAD about a middle of a half.
fn spread(values: Vec<f32>) -> Vec<f32> {
    let count = values.len() as f32;
    let mean = values.iter().sum::<f32>() / count;
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / count;
    let gain = SPREAD / max(variance.sqrt(), 1.0e-6);
    values
        .into_iter()
        .map(|value| 0.5 + (value - mean) * gain)
        .collect()
}

/// The map's levels as texels: the map, then each level averaging the one
/// above it two texels each way.
fn levels(map: Vec<[f32; 4]>) -> Vec<Vec<[u8; 4]>> {
    let mut levels = Vec::with_capacity(LEVELS as usize);
    let mut level = map;
    let mut side = TEXELS;
    loop {
        levels.push(
            level
                .iter()
                .map(|texel| texel.map(|value| (min(max(value, 0.0), 1.0) * 255.0 + 0.5) as u8))
                .collect(),
        );
        if side == 1 {
            return levels;
        }
        let half = side / 2;
        level = (0..half * half)
            .map(|index| {
                let (x, y) = (index % half * 2, index / half * 2);
                let at = |dx: usize, dy: usize| level[(y + dy) * side + x + dx];
                let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
                [0, 1, 2, 3]
                    .map(|channel| 0.25 * (a[channel] + b[channel] + c[channel] + d[channel]))
            })
            .collect();
        side = half;
    }
}
