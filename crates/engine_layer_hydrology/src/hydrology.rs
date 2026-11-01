use std::sync::Arc;

use engine_core::fields;
use engine_heightfield::bed::{HEAD_FLOOD, NARROWEST, bankfull};
use engine_heightfield::hydrology::{
    Drainage, FLOW_RANGE, Grid, HydrologyBake, HydrologyStats, LAKE_RANGE, NEIGHBORS,
    WETNESS_RANGE, WaterGrid, drainage, neighbor_steps,
};
use engine_heightfield::rivers::{RiverLine, Rivers};
use engine_heightfield::{HeightImage, MAX_LEVEL};
use engine_math::{max, min};
use serde::{Deserialize, Serialize};

use crate::given::{Drained, GivenLayers, courses};
use crate::rivers::{Channels, trace};

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct HydrologyLayer {
        pub level: u32 = 0 => range(0, MAX_LEVEL),
        pub discharge: f32 = DISCHARGE => range(0.001, 1000.0),
    }
}

const DISCHARGE: f32 = 1.0;

const SHORE_SAMPLES: usize = 2;
const SHORE_DROP: f32 = 0.5;
const DRY_DEPTH: f32 = 20.0;

const LAKE_DEPTH: f32 = 2.0;
const WET: f32 = 1.0e-3;

fn lakes(heights: &[f32], filled: &[f32], width: usize) -> Vec<bool> {
    let count = heights.len();
    let height = count / width;
    let depth = |sample: usize| filled[sample] - heights[sample];
    let mut lake = vec![false; count];
    let mut seen = vec![false; count];
    let mut region = Vec::new();
    for start in 0..count {
        if seen[start] || depth(start) <= WET {
            continue;
        }
        region.clear();
        region.push(start);
        seen[start] = true;
        let mut deepest = 0.0_f32;
        let mut next = 0;
        while next < region.len() {
            let sample = region[next];
            next += 1;
            deepest = max(deepest, depth(sample));
            let (x, z) = ((sample % width) as i32, (sample / width) as i32);
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x + dx, z + dz);
                if nx < 0 || nz < 0 || nx >= width as i32 || nz >= height as i32 {
                    continue;
                }
                let neighbor = nz as usize * width + nx as usize;
                if !seen[neighbor] && depth(neighbor) > WET {
                    seen[neighbor] = true;
                    region.push(neighbor);
                }
            }
        }
        if deepest >= LAKE_DEPTH {
            for &sample in &region {
                lake[sample] = true;
            }
        }
    }
    lake
}

fn encode(value: f32, range: [f32; 2]) -> u16 {
    let unit = (value - range[0]) / (range[1] - range[0]);
    (max(min(unit, 1.0), 0.0) * 65_535.0 + 0.5) as u16
}

fn spread(heights: &[f32], drained: &Drainage, width: usize, spacing: f32) -> (Vec<f32>, Vec<f32>) {
    let count = heights.len();
    let steps = neighbor_steps(width, spacing);
    let mut area = vec![spacing * spacing; count];
    let mut steepest = vec![0.0_f32; count];
    for &sample in drained.order.iter().rev() {
        let sample = sample as usize;
        let receiver = drained.receivers[sample];
        if receiver == u32::MAX {
            continue;
        }
        let ground = drained.filled[sample];
        let mut down = [0.0_f32; 8];
        let mut most = 0.0_f32;
        if ground <= heights[sample] {
            for (index, &(offset, inverse)) in steps.iter().enumerate() {
                let drop = ground - drained.filled[sample.wrapping_add_signed(offset)];
                if drop > 0.0 {
                    down[index] = drop * inverse;
                    most = max(most, down[index]);
                }
            }
        }
        let water = area[sample];
        if most == 0.0 {
            area[receiver as usize] += water;
            continue;
        }
        steepest[sample] = most;
        let scale = 1.0 / most;
        let mut total = 0.0_f32;
        for slope in &mut down {
            let ratio = *slope * scale;
            let squared = ratio * ratio;
            *slope = squared * squared;
            total += *slope;
        }
        let share = water / total;
        for (&(offset, _), &weight) in steps.iter().zip(&down) {
            if weight > 0.0 {
                area[sample.wrapping_add_signed(offset)] += weight * share;
            }
        }
    }
    (area, steepest)
}

pub(crate) fn bake(
    (heights, width, spacing): (&[f32], usize, f32),
    (landscape, across, apart): (&[f32], usize, f32),
    (layer, given, finest): (&HydrologyLayer, &GivenLayers<'_>, f32),
) -> (HydrologyBake, Vec<RiverLine>) {
    let count = landscape.len();
    let rows = count / across;
    let routed = drainage(landscape, across, apart);
    let lake = lakes(landscape, &routed.filled, across);
    let (spread_area, steepest) = spread(landscape, &routed, across, apart);
    let filled = routed.filled;
    let found = Found {
        filled: &filled,
        lakes: &lake,
    };
    let water = Arc::new(water(landscape, &found, across, apart));
    let mut lake_samples = 0;
    let mut flow = Vec::with_capacity(count);
    let mut wetness = Vec::with_capacity(count);
    let mut lakes = Vec::with_capacity(count);
    for sample in 0..count {
        let standing = if lake[sample] {
            lake_samples += 1;
            filled[sample] - landscape[sample]
        } else {
            0.0
        };
        flow.push(encode(libm::log10f(spread_area[sample]), FLOW_RANGE));
        let catchment = spread_area[sample] / apart;
        let index = libm::logf(catchment / max(steepest[sample], 1.0e-3));
        wetness.push(encode(index, WETNESS_RANGE));
        lakes.push(encode(standing, LAKE_RANGE));
    }
    let image = |samples: Vec<u16>| {
        HeightImage::new(across as u32, rows as u32, samples).expect("grid_two_samples_each_way")
    };
    let Drainage {
        order,
        receivers,
        area,
        ..
    } = drainage(heights, width, spacing);
    let in_lake: Vec<bool> = (0..heights.len())
        .map(|sample| {
            let (x, z) = (
                (sample % width) as f32 * spacing,
                (sample / width) as f32 * spacing,
            );
            water
                .lake_reach(x, z)
                .is_some_and(|(level, _)| heights[sample] < level)
        })
        .collect();
    let (lines, channel_samples, max_area, given) = rivers_over(
        (heights, width, spacing),
        (water.as_ref(), &in_lake),
        (order, receivers, area),
        (layer, given, finest),
    );
    let hydrology = HydrologyBake {
        spacing: apart,
        flow: image(flow),
        wetness: image(wetness),
        lakes: image(lakes),
        water,
        rivers: Arc::new(Rivers::default()),
        stats: HydrologyStats {
            samples: heights.len(),
            channel_samples,
            lake_samples,
            max_area,
            given,
            ..HydrologyStats::default()
        },
        falls: Arc::from(Vec::new()),
    };
    (hydrology, lines)
}

fn rivers_over(
    (heights, width, spacing): (&[f32], usize, f32),
    (water, in_lake): (&WaterGrid, &[bool]),
    (order, receivers, area): (Vec<u32>, Vec<u32>, Vec<f32>),
    (layer, given, finest): (&HydrologyLayer, &GivenLayers<'_>, f32),
) -> (Vec<RiverLine>, usize, f32, usize) {
    let mut rank = vec![0_u32; heights.len()];
    for (position, &sample) in order.iter().enumerate() {
        rank[sample as usize] = position as u32;
    }
    let runoff = layer.discharge * 1.0e-6;
    let channel_area = HEAD_FLOOD / runoff;
    let given = courses(
        given,
        (
            &Grid {
                heights,
                width,
                spacing,
            },
            water,
        ),
        &Drained {
            order: &order,
            receivers: &receivers,
            area: &area,
            lakes: in_lake,
        },
        (runoff, NARROWEST * finest),
    );
    drop(order);
    let (area, within, continued, mut lines) = match given {
        Some(given) => (given.area, Some(given.within), given.continued, given.lines),
        None => (area, None, Vec::new(), Vec::new()),
    };
    let channel_samples = (0..area.len())
        .filter(|&sample| {
            (area[sample] >= channel_area && !in_lake[sample])
                || within.as_ref().is_some_and(|within| within[sample])
        })
        .count();
    let max_area = lines
        .iter()
        .flat_map(|line| &line.area)
        .chain(&area)
        .fold(0.0_f32, |most, &drained| max(most, drained));
    let given = lines.len();
    lines.extend(trace(
        &Channels {
            area: &area,
            receivers: &receivers,
            lakes: in_lake,
            rank: &rank,
            within: within.as_deref(),
            continued: &continued,
        },
        (width, spacing),
        channel_area,
        |drained| bankfull(drained * runoff).0,
    ));
    (lines, channel_samples, max_area, given)
}

struct Found<'a> {
    filled: &'a [f32],
    lakes: &'a [bool],
}

fn water(heights: &[f32], found: &Found<'_>, width: usize, spacing: f32) -> WaterGrid {
    let Found { filled, lakes } = *found;
    let count = heights.len();
    let height = count / width;
    let (last_x, last_z) = (width as i32 - 1, height as i32 - 1);
    let mut surface = vec![f32::NAN; count];
    let mut lake = vec![0_u16; count];
    let mut levels = Vec::new();
    let mut areas = Vec::new();
    let mut region = Vec::new();
    for start in 0..count {
        if !lakes[start] || lake[start] != 0 || levels.len() >= usize::from(u16::MAX) {
            continue;
        }
        levels.push(filled[start]);
        let number = levels.len() as u16;
        region.clear();
        region.push(start);
        lake[start] = number;
        let mut next = 0;
        while next < region.len() {
            let sample = region[next];
            next += 1;
            surface[sample] = filled[sample];
            let (x, z) = ((sample % width) as i32, (sample / width) as i32);
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x + dx, z + dz);
                if nx < 0 || nz < 0 || nx > last_x || nz > last_z {
                    continue;
                }
                let neighbor = nz as usize * width + nx as usize;
                if lakes[neighbor] && lake[neighbor] == 0 {
                    lake[neighbor] = number;
                    region.push(neighbor);
                }
            }
        }
        areas.push(region.len() as f32 * spacing * spacing);
    }
    let depth: Vec<f32> = (0..count)
        .map(|sample| {
            if surface[sample].is_nan() {
                0.0
            } else {
                max(surface[sample] - heights[sample], 0.0)
            }
        })
        .collect();
    for _ in 0..SHORE_SAMPLES {
        let (previous, numbers) = (surface.clone(), lake.clone());
        for sample in 0..count {
            if !previous[sample].is_nan() {
                continue;
            }
            let (x, z) = ((sample % width) as i32, (sample / width) as i32);
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x + dx, z + dz);
                if nx < 0 || nz < 0 || nx > last_x || nz > last_z {
                    continue;
                }
                let neighbor = nz as usize * width + nx as usize;
                let level = previous[neighbor];
                let ground = heights[sample];
                if !level.is_nan()
                    && ground >= level - SHORE_DROP
                    && (surface[sample].is_nan() || level > surface[sample])
                {
                    surface[sample] = level;
                    lake[sample] = numbers[neighbor];
                }
            }
        }
    }
    for (sample, level) in surface.iter_mut().enumerate() {
        if level.is_nan() {
            *level = heights[sample] - DRY_DEPTH;
        }
    }
    WaterGrid {
        width,
        height,
        spacing,
        surface,
        depth,
        flow: vec![[0.0; 2]; count],
        lake,
        levels,
        areas,
    }
}

engine_codec::record!(HydrologyLayer { level, discharge });
