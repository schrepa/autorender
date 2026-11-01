use engine_core::fields;
use engine_heightfield::hydrology::{Drainage, Flood};
use engine_heightfield::layer::image_row;
use engine_heightfield::{HeightImage, MAX_LEVEL};
use engine_math::{max, min};
use serde::{Deserialize, Serialize};

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct ErosionLayer {
        pub level: u32 = 4 => range(0, MAX_LEVEL),
        pub rounds: u32 = 60 => range(1, 400),
        pub incision: f32 = 0.002 => range(0.0, 1.0),
        pub debris: f32 = 0.004 => range(0.0, 1.0),
        pub creep: f32 = 20.0 => range(0.0, 1000.0),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ErosionStats {
    pub samples: usize,
    pub rounds: u32,
    pub deepest: f32,
    pub highest: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Erosion {
    pub spacing: f32,
    pub change: HeightImage,
    pub range: [f32; 2],
    pub ground: HeightImage,
    pub ground_range: [f32; 2],
    pub stats: ErosionStats,
}

impl Erosion {
    pub(crate) fn wear_row(
        &self,
        xs: &[f32],
        z: f32,
        spacing: f32,
        heights: &[f32],
        mut each: impl FnMut(usize, f32),
    ) {
        let count = heights.len().min(xs.len());
        let mut started = vec![0.0_f32; count];
        image_row(
            &self.ground,
            self.ground_range,
            (self.spacing, spacing),
            xs[..count].iter().copied(),
            z,
            |index, ground| started[index] = ground,
        );
        image_row(
            &self.change,
            self.range,
            (self.spacing, spacing),
            xs[..count].iter().copied(),
            z,
            |index, change| {
                let filled = min(max(change, 0.0), max(started[index] - heights[index], 0.0));
                each(index, change + filled);
            },
        );
    }
}

#[derive(Default)]
pub(crate) struct Scratch {
    drainage: Drainage,
    flood: Flood,
}

pub(crate) fn erode_round(
    heights: &mut Vec<f32>,
    (width, spacing): (usize, f32),
    layer: &ErosionLayer,
    round: u32,
    scratch: &mut Scratch,
) {
    let Scratch { drainage, flood } = scratch;
    let count = heights.len();
    let height = count / width;
    if layer.creep > 0.0 {
        let rate = min(layer.creep / (spacing * spacing), 0.24);
        let ground = &mut drainage.filled;
        ground.resize(count, 0.0);
        ground[..width].copy_from_slice(&heights[..width]);
        ground[count - width..].copy_from_slice(&heights[count - width..]);
        for z in 1..height - 1 {
            let row = z * width;
            ground[row] = heights[row];
            ground[row + width - 1] = heights[row + width - 1];
            for at in row + 1..row + width - 1 {
                let around =
                    heights[at - 1] + heights[at + 1] + heights[at - width] + heights[at + width];
                ground[at] = heights[at] + rate * (around - 4.0 * heights[at]);
            }
        }
    } else {
        std::mem::swap(heights, &mut drainage.filled);
    }
    drainage.route(width, spacing, round, flood);
    let Drainage {
        filled: ground,
        order,
        receivers,
        slopes,
        area,
    } = drainage;
    let diagonal = std::f32::consts::SQRT_2 * spacing;
    for &sample in order.iter() {
        let sample = sample as usize;
        let receiver = receivers[sample];
        if receiver == u32::MAX {
            continue;
        }
        let receiver = receiver as usize;
        let step = sample.abs_diff(receiver);
        let length = if step == 1 || step == width {
            spacing
        } else {
            diagonal
        };
        let factor =
            area[sample].sqrt() * (layer.incision + layer.debris * slopes[sample]) / length;
        let lowered = (ground[sample] + factor * ground[receiver]) / (1.0 + factor);
        ground[sample] = max(lowered, ground[receiver]);
    }
    std::mem::swap(heights, ground);
}

fn encoded(values: &[f32], width: usize) -> (HeightImage, [f32; 2]) {
    let mut range = [f32::INFINITY, f32::NEG_INFINITY];
    for &value in values {
        range = [min(range[0], value), max(range[1], value)];
    }
    let span = max(range[1] - range[0], 1.0e-3);
    let samples = values
        .iter()
        .map(|&value| {
            let unit = (value - range[0]) / span;
            (max(min(unit, 1.0), 0.0) * 65_535.0 + 0.5) as u16
        })
        .collect();
    let image = HeightImage::new(width as u32, (values.len() / width) as u32, samples)
        .expect("grid_two_samples_each_way");
    (image, [range[0], range[0] + span])
}

pub(crate) fn finish(
    original: &[f32],
    eroded: &[f32],
    width: usize,
    spacing: f32,
    rounds: u32,
) -> Erosion {
    let changes: Vec<f32> = original
        .iter()
        .zip(eroded)
        .map(|(before, after)| after - before)
        .collect();
    let (change, range) = encoded(&changes, width);
    let (ground, ground_range) = encoded(original, width);
    Erosion {
        spacing,
        change,
        range,
        ground,
        ground_range,
        stats: ErosionStats {
            samples: original.len(),
            rounds,
            deepest: -min(range[0], 0.0),
            highest: max(range[1], 0.0),
        },
    }
}

engine_codec::record!(ErosionLayer {
    level,
    rounds,
    incision,
    debris,
    creep,
});
engine_codec::record!(ErosionStats {
    samples,
    rounds,
    deepest,
    highest,
});
engine_codec::record!(Erosion {
    spacing,
    change,
    range,
    ground,
    ground_range,
    stats,
});
