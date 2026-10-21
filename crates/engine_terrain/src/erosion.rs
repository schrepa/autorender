//! Erosion: rivers wear valleys into the layers below it, debris flows
//! scour its steep slopes into gullies and hillslopes creep toward them,
//! baked at one pyramid level. Each round lets hillslopes creep, fills the
//! grid's pits as sediment (Priority-Flood with epsilon), sends every
//! sample's water to one of its lower neighbors, drawn afresh each round
//! with odds in proportion to the slope down to each, sums upstream area,
//! and lowers each sample toward its receiver by the stream power law
//! solved implicitly downstream first (Braun and Willett 2013, with
//! incision growing as the square root of the area), cutting faster the
//! steeper the ground falls there, as debris flows do. What it changed from
//! the layers below is a signed 16-bit image with coarser levels, read
//! through its uniform cubic B-spline and added at every terrain level, so
//! the finer layers' detail rides the eroded shape without the bake's grid
//! printed into it. Only exact arithmetic, square roots and integer hashes
//! enter it: every target bakes the same bits.

use crate::MAX_LEVEL;
use engine_math::{max, min};
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use crate::hydrology::{Drainage, Flood, image_level};
use crate::layers::HeightImage;

settings! {
    /// A layer that erodes the layers below it.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct ErosionLayer {
        /// The pyramid level whose samples it erodes: finer levels wear finer
        /// valleys and take longer.
        pub level: u32 = 4 => range(0, MAX_LEVEL),
        /// Rounds of incision and creep.
        pub rounds: u32 = 60 => range(1, 400),
        /// How fast rivers cut each round: the stream power law's erodibility
        /// times its time step.
        pub incision: f32 = 0.002 => range(0.0, 1.0),
        /// How much faster water cuts steep ground, as the debris flows of
        /// steep headwaters scour it: `1 + debris * slope` times as fast as
        /// stream power alone, so steep slopes carve gullies rather than
        /// stand smooth.
        pub debris: f32 = 2.0 => range(0.0, 100.0),
        /// Square meters of hillslope creep each round, at most about a
        /// quarter of the square of the level's spacing: the most an
        /// explicit step keeps stable.
        pub creep: f32 = 20.0 => range(0.0, 1000.0),
    }
}

/// What an erosion bake did: samples it eroded, rounds, and the most it
/// wore away and filled (meters).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct ErosionStats {
    pub samples: usize,
    pub rounds: u32,
    pub deepest: f32,
    pub highest: f32,
}

/// What erosion changed: meters it added (negative where it wore ground
/// away), over `range`, at samples `spacing` meters apart from the
/// terrain's first corner.
#[derive(Clone, Debug, PartialEq)]
pub struct Erosion {
    pub spacing: f32,
    pub change: HeightImage,
    pub range: [f32; 2],
    pub stats: ErosionStats,
}

impl Erosion {
    /// Meters erosion adds at terrain-local `(x, z)`, as terrain samples
    /// `spacing` meters apart see it, for each of `xs` in turn, handed to
    /// `each` with its index: a row at a time, to the same bits as one
    /// point alone. The change is read through its image's uniform cubic
    /// B-spline, which never passes its samples, so it digs no pits beside
    /// the valleys' sharp cuts and leaves no creases along the image's rows
    /// and columns.
    #[inline]
    pub(crate) fn change_row(
        &self,
        xs: impl IntoIterator<Item = f32>,
        z: f32,
        spacing: f32,
        mut each: impl FnMut(usize, f32),
    ) {
        let level = image_level(self.spacing, self.change.levels(), spacing);
        let scale = 1.0 / self.spacing;
        let [low, high] = self.range;
        let us = xs.into_iter().map(|x| x * scale);
        self.change.sample_smooth_row(us, z * scale, level, |index, sample| {
            each(index, low + (high - low) * sample * (1.0 / 65_535.0));
        });
    }
}

/// What erosion's rounds keep from one to the next: how water drains over
/// the grid, whose filled heights hold the grid while a round lowers them,
/// and the flood's queues.
#[derive(Default)]
pub(crate) struct Scratch {
    drainage: Drainage,
    flood: Flood,
}

/// Round `round` of `layer` over the grid of `heights` (`width` samples
/// wide, `spacing` meters apart), with `scratch` from the rounds before:
/// hillslopes creep, then rivers cut, so every round ends draining without
/// pits.
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
    // Hillslope creep into the drainage's grid: each sample eases toward
    // its neighbors, within the explicit scheme's stable step; the edges
    // hold.
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
    // Pits fill as sediment; then, downstream first, each sample sinks
    // toward its receiver's new height, never below it, faster the steeper
    // it falls there.
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
            layer.incision * area[sample].sqrt() * (1.0 + layer.debris * slopes[sample]) / length;
        let lowered = (ground[sample] + factor * ground[receiver]) / (1.0 + factor);
        ground[sample] = max(lowered, ground[receiver]);
    }
    std::mem::swap(heights, ground);
}

/// What erosion changed from `original` to `eroded` (`width` samples
/// wide, `spacing` meters apart) over `rounds` rounds.
pub(crate) fn finish(
    original: &[f32],
    eroded: &[f32],
    width: usize,
    spacing: f32,
    rounds: u32,
) -> Erosion {
    let mut range = [0.0_f32, 0.0_f32];
    for (before, after) in original.iter().zip(eroded) {
        let change = after - before;
        range = [min(range[0], change), max(range[1], change)];
    }
    // A span of at least a millimeter keeps the encoding finite.
    let span = max(range[1] - range[0], 1.0e-3);
    let samples = original
        .iter()
        .zip(eroded)
        .map(|(before, after)| {
            let unit = ((after - before) - range[0]) / span;
            (max(min(unit, 1.0), 0.0) * 65_535.0 + 0.5) as u16
        })
        .collect();
    let height = original.len() / width;
    Erosion {
        spacing,
        change: HeightImage::new(width as u32, height as u32, samples)
            .expect("grids hold two samples each way"),
        range: [range[0], range[0] + span],
        stats: ErosionStats {
            samples: original.len(),
            rounds,
            deepest: -range[0],
            highest: range[1],
        },
    }
}

crate::codec::record!(ErosionLayer {
    level,
    rounds,
    incision,
    debris,
    creep,
});
crate::codec::record!(ErosionStats {
    samples,
    rounds,
    deepest,
    highest,
});
crate::codec::record!(Erosion {
    spacing,
    change,
    range,
    stats,
});
