//! Erosion: rivers wear valleys into the layers below it and
//! slopes creep toward them, baked at one pyramid level. Each round fills
//! the grid's pits as sediment (Priority-Flood with epsilon), routes every
//! sample's water to its steepest neighbor the flood reached first, sums
//! upstream area, lowers each sample toward its receiver by the stream
//! power law solved implicitly downstream first (Braun and Willett 2013,
//! with incision growing as the square root of the area), and lets
//! hillslopes creep. What it changed from the layers below is a signed
//! 16-bit image with coarser levels, added at every terrain level, so the
//! finer layers' detail rides the eroded shape. Only exact arithmetic and
//! square roots enter it: every target bakes the same bits.

use crate::MAX_LEVEL;
use engine_math::{max, min};
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use crate::hydrology::{Drainage, drainage, image_level};
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
        /// Square meters of hillslope creep each round.
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
    /// Meters erosion adds at terrain-local `(x, z)` as terrain samples
    /// `spacing` meters apart see it.
    #[inline]
    pub(crate) fn change_at(&self, x: f32, z: f32, spacing: f32) -> f32 {
        let level = image_level(self.spacing, self.change.levels(), spacing);
        let scale = 1.0 / self.spacing;
        // Bilinear: cubic interpolation would dig pits beside the valleys'
        // sharp cuts.
        let sample = self.change.sample_linear(x * scale, z * scale, level);
        self.range[0] + (self.range[1] - self.range[0]) * sample * (1.0 / 65_535.0)
    }
}

/// One round of `layer` over the grid of `heights` (`width` samples wide,
/// `spacing` meters apart): hillslopes creep, then rivers cut, so every
/// round ends draining without pits.
pub(crate) fn erode_round(heights: &mut [f32], width: usize, spacing: f32, layer: &ErosionLayer) {
    // Hillslope creep: each sample eases toward its neighbors, within the
    // explicit scheme's stable step.
    if layer.creep > 0.0 {
        let height = heights.len() / width;
        let rate = min(layer.creep / (spacing * spacing), 0.24);
        let before = heights.to_vec();
        for z in 1..height.saturating_sub(1) {
            for x in 1..width - 1 {
                let at = z * width + x;
                let around =
                    before[at - 1] + before[at + 1] + before[at - width] + before[at + width];
                heights[at] = before[at] + rate * (around - 4.0 * before[at]);
            }
        }
    }
    let Drainage {
        filled,
        order,
        receivers,
        area,
        ..
    } = drainage(heights, width, spacing);
    // Pits fill as sediment; then, downstream first, each sample sinks
    // toward its receiver's new height, never below it.
    heights.copy_from_slice(&filled);
    for &sample in &order {
        let sample = sample as usize;
        let receiver = receivers[sample];
        if receiver == u32::MAX {
            continue;
        }
        let receiver = receiver as usize;
        let (dx, dz) = (
            (sample % width) as f32 - (receiver % width) as f32,
            (sample / width) as f32 - (receiver / width) as f32,
        );
        let length = (dx * dx + dz * dz).sqrt() * spacing;
        let factor = layer.incision * area[sample].sqrt() / length;
        let lowered = (heights[sample] + factor * heights[receiver]) / (1.0 + factor);
        heights[sample] = max(lowered, heights[receiver]);
    }
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
