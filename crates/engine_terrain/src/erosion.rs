//! Erosion: rivers wear valleys into the layers below it, debris flows
//! scour its steep slopes into gullies and hillslopes creep toward them,
//! baked at one pyramid level. Each round lets hillslopes creep, fills the
//! grid's pits as sediment (Priority-Flood with epsilon), sends every
//! sample's water to one of its lower neighbors, drawn afresh each round
//! with odds in proportion to the slope down to each, sums upstream area,
//! and lowers each sample toward its receiver by the stream power law
//! solved implicitly downstream first (Braun and Willett 2013, with
//! incision growing as the square root of the area), debris flows scouring
//! on top of it as the square of the slope: steep ground gullies fastest,
//! and debris flows alone gully steep ground while hardly touching gentle
//! ground. What it changed from the layers below is a signed 16-bit image
//! with coarser levels, read through its uniform cubic B-spline and added
//! at every terrain level, so the finer layers' detail rides the eroded
//! shape without the bake's grid printed into it. Where it laid sediment
//! down, the sediment fills the hollows of that finer detail as deep as it
//! lies, as sediment buries the relief it settles on: valley floors lie
//! smooth instead of holding the finer detail's pits. Only exact
//! arithmetic, square roots and integer hashes enter it: every target
//! bakes the same bits.

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
        /// How fast the debris flows of steep headwaters scour each round,
        /// on top of the rivers: their erodibility times the time step for
        /// each unit of slope, so they cut as the square of the slope where
        /// stream power cuts as the slope, and steep slopes carve gullies
        /// rather than stand smooth.
        pub debris: f32 = 0.004 => range(0.0, 1.0),
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
/// away), over `range`, and the ground it started from (terrain-local
/// meters), over `ground_range`, at samples `spacing` meters apart from the
/// terrain's first corner.
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
        each: impl FnMut(usize, f32),
    ) {
        Self::read_row(
            &self.change,
            self.range,
            (self.spacing, spacing),
            xs,
            z,
            each,
        );
    }

    /// The meters erosion adds at terrain-local `(x, z)` for each of `xs`
    /// in turn, where the layers below it stand at `heights` (one each), as
    /// terrain samples `spacing` meters apart see it, handed to `each` with
    /// its index: the change ([`Self::change_row`]), and where erosion laid
    /// sediment down, the hollows the finer detail cut below the ground it
    /// started from filled as deep as the sediment lies.
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
        Self::read_row(
            &self.ground,
            self.ground_range,
            (self.spacing, spacing),
            xs[..count].iter().copied(),
            z,
            |index, ground| started[index] = ground,
        );
        self.change_row(xs[..count].iter().copied(), z, spacing, |index, change| {
            let filled = min(max(change, 0.0), max(started[index] - heights[index], 0.0));
            each(index, change + filled);
        });
    }

    /// `image`'s samples over `range`, its samples `own` meters apart, as
    /// terrain samples `spacing` meters apart see them at terrain-local
    /// `(x, z)` for each of `xs`, through its uniform cubic B-spline, which
    /// never passes its samples, so it digs no pits beside the valleys'
    /// sharp cuts and leaves no creases along the image's rows and columns.
    #[inline]
    fn read_row(
        image: &HeightImage,
        [low, high]: [f32; 2],
        (own, spacing): (f32, f32),
        xs: impl IntoIterator<Item = f32>,
        z: f32,
        mut each: impl FnMut(usize, f32),
    ) {
        let level = image_level(own, image.levels(), spacing);
        let scale = 1.0 / own;
        let us = xs.into_iter().map(|x| x * scale);
        image.sample_smooth_row(us, z * scale, level, |index, sample| {
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
    // toward its receiver's new height, never below it: as fast as the
    // rivers cut there, and faster the steeper it falls, as debris flows
    // scour.
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
        let factor = area[sample].sqrt() * (layer.incision + layer.debris * slopes[sample]) / length;
        let lowered = (ground[sample] + factor * ground[receiver]) / (1.0 + factor);
        ground[sample] = max(lowered, ground[receiver]);
    }
    std::mem::swap(heights, ground);
}

/// `values` (`width` a row) as a 16-bit image over their own range, and
/// that range: a millimeter at least, which keeps the encoding finite.
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
        .expect("grids hold two samples each way");
    (image, [range[0], range[0] + span])
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
    ground,
    ground_range,
    stats,
});
