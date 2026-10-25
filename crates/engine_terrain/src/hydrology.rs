//! Hydrology: where water runs over a terrain and the channels it
//! cuts. A bake reads the heights the layers below it give at one
//! pyramid level, fills their pits so that every sample drains off the
//! grid's edge (Priority-Flood with epsilon; Barnes, Lehman and Mulla
//! 2014), sends each sample's water to one of its lower neighbors, drawn
//! by a hash of the sample with odds in proportion to the slope down to
//! each, so courses follow the ground's fall rather than the grid's eight
//! ways, and sums upstream area in the reverse of the flood's order.
//! Inside a filled pit the flood spreads from the spill point over the
//! lowest ground first and water drains back the way the flood came, so
//! rivers crossing a pit follow its low line rather than straight rays from
//! the spill point. Channels then run where enough area drains, deeper and
//! wider downstream. The maps of flow and the wetness index spread each
//! sample's water over all its lower neighbors instead, and the depths of
//! filled pits follow from the same grids. Each result is a 16-bit image
//! with coarser levels, read through its uniform cubic B-spline, so every
//! terrain level samples it without aliasing or the image's grid, and
//! every sample is the same on every target: the bake is deterministic.
//! Rivers (`rivers.rs`) follow the channels as lines, each cutting its
//! channel into the ground with its water in it, the one place channels
//! are cut; the grid of standing water holds lakes alone.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::MAX_LEVEL;
use engine_math::{max, min};
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use std::sync::Arc;

use glam::Vec2;

use crate::bed::bankfull_width;
use crate::layers::HeightImage;
use crate::noise::hash;
use crate::rivers::{Channels, RiverLine, Rivers, trace};

settings! {
    /// A layer that bakes where water runs over the layers below it and cuts
    /// channels into them.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct HydrologyLayer {
        /// The pyramid level whose samples the bake reads: rivers are traced
        /// along its grid, and begin where their channels span two of its
        /// samples from bank to bank, the narrowest water settling over
        /// samples that far apart holds; its maps and lakes lie over the
        /// landscape [`MAP_LEVELS`] levels coarser. Finer levels trace narrower
        /// channels more closely and take longer.
        pub level: u32 = 0 => range(0, MAX_LEVEL),
        /// Cubic meters a second each square kilometer drained brings the
        /// channels full to their banks: every channel carries this times the
        /// area draining through it, which sets how wide and deep its channel
        /// stands, as channels full to their banks widen and deepen
        /// downstream, shaped by the slope of its reach.
        pub discharge: f32 = DISCHARGE => range(0.001, 1000.0),
    }
}

/// The default [`HydrologyLayer::discharge`]: a cubic meter a second each
/// square kilometer brings.
const DISCHARGE: f32 = 1.0;

/// What a bake found: samples read, samples channels run through, samples
/// of filled pits, the largest upstream area (square meters), the
/// waterfalls rivers drop over, the river points in each reach (Pool to
/// Falls, as [`crate::Reach`] numbers them), and the steps and knobs carved
/// into rivers' beds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct HydrologyStats {
    pub samples: usize,
    pub channel_samples: usize,
    pub lake_samples: usize,
    pub max_area: f32,
    pub falls: usize,
    pub reaches: [usize; 5],
    pub steps: usize,
    pub knobs: usize,
}

/// A waterfall: where a river leaves its bed over a drop, the lip's water
/// and where it lands at last (terrain-local meters), its water's path from
/// one to the other: through the air on each jet and down the chutes
/// between where one jet lands in the chute the next leaves (a cascade),
/// the seconds after leaving the lip its water passes each point of it,
/// whether it flies through the air from each point to the next, the
/// meters across its water there and how much of the sky it sees there (0
/// to 1, the ground around hiding the rest); the meters across the river at
/// its lip, the area draining over it (square meters), the water's speed as
/// it lands at last (meters a second) and the discharge it carries (cubic
/// meters a second).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Fall {
    pub top: [f32; 3],
    pub bottom: [f32; 3],
    pub path: Vec<[f32; 3]>,
    pub times: Vec<f32>,
    pub aloft: Vec<bool>,
    pub widths: Vec<f32>,
    pub sky: Vec<f32>,
    pub width: f32,
    pub area: f32,
    pub speed: f32,
    pub discharge: f32,
}

/// The ranges the baked images encode, from 0 to 65535: base-10 logarithms
/// of upstream area (square meters), wetness indexes and lake depths
/// (meters).
pub const FLOW_RANGE: [f32; 2] = [0.0, 12.0];
pub const WETNESS_RANGE: [f32; 2] = [-5.0, 25.0];
pub const LAKE_RANGE: [f32; 2] = [0.0, 200.0];

/// A bake's results over the terrain, from its first corner: images whose
/// samples lie `spacing` meters apart.
#[derive(Clone, Debug, PartialEq)]
pub struct Hydrology {
    pub spacing: f32,
    /// Upstream area, each sample's water spread over all its lower
    /// neighbors, as [`FLOW_RANGE`] encodes it.
    pub flow: HeightImage,
    /// The topographic wetness index ln(a / tan b): a the area draining
    /// through a meter of contour, spread as `flow`'s, b the ground's
    /// steepest slope down; as [`WETNESS_RANGE`] encodes it.
    pub wetness: HeightImage,
    /// How deep water would stand in filled pits, as [`LAKE_RANGE`]
    /// encodes it.
    pub lakes: HeightImage,
    /// Where lakes stand in filled pits.
    pub(crate) water: Arc<WaterGrid>,
    /// Rivers in the channels, their water set from the
    /// finest ground once the bake reads it.
    pub(crate) rivers: Arc<Rivers>,
    /// Where rivers fall.
    pub falls: Arc<[Fall]>,
    pub stats: HydrologyStats,
}

/// Where a bake's lakes stand, sample by sample of its grid: level at
/// their pits' spill heights. Surfaces reach a sample or two past the shore,
/// under the banks, so water meets the ground where they cross; elsewhere
/// they lie well below the ground. Each lake has a number: the samples
/// holding its water and those reaching past its shore name it (from 1; 0
/// for none), and `levels` and `areas` hold its level and the area its
/// water covers by number. Rivers stand on lines of their own ([`Rivers`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WaterGrid {
    pub width: usize,
    pub height: usize,
    /// Meters between samples, from the terrain's first corner.
    pub spacing: f32,
    /// The water's surface (terrain-local meters).
    pub surface: Vec<f32>,
    /// Meters of water above the ground; 0 where dry.
    pub depth: Vec<f32>,
    /// The water's velocity along x and z (meters a second); still in
    /// lakes.
    pub flow: Vec<[f32; 2]>,
    /// The lake each sample's surface belongs to, from 1; 0 for none.
    pub lake: Vec<u16>,
    /// Each lake's level (terrain-local meters), lake 1 first.
    pub levels: Vec<f32>,
    /// The square meters each lake's water covers, its connected samples'
    /// squares, lake 1 first.
    pub areas: Vec<f32>,
}

/// Samples past the shore water's surface reaches under the banks.
const SHORE_SAMPLES: usize = 2;
/// Meters below the water a dry sample's ground may stand and still take
/// its level.
const SHORE_DROP: f32 = 0.5;
/// Meters below the ground dry samples' surfaces lie.
const DRY_DEPTH: f32 = 20.0;

/// The eight neighbors, orthogonal first, and how far each lies in
/// samples.
const NEIGHBORS: [(i32, i32, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, std::f32::consts::SQRT_2),
    (-1, 1, std::f32::consts::SQRT_2),
    (1, -1, std::f32::consts::SQRT_2),
    (-1, -1, std::f32::consts::SQRT_2),
];

/// Meters of water a filled pit must hold somewhere to stand as a lake:
/// shallower pits stay dry and rivers run through them, such as the
/// hollows erosion's smoothly read change leaves, centimeters deep over the
/// pits its sediment filled and decimeters deep beside its sharpest cuts.
const LAKE_DEPTH: f32 = 2.0;
/// Meters of water a sample of a lake holds at least.
const WET: f32 = 1.0e-3;

/// Which samples lie in lakes: the connected samples a pit fills (eight
/// neighbors each), where it fills [`LAKE_DEPTH`] or deeper somewhere.
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
        // The pit's connected samples, and its deepest.
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

/// `height` as a key whose order as an unsigned integer is
/// [`f32::total_cmp`]'s: negative heights' bits flipped, the others' sign
/// bit set.
#[inline]
fn level_key(height: f32) -> u64 {
    let bits = height.to_bits();
    let flip = (((bits as i32) >> 31) as u32) | 0x8000_0000;
    u64::from(bits ^ flip)
}

/// Encodes `value` over `range` as a 16-bit sample.
fn encode(value: f32, range: [f32; 2]) -> u16 {
    let unit = (value - range[0]) / (range[1] - range[0]);
    (max(min(unit, 1.0), 0.0) * 65_535.0 + 0.5) as u16
}

/// Decodes a sample (0 to 65535, interpolated) over `range`.
pub fn decode(sample: f32, range: [f32; 2]) -> f32 {
    range[0] + (range[1] - range[0]) * sample * (1.0 / 65_535.0)
}

/// A flood's marks and queues, kept from one flood to the next: whether it
/// has reached each sample, the sample each one it raised spills into
/// (`u32::MAX` for the others), its open edge by height then sample, and
/// the raised samples of the pit it fills by their own ground, then by when
/// it reached them.
#[derive(Default)]
pub(crate) struct Flood {
    closed: Vec<bool>,
    parents: Vec<u32>,
    open: BinaryHeap<Reverse<u64>>,
    pits: BinaryHeap<Reverse<(u64, u32)>>,
}

impl Flood {
    /// Fills `heights`' pits so every sample drains off the edge of a grid
    /// `width` samples wide, each raised just above the sample it spills
    /// into; puts the samples in `order` in the order the flood reached
    /// them, and notes the sample each raised one spills into. The flood
    /// fills a pit from its spill point over the lowest ground first, in
    /// the order it reached samples of equal heights.
    fn fill(&mut self, heights: &mut [f32], width: usize, order: &mut Vec<u32>) {
        let count = heights.len();
        let height = count / width;
        let Self {
            closed,
            parents,
            open,
            pits,
        } = self;
        closed.clear();
        closed.resize(count, false);
        parents.clear();
        parents.resize(count, u32::MAX);
        open.clear();
        pits.clear();
        order.clear();
        order.reserve(count);
        let mut reached = 0_u32;
        let edges = (0..width)
            .flat_map(|x| [(x, 0), (x, height - 1)])
            .chain((0..height).flat_map(|z| [(0, z), (width - 1, z)]));
        for (x, z) in edges {
            let sample = z * width + x;
            if !closed[sample] {
                closed[sample] = true;
                open.push(Reverse((level_key(heights[sample]) << 32) | sample as u64));
            }
        }
        let (columns, rows) = (width as u32, height as u32);
        loop {
            let sample = match pits.pop() {
                Some(Reverse((_, sample))) => sample,
                None => match open.pop() {
                    // The sample is the key's low half.
                    Some(Reverse(key)) => key as u32,
                    None => break,
                },
            };
            order.push(sample);
            let (x, z) = (sample % columns, sample / columns);
            let spill = heights[sample as usize].next_up();
            for (dx, dz, _) in NEIGHBORS {
                let (nx, nz) = (x.wrapping_add_signed(dx), z.wrapping_add_signed(dz));
                if nx >= columns || nz >= rows {
                    continue;
                }
                let neighbor = (nz * columns + nx) as usize;
                if closed[neighbor] {
                    continue;
                }
                closed[neighbor] = true;
                let key = level_key(heights[neighbor]) << 32;
                if heights[neighbor] < spill {
                    pits.push(Reverse((key | u64::from(reached), neighbor as u32)));
                    reached += 1;
                    heights[neighbor] = spill;
                    parents[neighbor] = sample;
                } else {
                    open.push(Reverse(key | neighbor as u64));
                }
            }
        }
    }
}

/// Where water drains over a grid: its heights
/// with every pit filled to spill, the samples in the order the flood
/// reached them, each sample's receiver (`u32::MAX` on the edge, which
/// drains off the grid) and the slope down to it, and the area draining
/// through each sample (square meters).
#[derive(Default)]
pub(crate) struct Drainage {
    pub(crate) filled: Vec<f32>,
    pub(crate) order: Vec<u32>,
    pub(crate) receivers: Vec<u32>,
    pub(crate) slopes: Vec<f32>,
    pub(crate) area: Vec<f32>,
}

/// Seed of the hash drawing each sample's receiver among its lower
/// neighbors; each draw of receivers hashes under a seed of its own.
const RECEIVER_SEED: u32 = 0x2f6b_3c91;

/// The eight neighbors' offsets in a grid `width` samples wide, in
/// [`NEIGHBORS`]' order, and the inverse of the meters to each where its
/// samples lie `spacing` meters apart.
fn neighbor_steps(width: usize, spacing: f32) -> [(isize, f32); 8] {
    NEIGHBORS.map(|(dx, dz, distance)| {
        (
            dz as isize * width as isize + dx as isize,
            1.0 / (distance * spacing),
        )
    })
}

/// How water drains over the grid of `heights`, `width` samples wide
/// (row by row along z, at least two each way), whose samples lie
/// `spacing` meters apart.
pub(crate) fn drainage(heights: &[f32], width: usize, spacing: f32) -> Drainage {
    let mut drainage = Drainage {
        filled: heights.to_vec(),
        ..Drainage::default()
    };
    drainage.route(width, spacing, 0, &mut Flood::default());
    drainage
}

impl Drainage {
    /// How water drains over the grid its filled heights hold, `width`
    /// samples wide (row by row along z, at least two each way) and
    /// `spacing` meters apart, found in place: its pits filled, then its
    /// receivers, slopes and areas, in its own buffers and `flood`'s kept
    /// from the routing before. `draw` says which of the hash's draws picks
    /// the receivers.
    pub(crate) fn route(&mut self, width: usize, spacing: f32, draw: u32, flood: &mut Flood) {
        let Self {
            filled,
            order,
            receivers,
            slopes,
            area,
        } = self;
        let count = filled.len();
        let height = count / width;
        flood.fill(filled, width, order);
        // Each sample drains to one of its lower neighbors, drawn by a hash
        // of the sample with odds in proportion to the slope down to each.
        // The flood reached every lower neighbor of a sample it did not
        // raise before that sample, so courses only fall and the reverse of
        // its order visits every sample before the one it drains to. Each
        // raised sample drains to the one it spills into, back the way the
        // flood came; edge samples drain off the grid.
        receivers.clear();
        receivers.resize(count, u32::MAX);
        slopes.clear();
        slopes.resize(count, 0.0);
        let seed = RECEIVER_SEED.wrapping_add(draw.wrapping_mul(0x9e37_79b9));
        let steps = neighbor_steps(width, spacing);
        for z in 1..height - 1 {
            for x in 1..width - 1 {
                let sample = z * width + x;
                if flood.parents[sample] != u32::MAX {
                    receivers[sample] = flood.parents[sample];
                    continue;
                }
                let ground = filled[sample];
                let mut down = [0.0_f32; 8];
                let (mut total, mut first) = (0.0_f32, None);
                for (index, &(offset, inverse)) in steps.iter().enumerate() {
                    let drop = ground - filled[sample.wrapping_add_signed(offset)];
                    if drop > 0.0 {
                        down[index] = drop * inverse;
                        total += down[index];
                        first = first.or(Some(index));
                    }
                }
                let Some(first) = first else {
                    continue;
                };
                // A draw from 0 up to the slopes' sum falls within one of
                // theirs.
                let unit = (hash(x as i32, z as i32, seed) >> 8) as f32 * (1.0 / 16_777_216.0);
                let mut pick = unit * total;
                let mut chosen = first;
                for (index, &slope) in down.iter().enumerate().skip(first) {
                    if slope > 0.0 {
                        chosen = index;
                        if pick < slope {
                            break;
                        }
                        pick -= slope;
                    }
                }
                receivers[sample] = sample.wrapping_add_signed(steps[chosen].0) as u32;
                slopes[sample] = down[chosen];
            }
        }
        area.clear();
        area.resize(count, spacing * spacing);
        for &sample in order.iter().rev() {
            let receiver = receivers[sample as usize];
            if receiver != u32::MAX {
                area[receiver as usize] += area[sample as usize];
            }
        }
    }
}

/// The area draining through each sample of the grid `drained` routes over
/// `heights` (`width` samples wide, `spacing` meters apart, square meters)
/// as the maps read it, and the steepest slope down from each: every
/// sample's water spread over all its lower neighbors in proportion to the
/// fourth power of the slope down to each (Holmgren 1994), in the reverse
/// of the flood's order, so it neither runs in the grid's eight ways nor
/// gathers into single courses as rivers do. Samples the flood raised pass
/// theirs back the way the flood came, as they drain.
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
        // Shares of the steepest's, whose own is 1.
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

/// Levels coarser than a hydrology bake's own its maps and lakes are made
/// at: they lie smooth over the landscape, eight times as many meters a
/// sample as the grid rivers are traced along.
pub const MAP_LEVELS: u32 = 3;

/// Bakes `layer` over two grids of the ground (each its heights, samples
/// a row, row by row along z, at least two each way, and meters between
/// them): its own level's, along which water runs and rivers are traced,
/// and the landscape's, over which its maps and lakes lie. Its rivers'
/// lines come back for their channels and water to be set
/// (`crate::hydraulics::river`); it holds no rivers yet.
pub(crate) fn bake(
    (heights, width, spacing): (&[f32], usize, f32),
    (landscape, across, apart): (&[f32], usize, f32),
    layer: &HydrologyLayer,
) -> (Hydrology, Vec<RiverLine>) {
    let count = landscape.len();
    let rows = count / across;
    let routed = drainage(landscape, across, apart);
    let lake = lakes(landscape, &routed.filled, across);
    // The maps read water spread over every lower neighbor.
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
        // libm's logarithms, as browsers compute them: the platform's
        // differ.
        flow.push(encode(libm::log10f(spread_area[sample]), FLOW_RANGE));
        let catchment = spread_area[sample] / apart;
        let index = libm::logf(catchment / max(steepest[sample], 1.0e-3));
        wetness.push(encode(index, WETNESS_RANGE));
        lakes.push(encode(standing, LAKE_RANGE));
    }
    let image = |samples: Vec<u16>| {
        HeightImage::new(across as u32, rows as u32, samples)
            .expect("grids hold two samples each way")
    };
    // Rivers run where single courses gather water over the ground they
    // are cut into, its own grid: a sample lies in a lake where a lake's
    // level stands above its ground.
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
    let mut rank = vec![0_u32; heights.len()];
    for (position, &sample) in order.iter().enumerate() {
        rank[sample as usize] = position as u32;
    }
    drop(order);
    // Channels begin where their flood reaches the head's
    // ([`crate::bed::HEAD_FLOOD`]).
    let runoff = layer.discharge * 1.0e-6;
    let channel_area = crate::bed::HEAD_FLOOD / runoff;
    let channel_samples = area
        .iter()
        .zip(&in_lake)
        .filter(|&(&drained, &lake)| drained >= channel_area && !lake)
        .count();
    let max_area = area
        .iter()
        .fold(0.0_f32, |most, &drained| max(most, drained));
    let lines = trace(
        &Channels {
            area: &area,
            receivers: &receivers,
            lakes: &in_lake,
            rank: &rank,
        },
        (width, spacing),
        channel_area,
        |drained| bankfull_width(drained * runoff),
    );
    let hydrology = Hydrology {
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
            ..HydrologyStats::default()
        },
        falls: Arc::from(Vec::new()),
    };
    (hydrology, lines)
}

/// What a bake found about one place: the base-10 logarithm of its
/// upstream area (square meters), its wetness index, how deep water would
/// stand there in a filled pit (meters), and meters from the nearest
/// river's edge, negative in its water, [`RIVER_FAR`] beyond every river's
/// reach; and of the bed of the river holding it, the median size of its
/// grains (d50, meters) and the river's stream power at its bankfull flood
/// (watts per square meter of bed), 0 outside every river's bed, and meters
/// the ground stands above that flood's level, negative below it,
/// [`RIVER_FAR`] outside every river's bed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct HydrologySample {
    pub flow: f32,
    pub wetness: f32,
    pub lake: f32,
    pub river: f32,
    pub grain: f32,
    pub power: f32,
    pub flood: f32,
}

/// What [`HydrologySample::river`] reads beyond every river's reach.
pub const RIVER_FAR: f32 = 1.0e9;

impl Default for HydrologySample {
    fn default() -> Self {
        Self {
            flow: 0.0,
            wetness: 0.0,
            lake: 0.0,
            river: RIVER_FAR,
            grain: 0.0,
            power: 0.0,
            flood: RIVER_FAR,
        }
    }
}

impl Hydrology {
    /// The image level terrain samples `spacing` meters apart read: the
    /// coarsest whose samples lie no further apart than the terrain's.
    pub fn level_for(&self, spacing: f32) -> usize {
        image_level(self.spacing, self.flow.levels(), spacing)
    }

    /// What the bake found at terrain-local `(x, z)`, as terrain samples
    /// `spacing` meters apart see it.
    pub fn at(&self, x: f32, z: f32, spacing: f32) -> HydrologySample {
        let river = self.rivers.edge(x, z, spacing);
        self.masks_at(x, z, spacing, river)
    }

    /// [`Self::at`] with the meters to the nearest river's edge, `river`,
    /// already read: the images through their uniform cubic B-splines, as
    /// the GPU reads them.
    pub(crate) fn masks_at(&self, x: f32, z: f32, spacing: f32, river: f32) -> HydrologySample {
        let level = self.level_for(spacing);
        let (u, v) = (x / self.spacing, z / self.spacing);
        HydrologySample {
            flow: decode(self.flow.sample_smooth(u, v, level), FLOW_RANGE),
            wetness: decode(self.wetness.sample_smooth(u, v, level), WETNESS_RANGE),
            lake: decode(self.lakes.sample_smooth(u, v, level), LAKE_RANGE),
            river,
            ..HydrologySample::default()
        }
    }

    /// Its rivers, as a handle: what their flow and tiles' water are made
    /// from. Their lines and water are the crate's own: water is read from
    /// tiles ([`crate::Tile::water_at`]), courses from
    /// [`Rivers::courses`].
    pub fn rivers(&self) -> &Arc<Rivers> {
        &self.rivers
    }

    /// The water standing at terrain-local `(x, z)`, in a lake or a river,
    /// as bakes after this one read it before there are tiles: its surface
    /// and depth; `None` where dry.
    pub(crate) fn water_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        self.water.at(x, z).or_else(|| self.rivers.water_at(x, z))
    }

    /// Whether a lake or a river reaches into the terrain-local rectangle
    /// from `low` to `high`: a sample of a lake's water within a sample of
    /// it, or a river's channel, out as far again as it is wide, within a
    /// sample.
    pub(crate) fn water_reaches(&self, low: Vec2, high: Vec2) -> bool {
        let water = &self.water;
        let pad = Vec2::splat(water.spacing);
        let (first, last) = ((low - pad) / water.spacing, (high + pad) / water.spacing);
        let columns = max(first.x.floor(), 0.0) as usize
            ..=min(last.x.ceil(), water.width as f32 - 1.0) as usize;
        let lake = (max(first.y.floor(), 0.0) as usize
            ..=min(last.y.ceil(), water.height as f32 - 1.0) as usize)
            .any(|z| {
                columns.clone().any(|x| {
                    let sample = z * water.width + x;
                    water.lake[sample] != 0 && water.depth[sample] > 0.0
                })
            });
        lake || self.rivers.reaches(low - pad, high + pad)
    }
}

/// What a bake found that lakes stand by: the filled heights and lake
/// samples.
struct Found<'a> {
    filled: &'a [f32],
    lakes: &'a [bool],
}

/// Where lakes stand over the grid of `heights` (`width` samples wide,
/// `spacing` meters apart), from what the bake `found`, each numbered and
/// the area its water covers measured.
fn water(heights: &[f32], found: &Found<'_>, width: usize, spacing: f32) -> WaterGrid {
    let Found { filled, lakes } = *found;
    let count = heights.len();
    let height = count / width;
    let (last_x, last_z) = (width as i32 - 1, height as i32 - 1);
    // Lakes stand level at their spill heights, each connected one numbered
    // and its samples' squares summed, in the grid's order on every host.
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
    // Depths over the ground, then surfaces reaching past the
    // shore: dry samples beside water take its level, a sample at a time,
    // where their ground stands about as high, so it hides them. Ground
    // falling away below the water (where lakes spill and rivers leave
    // terrain's water for the sea) takes none: a surface reaching over it
    // would stand in the air.
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
    // Elsewhere well below the ground.
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

impl WaterGrid {
    /// The water at terrain-local `(x, z)`: its surface and depth,
    /// interpolated between the samples around it; `None` where dry.
    pub fn at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let (u, v) = (x / self.spacing, z / self.spacing);
        if !(u >= 0.0 && v >= 0.0) {
            return None;
        }
        let (x0, z0) = (u.floor() as usize, v.floor() as usize);
        if x0 + 1 >= self.width || z0 + 1 >= self.height {
            return None;
        }
        let (fx, fz) = (u - x0 as f32, v - z0 as f32);
        let at = |values: &[f32], dx: usize, dz: usize| values[(z0 + dz) * self.width + x0 + dx];
        let blend = |values: &[f32]| {
            let low = at(values, 0, 0) + (at(values, 1, 0) - at(values, 0, 0)) * fx;
            let high = at(values, 0, 1) + (at(values, 1, 1) - at(values, 0, 1)) * fx;
            low + (high - low) * fz
        };
        let depth = blend(&self.depth);
        (depth > 0.0).then(|| (blend(&self.surface), depth))
    }

    /// The highest level of the lakes holding the sample nearest
    /// terrain-local `(x, z)` or one beside it, if any do: where a river
    /// leaving a lake begins.
    pub fn lake_beside(&self, x: f32, z: f32) -> Option<f32> {
        let center = self.nearest(x, z)?;
        let (cx, cz) = ((center % self.width) as i64, (center / self.width) as i64);
        let mut level: Option<f32> = None;
        for dz in -1..=1 {
            for dx in -1..=1 {
                let (sx, sz) = (cx + dx, cz + dz);
                if sx < 0 || sz < 0 || sx >= self.width as i64 || sz >= self.height as i64 {
                    continue;
                }
                let sample = sz as usize * self.width + sx as usize;
                if self.depth[sample] > 0.0 && self.lake[sample] != 0 {
                    let lake = self.levels[usize::from(self.lake[sample]) - 1];
                    level = Some(level.map_or(lake, |level| max(level, lake)));
                }
            }
        }
        level
    }

    /// The level and the span of the lake whose water or shore holds
    /// terrain-local `(x, z)` by half or more, the four samples around it
    /// blended: the lake of the one of them nearest it ([`Self::span`]).
    /// Tiles' water stands at it there ([`crate::tile_water::water_rows`]):
    /// out under its banks, never past its outlet, whose ground falls away.
    pub(crate) fn lake_reach(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        if self.levels.is_empty() {
            return None;
        }
        let (u, v) = (x / self.spacing, z / self.spacing);
        if !(u.is_finite() && v.is_finite()) {
            return None;
        }
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i, v - j);
        let (last_x, last_z) = (self.width as i64 - 1, self.height as i64 - 1);
        let lake = |dx: i64, dz: i64| {
            let x = (i as i64 + dx).clamp(0, last_x) as usize;
            let z = (j as i64 + dz).clamp(0, last_z) as usize;
            self.lake[z * self.width + x]
        };
        let corners = [
            (lake(0, 0), (1.0 - fu) * (1.0 - fv)),
            (lake(1, 0), fu * (1.0 - fv)),
            (lake(0, 1), (1.0 - fu) * fv),
            (lake(1, 1), fu * fv),
        ];
        let held: f32 = corners
            .iter()
            .filter(|(lake, _)| *lake != 0)
            .map(|(_, weight)| weight)
            .sum();
        if held < 0.5 {
            return None;
        }
        let (lake, _) = corners
            .iter()
            .filter(|(lake, _)| *lake != 0)
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        Some((self.levels[usize::from(*lake) - 1], self.span(*lake)))
    }

    /// How far across lake `number` (from 1) is: the diameter of a circle
    /// as large as its water, 2√(A/π) (meters), the fetch the wind has over
    /// it whichever way it blows.
    fn span(&self, number: u16) -> f32 {
        2.0 * (self.areas[usize::from(number) - 1] / std::f32::consts::PI).sqrt()
    }

    /// The sample nearest terrain-local `(x, z)`; `None` off the grid.
    fn nearest(&self, x: f32, z: f32) -> Option<usize> {
        let (u, v) = ((x / self.spacing).round(), (z / self.spacing).round());
        if !(u >= 0.0 && v >= 0.0) || u as usize >= self.width || v as usize >= self.height {
            return None;
        }
        Some(v as usize * self.width + u as usize)
    }
}

/// The level of an image whose samples lie `image_spacing` meters apart,
/// with `levels` coarser levels, that terrain samples `spacing` meters
/// apart read: the coarsest whose samples lie no further apart.
pub(crate) fn image_level(image_spacing: f32, levels: usize, spacing: f32) -> usize {
    let mut level = 0;
    while level < levels && image_spacing * (2_u32 << level) as f32 <= spacing * 1.0001 {
        level += 1;
    }
    level
}

crate::codec::record!(HydrologyLayer { level, discharge });
crate::codec::record!(HydrologyStats {
    samples,
    channel_samples,
    lake_samples,
    max_area,
    falls,
    reaches,
    steps,
    knobs,
});
crate::codec::record!(Fall {
    top,
    bottom,
    path,
    times,
    aloft,
    widths,
    sky,
    width,
    area,
    speed,
    discharge,
});
crate::codec::record!(Hydrology {
    spacing,
    flow,
    wetness,
    lakes,
    water,
    rivers,
    falls,
    stats,
});
crate::codec::record!(WaterGrid {
    width,
    height,
    spacing,
    surface,
    depth,
    flow,
    lake,
    levels,
    areas,
});
