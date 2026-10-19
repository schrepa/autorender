//! Hydrology: where water runs over a terrain and the channels it
//! cuts. A bake reads the heights the layers below it give at one
//! pyramid level, fills their pits so that every sample drains off the
//! grid's edge (Priority-Flood with epsilon; Barnes, Lehman and Mulla
//! 2014), sends each sample's water to its steepest neighbor among those
//! the flood reached before it, and sums upstream area in the reverse of
//! the flood's order. Inside a filled pit the flood spreads from the spill
//! point over the lowest ground first and water drains back the way the
//! flood came, so rivers crossing a pit follow its low line rather than
//! straight rays from the spill point. Channels then run where enough
//! area drains, deeper and wider downstream, and a wetness index and the
//! depths of filled pits follow from the same grids. Each result is a
//! 16-bit image with coarser levels, so every terrain level samples it
//! without aliasing, and every sample is the same on every target: the bake
//! is deterministic. Rivers (`rivers.rs`) follow the channels as lines,
//! each cutting its channel into the ground with its water in it, the one
//! place channels are cut; the grid of standing water holds lakes alone.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::MAX_LEVEL;
use engine_math::{max, min};
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use std::sync::Arc;

use crate::layers::HeightImage;
use crate::rivers::{Channels, MIN_DEPTH, RiverLine, Rivers, trace};

settings! {
    /// A layer that bakes where water runs over the layers below it and cuts
    /// channels into them.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct HydrologyLayer {
        /// The pyramid level whose samples the bake reads: finer levels carve
        /// narrower channels and take longer.
        pub level: u32 = 3 => range(0, MAX_LEVEL),
        /// Square meters of upstream area where channels begin.
        pub channel_area: f32 = 200_000.0 => range(100.0, 1.0e10),
        /// Meters deep channels are where they begin; downstream their depth
        /// grows with the fourth root of the area they drain, up to `max_depth`.
        pub depth: f32 = 1.0 => range(0.0, 100.0),
        /// Meters wide channels are where they begin; downstream their width
        /// grows with the square root of the area they drain, up to `max_width`.
        pub width: f32 = 6.0 => range(0.1, 1000.0),
        /// The deepest channels grow, meters.
        pub max_depth: f32 = 6.0 => range(0.0, 100.0),
        /// The widest channels grow, meters.
        pub max_width: f32 = 60.0 => range(0.1, 1000.0),
        /// Square meters of upstream area where streams begin: channels
        /// draining less than `channel_area` run as streams down to the rivers
        /// they join, their width falling with the area they drain below a
        /// river's at its head and their depth with its square root, cut a little
        /// into the ground. 0, the default, has none, as has `channel_area` or
        /// more.
        #[serde(skip_serializing_if = "is_zero")]
        pub stream_area: f32 = 0.0 => range(0.0, 1.0e10),
        /// Cubic meters a second each square kilometer drained yields to the
        /// rivers: with their channels' width and slope and the roughness of
        /// their beds it sets how deep and fast they run. 0, the default,
        /// yields what runs rivers `fill` of the way up their channels where
        /// their channels' slope is typical.
        #[serde(skip_serializing_if = "is_zero")]
        pub discharge: f32 = 0.0 => range(0.0, 1000.0),
        /// The share of their channels' depth rivers run at where their
        /// channels' slope is typical, while `discharge` leaves their yield
        /// to their channels: fuller where they level out, shallower where
        /// they steepen.
        #[serde(skip_serializing_if = "is_default_fill")]
        pub fill: f32 = FILL => range(0.05, 1.0),
        /// Manning's roughness of riverbeds (seconds over the cube root of a
        /// meter): rougher beds hold water deeper and slower.
        #[serde(skip_serializing_if = "is_default_roughness")]
        pub roughness: f32 = ROUGHNESS => range(0.01, 0.3),
    }
}

/// The default [`HydrologyLayer::fill`] and [`HydrologyLayer::roughness`]:
/// rivers three quarters of the way up their channels, over a natural
/// gravel bed.
pub const FILL: f32 = 0.75;
pub const ROUGHNESS: f32 = 0.035;

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

fn is_default_fill(value: &f32) -> bool {
    *value == FILL
}

fn is_default_roughness(value: &f32) -> bool {
    *value == ROUGHNESS
}

/// What a bake found: samples read, samples channels run through, samples
/// of filled pits, the largest upstream area (square meters), and the
/// waterfalls rivers drop over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct HydrologyStats {
    pub samples: usize,
    pub channel_samples: usize,
    pub lake_samples: usize,
    pub max_area: f32,
    pub falls: usize,
}

/// A waterfall: where a river leaves its bed over a drop, the lip's water
/// and where it lands at last (terrain-local meters), its water's path from
/// one to the other: through the air on each jet and down the chutes
/// between where one jet lands in the chute the next leaves (a cascade),
/// the seconds after leaving the lip its water passes each point of it, and
/// whether it flies through the air from each point to the next; the
/// meters across the river at its lip, the area draining over it (square
/// meters), the water's speed as it lands at last (meters a second) and the
/// discharge it carries (cubic meters a second).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Fall {
    pub top: [f32; 3],
    pub bottom: [f32; 3],
    pub path: Vec<[f32; 3]>,
    pub times: Vec<f32>,
    pub aloft: Vec<bool>,
    pub width: f32,
    pub area: f32,
    pub speed: f32,
    pub discharge: f32,
}

/// The meters a river must fall from a lip to leave its bed.
pub const FALL_HEIGHT: f32 = 6.0;

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
    /// Upstream area, as [`FLOW_RANGE`] encodes it.
    pub flow: HeightImage,
    /// The topographic wetness index ln(a / tan b): a the area draining
    /// through a meter of contour, b the ground's slope; as
    /// [`WETNESS_RANGE`] encodes it.
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
/// for none), and `levels` holds its level by number. Rivers stand on lines
/// of their own ([`Rivers`]).
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

/// Meters of water a filled pit must hold somewhere, and square meters it
/// must cover, to stand as a lake: shallower or smaller pits, such as fine
/// detail leaves on flat ground, stay dry and rivers run through them.
const LAKE_DEPTH: f32 = 2.0;
const LAKE_AREA: f32 = 4000.0;
/// Meters of water a sample of a lake holds at least.
const WET: f32 = 1.0e-3;

/// Which samples lie in lakes: the connected samples a pit fills (eight
/// neighbors each, `spacing` meters apart), where it fills [`LAKE_DEPTH`] or
/// deeper somewhere over [`LAKE_AREA`] or more.
fn lakes(heights: &[f32], filled: &[f32], width: usize, spacing: f32) -> Vec<bool> {
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
        if deepest >= LAKE_DEPTH && region.len() as f32 * spacing * spacing >= LAKE_AREA {
            for &sample in &region {
                lake[sample] = true;
            }
        }
    }
    lake
}

/// A height ordered totally, lowest first in the flood's queue.
#[derive(Clone, Copy, PartialEq)]
struct Level(f32);

impl Eq for Level {}

impl PartialOrd for Level {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Level {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
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

/// Fills `heights`' pits so every sample drains off the edge of a grid
/// `width` samples wide, each raised just above the sample it spills into;
/// returns the samples in the order the flood reached them, and for each
/// raised sample the one it spills into (`u32::MAX` for the others). The
/// flood fills a pit from its spill point over the lowest ground first,
/// in the order it reached samples of equal heights.
fn flood(heights: &mut [f32], width: usize) -> (Vec<u32>, Vec<u32>) {
    let count = heights.len();
    let height = count / width;
    let mut closed = vec![false; count];
    let mut open = BinaryHeap::new();
    // Raised samples by their own ground, then by when the flood reached
    // them.
    let mut pits = BinaryHeap::new();
    let mut reached = 0_u32;
    let mut parents = vec![u32::MAX; count];
    let mut order = Vec::with_capacity(count);
    let edges = (0..width)
        .flat_map(|x| [(x, 0), (x, height - 1)])
        .chain((0..height).flat_map(|z| [(0, z), (width - 1, z)]));
    for (x, z) in edges {
        let sample = z * width + x;
        if !closed[sample] {
            closed[sample] = true;
            open.push(Reverse((Level(heights[sample]), sample as u32)));
        }
    }
    loop {
        let sample = match pits.pop() {
            Some(Reverse((_, _, sample))) => sample as usize,
            None => match open.pop() {
                Some(Reverse((_, sample))) => sample as usize,
                None => break,
            },
        };
        order.push(sample as u32);
        let (x, z) = ((sample % width) as i32, (sample / width) as i32);
        let spill = heights[sample].next_up();
        for (dx, dz, _) in NEIGHBORS {
            let (nx, nz) = (x + dx, z + dz);
            if nx < 0 || nz < 0 || nx >= width as i32 || nz >= height as i32 {
                continue;
            }
            let neighbor = nz as usize * width + nx as usize;
            if closed[neighbor] {
                continue;
            }
            closed[neighbor] = true;
            if heights[neighbor] < spill {
                pits.push(Reverse((
                    Level(heights[neighbor]),
                    reached,
                    neighbor as u32,
                )));
                reached += 1;
                heights[neighbor] = spill;
                parents[neighbor] = sample as u32;
            } else {
                open.push(Reverse((Level(heights[neighbor]), neighbor as u32)));
            }
        }
    }
    (order, parents)
}

/// Where water drains over a grid: its heights
/// with every pit filled to spill, the samples in the order the flood
/// reached them, each sample's receiver (`u32::MAX` on the edge, which
/// drains off the grid) and the slope down to it, and the area draining
/// through each sample (square meters).
pub(crate) struct Drainage {
    pub(crate) filled: Vec<f32>,
    pub(crate) order: Vec<u32>,
    pub(crate) receivers: Vec<u32>,
    pub(crate) slopes: Vec<f32>,
    pub(crate) area: Vec<f32>,
}

/// How water drains over the grid of `heights`, `width` samples wide
/// (row by row along z, at least two each way), whose samples lie
/// `spacing` meters apart.
pub(crate) fn drainage(heights: &[f32], width: usize, spacing: f32) -> Drainage {
    let count = heights.len();
    let height = count / width;
    let mut filled = heights.to_vec();
    let (order, parents) = flood(&mut filled, width);
    let mut rank = vec![0_u32; count];
    for (position, &sample) in order.iter().enumerate() {
        rank[sample as usize] = position as u32;
    }
    // Each sample drains to its steepest neighbor the flood reached before
    // it, and each raised sample to the one it spills into, back the way the
    // flood came, so the reverse of the flood's order visits every sample
    // before the one it drains to; edge samples drain off the grid.
    let (last_x, last_z) = (width as i32 - 1, height as i32 - 1);
    let mut receivers = vec![u32::MAX; count];
    let mut slopes = vec![0.0_f32; count];
    for sample in 0..count {
        let (x, z) = ((sample % width) as i32, (sample / width) as i32);
        if x == 0 || z == 0 || x == last_x || z == last_z {
            continue;
        }
        if parents[sample] != u32::MAX {
            receivers[sample] = parents[sample];
            continue;
        }
        let mut best = (u32::MAX, f32::NEG_INFINITY);
        for (dx, dz, distance) in NEIGHBORS {
            let neighbor = (z + dz) as usize * width + (x + dx) as usize;
            if rank[neighbor] >= rank[sample] {
                continue;
            }
            let slope = (filled[sample] - filled[neighbor]) / (distance * spacing);
            if slope > best.1 {
                best = (neighbor as u32, slope);
            }
        }
        receivers[sample] = best.0;
        slopes[sample] = max(best.1, 0.0);
    }
    let mut area = vec![spacing * spacing; count];
    for &sample in order.iter().rev() {
        let receiver = receivers[sample as usize];
        if receiver != u32::MAX {
            area[receiver as usize] += area[sample as usize];
        }
    }
    Drainage {
        filled,
        order,
        receivers,
        slopes,
        area,
    }
}

/// Bakes `layer` over the grid of `heights`, `width` samples wide (row by
/// row along z, at least two each way), whose samples lie `spacing` meters
/// apart; with its rivers' lines, whose channels and water the finest
/// ground sets once read (`crate::hydraulics::river`), and no rivers yet.
pub(crate) fn bake(
    heights: &[f32],
    width: usize,
    spacing: f32,
    layer: &HydrologyLayer,
) -> (Hydrology, Vec<RiverLine>) {
    let count = heights.len();
    let height = count / width;
    // Water runs over the ground smoothed a sample wide: hollows narrower
    // than that, such as the finest detail leaves on flat ground, hold no
    // lakes.
    let smoothed = smooth_grid(heights, width);
    let heights = smoothed.as_slice();
    let Drainage {
        filled,
        order,
        receivers,
        slopes,
        area,
    } = drainage(heights, width, spacing);
    let lake = lakes(heights, &filled, width, spacing);
    // Channels run where enough area drains; water crossing a filled pit
    // runs as a lake, not a channel.
    let channel_samples = area
        .iter()
        .zip(&lake)
        .filter(|&(&drained, &lake)| drained >= layer.channel_area && !lake)
        .count();
    let found = Found {
        filled: &filled,
        lakes: &lake,
    };
    // The grid keeps lakes alone, rivers standing on lines of their own whose
    // flow places their falls.
    let water = Arc::new(water(heights, &found, width, spacing));
    let mut rank = vec![0_u32; count];
    for (position, &sample) in order.iter().enumerate() {
        rank[sample as usize] = position as u32;
    }
    let lines = trace(
        &Channels {
            area: &area,
            receivers: &receivers,
            lakes: &lake,
            rank: &rank,
        },
        (width, spacing),
        if layer.stream_area > 0.0 {
            min(layer.stream_area, layer.channel_area)
        } else {
            layer.channel_area
        },
        |drained| {
            let ratio = drained / layer.channel_area;
            if ratio < 1.0 {
                // Streams narrow with the area they drain
                // and grow shallow with its square root, meeting a river's
                // size at its head.
                return (0.5 * layer.width * ratio, MIN_DEPTH * ratio.sqrt());
            }
            let half = 0.5 * min(layer.width * ratio.sqrt(), layer.max_width);
            let depth = min(layer.depth * ratio.sqrt().sqrt(), layer.max_depth);
            (half, max(depth, MIN_DEPTH))
        },
    );
    let mut lake_samples = 0;
    let mut max_area = 0.0_f32;
    let mut flow = Vec::with_capacity(count);
    let mut wetness = Vec::with_capacity(count);
    let mut lakes = Vec::with_capacity(count);
    for sample in 0..count {
        max_area = max(max_area, area[sample]);
        let standing = if lake[sample] {
            lake_samples += 1;
            filled[sample] - heights[sample]
        } else {
            0.0
        };
        // libm's logarithms, as browsers compute them: the platform's
        // differ.
        flow.push(encode(libm::log10f(area[sample]), FLOW_RANGE));
        let catchment = area[sample] / spacing;
        let index = libm::logf(catchment / max(slopes[sample], 1.0e-3));
        wetness.push(encode(index, WETNESS_RANGE));
        lakes.push(encode(standing, LAKE_RANGE));
    }
    let image = |samples: Vec<u16>| {
        HeightImage::new(width as u32, height as u32, samples)
            .expect("grids hold two samples each way")
    };
    let hydrology = Hydrology {
        spacing,
        flow: image(flow),
        wetness: image(wetness),
        lakes: image(lakes),
        water,
        rivers: Arc::new(Rivers::default()),
        stats: HydrologyStats {
            samples: count,
            channel_samples,
            lake_samples,
            max_area,
            falls: 0,
        },
        falls: Arc::from(Vec::new()),
    };
    (hydrology, lines)
}

/// What a bake found about one place: the base-10 logarithm of its
/// upstream area (square meters), its wetness index, how deep water would
/// stand there in a filled pit (meters), and meters from the nearest
/// river's edge, negative in its water; [`RIVER_FAR`] beyond every river's
/// reach.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct HydrologySample {
    pub flow: f32,
    pub wetness: f32,
    pub lake: f32,
    pub river: f32,
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
        let river = self
            .rivers
            .nearest(x, z)
            .map_or(RIVER_FAR, |river| river.distance - river.half);
        self.masks_at(x, z, spacing, river)
    }

    /// [`Self::at`] with the meters to the nearest river's edge, `river`,
    /// already read.
    pub(crate) fn masks_at(&self, x: f32, z: f32, spacing: f32, river: f32) -> HydrologySample {
        let level = self.level_for(spacing);
        let (u, v) = (x / self.spacing, z / self.spacing);
        HydrologySample {
            flow: decode(self.flow.sample(u, v, level), FLOW_RANGE),
            wetness: decode(self.wetness.sample(u, v, level), WETNESS_RANGE),
            lake: decode(self.lakes.sample(u, v, level), LAKE_RANGE),
            river,
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
}

/// What a bake found that lakes stand by: the filled heights and lake
/// samples.
struct Found<'a> {
    filled: &'a [f32],
    lakes: &'a [bool],
}

/// Where lakes stand over the grid of `heights` (`width` samples wide,
/// `spacing` meters apart), from what the bake `found`, each numbered.
fn water(heights: &[f32], found: &Found<'_>, width: usize, spacing: f32) -> WaterGrid {
    let Found { filled, lakes } = *found;
    let count = heights.len();
    let height = count / width;
    let (last_x, last_z) = (width as i32 - 1, height as i32 - 1);
    // Lakes stand level at their spill heights, each connected one numbered.
    let mut surface = vec![f32::NAN; count];
    let mut lake = vec![0_u16; count];
    let mut levels = Vec::new();
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

    /// The level of the lake holding the sample nearest terrain-local
    /// `(x, z)`, if one does.
    pub fn lake_level(&self, x: f32, z: f32) -> Option<f32> {
        let sample = self.nearest(x, z)?;
        (self.depth[sample] > 0.0).then(|| self.levels[usize::from(self.lake[sample]) - 1])
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

    /// The level of the lake whose water or shore holds terrain-local
    /// `(x, z)` by half or more, the four samples around it blended: the
    /// lake of the one of them nearest it. Tiles' water stands at it there
    /// ([`crate::tile_water::water_rows`]): out under its banks, never past
    /// its outlet, whose ground falls away.
    pub(crate) fn lake_reach(&self, x: f32, z: f32) -> Option<f32> {
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
        Some(self.levels[usize::from(*lake) - 1])
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

/// The grid of `heights` (`width` samples wide) blurred by binomial
/// weights `[1, 4, 6, 4, 1] / 16` along each axis, edges clamped.
fn smooth_grid(heights: &[f32], width: usize) -> Vec<f32> {
    const WEIGHTS: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let height = heights.len() / width;
    let pass = |source: &[f32], along_x: bool| -> Vec<f32> {
        let mut out = vec![0.0_f32; source.len()];
        for z in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                for (tap, weight) in WEIGHTS.iter().enumerate() {
                    let offset = tap as i64 - 2;
                    let (sx, sz) = if along_x {
                        ((x as i64 + offset).clamp(0, width as i64 - 1) as usize, z)
                    } else {
                        (x, (z as i64 + offset).clamp(0, height as i64 - 1) as usize)
                    };
                    sum += weight * source[sz * width + sx];
                }
                out[z * width + x] = sum;
            }
        }
        out
    };
    pass(&pass(heights, true), false)
}

crate::codec::record!(HydrologyLayer {
    level,
    channel_area,
    depth,
    width,
    max_depth,
    max_width,
    stream_area,
    discharge,
    fill,
    roughness,
});
crate::codec::record!(HydrologyStats {
    samples,
    channel_samples,
    lake_samples,
    max_area,
    falls,
});
crate::codec::record!(Fall {
    top,
    bottom,
    path,
    times,
    aloft,
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
});
