use engine_core::fields;
use engine_heightfield::hydrology::{Drainage, Flood, neighbor_steps};
use engine_heightfield::layer::image_row;
use engine_heightfield::noise::{gradient_value, hash, smooth};
use engine_heightfield::{HeightImage, MAX_LEVEL};
use engine_math::{max, min};
use serde::{Deserialize, Serialize};

pub const CLIFF: &str = "cliff";
pub const BRINK: &str = "brink";
pub const FOOT: &str = "foot";
pub const HARDNESS: &str = "hardness";

const TRACE: f32 = 0.05;
const STABLE_CREEP: f32 = 0.24;
const LEVELLED: f32 = 0.5;

const BED_WARP: u32 = 0x243f_6a88;
const BED_PATCHES: u32 = 0x85a3_08d3;
const BED_KINDS: u32 = 0x1319_8a2e;

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Beds {
        pub seed: u32 = 0 => any,
        pub thickness: f32 = 40.0 => range(1.0, 1000.0),
        pub hard: f32 = 0.0 => range(0.0, 1.0),
        pub resistance: f32 = 0.85 => range(0.0, 1.0),
        pub slopes: [f32; 2] = [38.0, 68.0] => range(1.0, 89.0),
        pub dip: [f32; 2] = [0.0, 0.0] => range(-1.0, 1.0),
        pub warp: [f32; 2] = [0.0, 1000.0] => range(0.0, 1.0e6),
        pub patches: f32 = 900.0 => range(1.0, 1.0e6),
        pub contrast: f32 = 0.0 => range(0.0, 100.0),
    }
}

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct ErosionLayer {
        pub level: u32 = 4 => range(0, MAX_LEVEL),
        pub rounds: u32 = 60 => range(1, 400),
        pub incision: f32 = 0.002 => range(0.0, 1.0),
        pub debris: f32 = 0.004 => range(0.0, 1.0),
        pub creep: f32 = 20.0 => range(0.0, 1000.0),
        pub trunk: f32 = 1.0e9 => range(1.0, 1.0e12),
        pub source: f32 = 0.0 => range(0.0, 1.0e12),
        pub banks: f32 = 0.0 => range(0.0, 1.0),
        pub channel: f32 = 4.0e4 => range(1.0, 1.0e12),
        pub floor: f32 = 0.02 => range(0.0, 10.0),
        pub weathering: f32 = 0.0 => range(0.0, 100.0),
        pub weathering_slope: f32 = 27.0 => range(1.0, 89.0),
        pub wash: f32 = 0.0 => range(0.0, 1.0),
        pub collapse: f32 = 0.0 => range(0.0, 1.0),
        pub talus: f32 = 0.35 => range(0.0, 1.0),
        pub repose: f32 = 34.0 => range(1.0, 89.0),
        pub mantle: f32 = 1.0 => range(0.01, 1000.0),
        pub cliff: [f32; 2] = [34.0, 42.0] => range(1.0, 89.0),
        pub beds: Beds = Beds::default() => any,
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
    pub cover: HeightImage,
    pub cover_range: [f32; 2],
    pub cliff: HeightImage,
    pub brink: HeightImage,
    pub foot: HeightImage,
    pub stats: ErosionStats,
}

impl Erosion {
    pub(crate) fn wear_row(
        &self,
        (xs, z, spacing): (&[f32], f32, f32),
        mantle: f32,
        heights: &[f32],
        mut each: impl FnMut(usize, f32),
    ) {
        let count = heights.len().min(xs.len());
        let spacings = (self.spacing, spacing);
        let mut started = vec![0.0_f32; count];
        image_row(
            &self.ground,
            self.ground_range,
            spacings,
            xs[..count].iter().copied(),
            z,
            |index, ground| started[index] = ground,
        );
        let mut settled = vec![0.0_f32; count];
        image_row(
            &self.cover,
            self.cover_range,
            spacings,
            xs[..count].iter().copied(),
            z,
            |index, cover| settled[index] = smooth(cover / mantle),
        );
        image_row(
            &self.change,
            self.range,
            spacings,
            xs[..count].iter().copied(),
            z,
            |index, change| {
                each(
                    index,
                    change + settled[index] * (started[index] - heights[index]),
                );
            },
        );
    }

    pub(crate) fn share_row(
        &self,
        name: &str,
        (xs, z, spacing): (&[f32], f32, f32),
        mut each: impl FnMut(usize, f32),
    ) -> bool {
        let image = match name {
            CLIFF => &self.cliff,
            BRINK => &self.brink,
            FOOT => &self.foot,
            _ => return false,
        };
        image_row(
            image,
            [0.0, 1.0],
            (self.spacing, spacing),
            xs.iter().copied(),
            z,
            |index, share| each(index, max(min(share, 1.0), 0.0)),
        );
        true
    }
}

fn unit(value: u32) -> f32 {
    (value >> 8) as f32 * (1.0 / 16_777_216.0)
}

fn tangent(degrees: f32) -> f32 {
    libm::tanf(degrees.to_radians())
}

impl Beds {
    fn datum(&self, x: f32, z: f32) -> f32 {
        let [amplitude, wavelength] = self.warp;
        let across = 1.0 / max(wavelength, 1.0);
        self.dip[0] * x
            + self.dip[1] * z
            + amplitude * gradient_value(x * across, z * across, self.seed ^ BED_WARP)
    }

    fn patch(&self, x: f32, z: f32) -> f32 {
        if self.contrast <= 0.0 {
            return 1.0;
        }
        let across = 1.0 / self.patches;
        let spread = gradient_value(x * across + 3.3, z * across + 8.1, self.seed ^ BED_PATCHES);
        smooth(0.5 + self.contrast * spread)
    }

    fn hardness(&self, level: f32) -> f32 {
        let bed = (level / self.thickness).floor() as i32;
        if unit(hash(bed, 0, self.seed ^ BED_KINDS)) < self.hard {
            1.0
        } else {
            0.0
        }
    }

    pub(crate) fn surface(&self, x: f32, z: f32, height: f32) -> f32 {
        self.hardness(height + self.datum(x, z)) * self.patch(x, z)
    }
}

pub(crate) struct Ground {
    width: usize,
    spacing: f32,
    heights: Vec<f32>,
    loose: Vec<f32>,
    crept: Vec<f32>,
    hard: Vec<f32>,
    datum: Vec<f32>,
    patch: Vec<f32>,
    drainage: Drainage,
    flood: Flood,
}

impl Ground {
    pub(crate) fn new(heights: Vec<f32>, width: usize, spacing: f32, beds: &Beds) -> Self {
        let count = heights.len();
        let at = |sample: usize| {
            (
                (sample % width) as f32 * spacing,
                (sample / width) as f32 * spacing,
            )
        };
        let datum = (0..count)
            .map(|sample| {
                let (x, z) = at(sample);
                beds.datum(x, z)
            })
            .collect();
        let patch = (0..count)
            .map(|sample| {
                let (x, z) = at(sample);
                beds.patch(x, z)
            })
            .collect();
        Self {
            width,
            spacing,
            loose: vec![0.0; count],
            crept: vec![0.0; count],
            hard: vec![0.0; count],
            heights,
            datum,
            patch,
            drainage: Drainage::default(),
            flood: Flood::default(),
        }
    }

    pub(crate) fn round(&mut self, layer: &ErosionLayer, round: u32) {
        self.harden(&layer.beds);
        self.creep(layer);
        self.route(round);
        self.weather(layer);
        self.incise(layer);
        self.collapse(layer);
    }

    fn harden(&mut self, beds: &Beds) {
        let levels = self.heights.iter().zip(&self.loose).zip(&self.datum);
        for ((hard, ((height, loose), datum)), patch) in
            self.hard.iter_mut().zip(levels).zip(&self.patch)
        {
            *hard = beds.hardness(height - loose + datum) * patch;
        }
    }

    fn creep(&mut self, layer: &ErosionLayer) {
        let (width, count) = (self.width, self.heights.len());
        let rows = count / width;
        let rate = min(layer.creep / (self.spacing * self.spacing), STABLE_CREEP);
        self.crept.copy_from_slice(&self.heights);
        if rate <= 0.0 {
            return;
        }
        let heights = &self.heights;
        for z in 1..rows - 1 {
            for at in z * width + 1..(z + 1) * width - 1 {
                let around =
                    heights[at - 1] + heights[at + 1] + heights[at - width] + heights[at + width];
                let mobile = if self.loose[at] > TRACE {
                    1.0
                } else {
                    1.0 - self.hard[at]
                };
                self.crept[at] = heights[at] + rate * mobile * (around - 4.0 * heights[at]);
            }
        }
    }

    fn route(&mut self, round: u32) {
        let filled = &mut self.drainage.filled;
        filled.clear();
        filled.extend_from_slice(&self.crept);
        self.drainage
            .route(self.width, self.spacing, round, &mut self.flood);
        let grounds = self.drainage.filled.iter().zip(&self.crept);
        for ((height, loose), (&filled, &crept)) in
            self.heights.iter_mut().zip(&mut self.loose).zip(grounds)
        {
            *loose += max(filled - crept, 0.0);
            *height = filled;
        }
    }

    fn weather(&mut self, layer: &ErosionLayer) {
        if layer.weathering <= 0.0 && layer.wash <= 0.0 {
            return;
        }
        let width = self.width;
        let steps = neighbor_steps(width, self.spacing);
        let resistance = layer.beds.resistance;
        let full = 1.0 / (2.0 * self.spacing * tangent(layer.weathering_slope));
        self.crept.copy_from_slice(&self.heights);
        for sample in 0..self.heights.len() {
            if self.drainage.receivers[sample] == u32::MAX || !self.inside(sample, 1) {
                continue;
            }
            let before = &self.crept;
            let east = before[sample + 1] - before[sample - 1];
            let south = before[sample + width] - before[sample - width];
            let steep = min((east * east + south * south).sqrt() * full, 1.0);
            let lowest = steps.iter().fold(before[sample], |lowest, &(offset, _)| {
                min(lowest, before[sample.wrapping_add_signed(offset)])
            });
            let mut room = LEVELLED * (before[sample] - lowest);
            if self.loose[sample] < TRACE {
                let worn = min(
                    layer.weathering * (1.0 - resistance * self.hard[sample]) * steep,
                    room,
                );
                self.heights[sample] -= worn;
                room -= worn;
            }
            let washed = min(
                min(self.loose[sample] * layer.wash * steep, room),
                self.loose[sample],
            );
            self.heights[sample] -= washed;
            self.loose[sample] -= washed;
        }
    }

    fn inside(&self, sample: usize, margin: usize) -> bool {
        let (x, z) = (sample % self.width, sample / self.width);
        let rows = self.heights.len() / self.width;
        x >= margin && z >= margin && x + margin < self.width && z + margin < rows
    }

    fn incise(&mut self, layer: &ErosionLayer) {
        let (width, spacing) = (self.width, self.spacing);
        let diagonal = std::f32::consts::SQRT_2 * spacing;
        let resistance = layer.beds.resistance;
        for index in 0..self.drainage.order.len() {
            let sample = self.drainage.order[index] as usize;
            let receiver = self.drainage.receivers[sample];
            if receiver == u32::MAX {
                continue;
            }
            let receiver = receiver as usize;
            let (height, below) = (self.heights[sample], self.heights[receiver]);
            if height <= below {
                continue;
            }
            let step = sample.abs_diff(receiver);
            let length = if step == 1 || step == width {
                spacing
            } else {
                diagonal
            };
            let area = self.drainage.area[sample];
            let trunk = min(area, layer.trunk);
            let carried = max(trunk - layer.source, 0.0);
            let factor = carried.sqrt()
                * (layer.incision + layer.debris * self.drainage.slopes[sample])
                / length;
            let mut lowered = height - (height + factor * below) / (1.0 + factor);
            let cover = self.loose[sample];
            if lowered > cover {
                lowered = cover + (lowered - cover) * (1.0 - resistance * self.hard[sample]);
            }
            self.loose[sample] = max(cover - lowered, 0.0);
            let height = max(height - lowered, below);
            self.heights[sample] = height;
            if layer.banks > 0.0 && area >= layer.channel && self.inside(sample, 2) {
                let reach = layer.banks * (trunk / layer.trunk).sqrt();
                self.cut_banks(sample, height, reach, layer.floor);
            }
        }
    }

    fn cut_banks(&mut self, sample: usize, height: f32, reach: f32, floor: f32) {
        for (offset, inverse) in neighbor_steps(self.width, self.spacing) {
            let beside = sample.wrapping_add_signed(offset);
            let mut bank = height + floor / inverse;
            let onward = self.drainage.receivers[beside];
            if onward != u32::MAX {
                bank = max(bank, self.heights[onward as usize]);
            }
            let cut = reach * (self.heights[beside] - bank);
            if cut > 0.0 {
                self.heights[beside] -= cut;
                self.loose[beside] = max(self.loose[beside] - cut, 0.0);
            }
        }
    }

    fn steepest(&self, sample: usize, height: f32, steps: &[(isize, f32); 8]) -> (f32, f32) {
        let mut found = (0.0_f32, self.spacing);
        for &(offset, inverse) in steps {
            let slope = (height - self.heights[sample.wrapping_add_signed(offset)]) * inverse;
            if slope > found.0 {
                found = (slope, 1.0 / inverse);
            }
        }
        found
    }

    fn shed(&mut self, sample: usize, (height, limit): (f32, f32), moved: f32) {
        let steps = neighbor_steps(self.width, self.spacing);
        let over = |heights: &[f32], (offset, inverse): (isize, f32)| {
            let beside = sample.wrapping_add_signed(offset);
            (beside, max((height - heights[beside]) * inverse - limit, 0.0))
        };
        let total: f32 = (steps.iter())
            .map(|&step| over(&self.heights, step).1)
            .sum();
        if total <= 0.0 {
            return;
        }
        for step in steps {
            let (beside, over) = over(&self.heights, step);
            if over > 0.0 && self.inside(beside, 1) {
                let part = moved * over / total;
                self.heights[beside] += part;
                self.loose[beside] += part;
            }
        }
    }

    fn collapse(&mut self, layer: &ErosionLayer) {
        let steps = neighbor_steps(self.width, self.spacing);
        let beds = &layer.beds;
        let repose = tangent(layer.repose);
        let [soft, hard] = beds.slopes.map(tangent);
        for index in (0..self.drainage.order.len()).rev() {
            let sample = self.drainage.order[index] as usize;
            if !self.inside(sample, 1) {
                continue;
            }
            let mut height = self.heights[sample];
            let mut cover = self.loose[sample];
            let (steepest, length) = self.steepest(sample, height, &steps);
            let slid = min(cover, LEVELLED * (steepest - repose) * length);
            if slid > 0.0 {
                self.shed(sample, (height, repose), slid);
                height -= slid;
                cover -= slid;
            }
            if cover <= TRACE {
                let (steepest, length) = self.steepest(sample, height, &steps);
                let level = height - cover + self.datum[sample];
                let holds = soft + (hard - soft) * beds.hardness(level) * self.patch[sample];
                let fallen = LEVELLED * layer.collapse * (steepest - holds) * length;
                if fallen > 0.0 {
                    self.shed(sample, (height, holds), fallen * layer.talus);
                    height -= fallen;
                }
            }
            self.heights[sample] = height;
            self.loose[sample] = cover;
        }
    }

    pub(crate) fn settle(&mut self, round: u32) {
        self.crept.copy_from_slice(&self.heights);
        self.route(round);
    }

    fn cliffs(&self, layer: &ErosionLayer) -> Vec<u16> {
        let width = self.width;
        let rows = self.heights.len() / width;
        let [from, to] = layer.cliff.map(tangent);
        let band = 1.0 / max(to - from, 1.0e-6);
        let across = 1.0 / (2.0 * self.spacing);
        let heights = &self.heights;
        let mut cliffs = vec![0_u16; heights.len()];
        for z in 1..rows - 1 {
            for at in z * width + 1..(z + 1) * width - 1 {
                let east = (heights[at + 1] - heights[at - 1]) * across;
                let south = (heights[at + width] - heights[at - width]) * across;
                let slope = (east * east + south * south).sqrt();
                let bare = 1.0 - smooth(self.loose[at] / layer.mantle);
                let cliff = bare * smooth((slope - from) * band);
                cliffs[at] = (cliff * 65_535.0 + 0.5) as u16;
            }
        }
        cliffs
    }

    fn edges(&self, cliffs: &[u16]) -> [Vec<u16>; 2] {
        let steps = neighbor_steps(self.width, self.spacing);
        let mut brinks = vec![0_u16; cliffs.len()];
        let mut feet = vec![0_u16; cliffs.len()];
        for sample in 0..cliffs.len() {
            if !self.inside(sample, 1) {
                continue;
            }
            let height = self.heights[sample];
            let (mut below, mut above) = (0_u16, 0_u16);
            for (offset, _) in steps {
                let beside = sample.wrapping_add_signed(offset);
                if self.heights[beside] < height {
                    below = below.max(cliffs[beside]);
                } else if self.heights[beside] > height {
                    above = above.max(cliffs[beside]);
                }
            }
            let open = u32::from(u16::MAX - cliffs[sample]);
            brinks[sample] = (u32::from(below) * open / 65_535) as u16;
            feet[sample] = (u32::from(above) * open / 65_535) as u16;
        }
        [brinks, feet]
    }

    pub(crate) fn finish(&self, original: &[f32], layer: &ErosionLayer) -> Erosion {
        let width = self.width;
        let rows = self.heights.len() / width;
        let changes: Vec<f32> = original
            .iter()
            .zip(&self.heights)
            .map(|(before, after)| after - before)
            .collect();
        let covers: Vec<f32> = changes
            .iter()
            .zip(&self.loose)
            .map(|(&change, &loose)| {
                if loose >= TRACE {
                    loose
                } else {
                    min(change, 0.0)
                }
            })
            .collect();
        let (change, range) = encoded(&changes, width);
        let (ground, ground_range) = encoded(original, width);
        let (cover, cover_range) = encoded(&covers, width);
        let image = |shares: Vec<u16>| {
            HeightImage::new(width as u32, rows as u32, shares).expect("grid_two_samples_each_way")
        };
        let cliffs = self.cliffs(layer);
        let [brink, foot] = self.edges(&cliffs).map(image);
        let cliff = image(cliffs);
        Erosion {
            spacing: self.spacing,
            change,
            range,
            ground,
            ground_range,
            cover,
            cover_range,
            cliff,
            brink,
            foot,
            stats: ErosionStats {
                samples: original.len(),
                rounds: layer.rounds,
                deepest: -min(range[0], 0.0),
                highest: max(range[1], 0.0),
            },
        }
    }
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

engine_codec::record!(Beds {
    seed,
    thickness,
    hard,
    resistance,
    slopes,
    dip,
    warp,
    patches,
    contrast,
});
engine_codec::record!(ErosionLayer {
    level,
    rounds,
    incision,
    debris,
    creep,
    trunk,
    source,
    banks,
    channel,
    floor,
    weathering,
    weathering_slope,
    wash,
    collapse,
    talus,
    repose,
    mantle,
    cliff,
    beds,
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
    cover,
    cover_range,
    cliff,
    brink,
    foot,
    stats,
});
