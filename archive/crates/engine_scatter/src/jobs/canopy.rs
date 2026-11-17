use bytemuck::{Pod, Zeroable};
use engine_core::gpu_consts;
use engine_heightfield::tile::Placed;
use engine_math::{clamp, ln, max, min, pow, sin_cos};

pub(crate) const MAX_CANOPY_TEXELS: u32 = 4096;
pub(crate) const CANOPY_BLOCK: u32 = 4;

gpu_consts! {
    pub(crate) const CANOPY_SIZE_LOW: f32 = -4.0;
    pub(crate) const CANOPY_SIZE_OCTAVES: f32 = 12.0;
    pub(crate) const CANOPY_TOP_OVER_MEAN: f32 = 2.0;
}

const SUB: usize = 4;
const GREY: f32 = 0.577_350_27;

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct CanopyAsset {
    pub(crate) color: [f32; 3],
    pub(crate) hue: f32,
    pub(crate) brightness: f32,
    pub(crate) center: [f32; 2],
    pub(crate) radius: f32,
    pub(crate) top: f32,
    pub(crate) diameter: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct CanopyAsk {
    pub(crate) recipe: [u8; 32],
    pub(crate) size: [u32; 2],
    pub(crate) origin: [f32; 2],
    pub(crate) variation: f32,
    pub(crate) assets: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct CanopyMade {
    pub(crate) size: [u32; 2],
    pub(crate) tallest: f32,
    pub(crate) span: f32,
}

pub(crate) fn ask_bytes(ask: &CanopyAsk, assets: &[CanopyAsset]) -> Vec<u8> {
    let mut bytes = bytemuck::bytes_of(ask).to_vec();
    bytes.extend_from_slice(bytemuck::cast_slice(assets));
    bytes
}

pub(crate) fn read_ask(input: &[u8]) -> Option<(CanopyAsk, Vec<CanopyAsset>)> {
    let (head, rest) = input.split_at_checked(size_of::<CanopyAsk>())?;
    let ask: CanopyAsk = bytemuck::try_pod_read_unaligned(head).ok()?;
    if rest.len() != ask.assets as usize * size_of::<CanopyAsset>() {
        return None;
    }
    let assets = rest
        .chunks_exact(size_of::<CanopyAsset>())
        .map(bytemuck::pod_read_unaligned)
        .collect();
    Some((ask, assets))
}

pub(crate) fn read_made(bytes: &[u8]) -> Option<(CanopyMade, &[u8], &[u8])> {
    let (head, rest) = bytes.split_at_checked(size_of::<CanopyMade>())?;
    let made: CanopyMade = bytemuck::try_pod_read_unaligned(head).ok()?;
    let texels = (made.size[0] as usize)
        .checked_mul(made.size[1] as usize)?
        .checked_mul(4)?;
    if texels == 0 || rest.len() != 2 * texels {
        return None;
    }
    let (color, shape) = rest.split_at(texels);
    Some((made, color, shape))
}

pub(crate) fn tint(x: f32, z: f32) -> u32 {
    let cell = |value: f32| (value * 4.0).floor() as i32 as u32;
    let mut hash = cell(x).wrapping_mul(0x8da6_b343) ^ cell(z).wrapping_mul(0xd816_3841);
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb_352d);
    hash = (hash ^ (hash >> 15)).wrapping_mul(0x846c_a68b);
    (hash ^ (hash >> 16)) & 0xffff_0000
}

fn varied(asset: &CanopyAsset, plant: u32, share: f32) -> [f32; 3] {
    let color = asset.color;
    let rolls = [
        (plant >> 24) as f32 / 127.5 - 1.0,
        ((plant >> 16) & 255) as f32 / 127.5 - 1.0,
    ];
    let (sin, cos) = sin_cos(asset.hue * share * rolls[0] * std::f32::consts::TAU);
    let along = GREY * (color[0] + color[1] + color[2]);
    let across = [
        GREY * (color[2] - color[1]),
        GREY * (color[0] - color[2]),
        GREY * (color[1] - color[0]),
    ];
    let gain = 1.0 + asset.brightness * share * rolls[1];
    std::array::from_fn(|channel| {
        let turned = color[channel] * cos + across[channel] * sin + GREY * along * (1.0 - cos);
        max(turned, 0.0) * gain
    })
}

fn octaves(meters: f32) -> f32 {
    ln(max(meters, 1.0e-6)) * std::f32::consts::LOG2_E
}

fn unorm(value: f32) -> u8 {
    (clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8
}

#[derive(Clone, Copy)]
struct Stood {
    color: [f32; 3],
    size: f32,
    width: f32,
}

pub(crate) struct Canopy {
    size: [usize; 2],
    cell: [f32; 2],
    origin: [f32; 2],
    variation: f32,
    covered: Vec<f32>,
    heights: Vec<f32>,
    colors: Vec<[f32; 3]>,
    sizes: Vec<f32>,
    widths: Vec<f32>,
    spanned: [f64; 2],
}

impl Canopy {
    pub(crate) fn new(ask: &CanopyAsk, extent: [f32; 2]) -> Self {
        let size = [ask.size[0] as usize, ask.size[1] as usize];
        let cells = size[0] * size[1];
        Self {
            size,
            cell: [extent[0] / size[0] as f32, extent[1] / size[1] as f32],
            origin: ask.origin,
            variation: ask.variation,
            covered: vec![0.0; cells],
            heights: vec![0.0; cells],
            colors: vec![[0.0; 3]; cells],
            sizes: vec![0.0; cells],
            widths: vec![0.0; cells],
            spanned: [0.0; 2],
        }
    }

    fn stand(&mut self, at: usize, (weight, height): (f32, f32), stood: &Stood) {
        self.covered[at] += weight;
        self.heights[at] += height;
        for (sum, channel) in self.colors[at].iter_mut().zip(stood.color) {
            *sum += weight * channel;
        }
        self.sizes[at] += weight * stood.size;
        self.widths[at] += weight * stood.width;
    }

    pub(crate) fn add(&mut self, assets: &[CanopyAsset], placed: &Placed) {
        let Some(asset) = assets.get(usize::from(placed.asset)) else {
            return;
        };
        let scale = placed.scale;
        let radius = asset.radius * scale;
        let top = asset.top * scale - placed.sink;
        if !(radius > 0.0 && top > 0.0) {
            return;
        }
        let (sin, cos) = sin_cos(placed.yaw);
        let [x, _, z] = placed.position;
        let center = [
            x + scale * (asset.center[0] * cos - asset.center[1] * sin),
            z + scale * (asset.center[0] * sin + asset.center[1] * cos),
        ];
        let diameter = asset.diameter * scale;
        let stood = Stood {
            color: varied(
                asset,
                tint(self.origin[0] + x, self.origin[1] + z),
                self.variation,
            ),
            size: octaves(diameter),
            width: octaves(2.0 * radius),
        };
        let [columns, rows] = self.size;
        let [wide, deep] = self.cell;
        let step = 1.0 / SUB as f32;
        let mut stood_on = 0.0_f32;
        if radius < step * min(wide, deep) {
            let (column, row) = ((center[0] / wide).floor(), (center[1] / deep).floor());
            if column >= 0.0 && row >= 0.0 && (column as usize) < columns && (row as usize) < rows
            {
                let weight = std::f32::consts::PI * radius * radius / (wide * deep);
                let at = row as usize * columns + column as usize;
                self.stand(at, (weight, weight * top / CANOPY_TOP_OVER_MEAN), &stood);
                stood_on = weight;
            }
        } else {
            let first = |low: f32, cell: f32| max((low / cell).floor(), 0.0) as usize;
            let last = |high: f32, cell: f32, count: usize| {
                (max((high / cell).floor(), -1.0) as isize).min(count as isize - 1)
            };
            let inverse = 1.0 / (radius * radius);
            let share = step * step;
            for row in first(center[1] - radius, deep) as isize..=last(center[1] + radius, deep, rows)
            {
                for column in
                    first(center[0] - radius, wide) as isize..=last(center[0] + radius, wide, columns)
                {
                    let (mut weight, mut height) = (0.0_f32, 0.0_f32);
                    for b in 0..SUB {
                        let dz = (row as f32 + (b as f32 + 0.5) * step) * deep - center[1];
                        for a in 0..SUB {
                            let dx = (column as f32 + (a as f32 + 0.5) * step) * wide - center[0];
                            let within = 1.0 - (dx * dx + dz * dz) * inverse;
                            if within > 0.0 {
                                weight += share;
                                height += share * top * within;
                            }
                        }
                    }
                    if weight > 0.0 {
                        let at = row as usize * columns + column as usize;
                        self.stand(at, (weight, height), &stood);
                        stood_on += weight;
                    }
                }
            }
        }
        self.spanned[0] += f64::from(stood_on);
        self.spanned[1] += f64::from(stood_on * diameter);
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        let cells = self.covered.len();
        let surface = |at: usize| {
            let weight = self.covered[at];
            if weight > 0.0 {
                min(weight, 1.0) * self.heights[at] / weight
            } else {
                0.0
            }
        };
        let tallest = (0..cells).map(surface).fold(0.0_f32, max);
        let made = CanopyMade {
            size: [self.size[0] as u32, self.size[1] as u32],
            tallest,
            span: if self.spanned[0] > 0.0 {
                (self.spanned[1] / self.spanned[0]) as f32
            } else {
                0.0
            },
        };
        let mut bytes = Vec::with_capacity(size_of::<CanopyMade>() + 8 * cells);
        bytes.extend_from_slice(bytemuck::bytes_of(&made));
        for at in 0..cells {
            let weight = self.covered[at];
            let cover = min(weight, 1.0);
            let mean = if weight > 0.0 { 1.0 / weight } else { 0.0 };
            let [red, green, blue] =
                self.colors[at].map(|sum| unorm(pow(clamp(sum * mean, 0.0, 1.0), 0.5) * cover));
            bytes.extend_from_slice(&[red, green, blue, unorm(cover)]);
        }
        for at in 0..cells {
            let weight = self.covered[at];
            let cover = min(weight, 1.0);
            let share = if weight > 0.0 { cover / weight } else { 0.0 };
            let ranged = |sum: f32| {
                unorm((sum * share - CANOPY_SIZE_LOW * cover) / CANOPY_SIZE_OCTAVES)
            };
            let height = if tallest > 0.0 {
                surface(at) / tallest
            } else {
                0.0
            };
            bytes.extend_from_slice(&[
                unorm(height),
                unorm(cover),
                ranged(self.sizes[at]),
                ranged(self.widths[at]),
            ]);
        }
        bytes
    }
}
