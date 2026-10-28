use std::sync::Arc;

use engine_math::{clamp, max, sin_cos};

use crate::erosion::Erosion;
use crate::layers::Layers;
use crate::noise::{gradient_value, smooth};
use crate::{Generator, MacroCauses, PathSample, Pyramid, TILE_QUADS};

pub const MAX_MACRO_TEXELS: u32 = 4096;
pub const MACRO_BLOCK: u32 = 4;

const RELIEF_REACH: f32 = 160.0;
const RELIEF_RANGE: f32 = 60.0;

const SEED: u32 = 0x6d61_6372;

const SUN_TANGENT: f32 = 0.577_350_3;

const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];

const WET: [f32; 3] = [-1.0, -1.0, -1.0];
const SANDY: [f32; 3] = [1.15, 1.0, 0.6];
const MINERAL: [f32; 3] = [0.95, 1.0, 1.15];
const WEATHERED: [f32; 3] = [1.0, 1.02, 0.85];
const DEEP_SOIL: [f32; 3] = [-0.85, -1.0, -1.4];
const SUNNY: [f32; 3] = [1.2, 1.0, 0.45];
const MOSSY: [f32; 3] = [-1.2, -0.93, -1.2];
const GRAY: [f32; 3] = [1.0, 1.0, 1.0];
const WARM: [f32; 3] = [1.0, -0.1, -1.95];

const LUSH_MIDDLING: f32 = 0.5;
const LUSH_PER_SHADE: f32 = 1.0 / 0.6;

struct Shape {
    columns: usize,
    rows: usize,
    spacing: f32,
    heights: Vec<f32>,
    gradients: Vec<f32>,
    facings: Vec<f32>,
    reliefs: Vec<f32>,
}

impl Shape {
    fn of(layers: &Arc<Layers>, pyramid: &Pyramid, level: u32, heading: f32) -> Option<Self> {
        let [tiles_x, tiles_z] = pyramid.tiles(level);
        let quads = TILE_QUADS as usize;
        let columns = tiles_x as usize * quads + 1;
        let rows = tiles_z as usize * quads + 1;
        let mut heights = vec![0.0_f32; columns * rows];
        for key in pyramid.level_tiles(level) {
            let mut budget = u64::MAX;
            let tile = Generator::new(Arc::clone(layers), pyramid, key).advance(&mut budget)?;
            let first = [usize::from(key.x) * quads, usize::from(key.z) * quads];
            for z in 0..=quads {
                for x in 0..=quads {
                    heights[(first[1] + z) * columns + first[0] + x] =
                        tile.height(x as i32, z as i32);
                }
            }
        }
        let spacing = pyramid.spacing(level);
        let mut shape = Self {
            columns,
            rows,
            spacing,
            heights,
            gradients: Vec::new(),
            facings: Vec::new(),
            reliefs: Vec::new(),
        };
        let reach = max(RELIEF_REACH, 2.0 * spacing);
        let ring: Vec<(f32, f32)> = (0..8)
            .map(|index| sin_cos((index as f32 + 0.5) * (std::f32::consts::TAU / 8.0)))
            .collect();
        let (along_x, along_z) = {
            let (sin, cos) = sin_cos(heading.to_radians());
            (sin, -cos)
        };
        let mut gradients = Vec::with_capacity(columns * rows);
        let mut facings = Vec::with_capacity(columns * rows);
        let mut reliefs = Vec::with_capacity(columns * rows);
        for z in 0..rows {
            for x in 0..columns {
                let at = |dx: isize, dz: isize| shape.sample(x as isize + dx, z as isize + dz);
                let gx = (at(1, 0) - at(-1, 0)) / (2.0 * spacing);
                let gz = (at(0, 1) - at(0, -1)) / (2.0 * spacing);
                gradients.push((gx * gx + gz * gz).sqrt());
                facings.push(-(gx * along_x + gz * along_z));
                let (px, pz) = (x as f32 * spacing, z as f32 * spacing);
                let around: f32 = ring
                    .iter()
                    .map(|&(sin, cos)| shape.height(px + cos * reach, pz + sin * reach))
                    .sum();
                reliefs.push(at(0, 0) - around * 0.125);
            }
        }
        shape.gradients = gradients;
        shape.facings = facings;
        shape.reliefs = reliefs;
        Some(shape)
    }

    fn sample(&self, x: isize, z: isize) -> f32 {
        let x = x.clamp(0, self.columns as isize - 1) as usize;
        let z = z.clamp(0, self.rows as isize - 1) as usize;
        self.heights[z * self.columns + x]
    }

    fn between(&self, values: &[f32], x: f32, z: f32) -> f32 {
        let fx = clamp(x / self.spacing, 0.0, (self.columns - 1) as f32);
        let fz = clamp(z / self.spacing, 0.0, (self.rows - 1) as f32);
        let x0 = (fx as usize).min(self.columns - 2);
        let z0 = (fz as usize).min(self.rows - 2);
        let (tx, tz) = (fx - x0 as f32, fz - z0 as f32);
        let at = z0 * self.columns + x0;
        let next = at + self.columns;
        let near = values[at] + (values[at + 1] - values[at]) * tx;
        let far = values[next] + (values[next + 1] - values[next]) * tx;
        near + (far - near) * tz
    }

    fn height(&self, x: f32, z: f32) -> f32 {
        self.between(&self.heights, x, z)
    }
}

struct Ground {
    moisture: f32,
    shore: f32,
    deposit: f32,
    worn: f32,
    steep: f32,
    dust: f32,
    relief: f32,
    facing: f32,
    broad: f32,
    patchy: f32,
    warmth: f32,
}

fn ramp(value: f32, low: f32, high: f32) -> f32 {
    smooth((value - low) / (high - low))
}

fn patches(x: f32, z: f32, wavelength: f32, index: u32) -> f32 {
    let shift = index as f32 * 17.13;
    let value = gradient_value(
        x / wavelength + shift,
        z / wavelength - shift,
        SEED.wrapping_add(index.wrapping_mul(0x9e37_79b9)),
    );
    clamp(0.5 + 0.5 * value, 0.0, 1.0)
}

fn ground(
    layers: &Layers,
    (shape, change): (&Shape, f32),
    (x, z): (f32, f32),
    footprint: f32,
) -> Ground {
    let (sx, sz) = (x + layers.offset.x, z + layers.offset.y);
    let shown = layers.shown(sx, sz);
    let height = shape.height(x, z);
    let (mut moisture, mut shore) = (0.0, 0.0);
    if shown > 0.0
        && let Some(hydrology) = layers.hydrology()
    {
        let water = hydrology.masks_at(sx, sz, footprint, crate::hydrology::RIVER_FAR);
        moisture = max(
            ramp(water.wetness, 7.0, 14.0),
            0.8 * ramp(water.flow, 4.0, 6.0),
        ) * shown;
        if let Some(level) = hydrology.water.lake_beside(sx, sz) {
            shore = (1.0 - ramp(height - level, 0.0, 2.5)) * shown;
        }
    }
    let change = change * shown;
    let road = if shown > 0.0 {
        layers.path_at(sx, sz)
    } else {
        PathSample::FAR
    };
    let gradient = shape.between(&shape.gradients, x, z);
    Ground {
        moisture,
        shore,
        deposit: ramp(change, 0.0, 2.0),
        worn: ramp(-change, 0.0, 4.0),
        steep: ramp(gradient, 0.58, 1.19),
        dust: (1.0 - ramp(road.edge(), 0.0, 6.0)) * shown,
        relief: clamp(
            shape.between(&shape.reliefs, x, z) / RELIEF_RANGE,
            -1.0,
            1.0,
        ),
        facing: shape.between(&shape.facings, x, z),
        broad: 0.65 * patches(sx, sz, 340.0, 0) + 0.35 * patches(sx, sz, 130.0, 1),
        patchy: patches(sx, sz, 45.0, 2),
        warmth: patches(sx, sz, 600.0, 3),
    }
}

fn shaded(tint: [f32; 3], hue: [f32; 3], amount: f32) -> [f32; 3] {
    [0, 1, 2].map(|channel| tint[channel] * max(1.0 + amount * hue[channel], 0.0))
}

fn texel(ground: &Ground, causes: &MacroCauses) -> [u8; 4] {
    let aspect = clamp(ground.facing / SUN_TANGENT, -1.0, 1.0);
    let lighter = |noise: f32| 2.0 * noise - 1.0;
    let shades = [
        (WET, causes.wet * max(ground.moisture, ground.shore)),
        (DEEP_SOIL, causes.deposit * ground.deposit),
        (SANDY, causes.deposit * 0.6 * ground.shore),
        (MINERAL, causes.worn * ground.worn),
        (MINERAL, causes.steep * ground.steep),
        (WEATHERED, causes.relief * max(ground.relief, 0.0)),
        (DEEP_SOIL, causes.relief * max(-ground.relief, 0.0)),
        (SANDY, causes.dust * ground.dust),
        (SUNNY, causes.aspect * max(aspect, 0.0)),
        (MOSSY, causes.aspect * max(-aspect, 0.0)),
        (GRAY, causes.broad * lighter(ground.broad)),
        (GRAY, causes.patchy * lighter(ground.patchy)),
    ];
    let tint = shades.iter().fold([1.0_f32; 3], |tint, &(hue, amount)| {
        shaded(tint, hue, amount)
    });
    let brightness: f32 = LUMINANCE
        .iter()
        .zip(tint)
        .map(|(weight, channel)| weight * channel)
        .sum();
    let lush = clamp(
        LUSH_MIDDLING + LUSH_PER_SHADE * (1.0 - brightness),
        0.0,
        1.0,
    );
    let tint = shaded(tint, WARM, causes.warmth * lighter(ground.warmth));
    let byte = |value: f32| (clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8;
    let scaled = |channel: f32| byte(channel * (128.0 / 255.0));
    [
        scaled(tint[0]),
        scaled(tint[1]),
        scaled(tint[2]),
        byte(lush),
    ]
}

pub fn macro_colors(
    layers: &Arc<Layers>,
    pyramid: &Pyramid,
    level: u32,
    size: u32,
    causes: &MacroCauses,
) -> Option<Vec<u8>> {
    let shape = Shape::of(layers, pyramid, level, causes.heading)?;
    let extent = pyramid.extent();
    let step = [extent.x / size as f32, extent.y / size as f32];
    let footprint = max(step[0], step[1]);
    let erosions: Vec<&Erosion> = crate::surface::erosions(layers).collect();
    let count = size as usize;
    let mut texels = Vec::with_capacity(count * count * 4);
    let mut changes = vec![0.0_f32; count];
    for z in 0..count {
        let pz = (z as f32 + 0.5) * step[1];
        changes.fill(0.0);
        let xs = (0..count).map(|x| (x as f32 + 0.5) * step[0] + layers.offset.x);
        for erosion in &erosions {
            erosion.change_row(xs.clone(), pz + layers.offset.y, footprint, |x, change| {
                changes[x] += change;
            });
        }
        for (x, &change) in changes.iter().enumerate() {
            let px = (x as f32 + 0.5) * step[0];
            let ground = ground(layers, (&shape, change), (px, pz), footprint);
            texels.extend(texel(&ground, causes));
        }
    }
    Some(texels)
}
