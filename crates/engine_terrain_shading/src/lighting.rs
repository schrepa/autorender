use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, TAU};

use engine_math::{clamp, max, min, sin_cos};

use crate::SampleShape;
use crate::codec::{self, Codec, Error, Reader, Writer};

pub const DIRECTIONS: usize = 16;

pub const SKY_PLANES: usize = 4;

pub const HORIZON_PLANES: usize = 2 * DIRECTIONS + 1;

pub const SKY_LAYERS: usize = 1;

pub const HORIZON_LAYERS: usize = DIRECTIONS / 2 + 1;

const SLOPE_TOLERANCE: f32 = 1.0e-4;

const FARTHEST: f32 = 6.0e4;
const NO_CREST: f32 = -6.0e4;

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainLighting {
    pub sky_size: u32,
    sky: Box<[u8]>,
    pub horizon_size: u32,
    horizons: Box<[u16]>,
    pub lowest: f32,
}

impl Codec for TerrainLighting {
    fn write(&self, out: &mut Writer) {
        out.put(&self.sky_size).put(&self.horizon_size);
        codec::put_planes(out, &self.sky, self.sky_size as usize);
        codec::put_wide_planes(out, &self.horizons, self.horizon_size as usize);
        out.put(&self.lowest);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, Error> {
        let sky_size: u32 = input.get()?;
        let horizon_size: u32 = input.get()?;
        let sky = codec::get_planes(input, sky_size as usize)?;
        let horizons = codec::get_wide_planes(input, horizon_size as usize)?;
        let area = |size: u32| (size as usize) * (size as usize);
        if sky.len() != SKY_PLANES * area(sky_size)
            || horizons.len() != HORIZON_PLANES * area(horizon_size)
        {
            return Err(codec::invalid("$.lighting"));
        }
        Ok(Self {
            sky_size,
            sky,
            horizon_size,
            horizons,
            lowest: input.get()?,
        })
    }
}

impl TerrainLighting {
    pub fn sky_layers(&self) -> Vec<u8> {
        let plane = |index: usize| Some(plane(&self.sky, self.sky_size, index));
        let layers = [[plane(0), plane(1), plane(2), plane(3)]];
        let mut texels = Vec::with_capacity(layers.len() * 4 * self.sky.len() / SKY_PLANES);
        for channels in &layers {
            interleave(channels, 255, &mut texels, |value| [value]);
        }
        texels
    }

    pub fn horizon_layers(&self) -> Vec<u8> {
        let plane = |index: usize| Some(plane(&self.horizons, self.horizon_size, index));
        let mut texels =
            Vec::with_capacity(HORIZON_LAYERS * 8 * self.horizons.len() / HORIZON_PLANES);
        for pair in 0..DIRECTIONS / 2 {
            let (first, second) = (2 * pair, 2 * pair + 1);
            let channels = [
                plane(first),
                plane(second),
                plane(DIRECTIONS + first),
                plane(DIRECTIONS + second),
            ];
            interleave(&channels, 0, &mut texels, u16::to_le_bytes);
        }
        let ground = [plane(2 * DIRECTIONS), None, None, None];
        interleave(&ground, 0, &mut texels, u16::to_le_bytes);
        texels
    }
}

fn plane<T>(planes: &[T], size: u32, index: usize) -> &[T] {
    let area = (size as usize) * (size as usize);
    &planes[index * area..(index + 1) * area]
}

fn interleave<T: Copy, const N: usize>(
    channels: &[Option<&[T]>; 4],
    fill: T,
    texels: &mut Vec<u8>,
    bytes: impl Fn(T) -> [u8; N],
) {
    let Some(area) = channels.iter().flatten().map(|plane| plane.len()).next() else {
        return;
    };
    for texel in 0..area {
        for plane in channels {
            texels.extend(bytes(plane.map_or(fill, |plane| plane[texel])));
        }
    }
}

pub struct Ground<'a> {
    pub samples: [u32; 2],
    pub spacing: f32,
    pub heights: &'a [f32],
}

impl Ground<'_> {
    pub(crate) fn height(&self, x: f32, z: f32) -> f32 {
        let [columns, rows] = self.samples.map(|count| count as usize);
        let fx = clamp(x / self.spacing, 0.0, (columns - 1) as f32);
        let fz = clamp(z / self.spacing, 0.0, (rows - 1) as f32);
        let x0 = (fx as usize).min(columns - 2);
        let z0 = (fz as usize).min(rows - 2);
        let (tx, tz) = (fx - x0 as f32, fz - z0 as f32);
        let at = z0 * columns + x0;
        let (a, b) = (self.heights[at], self.heights[at + 1]);
        let (c, d) = (self.heights[at + columns], self.heights[at + columns + 1]);
        let near = a + (b - a) * tx;
        let far = c + (d - c) * tx;
        near + (far - near) * tz
    }
}

struct Sweep {
    along: [f32; 2],
    across: [f32; 2],
    start: [f32; 2],
    step: f32,
    size: [usize; 2],
    slopes: Vec<f32>,
    rises: Vec<f32>,
    distances: Vec<f32>,
}

struct Between {
    first: usize,
    along: f32,
    across: f32,
}

impl Sweep {
    fn new(ground: &Ground<'_>, extent: [f32; 2], direction: usize, step: f32) -> Self {
        let (sin, cos) = sin_cos(direction as f32 * (TAU / DIRECTIONS as f32));
        let along = [cos, sin];
        let across = [-sin, cos];
        let corners = [
            [0.0, 0.0],
            [extent[0], 0.0],
            [0.0, extent[1]],
            [extent[0], extent[1]],
        ];
        let project = |axis: [f32; 2]| {
            corners
                .iter()
                .fold([f32::INFINITY, f32::NEG_INFINITY], |[low, high], corner| {
                    let at = corner[0] * axis[0] + corner[1] * axis[1];
                    [min(low, at), max(high, at)]
                })
        };
        let [along_low, along_high] = project(along);
        let [across_low, across_high] = project(across);
        let start = [along_low - step, across_low - step];
        let size = [
            ((along_high - along_low) / step).ceil() as usize + 2,
            ((across_high - across_low) / step).ceil() as usize + 2,
        ];
        let mut sweep = Self {
            along,
            across,
            start,
            step,
            size,
            slopes: vec![0.0; size[0] * size[1]],
            rises: vec![0.0; size[0] * size[1]],
            distances: vec![0.0; size[0] * size[1]],
        };
        let mut heights = vec![0.0_f32; size[0]];
        let mut hull: Vec<(f32, f32)> = Vec::with_capacity(size[0]);
        for line in 0..size[1] {
            let b = start[1] + (line as f32 + 0.5) * step;
            for (sample, height) in heights.iter_mut().enumerate() {
                let a = start[0] + (sample as f32 + 0.5) * step;
                *height = ground.height(along[0] * a + across[0] * b, along[1] * a + across[1] * b);
            }
            hull.clear();
            let row = line * size[0];
            for sample in (0..size[0]).rev() {
                let a = start[0] + (sample as f32 + 0.5) * step;
                let height = heights[sample];
                while hull.len() >= 2 {
                    let (near_at, near_height) = hull[hull.len() - 1];
                    let (far_at, far_height) = hull[hull.len() - 2];
                    let (near, far) = (near_at - a, far_at - a);
                    if (near_height - height) * far
                        <= (far_height - height) * near + SLOPE_TOLERANCE * near * far
                    {
                        hull.pop();
                    } else {
                        break;
                    }
                }
                let (rise, distance) = match hull.last() {
                    Some(&(crest_at, crest_height)) => (crest_height - height, crest_at - a),
                    None => (NO_CREST, 0.0),
                };
                sweep.slopes[row + sample] = if distance > 0.0 {
                    rise / distance
                } else {
                    NO_CREST
                };
                sweep.rises[row + sample] = rise;
                sweep.distances[row + sample] = distance;
                hull.push((a, height));
            }
        }
        sweep
    }

    fn between(&self, x: f32, z: f32) -> Between {
        let a = x * self.along[0] + z * self.along[1];
        let b = x * self.across[0] + z * self.across[1];
        let [columns, rows] = self.size;
        let fa = clamp(
            (a - self.start[0]) / self.step - 0.5,
            0.0,
            (columns - 2) as f32,
        );
        let fb = clamp(
            (b - self.start[1]) / self.step - 0.5,
            0.0,
            (rows - 2) as f32,
        );
        let (a0, b0) = (fa as usize, fb as usize);
        Between {
            first: b0 * columns + a0,
            along: fa - a0 as f32,
            across: fb - b0 as f32,
        }
    }

    fn mix(&self, values: &[f32], place: &Between) -> f32 {
        let at = place.first;
        let next = at + self.size[0];
        let near = values[at] + (values[at + 1] - values[at]) * place.along;
        let far = values[next] + (values[next + 1] - values[next]) * place.along;
        near + (far - near) * place.across
    }
}

#[derive(Clone, Copy, Default)]
struct SkyTexel {
    normal: [f32; 3],
    seen: f32,
    open: f32,
    bent: [f32; 3],
}

pub fn light(
    ground: &Ground<'_>,
    extent: [f32; 2],
    sky_size: u32,
    horizon_size: u32,
) -> TerrainLighting {
    let sky_count = sky_size as usize;
    let horizon_count = horizon_size as usize;
    let sky_step = [extent[0] / sky_size as f32, extent[1] / sky_size as f32];
    let horizon_step = [
        extent[0] / horizon_size as f32,
        extent[1] / horizon_size as f32,
    ];
    let center = |step: [f32; 2], x: usize, z: usize| {
        ((x as f32 + 0.5) * step[0], (z as f32 + 0.5) * step[1])
    };
    let finest = min(
        min(sky_step[0], sky_step[1]),
        min(horizon_step[0], horizon_step[1]),
    );
    let mut sky = vec![0_u8; SKY_PLANES * sky_count * sky_count];
    let sky_area = sky_count * sky_count;
    let mut texels = vec![SkyTexel::default(); sky_area];
    let across = max(sky_step[0], sky_step[1]);
    for z in 0..sky_count {
        for x in 0..sky_count {
            let (px, pz) = center(sky_step, x, z);
            let around = [-1.0, 0.0, 1.0].map(|dz| {
                [-1.0, 0.0, 1.0].map(|dx| ground.height(px + dx * across, pz + dz * across))
            });
            let [gx, gz] = SampleShape::of(around, across).gradient;
            texels[z * sky_count + x].normal = normalized([-gx, 1.0, -gz]);
        }
    }
    let horizon_area = horizon_count * horizon_count;
    let mut horizons = vec![0_u16; HORIZON_PLANES * horizon_area];
    for direction in 0..DIRECTIONS {
        let sweep = Sweep::new(ground, extent, direction, finest);
        let (sin, cos) = sin_cos(direction as f32 * (TAU / DIRECTIONS as f32));
        for z in 0..sky_count {
            for x in 0..sky_count {
                let (px, pz) = center(sky_step, x, z);
                let slope = sweep.mix(&sweep.slopes, &sweep.between(px, pz));
                accumulate(&mut texels[z * sky_count + x], [cos, sin], slope);
            }
        }
        for z in 0..horizon_count {
            for x in 0..horizon_count {
                let (px, pz) = center(horizon_step, x, z);
                let place = sweep.between(px, pz);
                let rise = sweep.mix(&sweep.rises, &place);
                let distance = sweep.mix(&sweep.distances, &place);
                let scale = if distance > FARTHEST {
                    FARTHEST / distance
                } else {
                    1.0
                };
                let at = z * horizon_count + x;
                horizons[direction * horizon_area + at] = half(max(rise * scale, NO_CREST));
                horizons[(DIRECTIONS + direction) * horizon_area + at] = half(distance * scale);
            }
        }
    }
    for (at, texel) in texels.iter().enumerate() {
        sky[at] = unorm(texel.seen / max(texel.open, 1.0e-6));
        let bent = normalized(texel.bent);
        for axis in 0..3 {
            sky[(1 + axis) * sky_area + at] = unorm(0.5 + 0.5 * (bent[axis] - texel.normal[axis]));
        }
    }
    let mut grounds = Vec::with_capacity(horizon_area);
    for z in 0..horizon_count {
        for x in 0..horizon_count {
            let (px, pz) = center(horizon_step, x, z);
            grounds.push(ground.height(px, pz));
        }
    }
    let lowest = grounds
        .iter()
        .fold(f32::INFINITY, |low, &height| min(low, height));
    for (at, &height) in grounds.iter().enumerate() {
        horizons[2 * DIRECTIONS * horizon_area + at] = half(height - lowest);
    }
    TerrainLighting {
        sky_size,
        sky: sky.into_boxed_slice(),
        horizon_size,
        horizons: horizons.into_boxed_slice(),
        lowest,
    }
}

fn accumulate(texel: &mut SkyTexel, direction: [f32; 2], slope: f32) {
    let [x, up, z] = texel.normal;
    let along = x * direction[0] + z * direction[1];
    let plane = -along / up;
    texel.open += slice(elevation(plane), along, up);
    let lowest = elevation(max(slope, plane));
    texel.seen += slice(lowest, along, up);
    let (_, sin, cos) = lowest;
    let (sin3, cos3) = (sin * sin * sin, cos * cos * cos);
    let level = up * cos3 / 3.0 + along * (2.0 / 3.0 - sin + sin3 / 3.0);
    let vertical = up * (1.0 - sin3) / 3.0 + along * cos3 / 3.0;
    texel.bent[0] += level * direction[0];
    texel.bent[1] += vertical;
    texel.bent[2] += level * direction[1];
}

fn slice((lowest, sin, cos): (f32, f32, f32), along: f32, up: f32) -> f32 {
    along * (FRAC_PI_4 - 0.5 * lowest - 0.5 * sin * cos) + up * 0.5 * (1.0 - sin * sin)
}

fn elevation(tangent: f32) -> (f32, f32, f32) {
    let cos = 1.0 / (1.0 + tangent * tangent).sqrt();
    (arctangent(tangent), tangent * cos, cos)
}

fn arctangent(x: f32) -> f32 {
    let wide = x.abs() > 1.0;
    let t = if wide { 1.0 / x } else { x };
    let t2 = t * t;
    let near = t
        * (0.999_866
            + t2 * (-0.330_299_5 + t2 * (0.180_141 + t2 * (-0.085_133 + t2 * 0.020_835_1))));
    if !wide {
        near
    } else if x > 0.0 {
        FRAC_PI_2 - near
    } else {
        -FRAC_PI_2 - near
    }
}

pub(crate) fn normalized(vector: [f32; 3]) -> [f32; 3] {
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    let scale = 1.0 / max(length, 1.0e-12);
    vector.map(|value| value * scale)
}

pub(crate) fn unorm(value: f32) -> u8 {
    (clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8
}

fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if (bits >> 23) & 0xff == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let (kept, shift) = if exponent > 0 {
        ((exponent as u32) << 10 | mantissa >> 13, 13)
    } else if exponent >= -10 {
        let shift = (14 - exponent) as u32;
        ((mantissa | 0x80_0000) >> shift, shift)
    } else {
        return sign;
    };
    let whole = if exponent > 0 {
        mantissa
    } else {
        mantissa | 0x80_0000
    };
    let rest = whole & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    let rounded = kept + u32::from(rest > halfway || (rest == halfway && kept & 1 == 1));
    sign | rounded.min(0x7bff) as u16
}
