//! A terrain's lighting: what its ground gives light by its shape alone,
//! made once from its heights (the job [`crate::jobs::LightingJob`]) and
//! kept as a record, so nothing builds it while frames run. Nothing in it
//! depends on the sun or the sky: as the time of day moves the sun, its
//! long shadows are a lookup.
//!
//! Its sky map is a grid over the whole terrain holding at every texel the
//! share of the sky its ground sees about its normal (cosine-weighted) and
//! the turn of that normal toward the middle of the sky it sees (its bent
//! normal less the normal); the ground's large-scale variety is its macro
//! colors' ([`crate::macro_colors`]). Its horizon map is a grid over the
//! terrain holding at every texel, in each of [`DIRECTIONS`] directions
//! around, how high above its ground and how far away lies the crest of
//! the highest ground ahead, the horizon's angle above level, and the
//! ground's height: a surface sees the sun where the sun's ray toward it
//! clears that crest, by as many meters as the crest's distance times the
//! sun's rise per meter, less the crest's height above it. Heights and
//! distances interpolate between texels as the meters the sun clears
//! crests by would; between directions, the slopes up to the crests do.
//!
//! Horizons come from sweeps, one per direction: lines along it a step apart
//! cross the terrain, sampled a step apart, and walking each line from its
//! far end, the upper convex hull of the ground passed gives every sample its
//! horizon exactly, however far its crest lies; texels read the samples
//! around them. Beyond the terrain its edge carries on.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, TAU};

use engine_math::{clamp, max, min, sin_cos};

use crate::codec::{self, Codec, CodecError, Reader, Writer};

/// Directions around each texel horizons are found in: direction `k` points
/// `k / DIRECTIONS` of a turn from +x toward +z.
pub const DIRECTIONS: usize = 16;

/// Planes of the sky map, a byte per texel each: the sky seen and the
/// normal's turn toward the sky along x, y and z.
pub const SKY_PLANES: usize = 4;

/// Planes of the horizon map, half floats: each direction's crest height
/// above the ground, then each direction's crest distance (meters), then the
/// ground's height above the lowest.
pub const HORIZON_PLANES: usize = 2 * DIRECTIONS + 1;

/// RGBA8 layers the sky map takes on the GPU: one, the sky seen (red) and
/// the normal's turn (green, blue and alpha: x, y and z).
pub const SKY_LAYERS: usize = 1;

/// RGBA16 float layers the horizon map takes on the GPU: per pair of
/// directions `2l` and `2l + 1`, layer `l` holds their crests' heights above
/// the ground (red, green) and their crests' distances (blue, alpha); the
/// last holds the ground's height above the lowest (red).
pub const HORIZON_LAYERS: usize = DIRECTIONS / 2 + 1;

/// How much steeper a nearer point must rise than a farther one to be the
/// crest: points on one slope keep the farthest, as a surface above the
/// ground sees it.
const SLOPE_TOLERANCE: f32 = 1.0e-4;

/// The farthest crest distance half floats hold, and the crest height
/// standing in for no ground ahead at all.
const FARTHEST: f32 = 6.0e4;
const NO_CREST: f32 = -6.0e4;

/// A terrain's lighting: its sky map, `sky_size` texels a side, as planes
/// of bytes ([`SKY_PLANES`]), and its horizon map, `horizon_size` texels a
/// side, as planes of half floats ([`HORIZON_PLANES`]), both over the
/// terrain's extent; and the lowest ground's height (meters above the
/// terrain's origin), which the horizon map's heights count from.
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

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let sky_size: u32 = input.get()?;
        let horizon_size: u32 = input.get()?;
        let sky = codec::get_planes(input, sky_size as usize)?;
        let horizons = codec::get_wide_planes(input, horizon_size as usize)?;
        let area = |size: u32| (size as usize) * (size as usize);
        if sky.len() != SKY_PLANES * area(sky_size)
            || horizons.len() != HORIZON_PLANES * area(horizon_size)
        {
            return Err(CodecError("lighting maps of another size"));
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
    /// The sky map as the GPU takes it: [`SKY_LAYERS`] RGBA8 layers, rows
    /// along z.
    pub fn sky_layers(&self) -> Vec<u8> {
        let plane = |index: usize| Some(plane(&self.sky, self.sky_size, index));
        let layers = [[plane(0), plane(1), plane(2), plane(3)]];
        let mut texels = Vec::with_capacity(layers.len() * 4 * self.sky.len() / SKY_PLANES);
        for channels in &layers {
            interleave(channels, 255, &mut texels, |value| [value]);
        }
        texels
    }

    /// The horizon map as the GPU takes it: [`HORIZON_LAYERS`] RGBA16
    /// float layers, rows along z, little-endian.
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

/// Plane `index` of `planes`, `size` texels a side.
fn plane<T>(planes: &[T], size: u32, index: usize) -> &[T] {
    let area = (size as usize) * (size as usize);
    &planes[index * area..(index + 1) * area]
}

/// Appends a layer of four-channel texels to `texels`, each channel from its
/// plane (`fill` where it has none) in the bytes `bytes` gives it.
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

/// A terrain's ground as its lighting reads it: `samples` along x and z,
/// `spacing` meters apart, row by row along z.
pub struct Ground<'a> {
    pub samples: [u32; 2],
    pub spacing: f32,
    pub heights: &'a [f32],
}

impl Ground<'_> {
    /// The height at terrain-local `(x, z)`, interpolated between samples,
    /// held within the grid: beyond it its edge carries on.
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

/// One direction's horizons over a grid aligned with it: lines along the
/// direction `step` meters apart, sampled `step` meters apart, the first
/// sample of the first line at `start` (along and across the direction),
/// each sample's slope up to its crest, the crest's height above its ground
/// and its distance (0 without ground ahead).
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

/// Where a point lies between a sweep's samples: the first of the four
/// around it, and how far it lies from it along and across the direction.
struct Between {
    first: usize,
    along: f32,
    across: f32,
}

impl Sweep {
    /// Direction `direction`'s sweep over `ground`'s `extent`, `step` meters
    /// apart, a step wider on every side.
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
        // The upper hull of the ground passed: each point's place along
        // the line and its height, farthest first.
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
                // Points the new one sees no higher than the next drop out:
                // the nearest left is the crest of its horizon, the farthest
                // of those along one slope (within a rounding's tolerance).
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

    /// Where terrain-local `(x, z)` lies between its samples.
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

    /// `values` at `place`, interpolated between the samples around it.
    fn mix(&self, values: &[f32], place: &Between) -> f32 {
        let at = place.first;
        let next = at + self.size[0];
        let near = values[at] + (values[at + 1] - values[at]) * place.along;
        let far = values[next] + (values[next + 1] - values[next]) * place.along;
        near + (far - near) * place.across
    }
}

/// What a sky texel accumulates over the directions: the sky seen and the
/// sky its ground's own plane leaves (each cosine-weighted about its
/// normal), and the sum of the directions seen, cosine-weighted.
#[derive(Clone, Copy, Default)]
struct SkyTexel {
    normal: [f32; 3],
    seen: f32,
    open: f32,
    bent: [f32; 3],
}

/// The lighting of `ground` over `extent` (meters along x and z from its
/// first sample), with a sky map `sky_size` and a horizon map
/// `horizon_size` texels a side.
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
    // Sweeps a texel of the finer map apart.
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
            // The ground's normal across a texel either way.
            texels[z * sky_count + x].normal = normalized([
                ground.height(px - across, pz) - ground.height(px + across, pz),
                2.0 * across,
                ground.height(px, pz - across) - ground.height(px, pz + across),
            ]);
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
                // Crests farther than half floats hold come nearer at
                // their slope.
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
    // The ground's height at each horizon texel, above the lowest.
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

/// Adds the sky of one direction's slice to `texel`: toward `direction`
/// (x and z), above both the horizon slope `slope` and the ground's own
/// plane, cosine-weighted about its normal. Elevations `e` from the slice's
/// lowest `e0` to the zenith hold the normal's cosine `up sin e + along cos
/// e`, weighted by `cos e`; the sky's share integrates that, and its middle
/// integrates the directions themselves the same way.
fn accumulate(texel: &mut SkyTexel, direction: [f32; 2], slope: f32) {
    let [x, up, z] = texel.normal;
    let along = x * direction[0] + z * direction[1];
    // The ground's own plane toward the direction, where the normal's
    // cosine falls to 0.
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

/// The sky of one direction's slice above the elevation `lowest` (its
/// angle, sine and cosine), cosine-weighted about a normal whose part along
/// the direction is `along` and whose upward part is `up`.
fn slice((lowest, sin, cos): (f32, f32, f32), along: f32, up: f32) -> f32 {
    along * (FRAC_PI_4 - 0.5 * lowest - 0.5 * sin * cos) + up * 0.5 * (1.0 - sin * sin)
}

/// The elevation rising `tangent` per meter: its angle (radians, within
/// about 1e-5), sine and cosine.
fn elevation(tangent: f32) -> (f32, f32, f32) {
    let cos = 1.0 / (1.0 + tangent * tangent).sqrt();
    (arctangent(tangent), tangent * cos, cos)
}

/// The arctangent of `x` within about 1e-5 radians: Abramowitz and
/// Stegun's polynomial (4.4.49) within -1 to 1, beyond from the
/// reciprocal's.
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

/// `value` within 0 to 1 as a byte, rounded.
pub(crate) fn unorm(value: f32) -> u8 {
    (clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8
}

/// The bits of the half float nearest `value` (ties to even), finite values
/// beyond the largest half float held to it.
fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if (bits >> 23) & 0xff == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    // The halves' mantissa before rounding, and the bits rounded away.
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
