use engine_math::{max, min};
use glam::{Mat3, Vec2, Vec3};

use crate::bed::{Bed, Reach};
use crate::grid::{TILE_QUADS, TILE_SAMPLES, TileKey};
use crate::hydrology::RIVER_FAR;
use crate::layers::Stamp;

pub use crate::grid::TILE_STRIDE;

pub const PATCH_QUADS: u32 = 16;
pub const PATCHES_PER_TILE: u32 = TILE_QUADS / PATCH_QUADS;

pub const RIVER_EDGE_STEP: f32 = 0.0625;
pub const RIVER_EDGE_ZERO: u16 = 32_768;
pub const RIVER_EDGE_FAR: u16 = u16::MAX;

const HOLE_CELLS: usize = TILE_SAMPLES as usize;

const SAMPLES: usize = (TILE_STRIDE * TILE_STRIDE) as usize;

const GRAIN: usize = 0;
const POWER: usize = 1;
const FLOOD: usize = 2;
const FORM: usize = 3;

const RATIO_STEP: f32 = 1.0 / 512.0;

pub fn shaded_elsewhere(level: u8, around: bool) -> bool {
    level > 0 && !around
}

pub fn river_meters(step: u16) -> f32 {
    if step == RIVER_EDGE_FAR {
        RIVER_FAR
    } else {
        (f32::from(step) - f32::from(RIVER_EDGE_ZERO)) * RIVER_EDGE_STEP
    }
}

fn from_half(code: u16) -> f32 {
    f32::from_bits(u32::from(code) << 16)
}

fn reach(form: u16) -> Reach {
    match form & 7 {
        0 => Reach::Pool,
        1 => Reach::Plane,
        2 => Reach::Steps,
        3 => Reach::Cascade,
        _ => Reach::Falls,
    }
}

pub fn bed_at(beds: &[u16], heights: &[f32], spacing: f32, local: Vec2) -> Option<Bed> {
    let (i, j, fu, fv) = tile_cell(spacing, local)?;
    let (x, z) = (i + i32::from(fu >= 0.5), j + i32::from(fv >= 0.5));
    let at = (z as usize + 1) * TILE_STRIDE as usize + x as usize + 1;
    let step = beds[FLOOD * SAMPLES + at];
    if step == RIVER_EDGE_FAR {
        return None;
    }
    let form = beds[FORM * SAMPLES + at];
    let d50 = from_half(beds[GRAIN * SAMPLES + at]);
    Some(Bed {
        reach: reach(form),
        d50,
        d84: d50 * (1.0 + f32::from(form >> 4) * RATIO_STEP),
        power: from_half(beds[POWER * SAMPLES + at]),
        flood: heights[at] + river_meters(step),
        carved: form & 8 != 0,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Solid {
    pub center: [f32; 3],
    pub half: [f32; 3],
    pub round: bool,
}

impl Solid {
    pub fn valid(&self) -> bool {
        self.center.iter().all(|value| value.is_finite())
            && self.half.iter().all(|&half| half.is_finite() && half > 0.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Footprint {
    pub center: [f32; 3],
    pub axes: [f32; 9],
    pub round: bool,
}

impl Footprint {
    fn axes(&self) -> Mat3 {
        Mat3::from_cols_array(&self.axes)
    }

    pub fn middle(&self) -> Vec2 {
        Vec2::new(self.center[0], self.center[2])
    }

    pub fn reach(&self) -> Vec2 {
        let rows = self.axes().transpose();
        let along = |row: Vec3| {
            if self.round {
                row.length()
            } else {
                row.abs().element_sum()
            }
        };
        Vec2::new(along(rows.x_axis), along(rows.z_axis))
    }

    pub fn span(&self, point: Vec2) -> Option<[f32; 2]> {
        let inverse = self.axes().inverse();
        let at = inverse * (Vec3::new(point.x, 0.0, point.y) - Vec3::from_array(self.center));
        let up = inverse.y_axis;
        if self.round {
            let rise = up.dot(up);
            let along = at.dot(up);
            let reach = along * along - rise * (at.dot(at) - 1.0);
            if !(rise > 0.0 && reach >= 0.0) {
                return None;
            }
            let root = reach.sqrt();
            return Some([(-along - root) / rise, (-along + root) / rise]);
        }
        let mut span = [f32::NEG_INFINITY, f32::INFINITY];
        for k in 0..3 {
            if up[k] != 0.0 {
                let (a, b) = ((-1.0 - at[k]) / up[k], (1.0 - at[k]) / up[k]);
                span = [max(span[0], min(a, b)), min(span[1], max(a, b))];
            } else if at[k].abs() > 1.0 {
                return None;
            }
        }
        (span[0] <= span[1]).then_some(span)
    }

    pub fn valid(&self) -> bool {
        self.center
            .iter()
            .chain(&self.axes)
            .all(|value| value.is_finite())
            && self.axes().determinant() != 0.0
    }
}

engine_codec::record!(Solid {
    center,
    half,
    round
});

engine_codec::record!(Footprint {
    center,
    axes,
    round
});

#[derive(Clone, Debug, PartialEq)]
pub struct TileJob {
    pub recipe: [u8; 32],
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
}

engine_codec::record!(TileJob {
    recipe,
    key,
    stamps,
});

impl TileJob {
    pub fn identity(&self) -> Vec<u8> {
        engine_codec::encode(self)
    }
}

pub fn hole_at(holes: &[u64], x: u32, z: u32) -> bool {
    if x as usize >= HOLE_CELLS || z as usize >= HOLE_CELLS {
        return false;
    }
    let index = z as usize * HOLE_CELLS + x as usize;
    holes[index / 64] & (1 << (index % 64)) != 0
}

pub fn tile_cell(spacing: f32, local: Vec2) -> Option<(i32, i32, f32, f32)> {
    let size = TILE_QUADS as f32;
    let u = local.x / spacing;
    let v = local.y / spacing;
    if !(u.is_finite() && v.is_finite()) || u < 0.0 || v < 0.0 || u > size || v > size {
        return None;
    }
    let i = (u.floor() as i32).min(TILE_QUADS as i32 - 1);
    let j = (v.floor() as i32).min(TILE_QUADS as i32 - 1);
    Some((i, j, u - i as f32, v - j as f32))
}

#[inline]
pub fn on_triangle([v00, v10, v01, v11]: [f32; 4], fu: f32, fv: f32) -> f32 {
    if fu + fv <= 1.0 {
        v00 + fu * (v10 - v00) + fv * (v01 - v00)
    } else {
        v11 + (1.0 - fu) * (v01 - v11) + (1.0 - fv) * (v10 - v11)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleShape {
    pub gradient: [f32; 2],
    pub curvature: f32,
}

impl SampleShape {
    pub fn of(around: [[f32; 3]; 3], spacing: f32) -> Self {
        let [[a, b, c], [d, e, f], [g, h, i]] = around;
        Self {
            gradient: smooth_gradient(around, spacing),
            curvature: (a + b + c + d + f + g + h + i - 8.0 * e) / (3.0 * spacing * spacing),
        }
    }
}

pub fn smooth_gradient(around: [[f32; 3]; 3], spacing: f32) -> [f32; 2] {
    let [[a, b, c], [d, _, f], [g, h, i]] = around;
    let across = 12.0 * spacing;
    [
        ((c - a) + 4.0 * (f - d) + (i - g)) / across,
        ((g - a) + 4.0 * (h - b) + (i - c)) / across,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Control {
    pub a: u8,
    pub b: u8,
    pub blend: u8,
    pub wet: u8,
}

impl Control {
    pub fn packed(self) -> u32 {
        u32::from(self.a)
            | u32::from(self.b) << 8
            | u32::from(self.blend) << 16
            | u32::from(self.wet) << 24
    }

    pub fn unpack(value: u32) -> Self {
        Self {
            a: value as u8,
            b: (value >> 8) as u8,
            blend: (value >> 16) as u8,
            wet: (value >> 24) as u8,
        }
    }
}

fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -25 {
        return sign;
    }
    let (full, shift, base) = if unbiased >= -14 {
        (mantissa, 13, ((unbiased + 15) as u32) << 10)
    } else {
        (mantissa | 0x80_0000, (-1 - unbiased) as u32, 0)
    };
    let kept = base | (full >> shift);
    let rest = full & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    let rounded = kept + u32::from(rest > half || (rest == half && (kept & 1) == 1));
    sign | rounded as u16
}

fn half_value(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from(bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x3ff);
    match exponent {
        0 => f32::from_bits(sign | (mantissa as f32 * (1.0 / 16_777_216.0)).to_bits()),
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

pub fn gradient_word([x, z]: [f32; 2]) -> u32 {
    u32::from(half_bits(x)) | u32::from(half_bits(z)) << 16
}

pub fn word_gradient(word: u32) -> [f32; 2] {
    [half_value(word as u16), half_value((word >> 16) as u16)]
}
