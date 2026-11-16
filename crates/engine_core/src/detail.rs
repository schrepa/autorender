use glam::{Mat4, Vec3};

use crate::{gpu_consts, math};

gpu_consts! {
    pub(crate) const DETAIL_FLOOR: f32 = 1.0e-30;
    pub(crate) const SEEN_FLOOR: f32 = 1.0e-6;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewDetail {
    pub eye: Vec3,
    pub pixel_angle: f32,
    pub texel: f32,
    pub pixel_error: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DetailLevel {
    pub level: u32,
    pub blend: f32,
}

impl ViewDetail {
    pub fn perspective(eye: Vec3, view_projection: Mat4, height: u32, pixel_error: f32) -> Self {
        let row = view_projection.row(1).truncate().length() * height.max(1) as f32;
        Self {
            eye,
            pixel_angle: 2.0 / math::max(row, 1.0e-12),
            texel: 0.0,
            pixel_error,
        }
    }

    pub fn orthographic(texel: f32, pixel_error: f32) -> Self {
        Self {
            eye: Vec3::ZERO,
            pixel_angle: 0.0,
            texel,
            pixel_error,
        }
    }

    pub fn across(&self, world: Vec3) -> f32 {
        (world - self.eye).length() * self.pixel_angle + self.texel
    }

    pub fn pixels(&self, length: f32, world: Vec3) -> f32 {
        length / self.across(world)
    }

    pub fn fits(&self, error: f32, world: Vec3) -> bool {
        error <= self.across(world) * self.pixel_error
    }

    pub fn handover(&self, error: f32, world: Vec3) -> f32 {
        band((self.across(world) * self.pixel_error / math::max(error, DETAIL_FLOOR)).log2() + 1.0)
    }

    pub fn reach(&self, length: f32) -> f32 {
        length / (self.pixel_angle * self.pixel_error)
    }

    #[must_use]
    pub fn scaled(self, factor: f32) -> Self {
        Self {
            pixel_error: self.pixel_error * factor,
            ..self
        }
    }
}

pub fn detail_level(span: f32, base: f32, levels: u32) -> DetailLevel {
    let coarsest = (levels.max(1) - 1) as f32;
    let ratio = math::max(span / math::max(base, DETAIL_FLOOR), DETAIL_FLOOR);
    let at = math::clamp(ratio.log2(), 0.0, coarsest);
    DetailLevel {
        level: at.floor() as u32,
        blend: band(at - at.floor()),
    }
}

pub fn kept_levels(levels: usize, room: usize) -> impl Iterator<Item = usize> {
    let stride = levels.div_ceil(room.max(1)).max(1);
    (0..levels.div_ceil(stride))
        .rev()
        .map(move |step| levels - 1 - step * stride)
}

pub fn band(at: f32) -> f32 {
    math::clamp(2.0 * at - 1.0, 0.0, 1.0)
}

pub fn seen_share(pixels: f32) -> f32 {
    math::clamp(1.0 + math::max(pixels, SEEN_FLOOR).log2(), 0.0, 1.0)
}

pub fn detail_wgsl() -> String {
    let mut text = String::from(WGSL);
    text.push_str(include_str!("detail.wgsl"));
    text
}
