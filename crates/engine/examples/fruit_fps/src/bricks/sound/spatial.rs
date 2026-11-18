use engine::math::{clamp, pow, sin_cos};
use glam::Vec3;

use crate::play::world::Arena;

use super::table::Curve;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Listener {
    pub eye: Vec3,
    pub right: Vec3,
}

impl Listener {
    pub fn new(eye: Vec3, yaw_radians: f32) -> Listener {
        let (sin, cos) = sin_cos(yaw_radians);
        Listener {
            eye,
            right: Vec3::new(cos, 0.0, -sin),
        }
    }

    pub fn attenuation(&self, curve: &Curve, at: Vec3) -> f32 {
        if curve.flat {
            return 1.0;
        }
        let distance = at.distance(self.eye);
        let near = clamp(
            (curve.far_m - distance) / (curve.far_m - curve.near_m),
            0.0,
            1.0,
        );
        curve.floor + (1.0 - curve.floor) * pow(near, curve.power)
    }

    pub fn pan(&self, at: Vec3, width: f32) -> f32 {
        let mut toward = at - self.eye;
        toward.y = 0.0;
        let length = toward.length();
        if length <= f32::EPSILON {
            return 0.0;
        }
        clamp(toward.dot(self.right) / length, -1.0, 1.0) * width
    }

    pub fn occlusion(&self, arena: &Arena, curve: &Curve, at: Vec3, pullback_m: f32) -> f32 {
        if curve.flat || !arena.ready() {
            return 1.0;
        }
        let back = (self.eye - at).normalize_or_zero() * pullback_m;
        if arena.sees(self.eye, at + back) {
            1.0
        } else {
            curve.occluded
        }
    }
}
