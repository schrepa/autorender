use engine_math::{clamp, pow};
use glam::Vec3;

use crate::table::Curve;
use crate::{Ear, Occluder};

impl Ear {
    pub(crate) fn attenuation(&self, curve: &Curve, at: Vec3) -> f32 {
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

    pub(crate) fn pan(&self, at: Vec3, width: f32) -> f32 {
        let mut toward = at - self.eye;
        toward.y = 0.0;
        let length = toward.length();
        if length <= f32::EPSILON {
            return 0.0;
        }
        clamp(toward.dot(self.right) / length, -1.0, 1.0) * width
    }

    pub(crate) fn occlusion(
        &self,
        occluder: &dyn Occluder,
        curve: &Curve,
        at: Vec3,
        pullback_m: f32,
    ) -> f32 {
        if curve.flat {
            return 1.0;
        }
        let back = (self.eye - at).normalize_or_zero() * pullback_m;
        if occluder.clear(self.eye, at + back) {
            1.0
        } else {
            curve.occluded
        }
    }
}
