use std::collections::BTreeSet;

use engine_core::{Projection, ViewFacts};
use glam::{Mat4, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub fn nearest_squared([low, high]: [Vec3; 2], eye: Vec3) -> f32 {
    eye.max(low).min(high).distance_squared(eye)
}

pub fn outside(view_projection: Mat4, [low, high]: [Vec3; 2]) -> bool {
    let mut beyond = [true; 5];
    for corner in 0..8_u32 {
        let point = Vec3::new(
            if corner & 1 == 0 { low.x } else { high.x },
            if corner & 2 == 0 { low.y } else { high.y },
            if corner & 4 == 0 { low.z } else { high.z },
        );
        let clip = view_projection * point.extend(1.0);
        beyond[0] &= clip.x < -clip.w;
        beyond[1] &= clip.x > clip.w;
        beyond[2] &= clip.y < -clip.w;
        beyond[3] &= clip.y > clip.w;
        beyond[4] &= clip.w <= 0.0;
    }
    beyond.contains(&true)
}

pub fn pixel_scale(facts: &ViewFacts) -> Option<(f32, bool)> {
    let half = facts.pixels[3] as f32 * 0.5;
    match facts.lens.projection {
        Projection::Perspective {
            vertical_fov_radians,
        } => {
            let slope = (vertical_fov_radians * 0.5).tan();
            (slope > 0.0).then(|| (half / slope, false))
        }
        Projection::Fitted { clip } => Some((half * clip[1][1].abs(), false)),
        Projection::Orthographic { height } => (height > 0.0).then(|| (half * 2.0 / height, true)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DetailLimit {
    pub least_px: f32,
    pub overlap: f32,
    pub never_m: f32,
}

impl DetailLimit {
    pub fn checked(&self, path: &str) -> Result<(), String> {
        if !(self.least_px.is_finite() && self.least_px >= 0.0) {
            return Err(format!("{path}.least_px: {} is not 0 or more", self.least_px));
        }
        if !(0.0..1.0).contains(&self.overlap) {
            return Err(format!("{path}.overlap: {} is not within 0..1", self.overlap));
        }
        if !(self.never_m.is_finite() && self.never_m > 0.0) {
            return Err(format!("{path}.never_m: {} is not above 0", self.never_m));
        }
        Ok(())
    }
}

impl Default for DetailLimit {
    fn default() -> Self {
        Self {
            least_px: 0.0,
            overlap: 0.0,
            never_m: 1.0,
        }
    }
}

pub struct SizeSkip<'a, K> {
    pub limit: DetailLimit,
    pub held: Option<&'a mut BTreeSet<K>>,
}

impl<K: Ord + Copy> SizeSkip<'_, K> {
    pub fn shown(
        &mut self,
        (key, size): (K, f32),
        near_squared: f32,
        scale: Option<(f32, bool)>,
    ) -> bool {
        let limit = self.limit;
        let Some((scale, flat)) = scale else {
            return true;
        };
        if size >= limit.never_m || limit.least_px <= 0.0 || near_squared <= 0.0 {
            return true;
        }
        let pixels = if flat {
            size * scale
        } else {
            size * scale / near_squared.sqrt()
        };
        let was = self
            .held
            .as_deref()
            .is_some_and(|held| held.contains(&key));
        let least = if was {
            limit.least_px * (1.0 - limit.overlap)
        } else {
            limit.least_px
        };
        let shown = pixels >= least;
        if let Some(held) = self.held.as_deref_mut() {
            if shown {
                held.insert(key);
            } else {
                held.remove(&key);
            }
        }
        shown
    }
}
