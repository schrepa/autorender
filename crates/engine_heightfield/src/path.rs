use engine_math::{max, min};
use glam::{Vec2, Vec3};

pub const PATH_FAR: f32 = 1.0e9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathSample {
    pub distance: f32,
    pub half: f32,
}

impl PathSample {
    pub const FAR: Self = Self {
        distance: PATH_FAR,
        half: 0.0,
    };

    #[inline]
    pub fn edge(&self) -> f32 {
        self.distance - self.half
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bridge {
    pub deck: Vec<Vec3>,
    pub width: f32,
    pub piers: Vec<(Vec3, f32)>,
    pub tunnel: bool,
}

#[inline]
pub fn distance_to_segment(point: Vec2, a: Vec2, b: Vec2) -> (f32, f32) {
    let along = b - a;
    let length = along.length_squared();
    let t = if length > 0.0 {
        min(max((point - a).dot(along) / length, 0.0), 1.0)
    } else {
        0.0
    };
    (point.distance(a + along * t), t)
}
