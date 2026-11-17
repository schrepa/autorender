use engine_math::{max, min};
use glam::{Vec2, Vec3};

use crate::grid::TILE_SAMPLES;

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

pub fn lay_decks(bridges: &[Bridge], first: [i64; 2], spacing: f32, heights: &mut [f32]) {
    let side = TILE_SAMPLES as usize;
    let position = |sample: [i64; 2]| Vec2::new(sample[0] as f32, sample[1] as f32) * spacing;
    let corner = position(first);
    let last = Vec2::splat((side - 1) as f32);
    for bridge in bridges.iter().filter(|bridge| !bridge.tunnel) {
        let half = bridge.width * 0.5;
        for pair in bridge.deck.windows(2) {
            let (a, b) = (
                Vec2::new(pair[0].x, pair[0].z),
                Vec2::new(pair[1].x, pair[1].z),
            );
            let low = (a.min(b) - Vec2::splat(half) - corner) / spacing;
            let high = (a.max(b) + Vec2::splat(half) - corner) / spacing;
            if high.x < 0.0 || high.y < 0.0 || low.x > last.x || low.y > last.y {
                continue;
            }
            let low = low.floor().clamp(Vec2::ZERO, last);
            let high = high.ceil().clamp(Vec2::ZERO, last);
            let along = b - a;
            let run = max(along.length_squared(), 1.0e-6);
            for row in low.y as usize..=high.y as usize {
                for column in low.x as usize..=high.x as usize {
                    let point = position([first[0] + column as i64, first[1] + row as i64]);
                    let t = min(max((point - a).dot(along) / run, 0.0), 1.0);
                    if point.distance(a + along * t) <= half {
                        heights[row * side + column] = pair[0].y + (pair[1].y - pair[0].y) * t;
                    }
                }
            }
        }
    }
}
