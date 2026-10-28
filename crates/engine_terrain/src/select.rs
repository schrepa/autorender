use engine_math::{max, sin_cos};
use glam::{Vec2, Vec3};

use crate::{Pyramid, TileKey};

#[derive(Clone, Debug, PartialEq)]
pub struct LodRanges {
    ranges: Vec<f32>,
}

impl LodRanges {
    pub const MIN_SPACINGS: f32 = 96.0;

    pub fn new(
        pyramid: &Pyramid,
        pixel_error: f32,
        viewport_height: f32,
        vertical_fov: f32,
    ) -> Self {
        let (sin, cos) = sin_cos(0.5 * vertical_fov);
        let tangent = sin / max(cos, 1.0e-6);
        let spacings = max(
            viewport_height / max(2.0 * tangent * pixel_error, 1.0e-6),
            Self::MIN_SPACINGS,
        );
        let spacings = if spacings.is_finite() {
            spacings
        } else {
            Self::MIN_SPACINGS
        };
        Self {
            ranges: (0..pyramid.levels())
                .map(|level| spacings * pyramid.spacing(level))
                .collect(),
        }
    }

    pub fn range(&self, level: u32) -> f32 {
        self.ranges[level as usize]
    }

    pub fn levels(&self) -> u32 {
        self.ranges.len() as u32
    }
}

const UNKNOWN_GROUND: f32 = 1.0e6;

fn cut_out(cutout: Option<(Vec2, Vec2)>, low: Vec2, high: Vec2) -> bool {
    cutout.is_some_and(|(first, last)| {
        low.x >= first.x && low.y >= first.y && high.x <= last.x && high.y <= last.y
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WantedTile {
    pub key: TileKey,
    pub score: f32,
    pub drawn: bool,
    pub seen: bool,
}

pub struct WantedView<'a> {
    pub eye: Vec3,
    pub forward: Vec2,
    pub ground: &'a dyn Fn(TileKey) -> Option<[f32; 2]>,
    pub seen: &'a dyn Fn(Vec3, Vec3) -> bool,
}

pub fn wanted_tiles(
    pyramid: &Pyramid,
    ranges: &LodRanges,
    view: &WantedView<'_>,
    reach: f32,
    pinned: u32,
    cutout: Option<(Vec2, Vec2)>,
    out: &mut Vec<WantedTile>,
) {
    let start = out.len();
    let eye = view.eye;
    let eye2 = Vec2::new(eye.x, eye.z);
    let forward = view.forward.try_normalize().unwrap_or(Vec2::ZERO);
    for level in (0..pyramid.levels()).rev() {
        let size = pyramid.tile_size(level);
        let [tiles_x, tiles_z] = pyramid.tiles(level);
        let everywhere = level + pinned >= pyramid.levels();
        let radius = ranges.range(level) * reach;
        let (first, last) = if everywhere {
            ([0, 0], [tiles_x - 1, tiles_z - 1])
        } else {
            let low = ((eye2 - radius) / size).floor();
            let high = ((eye2 + radius) / size).floor();
            if high.x < 0.0 || high.y < 0.0 || low.x >= tiles_x as f32 || low.y >= tiles_z as f32 {
                continue;
            }
            (
                [max(low.x, 0.0) as u32, max(low.y, 0.0) as u32],
                [
                    (high.x as u32).min(tiles_x - 1),
                    (high.y as u32).min(tiles_z - 1),
                ],
            )
        };
        for z in first[1]..=last[1] {
            for x in first[0]..=last[0] {
                let low = Vec2::new(x as f32, z as f32) * size;
                let high = low + size;
                if cut_out(cutout, low, high) {
                    continue;
                }
                let key = TileKey::new(level as u8, x as u16, z as u16);
                let nearest = eye2.clamp(low, high);
                let ground = (view.ground)(key);
                let rise = ground.map_or(0.0, |[bottom, top]| {
                    max(max(bottom - eye.y, eye.y - top), 0.0)
                });
                let across = nearest.distance(eye2);
                let distance = (across * across + rise * rise).sqrt();
                if !everywhere && distance > radius {
                    continue;
                }
                let toward = ((low + high) * 0.5 - eye2)
                    .try_normalize()
                    .unwrap_or(forward);
                let behind = 1.0 - forward.dot(toward);
                let score = if everywhere {
                    -1.0
                } else {
                    distance / max(ranges.range(level), 1.0e-3) * (1.0 + behind)
                };
                let [bottom, top] = ground.unwrap_or([-UNKNOWN_GROUND, UNKNOWN_GROUND]);
                out.push(WantedTile {
                    key,
                    score,
                    drawn: distance <= ranges.range(level),
                    seen: (view.seen)(
                        Vec3::new(low.x, bottom, low.y),
                        Vec3::new(high.x, top, high.y),
                    ),
                });
            }
        }
    }
    out[start..].sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then(b.key.level.cmp(&a.key.level))
    });
}
