use glam::Vec2;

pub use engine_tiles::{MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES, TileKey};

use crate::layers::Stamp;

pub const TILE_STRIDE: u32 = TILE_SAMPLES + 2;

pub fn reaches(stamp: &Stamp, pyramid: &Pyramid, key: TileKey) -> bool {
    let center = Vec2::from(stamp.center);
    let radius = Vec2::splat(stamp.radius + stamp.falloff);
    covers((center - radius, center + radius), pyramid, key)
}

pub fn reaches_span(stamp: &Stamp, pyramid: &Pyramid, (first, last): (TileKey, TileKey)) -> bool {
    let center = Vec2::from(stamp.center);
    let radius = Vec2::splat(stamp.radius + stamp.falloff);
    let (low, high) = (center - radius, center + radius);
    let near = tile_span(pyramid, first).0;
    let far = tile_span(pyramid, last).1;
    near.x <= high.x && near.y <= high.y && far.x >= low.x && far.y >= low.y
}

pub fn covers((low, high): (Vec2, Vec2), pyramid: &Pyramid, key: TileKey) -> bool {
    let (first, last) = tile_span(pyramid, key);
    first.x <= high.x && first.y <= high.y && last.x >= low.x && last.y >= low.y
}

fn tile_span(pyramid: &Pyramid, key: TileKey) -> (Vec2, Vec2) {
    let spacing = pyramid.spacing(u32::from(key.level));
    let size = TILE_QUADS as f32 * spacing;
    let first = Vec2::new(f32::from(key.x), f32::from(key.z)) * size - Vec2::splat(2.0 * spacing);
    (first, first + Vec2::splat(size + 4.0 * spacing))
}
