//! The street plan the city and the metro share: their grid, their
//! budgets, the river row and street loops, and placing parts in frames.

use std::f32::consts::PI;

use engine::prelude::*;

use super::StressConfig;
use super::random::Rng;

/// Street grid: 20-unit blocks between 4-unit streets, centered on multiples
/// of `PERIOD`.
pub(crate) const PERIOD: f32 = 24.0;
pub(crate) const HALF_BLOCK: f32 = 10.0;

/// `frame` then `local`, for an unscaled `frame`.
pub(crate) fn compose(frame: &Transform, local: &Transform) -> Transform {
    Transform {
        translation: frame.translation + frame.rotation * local.translation,
        rotation: frame.rotation * local.rotation,
        scale: local.scale,
    }
}

pub(crate) fn part_at(translation: Vec3, scale: Vec3) -> Transform {
    Transform {
        translation,
        rotation: Quat::IDENTITY,
        scale,
    }
}

/// Moving and static renderables of `config`'s city, and its radius in
/// blocks of `per_block` static renderables each.
pub(crate) fn budget(config: &StressConfig, per_block: f32) -> (u32, u32, f32) {
    let moving = config.renderables / 100 * config.moving_percent
        + (config.renderables % 100).min(config.moving_percent);
    let dense = if config.dense_colliders {
        config.colliders
    } else {
        0
    };
    let static_total = config.renderables - moving - dense.min(config.renderables - moving);
    let radius = math::max(
        (math::max(static_total as f32, moving as f32 * 0.5) / per_block / PI).sqrt(),
        1.0,
    );
    (moving, static_total, radius)
}

/// The block row the river runs along, for cities wide enough to have one.
pub(crate) fn river_row(radius: f32) -> Option<i32> {
    (radius >= 4.0).then(|| (radius * 0.45).round() as i32)
}

/// A random street loop around one or two of `blocks` (any block within
/// `radius` when there are none): its center and half extents.
pub(crate) fn street_loop(blocks: &[(i32, i32, f32)], radius: f32, rng: &mut Rng) -> (Vec3, Vec2) {
    let (bx, bz) = if blocks.is_empty() {
        let reach = radius.ceil() as i32;
        (
            rng.below(2 * reach as u32 + 1) as i32 - reach,
            rng.below(2 * reach as u32 + 1) as i32 - reach,
        )
    } else {
        let (bx, bz, _) = blocks[rng.below(blocks.len() as u32) as usize];
        (bx, bz)
    };
    let (wide, deep) = (1 + rng.below(2), 1 + rng.below(2));
    let center = Vec3::new(
        bx as f32 * PERIOD + wide as f32 * PERIOD * 0.5,
        0.0,
        bz as f32 * PERIOD + deep as f32 * PERIOD * 0.5,
    );
    (
        center,
        Vec2::new(wide as f32 * PERIOD * 0.5, deep as f32 * PERIOD * 0.5),
    )
}
