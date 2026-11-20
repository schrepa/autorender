use engine::prelude::*;
use glam::Mat3;

use super::Item;
use super::numbers::{EffectsNumbers, PreviewLook};
use crate::abilities::{DrawShape, Preview};
use crate::play::powers::Powers;

const ARC: f32 = 4.0;
const DOT: f32 = 5.0;
const RING: f32 = 6.0;
const FOOTPRINT: f32 = 7.0;

const HAND_RIGHT_M: f32 = 0.24;
const HAND_DOWN_M: f32 = 0.16;
const START_OFFSET_M: f32 = 0.65;
const DASH_STEP_M: f32 = 0.65;
const DASH_FILL: f32 = 0.60;
const BLEND_MIN_M: f32 = 1.6;
const BLEND_MAX_M: f32 = 5.5;
const MAX_DASHES: usize = 64;
const LANDING_DOT_M: f32 = 0.14;

fn dash_item(width: f32, color: [f32; 3], from: Vec3, to: Vec3) -> Item {
    Item {
        rows: [
            [from.x, from.y, from.z, 0.0],
            [to.x, to.y, to.z, 1.0],
            [color[0], color[1], color[2], width],
            [ARC, 0.0, 1.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
        ],
    }
}

fn dot_item(color: [f32; 3], at: Vec3, size: f32) -> Item {
    Item {
        rows: [
            [at.x, at.y, at.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [color[0], color[1], color[2], size],
            [DOT, 0.0, 1.0, 0.0],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
            [0.0; 4],
        ],
    }
}

fn mark(
    look: &PreviewLook,
    color: [f32; 3],
    kind: f32,
    centre: Vec3,
    radius: f32,
    half: Vec3,
    axes: Mat3,
) -> Item {
    Item {
        rows: [
            [centre.x, centre.y, centre.z, radius],
            [half.x, half.y, half.z, 0.0],
            [color[0], color[1], color[2], 0.0],
            [kind, 0.0, 0.0, look.rim_m],
            [axes.x_axis.x, axes.x_axis.y, axes.x_axis.z, 0.0],
            [axes.y_axis.x, axes.y_axis.y, axes.y_axis.z, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0; 4],
        ],
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = engine::math::max(edge1 - edge0, 1.0e-4);
    let t = engine::math::clamp((x - edge0) / span, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn sample_path(path: &[Vec3], lengths: &[f32], target: f32) -> Vec3 {
    let last = path.len().saturating_sub(1);
    for i in 1..path.len() {
        if lengths[i] >= target {
            let span = engine::math::max(lengths[i] - lengths[i - 1], 1.0e-5);
            let share = engine::math::clamp((target - lengths[i - 1]) / span, 0.0, 1.0);
            return path[i - 1].lerp(path[i], share);
        }
    }
    path[last]
}

fn hand_shift(preview: &Preview) -> Vec3 {
    let forward = preview.aim.direction.try_normalize().unwrap_or(Vec3::NEG_Z);
    let helper = if forward.y.abs() < 0.95 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = forward.cross(helper).try_normalize().unwrap_or(Vec3::X);
    let up = right.cross(forward);
    right * HAND_RIGHT_M - up * HAND_DOWN_M
}

fn arc_dashes(look: &PreviewLook, color: [f32; 3], preview: &Preview, glows: &mut Vec<Item>) {
    if preview.path.len() < 2 {
        return;
    }
    let mut lengths = Vec::with_capacity(preview.path.len());
    lengths.push(0.0_f32);
    for pair in preview.path.windows(2) {
        let prev = *lengths.last().unwrap_or(&0.0);
        lengths.push(prev + pair[0].distance(pair[1]));
    }
    let total = *lengths.last().unwrap_or(&0.0);
    let start = engine::math::min(START_OFFSET_M, total * 0.45);
    let span = total - start;
    if span <= 0.15 {
        return;
    }
    let shift = hand_shift(preview);
    let blend_end = engine::math::min(
        engine::math::clamp(total * 0.65, BLEND_MIN_M, BLEND_MAX_M),
        total,
    );
    let step = engine::math::max(span / MAX_DASHES as f32, DASH_STEP_M);
    let at_distance = |dist: f32| {
        let keep = 1.0 - smoothstep(start, blend_end, dist);
        sample_path(&preview.path, &lengths, dist) + shift * keep
    };
    let mut dist = start;
    while dist < total {
        let end = engine::math::min(dist + step * DASH_FILL, total);
        if end - dist > 0.05 {
            let mid = 0.5 * (dist + end);
            let width = look.width_m * (0.4 + 0.6 * smoothstep(start, start + 2.5, mid));
            glows.push(dash_item(width, color, at_distance(dist), at_distance(end)));
        }
        dist += step;
    }
}

fn shown(look: &PreviewLook, preview: &Preview, glows: &mut Vec<Item>, marks: &mut Vec<Item>) {
    let color = if preview.valid {
        look.valid
    } else {
        look.invalid
    };
    arc_dashes(look, color, preview, glows);
    match preview.shape {
        DrawShape::Sphere { centre, radius } => {
            glows.push(dot_item(color, centre, LANDING_DOT_M));
            marks.push(mark(
                look,
                color,
                RING,
                centre,
                radius,
                Vec3::new(radius + look.rim_m, look.height_m, radius + look.rim_m),
                Mat3::IDENTITY,
            ));
        }
        DrawShape::Box {
            centre,
            half_extents,
            axes,
        } => marks.push(mark(
            look,
            color,
            FOOTPRINT,
            centre,
            half_extents.max_element(),
            half_extents + Vec3::splat(look.rim_m),
            axes,
        )),
        DrawShape::Segment { a, b } => glows.push(dash_item(look.width_m, color, a, b)),
    }
}

pub(super) fn preview_items(
    look: &EffectsNumbers,
    powers: Option<&Powers>,
    viewer: Option<u32>,
    glows: &mut Vec<Item>,
    marks: &mut Vec<Item>,
) -> u32 {
    let (Some(preview), Some(powers), Some(viewer)) = (look.preview.as_ref(), powers, viewer)
    else {
        return 0;
    };
    let mut count = 0;
    for held in powers.held.iter().filter(|held| held.slot == viewer) {
        if let Some(preview_shape) = held.preview.as_ref() {
            shown(preview, preview_shape, glows, marks);
            count += 1;
        }
    }
    count
}
