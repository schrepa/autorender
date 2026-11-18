use engine::prelude::*;
use glam::Mat3;

use super::Item;
use super::numbers::{EffectsNumbers, PreviewLook};
use crate::abilities::{DrawShape, Preview};
use crate::play::powers::Powers;

const TRACER: f32 = 0.0;
const RING: f32 = 6.0;
const FOOTPRINT: f32 = 7.0;

fn streak(look: &PreviewLook, color: [f32; 3], from: Vec3, to: Vec3) -> Item {
    Item {
        rows: [
            [from.x, from.y, from.z, 0.0],
            [to.x, to.y, to.z, 1.0],
            [color[0], color[1], color[2], look.width_m],
            [TRACER, 0.0, 1.0, 0.0],
            [0.0; 4],
            [1.0, 0.0, 0.0, 0.0],
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

fn shown(look: &PreviewLook, preview: &Preview, glows: &mut Vec<Item>, marks: &mut Vec<Item>) {
    let color = if preview.valid {
        look.valid
    } else {
        look.invalid
    };
    let every = look.every_points.max(1) as usize;
    let mut points: Vec<Vec3> = preview.path.iter().copied().step_by(every).collect();
    if let Some(last) = preview.path.last()
        && points.last() != Some(last)
    {
        points.push(*last);
    }
    for pair in points.windows(2) {
        glows.push(streak(look, color, pair[0], pair[1]));
    }
    match preview.shape {
        DrawShape::Sphere { centre, radius } => marks.push(mark(
            look,
            color,
            RING,
            centre,
            radius,
            Vec3::new(radius + look.rim_m, look.height_m, radius + look.rim_m),
            Mat3::IDENTITY,
        )),
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
        DrawShape::Segment { a, b } => glows.push(streak(look, color, a, b)),
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
