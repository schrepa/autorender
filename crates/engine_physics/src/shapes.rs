use engine_places::physics::ColliderShape;
use glam::{Mat3, Quat, Vec3};
use rapier3d::prelude::{Pose, SharedShape};

use crate::solids::Solid;

const SQUARE: f32 = 1.0e-4;

pub(crate) struct BodyShape {
    pub shape: SharedShape,
    pub volume: f32,
    pub half: Vec3,
    pub round: bool,
}

fn sphere_points() -> impl Iterator<Item = Vec3> {
    (0..27).filter(|&index| index != 13).map(|index| {
        Vec3::new(
            (index % 3) as f32 - 1.0,
            ((index / 3) % 3) as f32 - 1.0,
            (index / 9) as f32 - 1.0,
        )
        .normalize()
    })
}

fn cube_points() -> impl Iterator<Item = Vec3> {
    (0..8).map(|corner| {
        Vec3::new(
            if corner & 1 == 0 { -1.0 } else { 1.0 },
            if corner & 2 == 0 { -1.0 } else { 1.0 },
            if corner & 4 == 0 { -1.0 } else { 1.0 },
        )
    })
}

fn uniform(radii: Vec3) -> bool {
    radii.max_element() - radii.min_element() <= SQUARE * radii.max_element()
}

fn ellipsoid(radii: Vec3) -> Option<SharedShape> {
    if uniform(radii) {
        return Some(SharedShape::ball(radii.x));
    }
    let points: Vec<Vec3> = sphere_points().map(|point| point * radii).collect();
    SharedShape::convex_hull(&points)
}

pub(crate) fn body_shape(shape: ColliderShape, scale: Vec3) -> Option<BodyShape> {
    let scale = scale.abs();
    if !scale.is_finite() || scale.min_element() < 1.0e-4 {
        return None;
    }
    match shape {
        ColliderShape::Cuboid { half_extents } => {
            let half = half_extents * scale;
            Some(BodyShape {
                shape: SharedShape::cuboid(half.x, half.y, half.z),
                volume: 8.0 * half.x * half.y * half.z,
                half,
                round: false,
            })
        }
        ColliderShape::Sphere { radius } => {
            let radii = Vec3::splat(radius) * scale;
            Some(BodyShape {
                shape: ellipsoid(radii)?,
                volume: 4.0 / 3.0 * std::f32::consts::PI * radii.x * radii.y * radii.z,
                half: radii,
                round: true,
            })
        }
    }
}

pub(crate) fn solid_shape(solid: &Solid) -> Option<(SharedShape, Pose)> {
    let columns = [solid.axes.x_axis, solid.axes.y_axis, solid.axes.z_axis];
    let lengths = columns.map(Vec3::length);
    if lengths
        .iter()
        .any(|length| !(length.is_finite() && *length > 1.0e-6))
    {
        return None;
    }
    let units = [0, 1, 2].map(|axis| columns[axis] / lengths[axis]);
    let square = units[0].dot(units[1]).abs() < SQUARE
        && units[0].dot(units[2]).abs() < SQUARE
        && units[1].dot(units[2]).abs() < SQUARE;
    let radii = Vec3::from_array(lengths);
    if square {
        let third = if units[0].cross(units[1]).dot(units[2]) < 0.0 {
            -units[2]
        } else {
            units[2]
        };
        let rotation = Quat::from_mat3(&Mat3::from_cols(units[0], units[1], third)).normalize();
        let shape = match solid.shape {
            ColliderShape::Cuboid { .. } => SharedShape::cuboid(radii.x, radii.y, radii.z),
            ColliderShape::Sphere { .. } => ellipsoid(radii)?,
        };
        return Some((shape, Pose::from_parts(solid.center, rotation)));
    }
    let points: Vec<Vec3> = match solid.shape {
        ColliderShape::Cuboid { .. } => cube_points().map(|point| solid.axes * point).collect(),
        ColliderShape::Sphere { .. } => sphere_points().map(|point| solid.axes * point).collect(),
    };
    Some((
        SharedShape::convex_hull(&points)?,
        Pose::from_translation(solid.center),
    ))
}
