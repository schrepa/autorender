use engine_core::valid_global_matrix;
use engine_places_simulation::physics::{
    ColliderShape, ColliderSolid, MAX_COLLIDER_EXTENT, MAX_COLLISION_COORDINATE,
    MIN_COLLIDER_EXTENT,
};
use engine_places_simulation::solids::Bounds;
use glam::{Mat3, Mat4, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Solid {
    pub shape: ColliderShape,
    pub center: Vec3,
    pub axes: Mat3,
    pub min: Vec3,
    pub max: Vec3,
}

impl Solid {
    pub(crate) fn new(shape: ColliderShape, matrix: Mat4) -> Option<Self> {
        if !placeable(shape, matrix) {
            return None;
        }
        let axes = Mat3::from_mat4(matrix) * Mat3::from_diagonal(shape.dimensions());
        Some(Self::shaped(shape, matrix.w_axis.truncate(), axes))
    }

    pub(crate) fn of(found: &ColliderSolid) -> Self {
        Self::shaped(found.shape, found.center, found.axes)
    }

    fn shaped(shape: ColliderShape, center: Vec3, axes: Mat3) -> Self {
        let bounds = Bounds::of_shape(shape, center, axes);
        Self {
            shape,
            center,
            axes,
            min: bounds.min,
            max: bounds.max,
        }
    }

    pub(crate) fn bounds(&self) -> Bounds {
        Bounds {
            min: self.min,
            max: self.max,
        }
    }
}

fn placeable(shape: ColliderShape, matrix: Mat4) -> bool {
    if shape.validate().is_err() {
        return false;
    }
    let position = matrix.w_axis.truncate();
    if !valid_global_matrix(matrix)
        || !(position.is_finite() && position.abs().max_element() <= MAX_COLLISION_COORDINATE)
    {
        return false;
    }
    let basis = Mat3::from_mat4(matrix) * Mat3::from_diagonal(shape.dimensions());
    let lengths = Vec3::new(
        basis.x_axis.length(),
        basis.y_axis.length(),
        basis.z_axis.length(),
    );
    if !lengths.is_finite()
        || lengths.min_element() < MIN_COLLIDER_EXTENT
        || lengths.max_element() > MAX_COLLIDER_EXTENT
    {
        return false;
    }
    let normalized = basis / lengths.max_element();
    let inverse = normalized.inverse();
    let row_sums = inverse.x_axis.abs() + inverse.y_axis.abs() + inverse.z_axis.abs();
    inverse.is_finite() && row_sums.max_element() <= 1000.0
}
